import mujoco
import pinocchio as pin
import numpy as np
from numpy.linalg import norm, solve
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R, Slerp
from scipy.optimize import minimize


class FrankaPandaRobot:

    def __init__(self):
        # --- Differential IK Constants ---
        self.DAMPING = 1e-4
        self.K_POS = 3.0      # Increased for tighter tracking
        self.K_ORI = 1.0
        self.K_NULL = 1.0     # Reduced so it doesn't fight the main task
        self.MAX_ANGVEL = 2.0
        self.INTEGRATION_DT = 0.01 # Smaller step for more stability


    def solve_differential_ik(
        self,
        model,
        data,
        target_pos,
        target_rot_matrix,
        q0,
        ee_site_id,
        qpos_indices=None,
        dof_indices=None,
    ):
        """Solve differential IK for one or many end-effectors.

        Single-robot mode (backwards compatible):
            - ee_site_id: int
            - target_pos: (3,)
            - target_rot_matrix: (3,3)

        Batched mode:
            - ee_site_id: (N,) array-like of site ids
            - target_pos: (N,3)
            - target_rot_matrix: (N,3,3) or (3,3) (will be broadcast)
            - qpos_indices: (N,7)/(N,9) or flat indices reshaped by caller
            - dof_indices:  (N,7)/(N,9)

        Returns:
            - dq: (7,) in single mode, or (N,7) in batched mode
        """

        # Ensure kinematics are up-to-date before reading derived quantities.
        # (site_xpos/site_xmat are undefined until mj_forward/mj_step() has run.)
        mujoco.mj_forward(model, data)

        site_ids = np.asarray(ee_site_id)
        is_batched = site_ids.ndim != 0 and site_ids.size > 1

        # -------------------- Single mode --------------------
        if not is_batched:
            ee_site_id_int = int(site_ids)

            current_pos = data.site_xpos[ee_site_id_int]
            current_rot_matrix = data.site_xmat[ee_site_id_int].reshape(3, 3)

            det = float(np.linalg.det(current_rot_matrix))
            if not np.isfinite(det) or det <= 0.0:
                raise ValueError(
                    "End-effector site rotation matrix is invalid (det <= 0). "
                    "Did you set data.qpos and call mj_forward()/mj_step() before IK? "
                    f"det={det}, ee_site_id={ee_site_id_int}"
                )

            error_pos = np.asarray(target_pos, dtype=np.float64) - current_pos
            error_rot_mat = np.asarray(target_rot_matrix, dtype=np.float64) @ current_rot_matrix.T
            error_rot_vec = R.from_matrix(error_rot_mat).as_rotvec()

            twist = np.zeros(6, dtype=np.float64)
            twist[:3] = error_pos * self.K_POS
            twist[3:] = error_rot_vec * self.K_ORI

            jac_p = np.zeros((3, model.nv))
            jac_r = np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, jac_p, jac_r, ee_site_id_int)
            J_full = np.vstack([jac_p, jac_r])

            if dof_indices is None:
                J = J_full[:, :7]
            else:
                dof_idx = np.asarray(dof_indices, dtype=np.int32).reshape(-1)
                J = J_full[:, dof_idx[:7]]

            vv = J @ J.T
            diag_indices = np.diag_indices_from(vv)
            vv[diag_indices] += self.DAMPING
            dq_task = J.T @ np.linalg.solve(vv, twist)

            if qpos_indices is None:
                current_q = data.qpos[:7]
            else:
                qpos_idx = np.asarray(qpos_indices, dtype=np.int32).reshape(-1)
                current_q = data.qpos[qpos_idx[:7]]

            q0_arr = np.asarray(q0, dtype=np.float64).reshape(-1)
            null_error = (q0_arr[:7] - current_q)
            J_pinv = np.linalg.pinv(J, rcond=1e-2)
            dq_null = (np.eye(7) - J_pinv @ J) @ (self.K_NULL * null_error)

            return np.clip(dq_task + dq_null, -self.MAX_ANGVEL, self.MAX_ANGVEL)

        # -------------------- Batched mode --------------------
        site_ids = site_ids.astype(np.int32).reshape(-1)
        n = int(site_ids.size)

        if qpos_indices is None or dof_indices is None:
            raise ValueError(
                "Batched differential IK requires qpos_indices and dof_indices per robot "
                "so we can select the correct joint columns in the Jacobian."
            )

        qpos_idx = np.asarray(qpos_indices, dtype=np.int32)
        dof_idx = np.asarray(dof_indices, dtype=np.int32)

        if qpos_idx.ndim != 2 or dof_idx.ndim != 2:
            raise ValueError(
                f"Expected qpos_indices and dof_indices to be 2D arrays shaped (N, 7/9). "
                f"Got qpos_indices.shape={qpos_idx.shape}, dof_indices.shape={dof_idx.shape}."
            )
        if qpos_idx.shape[0] != n or dof_idx.shape[0] != n:
            raise ValueError(
                f"Batch size mismatch: site_ids has N={n} but qpos_indices has {qpos_idx.shape[0]} "
                f"and dof_indices has {dof_idx.shape[0]}."
            )
        if qpos_idx.shape[1] < 7 or dof_idx.shape[1] < 7:
            raise ValueError(
                f"Need at least 7 arm joints per robot. Got qpos_indices.shape={qpos_idx.shape}, "
                f"dof_indices.shape={dof_idx.shape}."
            )

        qpos_arm = qpos_idx[:, :7]
        dof_arm = dof_idx[:, :7]

        target_pos_arr = np.asarray(target_pos, dtype=np.float64)
        if target_pos_arr.shape == (3,):
            target_pos_arr = np.repeat(target_pos_arr[None, :], n, axis=0)
        if target_pos_arr.shape != (n, 3):
            raise ValueError(f"target_pos must be shape (N,3). Got {target_pos_arr.shape}.")

        target_rot_arr = np.asarray(target_rot_matrix, dtype=np.float64)
        if target_rot_arr.shape == (3, 3):
            target_rot_arr = np.repeat(target_rot_arr[None, :, :], n, axis=0)
        if target_rot_arr.shape != (n, 3, 3):
            raise ValueError(
                f"target_rot_matrix must be shape (3,3) or (N,3,3). Got {target_rot_arr.shape}."
            )

        # Current site poses for all robots.
        current_pos = data.site_xpos[site_ids]
        current_rot = data.site_xmat[site_ids].reshape(n, 3, 3)

        dets = np.linalg.det(current_rot)
        bad = np.where(~np.isfinite(dets) | (dets <= 0.0))[0]
        if bad.size:
            i0 = int(bad[0])
            raise ValueError(
                "End-effector site rotation matrix is invalid (det <= 0) for at least one robot. "
                f"Example: batch_index={i0}, ee_site_id={int(site_ids[i0])}, det={float(dets[i0])}."
            )

        error_pos = target_pos_arr - current_pos
        # error_rot = target_rot @ current_rot.T (batched)
        error_rot = np.einsum("nij,njk->nik", target_rot_arr, np.transpose(current_rot, (0, 2, 1)))
        error_rot_vec = R.from_matrix(error_rot).as_rotvec()

        twist = np.zeros((n, 6), dtype=np.float64)
        twist[:, :3] = error_pos * self.K_POS
        twist[:, 3:] = error_rot_vec * self.K_ORI

        q0_arr = np.asarray(q0, dtype=np.float64)
        if q0_arr.ndim == 1:
            # Allow a flattened (9N,) or (7N,) vector.
            if q0_arr.size % n != 0:
                raise ValueError(
                    f"q0 length {q0_arr.size} is not divisible by batch size N={n}."
                )
            q0_arr = q0_arr.reshape(n, -1)
        if q0_arr.shape[0] != n or q0_arr.shape[1] < 7:
            raise ValueError(f"q0 must be shape (N,>=7). Got {q0_arr.shape}.")
        q0_arm = q0_arr[:, :7]

        dq_out = np.zeros((n, 7), dtype=np.float64)
        for i in range(n):
            jac_p = np.zeros((3, model.nv))
            jac_r = np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, jac_p, jac_r, int(site_ids[i]))
            J_full = np.vstack([jac_p, jac_r])
            J = J_full[:, dof_arm[i]]

            vv = J @ J.T
            diag_indices = np.diag_indices_from(vv)
            vv[diag_indices] += self.DAMPING
            dq_task = J.T @ np.linalg.solve(vv, twist[i])

            current_q = data.qpos[qpos_arm[i]]
            null_error = (q0_arm[i] - current_q)
            J_pinv = np.linalg.pinv(J, rcond=1e-2)
            dq_null = (np.eye(7) - J_pinv @ J) @ (self.K_NULL * null_error)

            dq_out[i] = np.clip(dq_task + dq_null, -self.MAX_ANGVEL, self.MAX_ANGVEL)

        return dq_out