import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R



# --- Differential IK Constants ---
DAMPING = 1e-4
K_POS = 3.0      # Increased for tighter tracking
K_ORI = 1.0
K_NULL = 1.0     # Reduced so it doesn't fight the main task
MAX_ANGVEL = 2.0
INTEGRATION_DT = 0.1 # Smaller step for more stability


def solve_differential_ik(model, data, target_pos, target_rot_matrix, q0, ee_site_id):
    # 1. Get Current Site State (TCP)
    current_pos = data.site_xpos[ee_site_id]
    current_rot_matrix = data.site_xmat[ee_site_id].reshape(3, 3)

    # 2. Calculate Error (Twist)
    error_pos = target_pos - current_pos
    
    # Orientation error
    error_rot_mat = target_rot_matrix @ current_rot_matrix.T
    error_rot_vec = R.from_matrix(error_rot_mat).as_rotvec()

    twist = np.zeros(6)
    twist[:3] = error_pos * K_POS
    twist[3:] = error_rot_vec * K_ORI

    # 3. Compute Site Jacobian
    jac_p = np.zeros((3, model.nv))
    jac_r = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jac_p, jac_r, ee_site_id)
    J = np.vstack([jac_p, jac_r])[:, :7] 

    # 4. Solve for dq using Damped Least Squares
    vv = J @ J.T
    diag_indices = np.diag_indices_from(vv)
    vv[diag_indices] += DAMPING
    dq = J.T @ np.linalg.solve(vv, twist)

    # 5. Nullspace Control (Bias toward home posture q0)
    current_q = data.qpos[:7]
    null_error = (q0[:7] - current_q)
    J_pinv = np.linalg.pinv(J, rcond=1e-2)
    dq_null = (np.eye(7) - J_pinv @ J) @ (K_NULL * null_error)
    
    return np.clip(dq + dq_null, -MAX_ANGVEL, MAX_ANGVEL)