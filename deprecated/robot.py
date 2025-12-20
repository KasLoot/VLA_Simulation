import pinocchio as pin
import numpy as np
import torch





class FrankaPandaRobot:
    def __init__(self, model_path):
        self.model = pin.buildModelFromMJCF(model_path)
        self.data = self.model.createData()
        self.frame_name = "hand"
        if self.model.existFrame("tcp"):
             self.frame_name = "tcp"
        elif not self.model.existFrame(self.frame_name):
             self.frame_name = self.model.frames[-1].name
        self.frame_id = self.model.getFrameId(self.frame_name)
        self.name = "franka_emika_panda"

        # The MuJoCo model uses 8 actuators (7 arm joints + 1 gripper tendon), while the
        # Pinocchio MJCF model exposes 9 DoFs (7 arm joints + 2 finger slide joints).
        # We keep an explicit mapping so callers can work in actuator-space (8) without
        # worrying about the internal Pinocchio configuration size.
        self._arm_n = 7
        self._finger_joint_names = ("finger_joint1", "finger_joint2")
        self._finger_q_indices = []
        for name in self._finger_joint_names:
            try:
                jid = self.model.getJointId(name)
                if jid != 0:
                    self._finger_q_indices.append(int(self.model.joints[jid].idx_q))
            except Exception:
                # Some models omit finger joints; keep mapping empty.
                pass
        self._has_two_fingers = len(self._finger_q_indices) == 2

        # Finger joint range in the MJCF (slide joints): [0, 0.04]
        self._finger_q_min = 0.0
        self._finger_q_max = 0.04

        # MuJoCo actuator8 (tendon "split") uses ctrlrange [0, 255] in this repo.
        # We treat the 8th actuator command as this ctrl value.
        self._gripper_ctrl_min = 0.0
        self._gripper_ctrl_max = 255.0
        

    def get_name(self):
        return self.name


    def _to_numpy(self, x) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy().astype(np.float64)
        return np.asarray(x, dtype=np.float64)

    def _gripper_ctrl_to_finger_q(self, ctrl: float) -> float:
        """Map a single MuJoCo gripper actuator control value (0..255) to finger joint q (0..0.04)."""
        ctrl = float(np.clip(ctrl, self._gripper_ctrl_min, self._gripper_ctrl_max))
        if self._gripper_ctrl_max == self._gripper_ctrl_min:
            return self._finger_q_min
        t = (ctrl - self._gripper_ctrl_min) / (self._gripper_ctrl_max - self._gripper_ctrl_min)
        return (1.0 - t) * self._finger_q_min + t * self._finger_q_max

    def _finger_q_to_gripper_ctrl(self, finger_q: float) -> float:
        """Map a finger joint q (0..0.04) back to MuJoCo gripper actuator control value (0..255)."""
        finger_q = float(np.clip(finger_q, self._finger_q_min, self._finger_q_max))
        if self._finger_q_max == self._finger_q_min:
            return self._gripper_ctrl_min
        t = (finger_q - self._finger_q_min) / (self._finger_q_max - self._finger_q_min)
        return (1.0 - t) * self._gripper_ctrl_min + t * self._gripper_ctrl_max

    def _pin_q_from_actuated(self, q_act: np.ndarray) -> np.ndarray:
        """Convert actuator-space vector (7 or 8) into a full Pinocchio configuration q (nq).

        - 7: arm only
        - 8: arm + gripper ctrl (0..255)
        - nq: already full q
        """
        q_act = np.asarray(q_act, dtype=np.float64)
        q_full = pin.neutral(self.model)

        if q_act.shape[-1] == self.model.nq:
            q_full[:] = q_act
            return q_full

        # Arm joints
        q_full[: self._arm_n] = q_act[: self._arm_n]

        # Fingers
        if self._has_two_fingers:
            if q_act.shape[-1] >= self._arm_n + 1:
                finger_q = self._gripper_ctrl_to_finger_q(q_act[self._arm_n])
            else:
                finger_q = 0.0
            q_full[self._finger_q_indices[0]] = finger_q
            q_full[self._finger_q_indices[1]] = finger_q

        return q_full

    def _actuated_from_pin_q(self, q_full: np.ndarray) -> np.ndarray:
        """Convert a Pinocchio configuration q (nq) into simulator actuator-space (8).

        Output layout: [joint1..joint7, gripper_ctrl]
        """
        q_full = np.asarray(q_full, dtype=np.float64)
        q_act = np.zeros((self._arm_n + 1,), dtype=np.float64)
        q_act[: self._arm_n] = q_full[: self._arm_n]

        if self._has_two_fingers:
            finger_q = 0.5 * (q_full[self._finger_q_indices[0]] + q_full[self._finger_q_indices[1]])
            q_act[self._arm_n] = self._finger_q_to_gripper_ctrl(finger_q)
        else:
            q_act[self._arm_n] = 0.0

        return q_act

    def _mujoco_joints_from_pin_q(self, q_full: np.ndarray) -> np.ndarray:
        """Convert a Pinocchio configuration q (nq) into MuJoCo joint targets (8).

        Output layout: [joint1..joint7, finger_q]
        where finger_q is the slide joint position in meters (0..0.04).
        """
        q_full = np.asarray(q_full, dtype=np.float64)
        q_j = np.zeros((self._arm_n + 1,), dtype=np.float64)
        q_j[: self._arm_n] = q_full[: self._arm_n]

        if self._has_two_fingers:
            finger_q = 0.5 * (q_full[self._finger_q_indices[0]] + q_full[self._finger_q_indices[1]])
            q_j[self._arm_n] = float(np.clip(finger_q, self._finger_q_min, self._finger_q_max))
        else:
            q_j[self._arm_n] = 0.0
        return q_j
    
    
        

    def forward_kinematics(self, joint_positions):
        q = self._to_numpy(joint_positions)
            
        # Handle batching
        if q.ndim == 1:
            q = q[None, :]
            is_batched = False
        else:
            is_batched = True
            
        batch_size = q.shape[0]
        results = []
        
        for b in range(batch_size):
            q_curr = q[b]
            q_full = self._pin_q_from_actuated(q_curr)
                
            pin.framesForwardKinematics(self.model, self.data, q_full)
            oMtool = self.data.oMf[self.frame_id]
            
            pos = oMtool.translation
            rot = oMtool.rotation
            rpy = pin.rpy.matrixToRpy(rot)
            
            # Return concatenated pos and rpy (6,)
            pose = np.concatenate([pos, rpy])
            results.append(pose)
            
        results_np = np.stack(results)
        
        if isinstance(joint_positions, torch.Tensor):
            res = torch.from_numpy(results_np).to(joint_positions.device).float()
        else:
            res = results_np
            
        return res if is_batched else res[0]

    def inverse_kinematics(
        self,
        target_pose=None,
        initial_guess=None,
        max_iter=100,
        tol=1e-4,
        alpha=0.1,
        damp=1e-4,
        return_format: str = "actuated",
        # Back-compat aliases used elsewhere in this repo
        target_pos=None,
        init_q=None,
        eps=None,
        dt=None,
    ):
        """Inverse kinematics for the end-effector frame.

        Args:
            target_pose: (.., 6) [x, y, z, roll, pitch, yaw] or (.., 3) [x, y, z].
            initial_guess/init_q: optional (.., 7|8|nq) starting point.
            return_format:
                - "actuated" (default): (.., 8) -> [joint1..joint7, gripper_ctrl]
                - "mujoco_joints"/"joints": (.., 8) -> [joint1..joint7, finger_q] (meters)
                - "arm": (.., 7)
                - "pinocchio"/"full": (.., nq) (e.g., 9)

        Notes:
            The solver optimizes only the 7 arm joints. Finger joints are kept fixed.
        """

        if target_pose is None and target_pos is not None:
            target_pose = target_pos
        if initial_guess is None and init_q is not None:
            initial_guess = init_q
        if eps is not None:
            tol = eps
        if dt is not None:
            alpha = dt

        t_pose = self._to_numpy(target_pose)
            
        if t_pose.ndim == 1:
            t_pose = t_pose[None, :]
            is_batched = False
        else:
            is_batched = True
            
        batch_size = t_pose.shape[0]
        results = []
        
        # Prepare initial guess
        if initial_guess is not None:
            q_guess_in = self._to_numpy(initial_guess)
            if q_guess_in.ndim == 1:
                q_guess_in = np.tile(q_guess_in, (batch_size, 1))
        else:
            q_guess_in = np.tile(pin.neutral(self.model), (batch_size, 1))

        for b in range(batch_size):
            pose_des = t_pose[b]
            q = self._pin_q_from_actuated(q_guess_in[b])

            pos_des = pose_des[:3]
            if pose_des.shape[0] >= 6:
                rpy_des = pose_des[3:6]
                rot_des = pin.rpy.rpyToMatrix(rpy_des[0], rpy_des[1], rpy_des[2])
            else:
                # If only a position target is provided, keep the current EE orientation.
                pin.framesForwardKinematics(self.model, self.data, q)
                rot_des = self.data.oMf[self.frame_id].rotation
            oMdes = pin.SE3(rot_des, pos_des)
            
            for i in range(max_iter):
                pin.framesForwardKinematics(self.model, self.data, q)
                oMtool = self.data.oMf[self.frame_id]
                dMi = oMdes.actInv(oMtool)
                err = pin.log(dMi).vector
                
                if np.linalg.norm(err) < tol:
                    break
                    
                J = pin.computeFrameJacobian(self.model, self.data, q, self.frame_id)
                J_arm = J[:, :7]

                lmbda = float(damp) if damp is not None else 1e-4
                JJt = J_arm @ J_arm.T
                J_pinv = J_arm.T @ np.linalg.solve(JJt + lmbda * np.eye(6), np.eye(6))
                
                v_task = J_pinv @ err

                # Null-space projection to minimize joint changes
                q_rest = self._pin_q_from_actuated(q_guess_in[b])
                v_null = 0.5 * (q_rest[:7] - q[:7]) # Move towards initial configuration
                P = np.eye(7) - J_pinv @ J_arm
                v_arm = v_task + P @ v_null
                
                v = np.zeros(self.model.nv)
                v[:7] = v_arm
                q = pin.integrate(self.model, q, v * alpha)
                
            results.append(q)
            
        results_np = np.stack(results)
        
        # Format output
        rf = (return_format or "actuated").lower()
        if rf in ("pinocchio", "full", "q"):
            out_np = results_np
        elif rf in ("mujoco_joints", "mujoco", "joints", "joint"):
            out_np = np.stack([self._mujoco_joints_from_pin_q(qi) for qi in results_np], axis=0)
        elif rf in ("arm", "arm7", "7"):
            out_np = results_np[:, : self._arm_n]
        else:
            # Default: actuator-space (8): [arm7, gripper_ctrl]
            out_np = np.stack([self._actuated_from_pin_q(qi) for qi in results_np], axis=0)

        if isinstance(target_pose, torch.Tensor) or isinstance(target_pos, torch.Tensor) or isinstance(initial_guess, torch.Tensor) or isinstance(init_q, torch.Tensor):
            # Prefer device from any tensor input
            dev = None
            for x in (target_pose, target_pos, initial_guess, init_q):
                if isinstance(x, torch.Tensor):
                    dev = x.device
                    break
            res = torch.from_numpy(out_np).to(dev if dev is not None else "cpu").float()
        else:
            res = out_np

        return res if is_batched else res[0]

    def generate_trajectory_from_ee_positions(
        self,
        start_pose,
        target_pose,
        num_steps: int = 50,
        initial_joint_config=None,
        # alias used in commented code in this repo
        init_joints=None,
        return_format: str = "actuated",
    ):
        """Generate a joint/control trajectory by interpolating in EE pose space.

        Supports single robot (6,) or batched robots (B,6). Output is:
            - single: (T, D)
            - batched: (B, T, D)
        where D depends on return_format (default 8 for actuators).
        """

        if initial_joint_config is None and init_joints is not None:
            initial_joint_config = init_joints

        is_torch = isinstance(start_pose, torch.Tensor) or isinstance(target_pose, torch.Tensor)
        device = start_pose.device if isinstance(start_pose, torch.Tensor) else (target_pose.device if isinstance(target_pose, torch.Tensor) else None)

        s_pose = self._to_numpy(start_pose)
        t_pose = self._to_numpy(target_pose)

        if initial_joint_config is not None:
            current_q = self._to_numpy(initial_joint_config)
        else:
            current_q = None

        # Normalize shapes
        if s_pose.ndim == 1:
            s_pose = s_pose[None, :]
        if t_pose.ndim == 1:
            t_pose = t_pose[None, :]
        if s_pose.shape[0] != t_pose.shape[0]:
            raise ValueError(f"start_pose and target_pose batch sizes must match. Got {s_pose.shape[0]} vs {t_pose.shape[0]}")
        batch_size = s_pose.shape[0]

        if current_q is not None and current_q.ndim == 1:
            current_q = np.tile(current_q, (batch_size, 1))

        alphas = np.linspace(0.0, 1.0, int(num_steps), dtype=np.float64)
        # (T, B, 6)
        traj_poses = s_pose[None, :, :] + alphas[:, None, None] * (t_pose - s_pose)[None, :, :]

        joint_traj_steps = []
        for i in range(int(num_steps)):
            pose_step = traj_poses[i]  # (B, 6)
            q_step = self.inverse_kinematics(
                pose_step,
                initial_guess=current_q,
                return_format=return_format,
            )
            q_step_np = self._to_numpy(q_step)
            joint_traj_steps.append(q_step_np)
            current_q = q_step_np

        # (T, B, D)
        results_np = np.stack(joint_traj_steps, axis=0)

        # Return (B, T, D) for batched, (T, D) for single
        if batch_size == 1:
            out_np = results_np[:, 0, :]
        else:
            out_np = results_np.transpose(1, 0, 2)

        if is_torch:
            return torch.from_numpy(out_np).to(device if device is not None else "cpu").float()
        return out_np


if __name__ == "__main__":
    model_path = "/home/yuxin/VLA_Simulation/robot_models/franka_emika_panda/robot.xml"
    robot = FrankaPandaRobot(model_path)
