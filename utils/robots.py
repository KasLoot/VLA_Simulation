import pinocchio as pin
import numpy as np
import torch
from numpy.linalg import norm, solve
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R, Slerp
from scipy.optimize import minimize

class FrankaPandaRobot:
    def __init__(self, model_path):
        self.model = pin.buildModelFromMJCF(model_path)
        self.data = self.model.createData()
        self.ee_body_name = "hand"
        self.ee_tip_name = "hand_tip_site"
        self.num_joints = self.model.nv
        print(f"self.model.lowerPositionLimit: {self.model.lowerPositionLimit}")
        print(f"self.model.upperPositionLimit: {self.model.upperPositionLimit}")

    def rotation_matrix_to_euler_angles(self, R: np.ndarray) -> np.ndarray:
        assert R.shape[-2:] == (3, 3), "Input must be a rotation matrix of shape (..., 3, 3)"
        sy = np.sqrt(R[..., 0, 0] ** 2 + R[..., 1, 0] ** 2)
        singular = sy < 1e-6

        x = np.where(~singular, np.arctan2(R[..., 2, 1], R[..., 2, 2]), np.arctan2(-R[..., 1, 2], R[..., 1, 1]))
        y = np.where(~singular, np.arctan2(-R[..., 2, 0], sy), np.arctan2(-R[..., 2, 0], sy))
        z = np.where(~singular, np.arctan2(R[..., 1, 0], R[..., 0, 0]), np.zeros_like(sy))

        return np.stack((x, y, z), axis=-1)
    
    def euler_angles_to_rotation_matrix(self, angles: np.ndarray) -> np.ndarray:
        assert angles.shape[-1] == 3, "Input must be of shape (..., 3) representing Euler angles"
        c = np.cos(angles)
        s = np.sin(angles)

        # R_x
        R_x = np.zeros(angles.shape[:-1] + (3, 3), dtype=angles.dtype)
        R_x[..., 0, 0] = 1
        R_x[..., 1, 1] = c[..., 0]
        R_x[..., 1, 2] = -s[..., 0]
        R_x[..., 2, 1] = s[..., 0]
        R_x[..., 2, 2] = c[..., 0]

        # R_y
        R_y = np.zeros(angles.shape[:-1] + (3, 3), dtype=angles.dtype)
        R_y[..., 0, 0] = c[..., 1]
        R_y[..., 0, 2] = s[..., 1]
        R_y[..., 1, 1] = 1
        R_y[..., 2, 0] = -s[..., 1]
        R_y[..., 2, 2] = c[..., 1]

        # R_z
        R_z = np.zeros(angles.shape[:-1] + (3, 3), dtype=angles.dtype)
        R_z[..., 0, 0] = c[..., 2]
        R_z[..., 0, 1] = -s[..., 2]
        R_z[..., 1, 0] = s[..., 2]
        R_z[..., 1, 1] = c[..., 2]
        R_z[..., 2, 2] = 1

        R = np.einsum('...ij,...jk,...kl->...il', R_z, R_y, R_x)
        return R

    def forward_kinematics(self, joint_positions: np.ndarray):
        positions = []
        orientations = []
        for q in joint_positions:
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            ee_frame_id = self.model.getFrameId(self.ee_tip_name)
            position = self.data.oMf[ee_frame_id].translation.copy()
            orientation = self.data.oMf[ee_frame_id].rotation.copy()
            positions.append(position)
            orientations.append(orientation)
        position = np.array(positions)
        orientation = np.array(orientations)

        return position, orientation

    def inverse_kinematics(self, target_translation: np.ndarray, target_orientation: np.ndarray, qs: np.ndarray=None, eps=1e-4, dt=1e-1, damp=1e-6, max_iterations=1000):
        ee_frame_id = self.model.getFrameId(self.ee_tip_name)
        if qs is None:
            qs = [pin.neutral(self.model)] * target_translation.shape[0]
        success = False
        for i in range(target_translation.shape[0]):
            j = 0
            oMdes = pin.SE3(target_orientation[i], target_translation[i])
            while j < max_iterations:
                pin.forwardKinematics(self.model, self.data, qs[i])
                pin.updateFramePlacements(self.model, self.data)
                dMi = oMdes.actInv(self.data.oMf[ee_frame_id])
                err = pin.log(dMi).vector
                if norm(err) < eps:
                    success = True
                    break
                if j >= max_iterations:
                    success = False
                    break
                J = pin.computeFrameJacobian(self.model, self.data, qs[i], ee_frame_id, pin.LOCAL)
                
                # Add a secondary objective to minimize joint changes
                w = 1e-3  
                v = - J.T.dot(solve(J.dot(J.T) + damp * np.eye(6), err))
                v += (np.eye(self.num_joints) - J.T.dot(solve(J.dot(J.T) + damp * np.eye(6), J))).dot(w * (pin.neutral(self.model) - qs[i]))
                
                qs[i] = pin.integrate(self.model, qs[i], v*dt)
                qs[i] = np.clip(qs[i], self.model.lowerPositionLimit, self.model.upperPositionLimit)
                j += 1
        return qs
    

    def inverse_kinematics_position_only(self, target_translation: np.ndarray, qs: np.ndarray=None, eps=1e-4, dt=1e-1, damp=1e-6, max_iterations=1000):
        ee_frame_id = self.model.getFrameId(self.ee_tip_name)
        if qs is None:
            qs = [pin.neutral(self.model)] * target_translation.shape[0]

        qs = np.array(qs) 

        for i in range(target_translation.shape[0]):
            j = 0
            target_pos = target_translation[i]

            while j < max_iterations:
                pin.forwardKinematics(self.model, self.data, qs[i])
                pin.updateFramePlacements(self.model, self.data)
                current_pos = self.data.oMf[ee_frame_id].translation
                err = target_pos - current_pos
                
                if norm(err) < eps:
                    break
                
                J = pin.computeFrameJacobian(self.model, self.data, qs[i], ee_frame_id, pin.LOCAL_WORLD_ALIGNED)
                J_pos = J[:3, :] 

                w = 1e-3 
                v = J_pos.T.dot(solve(J_pos.dot(J_pos.T) + damp * np.eye(3), err))
                null_space_projector = (np.eye(self.num_joints) - J_pos.T.dot(solve(J_pos.dot(J_pos.T) + damp * np.eye(3), J_pos)))
                v += null_space_projector.dot(w * (pin.neutral(self.model) - qs[i]))

                qs[i] = pin.integrate(self.model, qs[i], v * dt)
                qs[i] = np.clip(qs[i], self.model.lowerPositionLimit, self.model.upperPositionLimit)
                j += 1
                
        return qs

    def inverse_kinematics_optimization(self, target_pos, target_rot=None, initial_q=None, maxiter=500):
        ee_frame_id = self.model.getFrameId(self.ee_tip_name)
        
        target_pos = np.array(target_pos)
        if target_pos.ndim == 1:
            target_pos = target_pos[None, :]
            if target_rot is not None: target_rot = target_rot[None, :, :]
            if initial_q is not None: initial_q = np.array(initial_q)[None, :]

        n_robots = target_pos.shape[0]

        if initial_q is None:
            initial_q = np.array([pin.neutral(self.model)] * n_robots)
        else:
            initial_q = np.array(initial_q)

        solved_qs = np.zeros_like(initial_q)
        bounds = [(l, u) for l, u in zip(self.model.lowerPositionLimit, self.model.upperPositionLimit)]

        for i in range(n_robots):
            tgt_pos_i = target_pos[i]
            tgt_rot_i = target_rot[i] if target_rot is not None else None
            prev_q_i = initial_q[i]

            def cost_function(q):
                pin.forwardKinematics(self.model, self.data, q)
                pin.updateFramePlacements(self.model, self.data)
                
                curr_pos = self.data.oMf[ee_frame_id].translation
                curr_rot = self.data.oMf[ee_frame_id].rotation
                
                pos_err = np.sum((curr_pos - tgt_pos_i)**2)
                
                rot_err = 0.0
                if tgt_rot_i is not None:
                    diff = pin.log(pin.SE3(curr_rot.T @ tgt_rot_i, np.zeros(3))).vector
                    rot_err = np.sum(diff**2)

                reg_weight = 0.001 
                motion_err = np.sum((q - prev_q_i)**2)
                
                neutral_weight = 0.001
                neutral_err = np.sum((q - pin.neutral(self.model))**2)

                return 10000.0 * pos_err + 10.0 * rot_err + reg_weight * motion_err + neutral_weight * neutral_err

            res = minimize(
                fun=cost_function,
                x0=prev_q_i, 
                method='L-BFGS-B',
                bounds=bounds,
                options={'ftol': 1e-9, 'gtol': 1e-9, 'disp': False, 'maxiter': maxiter}
            )
            solved_qs[i] = res.x

        return solved_qs

    def generate_trajectory(self, start_pos, start_rot, target_pos, target_rot, init_joint_positions, num_steps=50, ik_maxiter=500):
        n_robots = start_pos.shape[0]
        
        t = np.linspace(0, 1, num_steps)
        s = 10 * t**3 - 15 * t**4 + 6 * t**5
        
        traj_pos = start_pos[None, :, :] + (target_pos - start_pos)[None, :, :] * s[:, None, None]

        traj_rot = np.zeros((num_steps, n_robots, 3, 3))
        for i in range(n_robots):
            r_start = R.from_matrix(start_rot[i])
            r_end = R.from_matrix(target_rot[i])
            key_rots = R.concatenate([r_start, r_end])
            slerp = Slerp([0, 1], key_rots)
            traj_rot[:, i, :, :] = slerp(t).as_matrix()

        joint_trajectory = np.zeros((n_robots, num_steps, self.num_joints))
        
        joint_trajectory[:, 0, :] = init_joint_positions
        current_qs = init_joint_positions.copy()

        print(f"Generating trajectory with IK maxiter={ik_maxiter}...")
        
        for step in range(1, num_steps):
            step_target_pos = traj_pos[step]
            step_target_rot = traj_rot[step]
            
            solved_qs = self.inverse_kinematics_optimization(
                target_pos=step_target_pos, 
                target_rot=step_target_rot,
                initial_q=current_qs,
                maxiter=ik_maxiter
            )
            
            joint_trajectory[:, step, :] = solved_qs
            current_qs = solved_qs

        return joint_trajectory


    def pd_control(self, target_positions, current_positions, current_velocities, kp=None, kd=None):
        """
        Computes PD control torques with support for joint-specific gains.
        
        Args:
            target_positions: (n_robots, 9)
            current_positions: (n_robots, 9)
            current_velocities: (n_robots, 9)
            kp: Scalar or Array (7,) for proportional gains. 
                Defaults to [100, 100, 100, 100, 50, 50, 10]
            kd: Scalar or Array (7,) for derivative gains.
                Defaults to [10, 10, 10, 10, 5, 5, 2]
        """
        n_robots = current_positions.shape[0]
        control_signals = np.zeros((n_robots, 9)) 

        # --- Handle Gains ---
        # If None, use tuned default profile for Franka Panda
        if kp is None:
            kp = np.array([100.0, 100.0, 100.0, 100.0, 50.0, 50.0, 10.0])
        elif np.isscalar(kp):
            kp = np.full(7, kp)
        else:
            kp = np.array(kp)

        if kd is None:
            kd = np.array([10.0, 50.0, 10.0, 10.0, 5.0, 5.0, 2.0])
            # kd = np.array([10.0, 10.0, 10.0, 10.0, 5.0, 5.0, 2.0])
        elif np.isscalar(kd):
            kd = np.full(7, kd)
        else:
            kd = np.array(kd)
        
        # --------------------
        
        for i in range(n_robots):
            q = current_positions[i]  # Shape (9,)
            v = current_velocities[i]
            q_des = target_positions[i]
            
            # 1. Compute Gravity Compensation
            pin.computeGeneralizedGravity(self.model, self.data, q)
            g = self.data.g  # Shape (9,)
            
            # 2. Extract Arm Components (First 7 Joints)
            q_arm = q[:7]
            v_arm = v[:7]
            q_des_arm = q_des[:7]
            g_arm = g[:7]
            
            # 3. PD Control Law for Arm (Broadcasting happens here automatically)
            # kp shape (7,) * difference shape (7,) -> (7,)
            tau_arm = g_arm + kp * (q_des_arm - q_arm) + kd * (0 - v_arm)
            
            # 4. Fill control signals
            control_signals[i, :7] = tau_arm
            # Fingers (indices 7,8) are left as 0.0 or handled separately if needed
            
        return control_signals