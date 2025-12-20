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
        self.ee_tip_name = "tcp"
        self.num_joints = self.model.nv
        print(f"self.model.lowerPositionLimit: {self.model.lowerPositionLimit}")
        print(f"self.model.upperPositionLimit: {self.model.upperPositionLimit}")

    def rotation_matrix_to_euler_angles(self, R: np.ndarray[np.ndarray]) -> np.ndarray[np.ndarray]:
        assert R.shape[-2:] == (3, 3), "Input must be a rotation matrix of shape (..., 3, 3)"
        sy = np.sqrt(R[..., 0, 0] ** 2 + R[..., 1, 0] ** 2)
        singular = sy < 1e-6

        x = np.where(~singular, np.arctan2(R[..., 2, 1], R[..., 2, 2]), np.arctan2(-R[..., 1, 2], R[..., 1, 1]))
        y = np.where(~singular, np.arctan2(-R[..., 2, 0], sy), np.arctan2(-R[..., 2, 0], sy))
        z = np.where(~singular, np.arctan2(R[..., 1, 0], R[..., 0, 0]), np.zeros_like(sy))

        return np.stack((x, y, z), axis=-1)
    
    def euler_angles_to_rotation_matrix(self, angles: np.ndarray[np.ndarray]) -> np.ndarray[np.ndarray]:
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

    def forward_kinematics(self, joint_positions: np.ndarray[np.ndarray]):
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


    def inverse_kinematics(self, target_translation: np.ndarray[np.ndarray], target_orientation: np.ndarray[np.ndarray], qs: np.ndarray[np.ndarray]=None, eps=1e-4, dt=1e-1, damp=1e-6, max_iterations=1000):
        ee_frame_id = self.model.getFrameId(self.ee_tip_name)
        if qs is None:
            qs = [pin.neutral(self.model)] * target_translation.shape[0]
            # print(f"qs initialized to neutral positions: {np.array(qs)}")
        success = False
        for i in range(target_translation.shape[0]):
            j = 0
            # print(f"target_translation[{i}]: {target_translation[i]}, target_orientation[{i}]: {target_orientation[i]}")
            oMdes = pin.SE3(target_orientation[i], target_translation[i])
            # print(f"oMdes for sample {i}:\n", oMdes)
            while j < max_iterations:
                pin.forwardKinematics(self.model, self.data, qs[i])
                pin.updateFramePlacements(self.model, self.data)
                dMi = oMdes.actInv(self.data.oMf[ee_frame_id])
                err = pin.log(dMi).vector
                if norm(err) < eps:
                    success = True
                    # print(f"IK Convergence achieved for sample {i} in {j} iterations, with error {err}")
                    # print(f"oMdes:\n", oMdes)
                    # print(f"dMi:\n", dMi)
                    break
                if j >= max_iterations:
                    success = False
                    break
                J = pin.computeFrameJacobian(self.model, self.data, qs[i], ee_frame_id, pin.LOCAL)
                
                # Add a secondary objective to minimize joint changes
                w = 1e-3  # Weight for the secondary objective
                v = - J.T.dot(solve(J.dot(J.T) + damp * np.eye(6), err))
                # Null-space projection for the secondary objective
                v += (np.eye(self.num_joints) - J.T.dot(solve(J.dot(J.T) + damp * np.eye(6), J))).dot(w * (pin.neutral(self.model) - qs[i]))
                
                qs[i] = pin.integrate(self.model, qs[i], v*dt)
                qs[i] = np.clip(qs[i], self.model.lowerPositionLimit, self.model.upperPositionLimit)
                # if not j % 10:
                #     print('%d: error = %s' % (j, err.T))
                j += 1
        
        # print("IK Success:", success)
        return qs
    

    def inverse_kinematics_position_only(self, target_translation: np.ndarray, qs: np.ndarray=None, eps=1e-4, dt=1e-1, damp=1e-6, max_iterations=1000):
        """
        Solves Inverse Kinematics for Position only (x, y, z), ignoring orientation.
        """
        ee_frame_id = self.model.getFrameId(self.ee_tip_name)
        if qs is None:
            qs = [pin.neutral(self.model)] * target_translation.shape[0]

        # Convert to numpy array if it isn't already, to ensure indexing works
        qs = np.array(qs) 

        for i in range(target_translation.shape[0]):
            j = 0
            target_pos = target_translation[i]

            while j < max_iterations:
                # 1. Forward Kinematics
                pin.forwardKinematics(self.model, self.data, qs[i])
                pin.updateFramePlacements(self.model, self.data)
                
                # 2. Get current end-effector position
                current_pos = self.data.oMf[ee_frame_id].translation
                
                # 3. Compute Error (Simple Euclidean difference)
                err = target_pos - current_pos
                
                if norm(err) < eps:
                    # Success
                    break
                
                # 4. Compute Jacobian
                J = pin.computeFrameJacobian(self.model, self.data, qs[i], ee_frame_id, pin.LOCAL_WORLD_ALIGNED)
                
                # 5. Slice Jacobian: We only care about the top 3 rows (Linear Velocity)
                J_pos = J[:3, :] 

                # 6. Compute Damped Least Squares solution
                # v = J_pos.T @ (J_pos @ J_pos.T + damp * I)^-1 @ err
                w = 1e-3 # Secondary task weight
                
                # Primary Task: Position
                # We solve J_pos * dq = err
                v = J_pos.T.dot(solve(J_pos.dot(J_pos.T) + damp * np.eye(3), err))
                
                # Secondary Task: Stay close to neutral configuration (Null-space projection)
                # P = (I - J_pos+ * J_pos)
                # null_space_motion = P * (q_neutral - q_current)
                null_space_projector = (np.eye(self.num_joints) - J_pos.T.dot(solve(J_pos.dot(J_pos.T) + damp * np.eye(3), J_pos)))
                v += null_space_projector.dot(w * (pin.neutral(self.model) - qs[i]))

                # 7. Integrate and Enforce Limits
                qs[i] = pin.integrate(self.model, qs[i], v * dt)
                qs[i] = np.clip(qs[i], self.model.lowerPositionLimit, self.model.upperPositionLimit)
                
                j += 1
                
        return qs


    def inverse_kinematics_optimization(self, target_pos, target_rot=None, initial_q=None, maxiter=500):
        """
        Now accepts 'maxiter' as an argument to control solver precision/speed.
        """
        ee_frame_id = self.model.getFrameId(self.ee_tip_name)
        
        # ... [Input shape handling code remains the same] ...
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

            # Optimization
            res = minimize(
                fun=cost_function,
                x0=prev_q_i, 
                method='L-BFGS-B',
                bounds=bounds,
                # USE THE PASSED MAXITER ARGUMENT HERE
                options={'ftol': 1e-9, 'gtol': 1e-9, 'disp': False, 'maxiter': maxiter}
            )
            solved_qs[i] = res.x

        return solved_qs

    def generate_trajectory(self, start_pos, start_rot, target_pos, target_rot, init_joint_positions, num_steps=50, ik_maxiter=500):
        """
        Added 'ik_maxiter' argument.
        """
        n_robots = start_pos.shape[0]
        
        # ... [Interpolation code (Quintic/Slerp) remains the same] ...
        # (Copy the interpolation parts from your previous successful version)
        
        # Time Parameterization (Quintic)
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

        # Solve Inverse Kinematics
        joint_trajectory = np.zeros((n_robots, num_steps, self.num_joints))
        
        joint_trajectory[:, 0, :] = init_joint_positions
        current_qs = init_joint_positions.copy()

        print(f"Generating trajectory with IK maxiter={ik_maxiter}...")
        
        for step in range(1, num_steps):
            step_target_pos = traj_pos[step]
            step_target_rot = traj_rot[step]
            
            # Pass the hyperparameter down
            solved_qs = self.inverse_kinematics_optimization(
                target_pos=step_target_pos, 
                target_rot=step_target_rot,
                initial_q=current_qs,
                maxiter=ik_maxiter # <--- Passed here
            )
            
            joint_trajectory[:, step, :] = solved_qs
            current_qs = solved_qs

        return joint_trajectory



if __name__ == "__main__":
    model_path = "robot_models/franka_emika_panda/robot.xml"
    robot = FrankaPandaRobot(model_path)
    
    joint_positions = np.array([[0.0, 0.3, 0.9, 0.93, 0.24, 0.04, 0.86, 0.0, 0.0],
                                        [0.25, 0.38, 0.134, 0.284, 0.894, 0.472, 0.412, 0.914, 0.37]])  # Initial guess for joint positions
    
    position, orientation = robot.forward_kinematics(joint_positions)
    print("End-Effector Position:", position)
    print("End-Effector Orientation:\n", orientation)
    
    solved_joint_positions = robot.inverse_kinematics(target_translation=position, target_orientation=orientation, qs=None)
    print("Solved Joint Positions from IK:", solved_joint_positions)

    fk_position, fk_orientation = robot.forward_kinematics(solved_joint_positions)
    print("FK Position from IK Joint Positions:", fk_position)
    print("FK Orientation from IK Joint Positions:\n", fk_orientation)

    print("Difference in Position:", fk_position - position)
    print("Difference in Orientation:\n", fk_orientation - orientation)

