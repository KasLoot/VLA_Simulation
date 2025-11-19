import mujoco
import mujoco.viewer
import time
import numpy as np

class RobotSim:
    def __init__(self, model_path, model_config, dt=0.002):
        """
        Initialize the MuJoCo simulation.
        
        Args:
            model_path (str): Path to the .xml or .urdf file.
            dt (float): Simulation timestep.
        """
        print(f"Loading model from: {model_path}")
        
        # Load the model
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.init_joint_angles = np.array(model_config["robot_pybullet"]["init_motor_angles"])

        
        # Simulation parameters
        self.dt = 0.002 # 500Hz
        self.model.opt.timestep = self.dt

    def reset(self):
        """Resets the simulation to the initial state."""
        mujoco.mj_resetData(self.model, self.data)
        
        # Set initial joint positions if they exist
        if hasattr(self, 'init_joint_angles') and self.init_joint_angles is not None:
            self.data.qpos = self.init_joint_angles.copy()

        mujoco.mj_forward(self.model, self.data)

    def get_joint_positions(self):
        """Returns the current joint positions."""
        return self.data.qpos.copy()

    def get_joint_velocities(self):
        """Returns the current joint velocities."""
        return self.data.qvel.copy()
    
    def get_ee_position(self, body_name="panda_link7"):
        """Returns the current end-effector position."""
        try:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        except mujoco.FatalError as exc:
            raise ValueError(f"Body '{body_name}' not found in the model.") from exc
        
        return self.data.xpos[body_id].copy()

    def get_ee_orientation(self, body_name="panda_link7"):
        """Returns the current end-effector orientation as a quaternion [w, x, y, z]."""
        try:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        except mujoco.FatalError as exc:
            raise ValueError(f"Body '{body_name}' not found in the model.") from exc
        
        return self.data.xquat[body_id].copy()

    def set_control(self, ctrl):
        """
        Apply control signals to the actuators.
        Args:
            ctrl (list or np.array): Control inputs (e.g., torques or positions depending on actuators)
        """
        # Safety check for control dimension
        if len(ctrl) != self.model.nu:
            print(f"Warning: Control input length {len(ctrl)} does not match model actuators {self.model.nu}")
            return
            
        self.data.ctrl[:] = ctrl

    def step(self):
        """Advances the simulation by one step."""
        mujoco.mj_step(self.model, self.data)


    def run_simulation(self, ref_object, duration=10.0, cam_config=None):
        """
        Launches the interactive viewer with a reference object for control.
        
        Args:
            ref_object: An object that has a method get_values(time) returning desired joint positions and velocities.
            duration (float): Duration to run the simulation in seconds.
            cam_config (dict): Optional dictionary with keys: 'lookat', 'distance', 'azimuth', 'elevation'.
        """
        print("Launching viewer... Press ESC to exit.")
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            
            # Initialize camera if config is provided
            if cam_config:
                if 'lookat' in cam_config:
                    viewer.cam.lookat[:] = cam_config['lookat']
                if 'distance' in cam_config:
                    viewer.cam.distance = cam_config['distance']
                if 'azimuth' in cam_config:
                    viewer.cam.azimuth = cam_config['azimuth']
                if 'elevation' in cam_config:
                    viewer.cam.elevation = cam_config['elevation']

            current_time = 0.0
            time_step = self.dt
            while viewer.is_running() and current_time < duration:
                step_start = time.time()

                # 1. Get reference values
                q_d, qd_d = ref_object.get_values(current_time)
                # print(f"Time: {current_time:.3f}, Desired Q: {q_d}, Desired QD: {qd_d}")

                # 2. Create control signal
                ctrl = np.zeros(self.model.nu)
                n_joints = min(len(q_d), self.model.nu)
                ctrl[:n_joints] = q_d[:n_joints]

                # 3. Apply control
                self.set_control(ctrl)

                # 4. Step Physics
                self.step()

                # 5. Sync Viewer
                viewer.sync()

                # 6. Time keeping (roughly real-time)
                time_until_next_step = time_step - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

                current_time += time_step

    def compute_ik(self, ee_traj, ee_ori_traj=None, body_name="panda_link7", max_iter=80, tol=1e-4, rot_tol=1e-2, restore_state=True):
        """
        Compute joint trajectories that track a Cartesian end-effector path.

        Args:
            ee_traj (array-like): Sequence of [x, y, z] target points with shape (N, 3).
            ee_ori_traj (array-like, optional): Sequence of [w, x, y, z] target quaternions with shape (N, 4).
            body_name (str): Name of the end-effector body in the MuJoCo model.
            max_iter (int): Maximum IK iterations for the first waypoint (subsequent waypoints reuse the previous solution).
            tol (float): Position tolerance for convergence in meters.
            rot_tol (float): Orientation tolerance for convergence in radians.
            restore_state (bool): Whether to restore the simulation state after computation.

        Returns:
            np.ndarray: Joint trajectory with shape (N, nq).
        """
        traj = np.asarray(ee_traj, dtype=float)
        if traj.ndim == 1:
            if traj.size != 3:
                raise ValueError("ee_traj entries must be 3D Cartesian points.")
            traj = traj.reshape(1, 3)
        if traj.shape[1] != 3:
            raise ValueError("ee_traj must have shape (N, 3).")

        use_orientation = False
        if ee_ori_traj is not None:
            ori_traj = np.asarray(ee_ori_traj, dtype=float)
            if ori_traj.ndim == 1:
                if ori_traj.size != 4:
                    raise ValueError("ee_ori_traj entries must be 4D quaternions.")
                ori_traj = ori_traj.reshape(1, 4)
            if ori_traj.shape[0] != traj.shape[0]:
                raise ValueError("ee_ori_traj must have the same length as ee_traj")
            if ori_traj.shape[1] != 4:
                raise ValueError("ee_ori_traj must have shape (N, 4) for quaternions [w, x, y, z].")
            use_orientation = True

        try:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        except mujoco.FatalError as exc:
            raise ValueError(f"Body '{body_name}' not found in the model.") from exc

        saved_qpos = self.data.qpos.copy()
        n_points = traj.shape[0]
        n_dof = self.model.nq
        joint_traj = np.empty((n_points, n_dof), dtype=float)

        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        damping = 0.05
        
        if use_orientation:
            jac = np.zeros((6, self.model.nv))
            error = np.zeros(6)
            reg_mat = np.eye(6) * (damping ** 2)
        else:
            reg_mat = np.eye(3) * (damping ** 2)

        def _compute_jacobian():
            if hasattr(mujoco, "mj_jacBody"):
                mujoco.mj_jacBody(self.model, self.data, jacp, jacr, body_id)
            else:
                mujoco.mj_jac(self.model, self.data, jacp, jacr, np.zeros(3), body_id)

        q_guess = self.data.qpos.copy()

        for idx, target_pos in enumerate(traj):
            self.data.qpos[:] = q_guess
            mujoco.mj_forward(self.model, self.data)

            iter_limit = max_iter if idx == 0 else max(max_iter // 2, 20)
            converged = False

            for _ in range(iter_limit):
                current_pos = self.data.xpos[body_id].copy()
                pos_error = target_pos - current_pos
                
                if use_orientation:
                    target_quat = ori_traj[idx]
                    current_quat = self.data.xquat[body_id].copy()
                    
                    rot_error = np.zeros(3)
                    mujoco.mju_subQuat(rot_error, target_quat, current_quat)

                    error[:3] = pos_error
                    error[3:] = rot_error
                    
                    if np.linalg.norm(pos_error) <= tol and np.linalg.norm(rot_error) <= rot_tol:
                        converged = True
                        break
                        
                    _compute_jacobian()
                    jac[:3] = jacp
                    jac[3:] = jacr
                    
                    jjt = jac @ jac.T + reg_mat
                    delta = jac.T @ np.linalg.solve(jjt, error)
                else:
                    if np.linalg.norm(pos_error) <= tol:
                        converged = True
                        break

                    _compute_jacobian()
                    jjt = jacp @ jacp.T + reg_mat
                    delta = jacp.T @ np.linalg.solve(jjt, pos_error)

                mujoco.mj_integratePos(self.model, self.data.qpos, delta, 1)
                
                # Enforce joint limits
                for j in range(self.model.njnt):
                    q_adr = self.model.jnt_qposadr[j]
                    min_val, max_val = self.model.jnt_range[j]
                    self.data.qpos[q_adr] = np.clip(self.data.qpos[q_adr], min_val, max_val)

                mujoco.mj_kinematics(self.model, self.data)

            if not converged:
                err_val = np.linalg.norm(error) if use_orientation else np.linalg.norm(pos_error)
                print(f"Warning: IK did not converge for waypoint {idx}; residual {err_val:.4f}.")

            q_guess = self.data.qpos.copy()
            joint_traj[idx] = q_guess

        if restore_state:
            self.data.qpos[:] = saved_qpos
            mujoco.mj_forward(self.model, self.data)
        
        return joint_traj
