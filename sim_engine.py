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
        self.init_joint_angles = model_config["robot_pybullet"]["init_motor_angles"]

        
        # Simulation parameters
        self.dt = 0.002 # 500Hz
        self.model.opt.timestep = self.dt

    def reset(self):
        """Resets the simulation to the initial state."""
        mujoco.mj_resetData(self.model, self.data)
        
        # Set initial joint positions if they exist
        if hasattr(self, 'init_joint_angles') and self.init_joint_angles is not None:
            # Ensure we don't exceed the number of joints in the model
            num_joints = min(len(self.init_joint_angles), self.model.nq)
            self.data.qpos[:num_joints] = self.init_joint_angles[:num_joints]

        mujoco.mj_forward(self.model, self.data)

    def get_joint_positions(self):
        """Returns the current joint positions."""
        return self.data.qpos.copy()

    def get_joint_velocities(self):
        """Returns the current joint velocities."""
        return self.data.qvel.copy()

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

    def compute_ik(self, target_pos, body_name="panda_link8", max_iter=100, tol=1e-3):
        """
        Simple Inverse Kinematics solver using Jacobian transpose / pseudo-inverse.
        
        Args:
            target_pos (np.array): Desired [x, y, z] position.
            body_name (str): Name of the end-effector body.
            max_iter (int): Maximum iterations.
            tol (float): Tolerance for convergence.
            
        Returns:
            np.array: Joint positions (qpos) that achieve the target.
        """
        # Get body ID
        try:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        except:
            print(f"Body '{body_name}' not found.")
            return self.data.qpos.copy()

        # Current qpos
        initial_qpos = self.data.qpos.copy()
        qpos = initial_qpos.copy()
        
        # Jacobian matrix
        jac = np.zeros((3, self.model.nv))
        
        # Iterative IK
        for _ in range(max_iter):
            # Forward kinematics
            mujoco.mj_kinematics(self.model, self.data)
            
            # Current EE position
            current_pos = self.data.xpos[body_id]
            
            # Error
            error = target_pos - current_pos
            if np.linalg.norm(error) < tol:
                break
                
            # Compute Jacobian (3xN for position only)
            # mj_jac(model, data, jacp, jacr, point, body)
            # jacp: 3 x nv, jacr: 3 x nv
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            
            mujoco.mj_jac(self.model, self.data, jacp, jacr, current_pos, body_id)
            
            # Solve J * dq = error
            # dq = pinv(J) * error
            # We only care about the arm joints (first 7 usually). 
            # If there are grippers, we might want to mask them, but let's assume all are active.
            
            # Damped Least Squares: dq = J.T * inv(J*J.T + lambda*I) * error
            # Or just numpy pinv
            dq = np.linalg.pinv(jacp) @ error
            
            # Update qpos
            qpos += dq * 0.5 # Step size
            
            # Set data to new qpos to recompute Jacobian in next step
            self.data.qpos[:] = qpos
            
        # Restore simulation state
        self.data.qpos[:] = initial_qpos
        mujoco.mj_kinematics(self.model, self.data)
        
        return qpos