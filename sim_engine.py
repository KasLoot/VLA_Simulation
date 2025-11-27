import mujoco
import numpy as np


class SimEngine():
    def __init__(self, model, sim_config, robots_config):
        self.model = model
        self.data = mujoco.MjData(model)
        self.dt = sim_config.time_step
        self.model.opt.timestep = self.dt
        self.robots_config = robots_config
        self.init_joint_angles = self.robots_config.init_joint_positions

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        # Set initial joint positions if they exist
        if hasattr(self, 'init_joint_angles') and self.init_joint_angles is not None:
            self.data.qpos = np.array(self.init_joint_angles).copy()
            mujoco.mj_forward(self.model, self.data)
    
    def step(self):
        mujoco.mj_step(self.model, self.data)

    def set_control(self, control):
        self.data.ctrl[:control.shape[0]] = control

    def get_joint_positions(self):
        return self.data.qpos.copy()
    
    def get_joint_velocities(self):
        return self.data.qvel.copy()
    
    def get_ee_position(self, body_name="robot_1/hand"):
        try:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        except mujoco.FatalError as exc:
            raise ValueError(f"Body '{body_name}' not found in the model.") from exc
        
        return self.data.xpos[body_id].copy()
    
    def get_ee_orientation_quat(self, body_name="robot_1/hand"):
        try:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        except mujoco.FatalError as exc:
            raise ValueError(f"Body '{body_name}' not found in the model.") from exc
        
        return self.data.xquat[body_id].copy()
    
    def get_ee_orientation_euler(self, body_name="robot_1/hand"):
        quat = self.get_ee_orientation_quat(body_name)
        # Convert quaternion to Euler angles (roll, pitch, yaw)
        w, x, y, z = quat
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = np.arctan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = np.arcsin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = np.arctan2(t3, t4)

        return np.array([roll_x, pitch_y, yaw_z])
