import mujoco
import numpy as np
from utils.robots import FrankaPandaRobot
# Avoid importing run_demo here to prevent circular imports.
# Accept generic config objects (typing.Any) instead of concrete classes.


class SimEngine():
    def __init__(self, sim_env, sim_config=None, robots_config=None):
        self.sim_env = sim_env
        self.data = mujoco.MjData(self.sim_env)
        self.sim_config = sim_config
        self.robots_config = robots_config
        self.init_joint_positions = self.robots_config.init_joint_positions
        self.__init__config__()
        self.reset()

    def __init__config__(self):
        self.base_body_ids = []
        self.robot_joint_indices = []
        self.robot_dof_indices = []
        self.robot_actuator_indices = []
        
        self.total_robots = sum(self.robots_config.quantities)
        
        for i in range(self.total_robots):
            # Base Body
            base_name = f"robot_{i}"
            base_id = mujoco.mj_name2id(self.sim_env, mujoco.mjtObj.mjOBJ_BODY, base_name)
            if base_id != -1:
                self.base_body_ids.append(base_id)
            
            # Find joints for this robot
            # Assuming standard panda joint names
            joint_names = [f"{base_name}/joint_{j}" for j in range(1, 8)] + \
                          [f"{base_name}/finger_joint_1", f"{base_name}/finger_joint_2"]
            
            for j_name in joint_names:
                j_id = mujoco.mj_name2id(self.sim_env, mujoco.mjtObj.mjOBJ_JOINT, j_name)
                if j_id != -1:
                    qpos_addr = self.sim_env.jnt_qposadr[j_id]
                    self.robot_joint_indices.append(qpos_addr)
                    
                    dof_addr = self.sim_env.jnt_dofadr[j_id]
                    self.robot_dof_indices.append(dof_addr)
            
            # Find actuators for this robot
            actuator_names = [f"{base_name}/joint_{j}_motor" for j in range(1, 8)] + \
                             [f"{base_name}/finger_1_position", f"{base_name}/finger_2_position"]
            for a_name in actuator_names:
                a_id = mujoco.mj_name2id(self.sim_env, mujoco.mjtObj.mjOBJ_ACTUATOR, a_name)
                if a_id != -1:
                    self.robot_actuator_indices.append(a_id)
        
        self.robot_joint_indices = np.array(self.robot_joint_indices, dtype=np.int32)
        self.robot_dof_indices = np.array(self.robot_dof_indices, dtype=np.int32)
        self.robot_actuator_indices = np.array(self.robot_actuator_indices, dtype=np.int32)

    def reset(self):
        print("Resetting simulation...")
        # print("Initial joint positions shape:", np.array(self.init_joint_positions).shape)
        mujoco.mj_resetData(self.sim_env, self.data)
        self.set_joint_controls(np.array(self.init_joint_positions).flatten())
        mujoco.mj_forward(self.sim_env, self.data)

    def step(self):
        mujoco.mj_step(self.sim_env, self.data)

    def forward(self):
        mujoco.mj_forward(self.sim_env, self.data)

    def set_joint_controls(self, joint_positions):
        self.data.qpos[self.robot_joint_indices] = joint_positions

    def set_actuator_controls(self, actuator_controls):
        self.data.ctrl[self.robot_actuator_indices] = actuator_controls

        
    def get_joint_positions(self):
        return self.data.qpos[self.robot_joint_indices].copy()

    def get_joint_velocities(self):
        return self.data.qvel[self.robot_dof_indices].copy()
    
    def get_body_position_from_name(self, body_name: str):
        body_id = self.sim_env.body(name=body_name).id
        return self.data.xpos[body_id].copy()
    
    def get_base_positions(self):
        base_positions = []
        for base_id in self.base_body_ids:
            base_positions.append(self.data.xpos[base_id].copy())
        return np.array(base_positions)
    
    def get_all_ee_global_positions(self):
        ee_positions = []
        ee_orientations = []
        for i in range(self.total_robots):
            site_name = f"robot_{i}/hand_tip_site"
            site_id = mujoco.mj_name2id(self.sim_env, mujoco.mjtObj.mjOBJ_SITE, site_name)
            if site_id != -1:
                ee_positions.append(self.data.site_xpos[site_id].copy())
                ee_orientations.append(self.data.site_xmat[site_id].reshape(3, 3).copy())

        return np.array(ee_positions), np.array(ee_orientations)


    def get_all_ee_local_positions(self):
        ee_positions = []
        ee_orientations = []
        for i in range(self.total_robots):
            site_name = f"robot_{i}/hand_tip_site"
            site_id = mujoco.mj_name2id(self.sim_env, mujoco.mjtObj.mjOBJ_SITE, site_name)
            if site_id != -1:
                ee_positions.append(self.data.site_xpos[site_id].copy())
                ee_orientations.append(self.data.site_xmat[site_id].reshape(3, 3).copy())
        base_positions = self.get_base_positions()
        ee_positions = np.array(ee_positions) - base_positions

        return np.array(ee_positions), np.array(ee_orientations)
        