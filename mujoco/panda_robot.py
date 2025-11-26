import mujoco
import numpy as np

class PandaRobot:
    def __init__(self, model, data, index):
        self.model = model
        self.data = data
        self.index = index
        self.prefix = f"robot{index+1}_"
        
        # Find actuator indices
        self.actuator_ids = []
        
        # We assume standard panda joint names prefixed
        # actuator names: actuator1, ..., actuator8
        for i in range(1, 9): # 7 joints + gripper
            name = f"{self.prefix}actuator{i}"
            id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if id != -1:
                self.actuator_ids.append(id)
            else:
                print(f"Warning: Actuator {name} not found for robot {index+1}")

        # Joints
        self.joint_ids = []
        for i in range(1, 8):
            name = f"{self.prefix}joint{i}"
            id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if id != -1:
                self.joint_ids.append(id)
        
        # Gripper joints
        self.gripper_joint_ids = []
        for i in range(1, 3):
            name = f"{self.prefix}finger_joint{i}"
            id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if id != -1:
                self.gripper_joint_ids.append(id)

        # Camera
        self.camera_name = f"{self.prefix}ee_cam"
        self.camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera_name)
        
    def set_control(self, ctrl_values):
        """
        Set control values for the robot.
        ctrl_values: list or array of control inputs.
        """
        # Handle 9-DOF input (7 arm + 2 gripper) -> 8-DOF control (7 arm + 1 gripper)
        if len(ctrl_values) == 9 and len(self.actuator_ids) == 8:
            arm_ctrl = ctrl_values[:7]
            # Average the two gripper finger positions
            avg_finger_pos = (ctrl_values[7] + ctrl_values[8]) / 2.0
            # Map 0-0.04m to 0-255 control range
            gripper_ctrl = (avg_finger_pos / 0.04) * 255
            
            if isinstance(ctrl_values, np.ndarray):
                ctrl_values = np.append(arm_ctrl, gripper_ctrl)
            else:
                ctrl_values = list(arm_ctrl) + [gripper_ctrl]

        if len(ctrl_values) > len(self.actuator_ids):
            print(f"Warning: Too many control values for robot {self.index+1}. Expected {len(self.actuator_ids)}, got {len(ctrl_values)}")
            return
            
        for i, val in enumerate(ctrl_values):
            if i < len(self.actuator_ids):
                self.data.ctrl[self.actuator_ids[i]] = val

    def get_camera_view(self, renderer):
        """
        Render the view from the robot's end-effector camera.
        """
        if self.camera_id != -1:
            renderer.update_scene(self.data, camera=self.camera_id)
            return renderer.render()
        return None

    def get_base_position(self):
        """
        Get the position of the robot's base link.
        """
        body_name = f"{self.prefix}link0"
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id != -1:
            return self.data.xpos[body_id]
        return np.array([0, 0, 0])

    def get_joint_positions(self):
        """
        Get the joint positions of the robot.
        Returns:
            np.ndarray: Array of joint positions (7 arm joints + 2 gripper joints).
        """
        qpos = []
        # Arm joints
        for id in self.joint_ids:
            qpos.append(self.data.qpos[self.model.jnt_qposadr[id]])
        
        # Gripper joints
        for id in self.gripper_joint_ids:
            qpos.append(self.data.qpos[self.model.jnt_qposadr[id]])
            
        return np.array(qpos)

    def get_ee_pose(self):
        """
        Get the position and orientation of the end-effector (hand).
        Returns:
            pos (np.ndarray): [x, y, z]
            quat (np.ndarray): [w, x, y, z]
        """
        body_name = f"{self.prefix}hand"
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id != -1:
            return self.data.xpos[body_id], self.data.xquat[body_id]
        return np.zeros(3), np.array([1, 0, 0, 0])
