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
        self._init_robot_joints()

    def _init_robot_joints(self):
        self.robot_joint_ids = []
        self.qpos_indices = []
        self.dof_indices = []
        self.robot_joint_names = []
        # Optional gripper joint (we expose it as one DoF; internally there are two finger joints).
        self.gripper_joint_ids = []
        self.gripper_qpos_indices = []
        self.gripper_dof_indices = []
        self.gripper_joint_names = []
        self._finger2_qpos_indices = []
        self._finger2_dof_indices = []

        # Actuator ids (for joint-target control)
        self.actuator_ids = []  # (total_robots, 8)
        # End-effector: prefer the true TCP site (robot_i/tcp). Fall back to the hand body.
        self.ee_site_ids = []
        self.ee_body_ids = []
        self._ee_is_site = []  # True if TCP is a site, False if we fall back to a body
        self.base_body_ids = []
        
        total_robots = sum(self.robots_config.quantities)
        
        for i in range(total_robots):
            # Base Body
            base_name = f"robot_{i}"
            base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, base_name)
            if base_id != -1:
                self.base_body_ids.append(base_id)
            
            # EE: prefer tcp site
            tcp_name = f"robot_{i}/tcp"
            tcp_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, tcp_name)
            if tcp_site_id != -1:
                self.ee_site_ids.append(tcp_site_id)
                self.ee_body_ids.append(-1)
                self._ee_is_site.append(True)
            else:
                # Fallback to hand body
                ee_name = f"robot_{i}/hand"
                ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, ee_name)
                if ee_id == -1:
                    raise ValueError(
                        f"Neither tcp site '{tcp_name}' nor hand body '{ee_name}' was found in the model."
                    )
                self.ee_site_ids.append(-1)
                self.ee_body_ids.append(ee_id)
                self._ee_is_site.append(False)

            for j in range(1, 8): # joint1 to joint7
                joint_name = f"robot_{i}/joint{j}"
                joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                if joint_id != -1:
                    self.robot_joint_ids.append(joint_id)
                    self.qpos_indices.append(self.model.jnt_qposadr[joint_id])
                    self.dof_indices.append(self.model.jnt_dofadr[joint_id])
                    self.robot_joint_names.append(joint_name)

            # Gripper: expose as a single joint using finger_joint1 (finger_joint2 is kept equal by equality).
            finger1_name = f"robot_{i}/finger_joint1"
            finger2_name = f"robot_{i}/finger_joint2"
            finger1_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, finger1_name)
            finger2_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, finger2_name)
            if finger1_id != -1:
                self.gripper_joint_ids.append(finger1_id)
                self.gripper_qpos_indices.append(self.model.jnt_qposadr[finger1_id])
                self.gripper_dof_indices.append(self.model.jnt_dofadr[finger1_id])
                self.gripper_joint_names.append(finger1_name)
            if finger2_id != -1:
                self._finger2_qpos_indices.append(self.model.jnt_qposadr[finger2_id])
                self._finger2_dof_indices.append(self.model.jnt_dofadr[finger2_id])

            # Cache actuator ids (robot_i/actuator1..8) for robust mapping
            act_ids_i = []
            for a in range(1, 9):
                act_name = f"robot_{i}/actuator{a}"
                act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
                if act_id == -1:
                    raise ValueError(f"Actuator '{act_name}' not found in the model.")
                act_ids_i.append(act_id)
            self.actuator_ids.append(act_ids_i)

        self.actuator_ids = np.asarray(self.actuator_ids, dtype=np.int32)
        self.ee_site_ids = np.asarray(self.ee_site_ids, dtype=np.int32)
        self.ee_body_ids = np.asarray(self.ee_body_ids, dtype=np.int32)
        self._ee_is_site = np.asarray(self._ee_is_site, dtype=bool)

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

    def set_joint_targets(self, joint_targets):
        """Set desired joint positions (targets) for each robot.

        This is a *joint-space* interface:
            - joints 1..7 are in radians
            - gripper joint is finger slide position in meters (0..0.04)

        Internally this maps to actuator controls:
            - actuator1..7: pass-through (their ctrlrange matches joint ranges)
            - actuator8: expects ctrlrange 0..255 in this repo; we linearly map 0..0.04 -> 0..255
        """
        jt = np.asarray(joint_targets, dtype=np.float64)
        total_robots = sum(self.robots_config.quantities)
        dof = 8

        if jt.ndim == 1:
            if jt.size != total_robots * dof:
                raise ValueError(f"joint_targets must have size {total_robots*dof}, got {jt.size}")
            jt = jt.reshape(total_robots, dof)
        elif jt.ndim == 2:
            if jt.shape != (total_robots, dof):
                raise ValueError(f"joint_targets must have shape ({total_robots},{dof}), got {jt.shape}")
        else:
            raise ValueError(f"joint_targets must be 1D or 2D, got ndim={jt.ndim}")

        # Compose actuator ctrl values in actuator id order
        ctrl = np.zeros((total_robots, dof), dtype=np.float64)
        ctrl[:, :7] = jt[:, :7]

        # Gripper mapping: finger_q (0..0.04) -> ctrl (0..255)
        finger_q = np.clip(jt[:, 7], 0.0, 0.04)
        ctrl[:, 7] = (finger_q / 0.04) * 255.0
        ctrl[:, 7] = np.clip(ctrl[:, 7], 0.0, 255.0)

        # Write into data.ctrl by actuator ids
        # data.ctrl is indexed by actuator id
        for i in range(total_robots):
            self.data.ctrl[self.actuator_ids[i]] = ctrl[i]

    def set_joint_positions_direct(self, joint_positions, zero_vel: bool = True):
        """Directly set joint positions in qpos (bypasses actuators).

        This is *state setting* (kinematic teleport), not physical control.

        Args:
            joint_positions: shape (total_robots, 7) or (total_robots, 8) or flat.
                - first 7 are arm joints (rad)
                - optional 8th is gripper finger slide position (meters, 0..0.04)
            zero_vel: if True, also zero the corresponding qvel DoFs.
        """
        jp = np.asarray(joint_positions, dtype=np.float64)
        total_robots = sum(self.robots_config.quantities)

        if jp.ndim == 1:
            if jp.size not in (total_robots * 7, total_robots * 8):
                raise ValueError(
                    f"joint_positions must have size {total_robots*7} or {total_robots*8}, got {jp.size}"
                )
            dof = 7 if jp.size == total_robots * 7 else 8
            jp = jp.reshape(total_robots, dof)
        elif jp.ndim == 2:
            if jp.shape[0] != total_robots or jp.shape[1] not in (7, 8):
                raise ValueError(
                    f"joint_positions must have shape ({total_robots}, 7) or ({total_robots}, 8), got {jp.shape}"
                )
        else:
            raise ValueError(f"joint_positions must be 1D or 2D, got ndim={jp.ndim}")

        # Arm qpos
        arm_flat = jp[:, :7].reshape(-1)
        self.data.qpos[self.qpos_indices] = arm_flat

        # Optional gripper: set both finger joints to the same value
        if jp.shape[1] == 8 and len(self.gripper_qpos_indices) == total_robots:
            finger_q = np.clip(jp[:, 7], 0.0, 0.04)
            self.data.qpos[self.gripper_qpos_indices] = finger_q
            if len(self._finger2_qpos_indices) == total_robots:
                self.data.qpos[self._finger2_qpos_indices] = finger_q

        if zero_vel:
            self.data.qvel[self.dof_indices] = 0.0
            if len(self.gripper_dof_indices) == total_robots:
                self.data.qvel[self.gripper_dof_indices] = 0.0
            if len(self._finger2_dof_indices) == total_robots:
                self.data.qvel[self._finger2_dof_indices] = 0.0

        mujoco.mj_forward(self.model, self.data)

    def get_joint_positions(self, include_gripper: bool = False):
        """Return joint positions.

        Args:
            include_gripper: if True, returns 8 DoFs per robot (7 arm + finger_joint1).
                             if False (default), returns arm joints only (7 DoFs per robot).
        """
        arm = self.data.qpos[self.qpos_indices].copy()
        if not include_gripper:
            return arm

        total_robots = sum(self.robots_config.quantities)
        if len(self.gripper_qpos_indices) != total_robots:
            # No gripper joints found (or inconsistent model). Fall back to arm-only.
            return arm

        arm_mat = arm.reshape(total_robots, 7)
        grip = self.data.qpos[self.gripper_qpos_indices].copy().reshape(total_robots, 1)
        # Return grouped per robot: [r0(8), r1(8), ...]
        return np.concatenate([arm_mat, grip], axis=1).reshape(-1)
    
    def get_joint_velocities(self, include_gripper: bool = False):
        arm = self.data.qvel[self.dof_indices].copy()
        if not include_gripper:
            return arm

        total_robots = sum(self.robots_config.quantities)
        if len(self.gripper_dof_indices) != total_robots:
            return arm

        arm_mat = arm.reshape(total_robots, 7)
        grip = self.data.qvel[self.gripper_dof_indices].copy().reshape(total_robots, 1)
        return np.concatenate([arm_mat, grip], axis=1).reshape(-1)

    def get_joint_limits(self):
        limits = self.model.jnt_range[self.robot_joint_ids].copy()
        num_robots = sum(self.robots_config.quantities)
        arm_limits = limits.reshape(num_robots, -1, 2)
        if len(self.gripper_joint_ids) == 0:
            return arm_limits
        grip_limits = self.model.jnt_range[self.gripper_joint_ids].copy().reshape(num_robots, 1, 2)
        return np.concatenate([arm_limits, grip_limits], axis=1)

    def get_joint_names(self):
        return self.robot_joint_names

    def get_joint_names_with_gripper(self):
        return self.robot_joint_names + self.gripper_joint_names

    def get_joint_damping(self):
        return self.model.dof_damping[self.dof_indices].copy()

    def get_joint_stiffness(self):
        return self.model.jnt_stiffness[self.robot_joint_ids].copy()

    def get_joint_frictionloss(self):
        return self.model.dof_frictionloss[self.dof_indices].copy()
    
    def get_body_position_from_name(self, body_name):
        try:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id == -1:
                 raise ValueError(f"Body '{body_name}' not found in the model.")
        except Exception as exc:
            raise ValueError(f"Body '{body_name}' not found in the model.") from exc
        
        return self.data.xpos[body_id].copy()

    def get_site_position_from_name(self, site_name: str):
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id == -1:
            raise ValueError(f"Site '{site_name}' not found in the model.")
        return self.data.site_xpos[site_id].copy()

    def get_body_orientation_quat_from_name(self, body_name):
        try:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id == -1:
                 raise ValueError(f"Body '{body_name}' not found in the model.")
        except Exception as exc:
            raise ValueError(f"Body '{body_name}' not found in the model.") from exc
        
        return self.data.xquat[body_id].copy()

    def get_site_orientation_quat_from_name(self, site_name: str):
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id == -1:
            raise ValueError(f"Site '{site_name}' not found in the model.")
        # MuJoCo stores site_xmat as 9 floats; convert to quaternion.
        quat = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quat, self.data.site_xmat[site_id])
        return quat

    def get_ee_position_from_name(self, ee_name: str = "robot_1/tcp"):
        """Get EE position relative to the robot base.

        Prefers tcp site names (robot_i/tcp). If a body name is passed, it will be used as a fallback.
        """
        # Parse robot index from name (assuming format robot_{i}/...)
        try:
            robot_idx = int(ee_name.split('/')[0].split('_')[1])
            base_name = f"robot_{robot_idx}"
            base_pos = self.get_body_position_from_name(base_name)
            try:
                ee_pos = self.get_site_position_from_name(ee_name)
            except ValueError:
                ee_pos = self.get_body_position_from_name(ee_name)
            return ee_pos - base_pos
        except Exception:
            # Fallback to global if parsing fails
            try:
                return self.get_site_position_from_name(ee_name)
            except ValueError:
                return self.get_body_position_from_name(ee_name)
    
    def get_ee_orientation_quat_from_name(self, ee_name=None):
        assert ee_name is not None, "ee_name must be specified"
        try:
            return self.get_site_orientation_quat_from_name(ee_name)
        except ValueError:
            return self.get_body_orientation_quat_from_name(ee_name)
    
    def get_ee_orientation_euler_from_name(self, ee_name=None):
        assert ee_name is not None, "ee_name must be specified"
        quat = self.get_ee_orientation_quat_from_name(ee_name)
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

    def get_all_ee_positions(self):
        base_pos = self.data.xpos[self.base_body_ids]
        ee_pos = np.zeros_like(base_pos)

        # Per-robot to allow mixed site/body fallback
        for i in range(base_pos.shape[0]):
            if self._ee_is_site[i]:
                ee_pos[i] = self.data.site_xpos[self.ee_site_ids[i]]
            else:
                ee_pos[i] = self.data.xpos[self.ee_body_ids[i]]

        return (ee_pos - base_pos).copy()

    def get_all_ee_orientations_quat(self):
        n = len(self._ee_is_site)
        quats = np.zeros((n, 4), dtype=np.float64)
        for i in range(n):
            if self._ee_is_site[i]:
                mujoco.mju_mat2Quat(quats[i], self.data.site_xmat[self.ee_site_ids[i]])
            else:
                quats[i] = self.data.xquat[self.ee_body_ids[i]]
        return quats

    def get_all_ee_orientations_euler(self):
        quats = self.get_all_ee_orientations_quat()
        w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
        
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = np.arctan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = np.clip(t2, -1.0, 1.0)
        pitch_y = np.arcsin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = np.arctan2(t3, t4)

        return np.stack([roll_x, pitch_y, yaw_z], axis=1)
