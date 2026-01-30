import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation as R
import time
from utils.robots import FrankaPandaRobot
from utils.env_builder import EnvironmentBuilder
from utils.scene_generator import SceneGenerator
import xml.etree.ElementTree as ET
from utils.tasks import State_Manager



robot = FrankaPandaRobot()

class EnvironmentConfig:
    row_spacing = 1.5 # Increased spacing for desks
    column_spacing = 1.5
    maximum_robots_per_row = 4
    env_template_path = "environments/templets/env_temp_1.xml"
    seed = 20

class RobotsConfig:
    names = ["franka_emika_panda"]
    quantities = [1]
    init_joint_positions: np.ndarray = np.array([[-0.0, 0.0, 0.0, -0.0, 0.0, 1.0, 0.0, 1.0, 1.0]*quantities[0]])  # Panda default pose

class SimulationConfig:
    time_step: float = 0.01
    # gui_refresh_rate: int = 1
    # physics_steps_per_control_step: int = 10

from utils.sim_engine import SimEngine
import json
import os
from pathlib import Path
def get_target_object_positions(robots_config: RobotsConfig, sim: SimEngine, scene_data: dict, *, local: bool = True):
    assert robots_config.quantities[0] == len(scene_data), "Mismatch between number of robots and scene data entries."
    target_object_locations = []
    for i, scene in enumerate(scene_data):
        locations = []
        base_name = f"robot_{i}"
        base_pos = sim.get_body_position_from_name(base_name)
        for obj in scene["objects"]:
            target_object_name = obj
            target_object_pos = sim.get_body_position_from_name(f"{base_name}/{target_object_name}")
            if local:
                locations.append(target_object_pos - base_pos)
            else:
                locations.append(target_object_pos)
        target_object_locations.append(locations)
    return np.array(target_object_locations)


def main():
    robots_config = RobotsConfig()
    robot_xml_paths = [os.path.join(Path(__file__).parent.resolve(),"robot_models", robot_name, "robot.xml") for robot_name in robots_config.names]

    env_config = EnvironmentConfig()
    xml_path = os.path.join(Path(__file__).parent, env_config.env_template_path)
    print(xml_path)

    # Generate Scene JSON
    scene_json_path = os.path.join(Path(__file__).parent, "scene", "pick_and_place_scene.json")
    generator = SceneGenerator(output_path=scene_json_path, num_robots=robots_config.quantities[0], seed=env_config.seed)
    
    # Tunable surface position [x, y, z] relative to robot
    surface_position = [0.6, 0.0, 0.0]
    generator.generate_scene(task="pick_and_place", surface_position=surface_position, min_objects=1, max_objects=1, collision=True)

    with open(scene_json_path, 'r') as f:
        scene_data = json.load(f)
    print(f"scene_data:\n{scene_data}")

    objects_dir = os.path.join(Path(__file__).parent, "objects")
    builder = EnvironmentBuilder(robot_xml_paths[0], 
                                 xml_path, 
                                 robots_config, 
                                 env_config, 
                                 scene_json_path=scene_json_path, 
                                 objects_dir=objects_dir,
                                 seed=env_config.seed)
    env_tree = builder.build(save_path="environments/built_envs/built_environment.xml")

    model = mujoco.MjModel.from_xml_path("environments/built_envs/built_environment.xml")

    # target_pos = np.array([0.0, 0.0, 0.0])
    target_rot_matrix = R.from_euler('xyz', [np.pi, 0, 0]).as_matrix()
    print("Target Rotation Matrix:\n", target_rot_matrix)

    scene_json_path = os.path.join(Path(__file__).parent, "scene", "pick_and_place_scene.json")
    with open(scene_json_path, 'r') as f:
        scene_data = json.load(f)
    print(f"scene_data:\n{scene_data}")

    sim_config = SimulationConfig()
    sim = SimEngine(sim_env=model, sim_config=sim_config, robots_config=robots_config)
    # Use the SimEngine-owned MjData everywhere to avoid desync between simulation state and viewer.
    data = sim.data
    n_robots = int(sim.total_robots)
    init_q = robots_config.init_joint_positions

    # Targets: use GLOBAL positions for IK (site_xpos is global). Pick the first object for each robot.
    target_pos_all = get_target_object_positions(robots_config, sim, scene_data, local=False)
    if target_pos_all.ndim != 3 or target_pos_all.shape[0] != n_robots or target_pos_all.shape[2] != 3:
        raise ValueError(f"Unexpected target_pos_all shape {target_pos_all.shape}; expected (N, num_obj, 3) with N={n_robots}.")
    if target_pos_all.shape[1] < 1:
        raise ValueError("Scene data contains no objects per robot; cannot build IK targets.")
    target_pos = target_pos_all[:, 0, :]  # (N,3)
    target_rot = np.repeat(target_rot_matrix[None, :, :], n_robots, axis=0)
    print(f"Target Positions batch shape: {target_pos.shape}")

    target_grip = 0.04  # OPEN
    gripper_max_speed = 0.02  # meters per second of joint position change


    # Gather site ids for all robots.
    ee_site_ids = []
    for i in range(n_robots):
        site_name = f"robot_{i}/hand_tip_site"
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if sid < 0:
            raise ValueError(
                f"Could not find site '{site_name}' in the model. "
                "Check the generated XML for the exact site name and robot count."
            )
        ee_site_ids.append(sid)
    ee_site_ids = np.asarray(ee_site_ids, dtype=np.int32)

    # Per-robot qpos/dof address tables (shape: N x 9)
    if sim.robot_joint_indices.size % n_robots != 0 or sim.robot_dof_indices.size % n_robots != 0:
        raise ValueError(
            f"Index arrays are not divisible by number of robots N={n_robots}. "
            f"robot_joint_indices.size={sim.robot_joint_indices.size}, robot_dof_indices.size={sim.robot_dof_indices.size}."
        )
    qpos_idx = sim.robot_joint_indices.reshape(n_robots, -1)
    dof_idx = sim.robot_dof_indices.reshape(n_robots, -1)
    if qpos_idx.shape[1] < 9 or dof_idx.shape[1] < 9:
        raise ValueError(f"Expected 9 joints per robot. Got qpos_idx.shape={qpos_idx.shape}, dof_idx.shape={dof_idx.shape}.")

    print(f"robot_joint_indices shape: {sim.robot_joint_indices.shape}")
    # Initialize all robots joint configuration (7 arm + 2 fingers) in the sim's joint index order.
    if init_q.size != sim.robot_joint_indices.size:
        raise ValueError(
            f"Expected init_q to have {sim.robot_joint_indices.size} elements (one per robot joint), "
            f"but got {init_q.size}."
        )
    sim.set_joint_controls(init_q)
    data.qvel[:] = 0.0
    sim.step()

    print(f"init_q reshaped: {init_q.reshape(n_robots, 9)}")
    save_data = {
        "init_q": init_q.copy(),
        "target_cart_pos": target_pos.copy(),
        "target_cart_rot": target_rot.copy(),
        "trajectory_include_init": [init_q.copy()],

    }

    state_manager = State_Manager(task_name="pick_and_place")
    full_state = state_manager.get_full_state()
    print(f"Full State Sequence: {full_state}")
    current_state = full_state[0]
    print(f"Current State: {current_state}")


    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.sync()

        while viewer.is_running():

            curr_ee_pos = data.site_xpos[ee_site_ids].copy()
            curr_grip = data.qpos[qpos_idx[:, 7:9]].copy()  # (N,2)
            diff  = np.linalg.norm(curr_ee_pos - target_pos)
            print(f"ee pos and target pos diff: {diff}")
            if diff < 0.01 and current_state != full_state[-1]:
                # Move to next state
                state_index = full_state.index(current_state)
                current_state = full_state[state_index + 1]
                print(f"Transitioning to next state: {current_state}")
                target_pos, target_grip = state_manager.get_state_target_qpos(current_state, target_pos)
                print(f"Current State: {current_state}, Target Position: {target_pos}, Target Grip: {target_grip}")
            elif diff < 0.01 and current_state == full_state[-1]:
                print("Final state reached and target achieved.")
                while True:
                    viewer.sync()
                    time.sleep(0.1)

            d_q = robot.solve_differential_ik(
                model=model,
                data=data,
                target_pos=target_pos,
                target_rot_matrix=target_rot,
                q0=init_q.reshape(n_robots, 9),
                ee_site_id=ee_site_ids,
                qpos_indices=qpos_idx,
                dof_indices=dof_idx,
            )

            # robot_joint_indices is an integer array of qpos addresses; index with it.
            curr_q_all = data.qpos[sim.robot_joint_indices].copy().reshape(n_robots, 9)
            


            # Update only arm joints for each robot; smoothly move finger joints to target grip.
            next_q_all = curr_q_all.copy()
            next_q_all[:, :7] = curr_q_all[:, :7] + d_q * robot.INTEGRATION_DT
            target_grip_arr = np.full((n_robots, 2), target_grip, dtype=np.float64)
            max_grip_step = gripper_max_speed * robot.INTEGRATION_DT
            grip_delta = np.clip(target_grip_arr - curr_grip, -max_grip_step, max_grip_step)
            next_q_all[:, 7:] = curr_grip + grip_delta

                
            sim.set_joint_controls(next_q_all.reshape(-1))
            data.qvel[sim.robot_dof_indices] = 0.0
            sim.step()
            viewer.sync()
            time.sleep(robot.INTEGRATION_DT)

if __name__ == "__main__":
    main()