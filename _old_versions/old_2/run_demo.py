import mujoco as mj
import mujoco.viewer
import os
import xml.etree.ElementTree as ET
from pathlib import Path


from utils.sim_engine import SimEngine
from utils.scene_builder import EnvironmentBuilder
from utils.scene_generator import SceneGenerator
import json


from dataclasses import dataclass, field
import time
import numpy as np
from robot_models.robots import FrankaPandaRobot


class EnvironmentConfig:
    row_spacing = 1.5 # Increased spacing for desks
    column_spacing = 1.5
    maximum_robots_per_row = 4
    env_template_path = "environments/templets/environment.xml"
    seed = 20

class RobotsConfig:
    names = ["franka_emika_panda"]
    quantities = [10]
    init_joint_positions = [[0.0]*9]

class SimulationConfig:
    time_step: float = 0.01
    # gui_refresh_rate: int = 1
    # physics_steps_per_control_step: int = 10


def get_target_object_positions(robots_config: RobotsConfig, sim: SimEngine, scene_data: dict):
    assert robots_config.quantities[0] == len(scene_data), "Mismatch between number of robots and scene data entries."
    target_object_locations = []
    for i, scene in enumerate(scene_data):
        target_object_name = scene["objects"][0]
        base_name = f"robot_{i}"
        base_pos = sim.get_body_position_from_name(base_name)
        target_object_pos = sim.get_body_position_from_name(f"{base_name}/{target_object_name}")
        target_object_locations.append(target_object_pos - base_pos)
    return np.array(target_object_locations)


def run_simulation():
    robots_config = RobotsConfig()
    robot_xml_paths = [os.path.join(Path(__file__).parent.resolve(),"robot_models", robot_name, "robot.xml") for robot_name in robots_config.names]
    print(robot_xml_paths)
    
    env_config = EnvironmentConfig()
    xml_path = os.path.join(Path(__file__).parent, env_config.env_template_path)
    print(xml_path)

    robot = FrankaPandaRobot(model_path=robot_xml_paths[0])
    
    # Generate Scene JSON
    scene_json_path = os.path.join(Path(__file__).parent, "scene", "pick_and_place_scene.json")
    generator = SceneGenerator(output_path=scene_json_path, num_robots=robots_config.quantities[0], seed=env_config.seed)
    
    # Tunable surface position [x, y, z] relative to robot
    surface_position = [0.6, 0.0, 0.0]
    generator.generate_scene(task="pick_and_place", surface_position=surface_position, min_objects=1, max_objects=2, collision=False)

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

    env_tree = ET.tostring(env_tree, encoding='unicode')
    sim_env = mj.MjModel.from_xml_string(env_tree)

    sim_config = SimulationConfig()
    sim = SimEngine(sim_env=sim_env, sim_config=sim_config, robots_config=robots_config)
    sim.reset()

    with mujoco.viewer.launch_passive(sim.sim_env, sim.data) as viewer:
        # Set initial camera view
        viewer.cam.lookat[:] = [2.0, 2.0, 0.0] # Look at center of scene
        viewer.cam.distance = 8.0
        viewer.cam.azimuth = 135 # Angle around Z axis
        viewer.cam.elevation = -30 # Angle from horizon

        target_object_cartesian_positions = get_target_object_positions(robots_config, sim, scene_data)
        target_orientations = robot.euler_angles_to_rotation_matrix(np.array([[0.0, np.pi, 0.0]] * robots_config.quantities[0]))

        start_joint_positions = sim.get_joint_positions()
        start_joint_positions = start_joint_positions.reshape(robots_config.quantities[0], -1)
        print(f"Start Joint Positions shape: {start_joint_positions.shape}")

        end_joint_positions = robot.inverse_kinematics(target_translation=target_object_cartesian_positions, target_orientation=target_orientations, qs=start_joint_positions)
        print(f"End Joint Positions shape: {end_joint_positions.shape}")
        print(f"End Joint Positions:\n{end_joint_positions}")

        calculated_ee_translation, calculated_ee_orientation = robot.forward_kinematics(np.array(end_joint_positions))
        print(f"Calculated EE Translation:\n{calculated_ee_translation}")
        print(f"Calculated EE Orientation:\n{calculated_ee_orientation}")
        print(f"Target EE Translation:\n{target_object_cartesian_positions}")
        print(f"Target EE Orientation:\n{target_orientations}")

        trans_diff = calculated_ee_translation - target_object_cartesian_positions
        print(f"Translation Difference:\n{trans_diff}")
        ori_diff = calculated_ee_orientation - target_orientations
        print(f"Orientation Difference:\n{ori_diff}")
        sim.set_joint_controls(end_joint_positions.flatten())

        while True:
            sim.set_joint_controls(end_joint_positions.flatten())
            # sim.step()
            viewer.sync()



def main():
    run_simulation()


if __name__ == "__main__":
    main()