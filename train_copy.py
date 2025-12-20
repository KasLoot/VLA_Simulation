import torch
import mujoco as mj
import mujoco.viewer
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from VLA_models.vla_1 import VLAModel_1


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

class TrainConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")


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

    train_config = TrainConfig()
    vla_model = VLAModel_1(in_features=9, hidden_features=128, out_features=6).to(train_config.device).to(torch.float32)
    optimizer = optim.AdamW(vla_model.parameters(), lr=1e-3, amsgrad=True)

    with mujoco.viewer.launch_passive(sim.sim_env, sim.data) as viewer:
        # Set initial camera view
        viewer.cam.lookat[:] = [2.0, 2.0, 0.0] # Look at center of scene
        viewer.cam.distance = 8.0
        viewer.cam.azimuth = 135 # Angle around Z axis
        viewer.cam.elevation = -30 # Angle from horizon
        
        mes_ee_positions, mes_ee_orientations = sim.get_all_ee_positions()
        print(f"mes_ee_positions shape: {mes_ee_positions.shape}, mes_ee_orientations shape: {mes_ee_orientations.shape}")

        target_object_cartesian_positions = get_target_object_positions(robots_config, sim, scene_data)
        target_orientations = robot.euler_angles_to_rotation_matrix(np.array([[0.0, np.pi, 0.0]] * robots_config.quantities[0]))
        target_object_cartesian_positions = torch.tensor(target_object_cartesian_positions, dtype=torch.float32).to(train_config.device)
        target_orientations = torch.tensor(target_orientations, dtype=torch.float32).to(train_config.device)

        while True:
            mes_ee_positions, mes_ee_orientations = sim.get_all_ee_positions()
            mes_ee_orientations_euler = robot.rotation_matrix_to_euler_angles(mes_ee_orientations)
            mes_states = np.concatenate((mes_ee_positions, mes_ee_orientations_euler), axis=-1)
            # print(f"mes_states shape: {mes_states.shape}")

            mes_joint_positions = sim.get_joint_positions().reshape(robots_config.quantities[0], -1)
            # print(f"mes_joint_positions shape: {mes_joint_positions.shape}")
            
            input_tensor = torch.concat((torch.tensor(mes_states).to(train_config.device), target_object_cartesian_positions), dim=-1).to(torch.float32)
            # print(f"input_tensor shape: {input_tensor.shape}")

            

            optimizer.zero_grad()
            predicted_ee_positions = vla_model(input_tensor).detach().cpu().numpy()
            # print(f"predicted_ee_positions shape: {predicted_ee_positions.shape}")
            # print(f"predicted_ee_positions[:, :3]: {predicted_ee_positions[:, :3]}")
            # print(f"predicted_ee_positions[:, 3:]: {predicted_ee_positions[:, 3:]}")

            predicted_orientations_matrix = robot.euler_angles_to_rotation_matrix(predicted_ee_positions[:, 3:])
            joint_controls = robot.inverse_kinematics(predicted_ee_positions[:, :3], predicted_orientations_matrix, qs=mes_joint_positions)
            # print(f"joint_controls shape: {joint_controls.shape}")

            sim.set_joint_controls(joint_controls.flatten())

            viewer.sync()
            # break



def main():
    run_simulation()


if __name__ == "__main__":
    main()