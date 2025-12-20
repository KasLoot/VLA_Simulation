import mujoco as mj
import mujoco.viewer
import os
import json
import xml.etree.ElementTree as ET
from pathlib import Path


from sim_engine import SimEngine
from scene_builder import EnvironmentBuilder
from scene_generator import SceneGenerator
from dashboard_cv2 import RobotDashboardCV2

from traj_genenrator import SinusoidalReference

from dataclasses import dataclass, field
import time
import numpy as np

from rl_model import RLModel_2

import torch
import torch.nn as nn
import torch.nn.functional as F
import pinocchio as pin

from robot import FrankaPandaRobot

from tqdm import tqdm

import matplotlib.pyplot as plt




class EnvironmentConfig:
    row_spacing = 1.5 # Increased spacing for desks
    column_spacing = 1.5
    maximum_robots_per_row = 4
    env_template_path = "environments/templets/environment.xml"
    seed = 20

class RobotsConfig:
    names = ["franka_emika_panda"]
    quantities = [10]
    init_joint_positions = None

class SimulationConfig:
    time_step: float = 0.002


class RLConfig:
    dtype = torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    iterations = 100
    step_per_iteration = 5000



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


def compute_reward(ee_pos_mes_all, target_pos):
    pass


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
    generator.generate_scene(task="pick_and_place", surface_position=surface_position, min_objects=1, collision=False)

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
    model = mj.MjModel.from_xml_string(env_tree)

    sim_config = SimulationConfig()
    sim = SimEngine(model, sim_config, robots_config)
    sim.reset()


    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        sim.reset()
        # Set initial camera view
        viewer.cam.lookat[:] = [2.0, 2.0, 0.0] # Look at center of scene
        viewer.cam.distance = 8.0
        viewer.cam.azimuth = 135 # Angle around Z axis
        viewer.cam.elevation = -30 # Angle from horizon

        target_object_locations = get_target_object_positions(robots_config, sim, scene_data)
        print(f"Target Object Locations shape: {target_object_locations.shape}")
        print(f"Target Object Locations\n: {target_object_locations}")

        # add orientation to target positions
        # Use [pi, 0, 0] to make the gripper point downwards
        target_object_locations = np.concatenate([target_object_locations, np.tile(np.array([np.pi, 0.0, 0.0]), (target_object_locations.shape[0], 1))], axis=1)
        print(f"Target Object Locations with Orientation shape: {target_object_locations.shape}")
        print(f"Target Object Locations with Orientation\n: {target_object_locations}")

        ee_pos = sim.get_all_ee_positions()
        print(f"EE Positions shape: {ee_pos.shape}")
        print(f"EE Positions:\n{ee_pos}")

        start_joint_positions = sim.get_joint_positions(include_gripper=True).reshape(robots_config.quantities[0], -1)
        print(f"Start Joint Positions shape: {start_joint_positions.shape}")
        print(f"Start Joint Positions:\n{start_joint_positions}")

        end_joint_positions = robot.inverse_kinematics(target_object_locations, start_joint_positions)
        print(f"End Joint Positions shape: {end_joint_positions.shape}")
        print(f"End Joint Positions:\n{end_joint_positions}")

        ee_start_positions = robot.forward_kinematics(start_joint_positions)
        print(f"Start EE Positions from FK shape: {ee_start_positions.shape}")
        print(f"Start EE Positions from FK:\n{ee_start_positions}")
        trajs = robot.generate_trajectory_from_ee_positions(ee_start_positions, target_object_locations, num_steps=200)
        print(f"Trajs shape before adding gripper joint: {trajs.shape}")
        trajs = trajs.transpose(1, 0, 2)  # (num_robots, num_steps, 7)
        # add 0.0 for gripper joint to each step
        gripper_joint = np.zeros((trajs.shape[0], trajs.shape[1], 1))
        trajs = np.concatenate([trajs, gripper_joint], axis=2).transpose(1, 0, 2)  # (num_robots, num_steps, 8)
        print(f"Trajs shape: {trajs.shape}")
        print(f"Trajs:\n{trajs}")


        
        for step in tqdm(range(trajs.shape[1])):
            q_d = trajs[:, step, :7]
            q_d = q_d.flatten()
            sim.set_joint_positions_direct(q_d)
            sim.step()

            # Visualization
            if viewer.is_running():
                viewer.user_scn.ngeom = 0 # Reset user geometries
                
                # Visualize EE positions (TCP)
                for i in range(robots_config.quantities[0]):
                    site_name = f"robot_{i}/tcp"
                    site_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
                    if site_id != -1:
                        pos = sim.data.site_xpos[site_id]
                    else:
                        # Fallback to hand body
                        pos = sim.data.xpos[sim.ee_body_ids[i]]

                    mujoco.mjv_initGeom(
                        viewer.user_scn.geoms[viewer.user_scn.ngeom],
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=[0.02, 0, 0],
                        pos=pos,
                        mat=np.eye(3).flatten(),
                        rgba=[0, 1, 0, 0.5] # Green for EE
                    )
                    viewer.user_scn.ngeom += 1

                # Visualize Target positions
                base_pos_global = sim.data.xpos[sim.base_body_ids]
                target_pos_global = target_object_locations[:, :3] + base_pos_global
                
                for pos in target_pos_global:
                    mujoco.mjv_initGeom(
                        viewer.user_scn.geoms[viewer.user_scn.ngeom],
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=[0.05, 0, 0],
                        pos=pos,
                        mat=np.eye(3).flatten(),
                        rgba=[1, 0, 0, 0.5] # Red for Target
                    )
                    viewer.user_scn.ngeom += 1

            viewer.sync()
            time.sleep(0.01)
        
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.01)

                

        






if __name__ == "__main__":
    run_simulation()