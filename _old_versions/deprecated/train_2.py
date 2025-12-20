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





class EnvironmentConfig:
    row_spacing = 1.5 # Increased spacing for desks
    column_spacing = 1.5
    maximum_robots_per_row = 4
    env_template_path = "environments/templets/environment.xml"
    seed = 20

class RobotsConfig:
    names = ["franka_emika_panda"]
    quantities = [1]
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

    train_config = RLConfig()
    # Use RLModel for stochastic policy (Gaussian)
    rl_model = RLModel_2(state_dim=6, action_dim=3).to(train_config.dtype).to(train_config.device)
    optimizer = torch.optim.AdamW(rl_model.parameters(), lr=1e-4, amsgrad=True)

    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        # Set initial camera view
        viewer.cam.lookat[:] = [2.0, 2.0, 0.0] # Look at center of scene
        viewer.cam.distance = 8.0
        viewer.cam.azimuth = 135 # Angle around Z axis
        viewer.cam.elevation = -30 # Angle from horizon

        target_object_locations = get_target_object_positions(robots_config, sim, scene_data)
        target_object_locations = torch.tensor(target_object_locations, dtype=train_config.dtype, device=train_config.device)
        print(f"Target Object Locations: {target_object_locations}")
        prebuilt_target_object_locations = target_object_locations.repeat(train_config.step_per_iteration, 1, 1)
        print(f"Prebuilt Target Object Locations shape: {prebuilt_target_object_locations.shape}")
        # print(f"Prebuilt Target Object Locations: {prebuilt_target_object_locations[:10]}")

        step_reward_scheduler = torch.logspace(0, 1, steps=train_config.step_per_iteration, base=10.0).to(train_config.dtype).to(train_config.device)
        print(f"Scheduler values.shape: {step_reward_scheduler.shape}")

        currerent_iteration = 0
        ee_pos_mes_all = []

        # Wrap the iteration loop with tqdm for progress tracking
        with tqdm(total=train_config.iterations, desc="Training Progress") as pbar:
            while viewer.is_running() and currerent_iteration < train_config.iterations:
                sim.reset()
                ee_pos_mes_iter = []
                states, actions, rewards, log_probs = [], [], [], []

                for step in range(train_config.step_per_iteration):
                    curr_ee_pos = torch.tensor(sim.get_all_ee_positions()).to(train_config.dtype).to(train_config.device)
                    ee_pos_mes_iter.append(curr_ee_pos)

                    curr_q_pos = torch.tensor(sim.get_joint_positions()).to(train_config.dtype).to(train_config.device)

                    obs = torch.hstack([target_object_locations, curr_ee_pos])

                    # Get action distribution and sample
                    dist = rl_model.get_dist(obs)
                    action = dist.sample()
                    log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
                    log_probs.append(log_prob)

                    pred_action = action.detach().cpu().numpy()

                    ee_control = robot.inverse_kinematics(target_pos=pred_action, 
                                                          init_q=curr_q_pos, 
                                                          max_iter=10, eps=1e-4, dt=0.1, damp=1e-6).detach().cpu().numpy()

                    sim.set_control(ee_control.flatten())
                    sim.step()
                    viewer.sync()


                ee_pos_mes_iter = torch.stack(ee_pos_mes_iter)
                
                distances = torch.norm(ee_pos_mes_iter - prebuilt_target_object_locations, dim=-1)
                squared_diff = distances ** 2
                rewards = -squared_diff * step_reward_scheduler[:, None]

                # --- RL Update (REINFORCE) ---
                log_probs = torch.cat(log_probs) # (T, 1)
                
                # Compute returns
                returns = []
                G = 0
                gamma = 0.99
                for r in reversed(rewards):
                    G = r + gamma * G
                    returns.insert(0, G)
                returns = torch.stack(returns)
                
                # Normalize returns
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)
                
                # Loss
                loss = -(log_probs * returns).mean()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                currerent_iteration += 1
                pbar.set_postfix({"Reward": rewards.sum().item(), "Loss": loss.item()})
                pbar.update(1)  # Update tqdm progress bar

                ee_pos_mes_all.append(ee_pos_mes_iter)

        






if __name__ == "__main__":
    run_simulation()