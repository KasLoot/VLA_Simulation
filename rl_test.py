import mujoco as mj
import mujoco.viewer
import os
import xml.etree.ElementTree as ET
from pathlib import Path


from utils.sim_engine import SimEngine
from utils.env_builder import EnvironmentBuilder
from utils.scene_generator import SceneGenerator
import json


from dataclasses import dataclass, field
import time
import numpy as np
from utils.robots import FrankaPandaRobot

import matplotlib.pyplot as plt

from test_1 import SimpleModel, get_reward
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm

torch.manual_seed(42)
np.random.seed(42)


class EnvironmentConfig:
    row_spacing = 1.5 # Increased spacing for desks
    column_spacing = 1.5
    maximum_robots_per_row = 4
    env_template_path = "environments/templets/env_temp_1.xml"
    seed = 20

class RobotsConfig:
    names = ["franka_emika_panda"]
    quantities = [10]
    init_joint_positions: list = [[-0.0, 0.0, 0.0, -0.0, 0.0, 1.0, 0.0, 0.04, 0.04]*quantities[0]]  # Panda default pose

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

def get_target_object_positions(robots_config: RobotsConfig, sim: SimEngine, scene_data: dict):
    assert robots_config.quantities[0] == len(scene_data), "Mismatch between number of robots and scene data entries."
    target_object_locations = []
    for i, scene in enumerate(scene_data):
        locations = []
        base_name = f"robot_{i}"
        base_pos = sim.get_body_position_from_name(base_name)
        for obj in scene["objects"]:
            target_object_name = obj
            target_object_pos = sim.get_body_position_from_name(f"{base_name}/{target_object_name}")
            locations.append(target_object_pos - base_pos)
        target_object_locations.append(locations)
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
    generator.generate_scene(task="pick_and_place", surface_position=surface_position, min_objects=1, max_objects=1, collision=False)

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

    with mujoco.viewer.launch_passive(sim.sim_env, sim.data) as viewer:
        viewer.sync()
        # Set initial camera view
        viewer.cam.lookat[:] = [2.0, 2.0, 0.0] # Look at center of scene
        viewer.cam.distance = 8.0
        viewer.cam.azimuth = 135 # Angle around Z axis
        viewer.cam.elevation = -30 # Angle from horizon

        target_object_cartesian_positions = get_target_object_positions(robots_config, sim, scene_data).transpose((1,0,2))
        target_orientations = np.array([[0.0, np.pi/2, 0.0]] * robots_config.quantities[0])

        print(f"Target Positions shape:\n{target_object_cartesian_positions.shape}")
        print(f"Target Orientations shape:\n{target_orientations.shape}")


        start_ee_positions_local, start_ee_orientations = sim.get_all_ee_local_positions()
        print(f"Start EE Positions shape: {start_ee_positions_local.shape}, Start EE Orientations shape: {start_ee_orientations.shape}")
        start_joint_positions = sim.get_joint_positions()


        init_joint_positions = start_joint_positions.reshape(robots_config.quantities[0], -1)
        assert init_joint_positions.all() == np.array(robots_config.init_joint_positions[0]).reshape(robots_config.quantities[0], -1).all(), "Initial joint positions do not match configuration."
        print(f"Start Joint Positions shape: {start_joint_positions.shape}")

        target_pos_input = target_object_cartesian_positions[0] # Shape becomes (10, 3)
        print(f"Target Position Input:\n {target_pos_input}")

        # trajectory_joint_positions = robot.generate_trajectory(
        #     start_pos=start_ee_positions_local, 
        #     start_rot=start_ee_orientations, 
        #     target_pos=target_pos_input, # Use the specific object target
        #     target_rot=target_orientations,
        #     init_joint_positions=init_joint_positions,
        #     num_steps=200,
        #     ik_maxiter=10
        # )

        # print(f"Trajectory Joint Positions shape: {trajectory_joint_positions.shape}")

        # # plot joint trajectories for each robot in separate subplots, 4 colomns
        # num_robots = robots_config.quantities[0]
        # num_cols = 4
        # num_rows = (num_robots + num_cols - 1) // num_cols
        # fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 3*num_rows))
        # for i in range(num_robots):
        #     row = i // num_cols
        #     col = i % num_cols
        #     ax = axs[row, col] if num_rows > 1 else axs[col]
        #     for j in range(7): # 7 joints
        #         ax.plot(trajectory_joint_positions[i, :, j], label=f'Joint {j+1}')
        #     ax.set_title(f'Robot {i} Joint Trajectories')
        #     ax.set_xlabel('Time Step')
        #     ax.set_ylabel('Joint Position (rad)')
        #     ax.legend()
        # plt.tight_layout()
        # plt.show()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", device)
        model = SimpleModel().to(device).train()
        optimizer = optim.AdamW(model.parameters(), lr=0.001)
        # pos, ori = robot.forward_kinematics(init_joint_positions)
        # ori = robot.rotation_matrix_to_euler_angles(ori)
        # print(f"Initial FK Position shape: {pos.shape}, Initial FK Orientation shape: {ori.shape}")

        # state = torch.hstack([torch.tensor(pos, dtype=torch.float32), torch.tensor(ori, dtype=torch.float32)]).to(device)
        # print(f"Initial State shape: {state.shape}")
        # target = torch.hstack([torch.tensor(target_pos_input, dtype=torch.float32), torch.tensor(target_orientations, dtype=torch.float32)]).to(device)
        # print(f"target shape: {target.shape}")


        for epoch in range(100):
            sim.reset()
            viewer.sync()
            pos, ori = robot.forward_kinematics(init_joint_positions)
            ori = robot.rotation_matrix_to_euler_angles(ori)
            state = torch.hstack([torch.tensor(pos, dtype=torch.float32), torch.tensor(ori, dtype=torch.float32)]).to(device)
            # print(f"Initial State shape: {state.shape}")
            target = torch.hstack([torch.tensor(target_pos_input, dtype=torch.float32), torch.tensor(target_orientations, dtype=torch.float32)]).to(device)
            # print(f"target shape: {target.shape}")

            state_all = []
            log_probs_all = []
            reward_all = []
            state_all.append(state)
            current_joint_positions = init_joint_positions.copy()

            best_reward_mean = -float('inf')
            for step in tqdm(range(200), desc=f"Epoch {epoch+1}/100", leave=True):
                
                input_state = torch.hstack([state, target])
                # print(f"input_state shape: {input_state.shape}")

                action_mean, action_std = model(input_state)
                dist = torch.distributions.Normal(action_mean, action_std)
                delta_actions = dist.rsample() * 0.05
                
                pos, ori = pos + delta_actions[:, :3].detach().cpu().numpy(), ori + delta_actions[:, 3:].detach().cpu().numpy()
                actions = torch.hstack([torch.tensor(pos, dtype=torch.float32), torch.tensor(ori, dtype=torch.float32)]).to(device)

                log_probs = dist.log_prob(actions).sum(-1)
                log_probs_all.append(log_probs)

                joint_actions = robot.inverse_kinematics_optimization(
                    target_pos=actions[:, :3].detach().cpu().numpy(),
                    target_rot=robot.euler_angles_to_rotation_matrix(actions[:, 3:].detach().cpu().numpy()),
                    initial_q=current_joint_positions,
                    maxiter=10
                )
                joint_actions = joint_actions.reshape(robots_config.quantities[0], -1)

                lower_limits = robot.model.lowerPositionLimit
                upper_limits = robot.model.upperPositionLimit
                eps = 1e-5
                joint_actions = np.clip(joint_actions, lower_limits + eps, upper_limits - eps)

                current_joint_positions = joint_actions
                sim.set_joint_controls(joint_actions.flatten())
                sim.forward()
                viewer.sync()

                pos, ori = sim.get_all_ee_local_positions()
                pos = pos + delta_actions[:, :3].detach().cpu().numpy()
                ori = robot.rotation_matrix_to_euler_angles(ori)
                ori = ori + delta_actions[:, 3:].detach().cpu().numpy()
                state = torch.hstack([torch.tensor(pos, dtype=torch.float32), torch.tensor(ori, dtype=torch.float32)]).to(device)
                state_all.append(state)
                # time.sleep(0.01)
            
            state_all = torch.stack(state_all, dim=1)
            log_probs_all = torch.stack(log_probs_all, dim=1)
            # print(f"state_all shape after stacking: {state_all.shape}")
            # print(f"log_probs_all shape after stacking: {log_probs_all.shape}")

            reward = get_reward(state_all, target) * 0.1
            reward_mean = reward.mean().item()
            # print(f"reward shape: {reward.shape}")

            total_return = reward.sum(dim=1)
            returns_mean = total_return.mean()
            returns_std = total_return.std()
            normalized_returns = (total_return - returns_mean) / (returns_std + 1e-8)
            normalized_returns = normalized_returns.clamp(-2.0, 2.0)

            loss = - (log_probs_all * normalized_returns.unsqueeze(-1).detach()).mean()

            print(f"Std before backprop: {model.log_std.exp()}")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            print(f"Std after backprop: {model.log_std.exp()}")

            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Reward Mean: {reward_mean:.4f}, Std of Return: {returns_std.item():.4f}")

            if reward_mean > best_reward_mean:
                best_reward_mean = reward_mean

                model_to_save = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'reward_mean': reward_mean
                }

                torch.save(model_to_save, f"checkpoints/rl_model.pth")
        
        # 1. Get Global Final Positions
        final_ee_positions_local, final_ee_orientations = sim.get_all_ee_local_positions()

        # 4. Calculate True Error
        # We compare Local Actuals vs Local Targets
        diffs = np.abs(final_ee_positions_local - target_pos_input)

        print("Final Local EE Positions (Corrected):\n", final_ee_positions_local)
        print("Target Local EE Positions:\n", target_pos_input)
        print("True Position Differences (Local vs Local):\n", diffs)

        while viewer.is_running():
            continue
    
    time.sleep(1.0)

def main():
    run_simulation()


if __name__ == "__main__":
    main()