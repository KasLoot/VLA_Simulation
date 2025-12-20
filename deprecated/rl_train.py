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

from rl_model import RLModel

import torch
import torch.nn as nn
import torch.nn.functional as F
import pinocchio as pin

class PinocchioIK:
    def __init__(self, model_path):
        self.model = pin.buildModelFromMJCF(model_path)
        self.data = self.model.createData()
        self.frame_name = "hand"
        if not self.model.existFrame(self.frame_name):
             self.frame_name = self.model.frames[-1].name
        self.frame_id = self.model.getFrameId(self.frame_name)
        
    def solve(self, target_pos, init_q, max_iter=20, eps=1e-4, dt=0.1, damp=1e-6):
        q = init_q.clone()
        # Pinocchio expects numpy array
        if isinstance(q, torch.Tensor):
            q = q.detach().cpu().numpy()
        if isinstance(target_pos, torch.Tensor):
            target_pos = target_pos.detach().cpu().numpy()
            
        q_full = pin.neutral(self.model)
        
        if len(q) == 7:
            q_full[:7] = q
        else:
            q_full[:len(q)] = q 
            
        for i in range(max_iter):
            pin.framesForwardKinematics(self.model, self.data, q_full)
            oMtool = self.data.oMf[self.frame_id]
            current_pos = oMtool.translation
            
            err = target_pos - current_pos
            if np.linalg.norm(err) < eps:
                break
                
            J = pin.computeFrameJacobian(self.model, self.data, q_full, self.frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            J_pos = J[:3, :]
            
            # We only want to update the first 7 joints.
            J_arm = J_pos[:, :7]
            
            jj_t = J_arm @ J_arm.T
            rhs = err
            lambda_eye = damp * np.eye(3)
            
            x = np.linalg.solve(jj_t + lambda_eye, rhs)
            dq_arm = J_arm.T @ x
            
            v = np.zeros(self.model.nv)
            v[:7] = dq_arm * dt
            q_full = pin.integrate(self.model, q_full, v)
            
        return q_full[:7]




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
    # gui_refresh_rate: int = 1
    # physics_steps_per_control_step: int = 10

@torch.no_grad()
def compute_reward(
    ee_pos: torch.Tensor,
    target_pos: torch.Tensor,
    action: torch.Tensor,
    prev_dist: torch.Tensor | None = None,
    dist_threshold: float = 0.05,
):
    """Per-step reward.

    Shapes (single robot):
      ee_pos: (3,), target_pos: (3,), action: (7,)
    Shapes (N robots):
      ee_pos: (N,3), target_pos: (N,3), action: (N,7)
    """
    # Distance to goal
    dist = torch.norm(ee_pos - target_pos, dim=-1)  # (N,)

    # 1) Distance reward: close -> near 1, far -> near 0 (prevents inflated rewards at large distances)
    # Tune k depending on typical distance scale in meters.
    k = 10.0
    r_dist = torch.exp(-k * dist)

    # 2) Progress shaping: reward reduction in distance since previous step
    if prev_dist is None:
        r_progress = torch.zeros_like(dist)
    else:
        r_progress = 5.0 * (prev_dist - dist)

    # 3) Action magnitude penalty (assumes action is a *delta* command)
    r_action = -0.01 * torch.sum(action.detach()**2, dim=-1)  # (N,)

    # 4) Success bonus
    is_success = dist < dist_threshold
    r_bonus = is_success.to(dtype=torch.float32) * 10.0  # Increased bonus for reaching goal

    total_reward = r_dist + r_progress + r_action + r_bonus
    return total_reward, is_success, dist


def compute_discounted_returns(rewards: torch.Tensor, gamma: float) -> torch.Tensor:
    """Compute discounted returns.

    rewards: (T,) or (T,N)
    returns: same shape
    """
    returns = torch.zeros_like(rewards)
    running = torch.zeros_like(rewards[-1])
    for t in reversed(range(rewards.shape[0])):
        running = rewards[t] + gamma * running
        returns[t] = running
    return returns


def run_simulation():


    robots_config = RobotsConfig()
    robot_xml_path = [os.path.join(Path(__file__).parent.resolve(),"robot_models", robot_name, "robot.xml") for robot_name in robots_config.names]
    print(robot_xml_path)
    
    ik_solver = PinocchioIK(robot_xml_path[0])

    env_config = EnvironmentConfig()
    xml_path = os.path.join(Path(__file__).parent, env_config.env_template_path)
    print(xml_path)
    
    # Generate Scene JSON
    scene_json_path = os.path.join(Path(__file__).parent, "scene", "pick_and_place_scene.json")
    generator = SceneGenerator(output_path=scene_json_path, num_robots=robots_config.quantities[0], seed=env_config.seed)
    
    # Tunable surface position [x, y, z] relative to robot
    surface_position = [0.6, 0.0, 0.0]
    generator.generate_scene(task="pick_and_place", surface_position=surface_position, min_objects=1, collision=False)
    
    # Load scene data for object tracking
    with open(scene_json_path, 'r') as f:
        scene_data = json.load(f)
    scene_dict = {item['robot_index']: item for item in scene_data}

    # Objects Directory
    objects_dir = os.path.join(Path(__file__).parent, "objects")

    builder = EnvironmentBuilder(robot_xml_path[0], 
                                 xml_path, 
                                 robots_config, 
                                 env_config, 
                                 scene_json_path=scene_json_path, 
                                 objects_dir=objects_dir,
                                 seed=env_config.seed)
    env_tree = builder.build(save_path="environments/built_envs/built_environment.xml")
    # print(env_tree)
    
    xml_string = ET.tostring(env_tree, encoding='unicode')
    model = mj.MjModel.from_xml_string(xml_string)

    sim_config = SimulationConfig()
    sim = SimEngine(model, sim_config, robots_config)

    sim.reset()


    # Dashboard Setup
    show_names = False
    def toggle_names(state):
        nonlocal show_names
        show_names = state

    dashboard = RobotDashboardCV2(num_robots=robots_config.quantities[0], toggle_names_callback=toggle_names, ui=False)

    # Initialize Renderer for cameras (smaller size for speed)
    renderer = mj.Renderer(sim.model, height=120, width=160)

    joint_limits = sim.get_joint_limits()
    print("Joint Names:\n", sim.get_joint_names())
    print("Joint Limits Shape:", joint_limits.shape)
    print("Joint Limits:\n", joint_limits)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    rl_model = RLModel(state_dim=20, action_dim=3).to(torch.float32).to(device)
    # Use lower learning rate for more stable training
    optimizer = torch.optim.Adam(rl_model.parameters(), lr=3e-4, eps=1e-5)

    rl_iterations = 500
    episodes_per_batch = 8
    steps_per_rollout = int(3 / sim_config.time_step)

    # Use actuator ctrlrange for action bounds (more correct than joint limits)
    # For Panda this usually includes 8 actuators (7 joints + gripper). We control the first 7.
    total_robots = robots_config.quantities[0]
    nu = sim.model.nu
    per_robot_nu = max(1, nu // max(1, total_robots))
    ctrlrange = torch.as_tensor(sim.model.actuator_ctrlrange, dtype=torch.float32, device=device)  # (nu,2)
    ctrlrange = ctrlrange[: total_robots * per_robot_nu].reshape(total_robots, per_robot_nu, 2)
    act_low = ctrlrange[:, :7, 0]
    act_high = ctrlrange[:, :7, 1]

    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        # Set initial camera view
        viewer.cam.lookat[:] = [2.0, 2.0, 0.0] # Look at center of scene
        viewer.cam.distance = 8.0
        viewer.cam.azimuth = 135 # Angle around Z axis
        viewer.cam.elevation = -30 # Angle from horizon

        current_time = 0.0
        start_time = time.time()
        last_dashboard_update = 0

        # Get object positions for each robot scene
        def get_all_object_positions():
            object_positions = []
            for i in range(robots_config.quantities[0]):
                robot_obj_pos = {}
                if i in scene_dict:
                    objects = scene_dict[i].get('objects', [])
                    
                    # Get base position for this robot
                    base_name = f"robot_{i}"
                    try:
                        base_pos = sim.get_body_position_from_name(base_name)
                    except ValueError:
                        object_positions.append({})
                        continue

                    for obj_name in objects:
                        full_obj_name = f"robot_{i}/{obj_name}"
                        try:
                            pos = sim.get_body_position_from_name(full_obj_name)
                            robot_obj_pos[obj_name] = pos - base_pos
                        except ValueError:
                            pass
                object_positions.append(robot_obj_pos)
            
            print(f"Object Positions: {object_positions}")
            return object_positions
        
        object_positions = get_all_object_positions()
        
        # Get the first object position for each robot as target
        targets = []
        for robot_objs in object_positions:
            if len(robot_objs) > 0:
                # Use the first object as target (avoid index errors when only 1 object exists)
                targets.append(list(robot_objs.values())[-1])
            else:
                targets.append(np.array([0.6, 0.0, 0.0])) # Default

        ee_pos_target = torch.as_tensor(np.array(targets), dtype=torch.float32, device=device)

        q_mes = torch.as_tensor(sim.get_joint_positions(), dtype=torch.float32, device=device).reshape(total_robots, -1)
        qd_mes = torch.as_tensor(sim.get_joint_velocities(), dtype=torch.float32, device=device).reshape(total_robots, -1)
        ee_pos_mes = torch.as_tensor(sim.get_all_ee_positions(), dtype=torch.float32, device=device).reshape(total_robots, -1)
        print(f"targets: {ee_pos_target}")
        
        while viewer.is_running() and dashboard.running:

            for iter in range(rl_iterations):
                
                batch_log_probs = []
                batch_rewards = []
                batch_entropies = []
                
                # Metrics for logging
                batch_ep_returns = []
                batch_final_dists = []

                for ep in range(episodes_per_batch):
                    sim.reset()
                    
                    # Update state after reset
                    q_mes = torch.as_tensor(sim.get_joint_positions(), dtype=torch.float32, device=device).reshape(total_robots, -1)
                    qd_mes = torch.as_tensor(sim.get_joint_velocities(), dtype=torch.float32, device=device).reshape(total_robots, -1)
                    ee_pos_mes = torch.as_tensor(sim.get_all_ee_positions(), dtype=torch.float32, device=device).reshape(total_robots, -1)

                    episode_log_probs = []
                    episode_rewards = []
                    episode_entropies = []

                    # Track distance for progress-based shaping
                    prev_dist = torch.norm(ee_pos_mes - ee_pos_target, dim=-1)

                    # Max change in joint target per step (radians). Tune for stability.
                    max_delta = 0.05
                    
                    for step in range(steps_per_rollout):

                        # step_start = time.time()
                        
                        # Set Controls for all robots here: use sim.set_control(control_cmds)


                        state = torch.hstack((q_mes, qd_mes, ee_pos_mes, ee_pos_target)).to(torch.float32)

                        # Sample stochastic action for exploration and log_prob for policy gradient.
                        # IMPORTANT: use a bounded action with a consistent log_prob (tanh-squashed Gaussian).
                        base_dist = rl_model.get_dist(state)
                        raw = base_dist.rsample()  # (N,3)
                        squashed = torch.tanh(raw)  # (N,3) in (-1,1)
                        delta = squashed * max_delta  # (N,3) small EE increments

                        # Change-of-variables correction for tanh squashing (Jacobian correction)
                        # Note: sum log-prob per action dim, then subtract sum of log-jacobian corrections
                        eps = 1e-6
                        log_prob_raw = base_dist.log_prob(raw)  # (N, 3)
                        # Jacobian correction: log(1 - tanh^2(x)) for each dimension
                        log_jacobian = torch.log(1.0 - squashed.pow(2) + eps)  # (N, 3)
                        log_prob = (log_prob_raw - log_jacobian).sum(dim=-1)  # (N,)
                        entropy = base_dist.entropy().sum(dim=-1)

                        # Convert delta -> absolute EE target
                        target_ee_next = ee_pos_mes + delta
                        
                        # Run IK for each robot
                        controls_list = []
                        for i in range(total_robots):
                            q_sol = ik_solver.solve(target_ee_next[i], q_mes[i])
                            controls_list.append(q_sol)
                        
                        controls = np.concatenate(controls_list)
                        
                        # Clamp controls to joint limits
                        act_low_np = act_low.cpu().numpy().reshape(-1)
                        act_high_np = act_high.cpu().numpy().reshape(-1)
                        controls = np.clip(controls, act_low_np, act_high_np)
                        
                        sim.set_control(controls)
                        
                        sim.step() 

                        q_mes = torch.as_tensor(sim.get_joint_positions(), dtype=torch.float32, device=device).reshape(total_robots, -1)
                        qd_mes = torch.as_tensor(sim.get_joint_velocities(), dtype=torch.float32, device=device).reshape(total_robots, -1)
                        ee_pos_mes = torch.as_tensor(sim.get_all_ee_positions(), dtype=torch.float32, device=device).reshape(total_robots, -1)
                        # ee_pos_target is already calculated outside
                        # print("q_mes shape:", q_mes.shape)
                        # print("qd_mes shape:", qd_mes.shape)
                        # print("ee_pos_mes shape:", ee_pos_mes.shape)
                        # print("ee_pos_target shape:", ee_pos_target.shape)
                        
                        # Per-step reward (computed after the environment transition)
                        r_t, _, dist_t = compute_reward(ee_pos_mes, ee_pos_target, delta, prev_dist=prev_dist)
                        prev_dist = dist_t

                        episode_rewards.append(r_t)
                        episode_log_probs.append(log_prob)
                        episode_entropies.append(entropy)

                        # Sync Viewer
                        # --- Viewer overlays (user scene) ---
                        # Clear every frame to avoid accumulating geoms.
                        viewer.user_scn.ngeom = 0

                        # 1) Optional robot name labels
                        if show_names:
                            for i in range(robots_config.quantities[0]):
                                body_id = mj.mj_name2id(sim.model, mj.mjtObj.mjOBJ_BODY, f"robot_{i}")
                                if body_id >= 0:
                                    pos = sim.data.xpos[body_id]
                                    mj.mjv_initGeom(
                                        viewer.user_scn.geoms[viewer.user_scn.ngeom],
                                        type=mj.mjtGeom.mjGEOM_LABEL,
                                        size=np.zeros(3),
                                        pos=pos + np.array([0, 0, 1.0]),
                                        mat=np.eye(3).flatten(),
                                        rgba=np.array([1, 1, 1, 1], dtype=np.float32),
                                    )
                                    viewer.user_scn.geoms[viewer.user_scn.ngeom].label = f"Robot {i}"
                                    viewer.user_scn.ngeom += 1

                        # 2) End-effector marker (small sphere at ee position)
                        ee_radius = 0.03
                        for i in range(robots_config.quantities[0]):
                            # Prefer TCP site at gripper tip; fall back to hand body.
                            tcp_site_id = mj.mj_name2id(sim.model, mj.mjtObj.mjOBJ_SITE, f"robot_{i}/tcp")
                            if tcp_site_id >= 0:
                                ee_pos_world = sim.data.site_xpos[tcp_site_id]
                            else:
                                ee_body_id = mj.mj_name2id(sim.model, mj.mjtObj.mjOBJ_BODY, f"robot_{i}/hand")
                                if ee_body_id < 0:
                                    continue
                                ee_pos_world = sim.data.xpos[ee_body_id]

                            mj.mjv_initGeom(
                                viewer.user_scn.geoms[viewer.user_scn.ngeom],
                                type=mj.mjtGeom.mjGEOM_SPHERE,
                                size=np.array([ee_radius, 0.0, 0.0]),
                                pos=ee_pos_world,
                                mat=np.eye(3).flatten(),
                                rgba=np.array([0.2, 1.0, 0.2, 0.9], dtype=np.float32),
                            )
                            viewer.user_scn.ngeom += 1

                        # 3) Target marker (small sphere at target position)
                        target_radius = 0.05
                        targets_np = ee_pos_target.detach().cpu().numpy()
                        for i in range(robots_config.quantities[0]):
                            target_pos_world = targets_np[i]

                            mj.mjv_initGeom(
                                viewer.user_scn.geoms[viewer.user_scn.ngeom],
                                type=mj.mjtGeom.mjGEOM_SPHERE,
                                size=np.array([target_radius, 0.0, 0.0]),
                                pos=target_pos_world,
                                mat=np.eye(3).flatten(),
                                rgba=np.array([1.0, 0.2, 0.2, 0.9], dtype=np.float32),
                            )
                            viewer.user_scn.ngeom += 1

                        viewer.sync()

                        # Update Dashboard (limit to 30Hz for smoother video)
                        if time.time() - last_dashboard_update > 0.033:
                            last_dashboard_update = time.time()
                            
                            for i in range(robots_config.quantities[0]):
                                # 1. Render Camera
                                cam_name = f"robot_{i}/ee_cam"
                                renderer.update_scene(sim.data, camera=cam_name)
                                img = renderer.render()
                                
                                # 2. Get Telemetry
                                # Joints: qpos. Need to find address.
                                # The robot joints are named robot_{i}/joint1 ... joint7
                                # We can find the qpos address of the first joint
                                j1_id = mj.mj_name2id(sim.model, mj.mjtObj.mjOBJ_JOINT, f"robot_{i}/joint1")
                                if j1_id >= 0:
                                    qpos_adr = sim.model.jnt_qposadr[j1_id]
                                    joints = sim.data.qpos[qpos_adr:qpos_adr+7]
                                else:
                                    joints = np.zeros(7)

                                # EE Pose
                                ee_pos = sim.get_ee_position_from_name(body_name=f"robot_{i}/hand")
                                ee_quat = sim.get_ee_orientation_quat_from_name(body_name=f"robot_{i}/hand")

                                dashboard.update_robot(i, img, joints, ee_pos, ee_quat)
                            
                            dashboard.update()

                        # # Time keeping
                        # time_until_next_step = sim.model.opt.timestep - (time.time() - step_start)
                        # if time_until_next_step > 0:
                        #     time.sleep(time_until_next_step) 

                        current_time += sim.model.opt.timestep

                    # End of episode processing
                    batch_log_probs.append(torch.stack(episode_log_probs)) # (T, N)
                    batch_rewards.append(torch.stack(episode_rewards))     # (T, N)
                    batch_entropies.append(torch.stack(episode_entropies)) # (T, N)
                    
                    batch_ep_returns.append(torch.stack(episode_rewards).sum(dim=0).mean().item())
                    batch_final_dists.append(dist_t.mean().item())

                # === Policy Update (after batch) ===
                # Compute returns for each episode
                batch_returns = []
                gamma = 0.99
                for rewards_t in batch_rewards:
                    returns_t = compute_discounted_returns(rewards_t, gamma=gamma)
                    batch_returns.append(returns_t)
                
                # Concatenate all data
                all_log_probs = torch.cat(batch_log_probs, dim=0) # (Batch*T, N)
                all_returns = torch.cat(batch_returns, dim=0)     # (Batch*T, N)
                all_entropies = torch.cat(batch_entropies, dim=0) # (Batch*T, N)
                
                # Normalize returns across the entire batch
                returns_mean = all_returns.mean()
                returns_std = all_returns.std() + 1e-8
                returns_normalized = (all_returns - returns_mean) / returns_std
                returns_normalized = returns_normalized.detach()

                ent_coef = 0.01
                policy_loss = -(all_log_probs * returns_normalized).mean()
                entropy_bonus = all_entropies.mean()
                loss = policy_loss - ent_coef * entropy_bonus

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(rl_model.parameters(), max_norm=0.5)
                optimizer.step()

                # Clear memory
                del batch_log_probs, batch_rewards, batch_entropies, batch_returns, all_log_probs, all_returns, all_entropies
                torch.cuda.empty_cache() if device == "cuda" else None

                if iter % 1 == 0:
                    avg_ep_return = np.mean(batch_ep_returns)
                    avg_final_dist = np.mean(batch_final_dists)
                    print(
                        f"iter={iter:03d}  loss={loss.item():.4f}  "
                        f"avg_return={avg_ep_return:.4f}  "
                        f"final_dist={avg_final_dist:.4f}  grad_norm={grad_norm:.4f}"
                    )
                
                

    print("Simulation Ended.")





if __name__ == "__main__":
    run_simulation()