import mujoco as mj
import mujoco.viewer
import os
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


class EnvironmentConfig:
    row_spacing = 1.5 # Increased spacing for desks
    column_spacing = 1.5
    maximum_robots_per_row = 4
    env_template_path = "environments/templets/environment.xml"

class RobotsConfig:
    names = ["franka_emika_panda"]
    quantities = [10]
    init_joint_positions = None

class SimulationConfig:
    time_step: float = 0.002
    # gui_refresh_rate: int = 1
    # physics_steps_per_control_step: int = 10


def run_simulation():


    robots_config = RobotsConfig()
    robot_xml_path = [os.path.join(Path(__file__).parent.resolve(),"robot_models", robot_name, "robot.xml") for robot_name in robots_config.names]
    print(robot_xml_path)

    env_config = EnvironmentConfig()
    xml_path = os.path.join(Path(__file__).parent, env_config.env_template_path)
    print(xml_path)
    
    # Generate Scene JSON
    scene_json_path = os.path.join(Path(__file__).parent, "scene", "pick_and_place_scene.json")
    generator = SceneGenerator(output_path=scene_json_path, num_robots=robots_config.quantities[0])
    
    # Tunable surface position [x, y, z] relative to robot
    surface_position = [0.6, 0.0, 0.0]
    generator.generate_scene(task="pick_and_place", surface_position=surface_position)
    
    # Objects Directory
    objects_dir = os.path.join(Path(__file__).parent, "objects")

    builder = EnvironmentBuilder(robot_xml_path[0], xml_path, robots_config, env_config, scene_json_path=scene_json_path, objects_dir=objects_dir)
    env_tree = builder.build(save_path="environments/built_envs/built_environment.xml")
    # print(env_tree)
    
    xml_string = ET.tostring(env_tree, encoding='unicode')
    model = mj.MjModel.from_xml_string(xml_string)

    sim_config = SimulationConfig()
    sim = SimEngine(model, sim_config, robots_config)

    sim.reset()

    # Define Home Pose (7 arm joints + 1 gripper actuator)
    # Note: The robot has 8 actuators: 7 for the arm and 1 for the gripper (which controls both fingers).
    # Joint 4 range is [-3.0718, -0.0698], so we center it around -1.57.
    # Gripper actuator range is [0, 255].
    q_init = [0, 0, 0, -0.8, 0, 1.57079, -0.7853, 128]
    amplitude = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 50]
    frequency = [0.5] * 8

    sin_ref = SinusoidalReference(amplitude=amplitude, frequency=frequency, q_init=q_init)

    # Dashboard Setup
    show_names = False
    def toggle_names(state):
        nonlocal show_names
        show_names = state

    dashboard = RobotDashboardCV2(num_robots=robots_config.quantities[0], toggle_names_callback=toggle_names, ui=False)

    # Initialize Renderer for cameras (smaller size for speed)
    renderer = mj.Renderer(sim.model, height=120, width=160)

    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        # Set initial camera view
        viewer.cam.lookat[:] = [2.0, 2.0, 0.0] # Look at center of scene
        viewer.cam.distance = 8.0
        viewer.cam.azimuth = 135 # Angle around Z axis
        viewer.cam.elevation = -30 # Angle from horizon

        current_time = 0.0
        start_time = time.time()
        last_dashboard_update = 0
        
        while viewer.is_running() and dashboard.running:
            # step_start = time.time()
            
            # Set Controls for all robots here: use sim.set_control(control_cmds)
            
            q_d, qd_d = sin_ref.get_values(current_time)
            
            # Replicate control for all robots
            all_controls = np.tile(q_d, robots_config.quantities[0])
            sim.set_control(all_controls)
            
            sim.step() 

            # Sync Viewer
            viewer.sync()
            





            # Update Robot Names in Viewer
            if show_names:
                for i in range(robots_config.quantities[0]):
                    # Find robot base position
                    # We named the container body "robot_{i}"
                    body_id = mj.mj_name2id(sim.model, mj.mjtObj.mjOBJ_BODY, f"robot_{i}")
                    if body_id >= 0:
                        pos = sim.data.xpos[body_id]
                        mj.mjv_initGeom(
                            viewer.user_scn.geoms[viewer.user_scn.ngeom],
                            type=mj.mjtGeom.mjGEOM_LABEL,
                            size=np.zeros(3),
                            pos=pos + np.array([0, 0, 1.0]),
                            mat=np.eye(3).flatten(),
                            rgba=np.array([1, 1, 1, 1], dtype=np.float32)
                        )
                        viewer.user_scn.geoms[viewer.user_scn.ngeom].label = f"Robot {i}"
                        viewer.user_scn.ngeom += 1
            else:
                # Clear user scene if names are hidden
                viewer.user_scn.ngeom = 0

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
                    ee_pos = sim.get_ee_position(body_name=f"robot_{i}/hand")
                    ee_quat = sim.get_ee_orientation_quat(body_name=f"robot_{i}/hand")

                    dashboard.update_robot(i, img, joints, ee_pos, ee_quat)
                
                dashboard.update()

            # # Time keeping
            # time_until_next_step = sim.model.opt.timestep - (time.time() - step_start)
            # if time_until_next_step > 0:
            #     time.sleep(time_until_next_step) 

            current_time += sim.model.opt.timestep





if __name__ == "__main__":
    run_simulation()