import mujoco as mj
import mujoco.viewer
import os
import xml.etree.ElementTree as ET
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import time

from sim_engine import SimEngine
from scene_builder import EnvironmentBuilder
from scene_generator import SceneGenerator

class EnvironmentConfig:
    row_spacing: float = 1.5
    column_spacing: float = 1.5
    maximum_robots_per_row: int = 4
    env_template_path: str = "environments/templets/environment.xml"

class RobotsConfig:
    names: list[str] = ["franka_emika_panda"]
    quantities: list[int] = [1] # Only 1 robot
    init_joint_positions: list[float] | None = None

class SimulationConfig:
    time_step: float = 0.002

def nothing(x):
    pass

def manual_tune():
    # 1. Setup Configuration
    robots_config = RobotsConfig()
    robot_xml_path = [os.path.join(Path(__file__).parent.resolve(),"robot_models", robot_name, "robot.xml") for robot_name in robots_config.names]
    
    env_config = EnvironmentConfig()
    xml_path = os.path.join(Path(__file__).parent, env_config.env_template_path)
    
    # 2. Generate Scene
    scene_json_path = os.path.join(Path(__file__).parent, "scene", "manual_tune_scene.json")
    generator = SceneGenerator(output_path=scene_json_path, num_robots=robots_config.quantities[0])
    surface_position = [0.6, 0.0, 0.0]
    generator.generate_scene(task="pick_and_place", surface_position=surface_position)
    
    objects_dir = os.path.join(Path(__file__).parent, "objects")

    # 3. Build Environment
    builder = EnvironmentBuilder(robot_xml_path[0], xml_path, robots_config, env_config, scene_json_path=scene_json_path, objects_dir=objects_dir)
    env_tree = builder.build(save_path="/home/yuxin/VLA_Simulation/test/environments/built_envs/manual_tune_env.xml")
    
    xml_string = ET.tostring(env_tree, encoding='unicode')
    model = mj.MjModel.from_xml_string(xml_string)

    sim_config = SimulationConfig()
    sim = SimEngine(model, sim_config, robots_config)
    sim.reset()

    # 4. Setup UI
    window_name = "Manual Tuning"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 640, 600) # Make it large enough for trackbars
    
    # Joint Limits (approximate for slider mapping)
    # We map 0-1000 to [min, max]
    joint_limits = [
        (-2.9, 2.9), # J1
        (-1.76, 1.76), # J2
        (-2.9, 2.9), # J3
        (-3.07, -0.07), # J4
        (-2.9, 2.9), # J5
        (-0.02, 3.75), # J6
        (-2.9, 2.9), # J7
        (0, 255) # Gripper
    ]
    
    # Initial values (Home pose)
    init_q = [0, 0, 0, -0.5, 0, 1.57, 0.626, 128]
    
    sliders = []
    for i in range(7):
        # Map init value to 0-1000
        min_val, max_val = joint_limits[i]
        val_norm = (init_q[i] - min_val) / (max_val - min_val)
        slider_val = int(val_norm * 1000)
        cv2.createTrackbar(f"Joint {i+1}", window_name, slider_val, 1000, nothing)
        sliders.append(f"Joint {i+1}")
        
    cv2.createTrackbar("Gripper", window_name, int(init_q[7]), 255, nothing)
    sliders.append("Gripper")

    # Renderer
    renderer = mj.Renderer(sim.model, height=480, width=640)
    
    print("Starting Manual Tuning...")
    print("Press 'Esc' to exit.")

    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        # Set initial camera view
        viewer.cam.lookat[:] = [0.35, 0.45, 0.0]
        viewer.cam.distance = 3.0
        viewer.cam.azimuth = 115
        viewer.cam.elevation = -30

        while viewer.is_running():
            # 1. Read Sliders
            ctrl = []
            for i in range(7):
                val = cv2.getTrackbarPos(sliders[i], window_name)
                min_val, max_val = joint_limits[i]
                angle = min_val + (val / 1000.0) * (max_val - min_val)
                ctrl.append(angle)
            
            gripper_val = cv2.getTrackbarPos("Gripper", window_name)
            ctrl.append(gripper_val)
            
            # 2. Apply Control
            sim.set_control(np.array(ctrl))
            
            # 3. Step Simulation
            sim.step()
            viewer.sync()
            
            # 4. Render & Update UI
            # Render camera
            renderer.update_scene(sim.data, camera="robot_0/ee_cam")
            img = renderer.render()
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            # Get EE Pose
            ee_pos = sim.get_ee_position(body_name="robot_0/hand")
            ee_quat = sim.get_ee_orientation_quat(body_name="robot_0/hand")
            
            # Convert Quat to Euler
            r = R.from_quat([ee_quat[1], ee_quat[2], ee_quat[3], ee_quat[0]])
            euler = r.as_euler('xyz', degrees=True)
            
            # Overlay Info
            info_text = [
                f"EE Pos: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]",
                f"EE Rot (Euler): [{euler[0]:.1f}, {euler[1]:.1f}, {euler[2]:.1f}]"
            ]
            
            for i, line in enumerate(info_text):
                cv2.putText(img, line, (10, 30 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Show Image
            cv2.imshow(window_name, img)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27: # Esc
                break
                
    cv2.destroyAllWindows()

if __name__ == "__main__":
    manual_tune()
