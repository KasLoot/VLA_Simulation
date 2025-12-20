import mujoco as mj
import mujoco.viewer
import time
import numpy as np
import os
from pathlib import Path
import xml.etree.ElementTree as ET



from utils.scene_generator import SceneGenerator
from utils.objects import ObjectLibrary
from utils.env_builder import EnvironmentBuilder


class EnvironmentConfig:
    row_spacing = 1.5 # Increased spacing for desks
    column_spacing = 1.5
    maximum_robots_per_row = 4
    env_template_path = "environments/templets/env_temp_1.xml"
    seed = 20

class RobotsConfig:
    names = ["franka_emika_panda"]
    quantities = [10]
    init_joint_positions = [[0.0]*9]

class SimulationConfig:
    time_step: float = 0.01


robots_config = RobotsConfig()
robot_xml_paths = [os.path.join(Path(__file__).parent.resolve(),"robot_models", robot_name, "robot.xml") for robot_name in robots_config.names]
print(robot_xml_paths)

env_config = EnvironmentConfig()
xml_path = os.path.join(Path(__file__).parent, env_config.env_template_path)
print(xml_path)

scene_json_path = os.path.join(Path(__file__).parent, "scene", "pick_and_place_scene.json")
scene_generator = SceneGenerator(output_path=scene_json_path, num_robots=robots_config.quantities[0], seed=42)
scene_generator.generate_scene(task="pick_and_place", surface_position=[0.5, 0, 0], min_objects=1, max_objects=3)

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
data = mj.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        viewer.sync()
        time.sleep(0.01)


time.sleep(2)