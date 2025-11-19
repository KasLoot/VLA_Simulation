import mujoco
import numpy as np
import json
from sim_engine import RobotSim



def run_simulation(self, duration=10.0, model_config_path="configs/pandaconfig.json", cam_config: dict=None):
    model_config = json.load(open(model_config_path))
    sim = RobotSim(model_path="robot_model.xml", model_config=model_config)

    sim.reset()

    dt = sim.dt

    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        if cam_config is not None:
            viewer.cam.lookat[:] = cam_config.get('lookat', [0, 0, 0])
            viewer.cam.distance = cam_config.get('distance', 2.0)
            viewer.cam.azimuth = cam_config.get('azimuth', 90)
            viewer.cam.elevation = cam_config.get('elevation', -30)

    current_time = 0.0
    while viewer.is_running() and current_time < duration:
        pass

        