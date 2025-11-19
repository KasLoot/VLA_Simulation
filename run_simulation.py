import mujoco
import mujoco.viewer
import numpy as np
import json
import time
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from sim_engine import RobotSim
from traj_gen import SinusoidalReference


def get_trajectory(start_pos, end_pos, dt, duration=5.0, sim=None, body_name="panda_link7", start_ori=None, end_ori=None):
    """
    Minimum-jerk trajectory between two Cartesian points.
    If sim is provided, also computes the joint trajectory using IK, respecting joint limits.
    """
    start = np.asarray(start_pos, dtype=float)
    end = np.asarray(end_pos, dtype=float)
    if start.shape != end.shape:
        raise ValueError("start_pos and end_pos must have the same shape")
    if dt <= 0:
        raise ValueError("dt must be positive")

    steps = max(2, int(np.ceil(duration / dt)))
    tau = np.linspace(0.0, 1.0, steps, dtype=float)
    blend = tau**3 * (10 - 15 * tau + 6 * tau**2)
    traj = start + (end - start) * blend[:, None]
    
    ori_traj = None
    if start_ori is not None and end_ori is not None:
        # Convert [w, x, y, z] (MuJoCo) to [x, y, z, w] (Scipy)
        s_quat = np.array([start_ori[1], start_ori[2], start_ori[3], start_ori[0]])
        e_quat = np.array([end_ori[1], end_ori[2], end_ori[3], end_ori[0]])
        
        key_rots = R.from_quat([s_quat, e_quat])
        key_times = [0, 1]
        slerp = Slerp(key_times, key_rots)
        
        interp_rots = slerp(blend)
        quats = interp_rots.as_quat() # [x, y, z, w]
        
        # Convert back to [w, x, y, z]
        ori_traj = np.zeros((steps, 4))
        ori_traj[:, 0] = quats[:, 3]
        ori_traj[:, 1] = quats[:, 0]
        ori_traj[:, 2] = quats[:, 1]
        ori_traj[:, 3] = quats[:, 2]

    if sim is not None:
        # Compute IK without restoring state, so the next segment starts from the correct configuration
        joint_traj = sim.compute_ik(traj, ee_ori_traj=ori_traj, body_name=body_name, restore_state=False)
        return traj, joint_traj

    return traj




def run_simulation(duration=10.0, model_config_path="configs/pandaconfig.json", cam_config: dict=None):
    model_config = json.load(open(model_config_path))
    sim = RobotSim(model_path="panda_scene.xml", model_config=model_config)

    sim.reset()

    print(f"sim.get_joint_positions(): {sim.get_ee_position()}")

    # start_pos = sim.get_ee_position()
    # end_pos = [-0.02, -0.587, 0.817]
    # start_ori = sim.get_ee_orientation()
    # end_ori = [1.0, 0.0, 0.0, 0.0]
    # print(f"Start Pos: {start_pos}")
    # print(f"End Pos: {end_pos}")
    # print(f"Start Ori: {start_ori}")
    # print(f"End Ori: {end_ori}")
    # traj, joint_traj = get_trajectory(start_pos, end_pos, sim.dt, sim=sim, body_name="panda_link7", start_ori=start_ori, end_ori=end_ori)

    # # plot end-effector trajectory
    # plt.figure(figsize=(10, 6))
    # plt.plot(np.arange(len(traj)) * sim.dt, traj[:, 0], label='X')
    # plt.plot(np.arange(len(traj)) * sim.dt, traj[:, 1], label='Y')
    # plt.plot(np.arange(len(traj)) * sim.dt, traj[:, 2], label='Z')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Position (m)')
    # plt.title('End-Effector Trajectory')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # # Reset simulation to initial state for playback
    # sim.reset()

    # # Plot joint trajectories
    # plt.figure(figsize=(10, 6))
    # for i in range(7):
    #     plt.plot(np.arange(len(joint_traj)) * sim.dt, joint_traj[:, i], label=f'Joint {i+1}')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Joint Angle (rad)')
    # plt.title('Joint Trajectories')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    sin_ref = SinusoidalReference(amplitude=[1, 0.5, 1, 1, 1, 1, 1],
                                  frequency=[1, 1, 1, 1, 1, 1, 1],
                                  q_init=sim.get_joint_positions())

    


    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        if cam_config is not None:
            viewer.cam.lookat[:] = cam_config.get('lookat', [0, 0, 0])
            viewer.cam.distance = cam_config.get('distance', 2.0)
            viewer.cam.azimuth = cam_config.get('azimuth', 90)
            viewer.cam.elevation = cam_config.get('elevation', -30)

        dt = sim.dt
        current_time = 0.0
        i = 0
        while viewer.is_running():
            
            # joint_positions = joint_traj[i]
            # i += 1

            joint_positions, _ = sin_ref.get_values(current_time)
            current_time += dt

            sim.set_control(joint_positions)
            sim.step()
            viewer.sync()


            # sync to real time
            time_until_next_step = dt - (time.time() % dt)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

        # Keep the viewer open after the trajectory is finished
        while viewer.is_running():
            sim.step()
            viewer.sync()
            time.sleep(dt)


if __name__ == "__main__":
    cam_config = {
        'lookat': [0.0, 0.0, 0.2],
        'distance': 2.2,
        'azimuth': 45,
        'elevation': -30
    }
    run_simulation(cam_config=cam_config)