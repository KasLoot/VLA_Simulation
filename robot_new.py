import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation as R
import time


# --- Differential IK Constants ---
DAMPING = 1e-4
K_POS = 3.0      # Increased for tighter tracking
K_ORI = 1.0
K_NULL = 1.0     # Reduced so it doesn't fight the main task
MAX_ANGVEL = 2.0
INTEGRATION_DT = 0.01 # Smaller step for more stability


def solve_differential_ik(model, data, target_pos, target_rot_matrix, q0, ee_site_id):
    # Ensure kinematics are up-to-date before reading derived quantities.
    # (site_xpos/site_xmat are undefined until mj_forward/mj_step has run.)
    mujoco.mj_forward(model, data)

    # 1. Get Current Site State (TCP)
    current_pos = data.site_xpos[ee_site_id]
    current_rot_matrix = data.site_xmat[ee_site_id].reshape(3, 3)
    # print("Current Position:", current_pos)
    # print("Current Rotation Matrix:\n", current_rot_matrix)

    # Basic sanity check: rotation matrix must be right-handed with det ~ +1.
    det = float(np.linalg.det(current_rot_matrix))
    if not np.isfinite(det) or det <= 0.0:
        raise ValueError(
            "End-effector site rotation matrix is invalid (det <= 0). "
            "Did you set data.qpos and call mj_forward()/mj_step() before IK? "
            f"det={det}, ee_site_id={ee_site_id}"
        )

    # 2. Calculate Error (Twist)
    error_pos = target_pos - current_pos
    
    # Orientation error
    error_rot_mat = target_rot_matrix @ current_rot_matrix.T
    error_rot_vec = R.from_matrix(error_rot_mat).as_rotvec()

    twist = np.zeros(6)
    twist[:3] = error_pos * K_POS
    twist[3:] = error_rot_vec * K_ORI

    # 3. Compute Site Jacobian
    jac_p = np.zeros((3, model.nv))
    jac_r = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jac_p, jac_r, ee_site_id)
    J = np.vstack([jac_p, jac_r])[:, :7] 

    # 4. Solve for dq using Damped Least Squares
    vv = J @ J.T
    diag_indices = np.diag_indices_from(vv)
    vv[diag_indices] += DAMPING
    dq = J.T @ np.linalg.solve(vv, twist)

    # 5. Nullspace Control (Bias toward home posture q0)
    current_q = data.qpos[:7]
    null_error = (q0[:7] - current_q)
    J_pinv = np.linalg.pinv(J, rcond=1e-2)
    dq_null = (np.eye(7) - J_pinv @ J) @ (K_NULL * null_error)
    
    return np.clip(dq + dq_null, -MAX_ANGVEL, MAX_ANGVEL)




class RobotsConfig:
    names = ["franka_emika_panda"]
    quantities = [1]
    init_joint_positions: list = [[-0.0, 0.0, 0.0, -0.0, 0.0, 1.0, 0.0, 0.04, 0.04]*quantities[0]]  # Panda default pose

class SimulationConfig:
    time_step: float = 0.01
    # gui_refresh_rate: int = 1
    # physics_steps_per_control_step: int = 10

from utils.sim_engine import SimEngine
import json
import os
from pathlib import Path
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


def main():
    model = mujoco.MjModel.from_xml_path("environments/built_envs/built_environment.xml")
    init_q = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785, 0.04, 0.04])

    # target_pos = np.array([0.0, 0.0, 0.0])
    target_rot_matrix = R.from_euler('xyz', [np.pi, 0, 0]).as_matrix()
    print("Target Rotation Matrix:\n", target_rot_matrix)

    scene_json_path = os.path.join(Path(__file__).parent, "scene", "pick_and_place_scene.json")
    with open(scene_json_path, 'r') as f:
        scene_data = json.load(f)
    print(f"scene_data:\n{scene_data}")
    sim = SimEngine(sim_env=model, sim_config=SimulationConfig(), robots_config=RobotsConfig())
    # Use the SimEngine-owned MjData everywhere to avoid desync between simulation state and viewer.
    data = sim.data
    target_pos = get_target_object_positions(RobotsConfig(), sim, scene_data)[0][0]  # First robot, first object
    print(f"Target Positions shape:\n{target_pos.shape}")



    ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "robot_0/hand_tip_site")

    if ee_site_id < 0:
        raise ValueError(
            "Could not find site 'robot_0/hand_tip_site' in the model. "
            "Check the generated XML for the exact site name."
        )

    # Initialize robot_0 joint configuration (7 arm + 2 fingers)
    if init_q.size != sim.robot_joint_indices.size:
        raise ValueError(
            f"Expected init_q to have {sim.robot_joint_indices.size} elements (one per robot joint), "
            f"but got {init_q.size}. robot_joint_indices={sim.robot_joint_indices}"
        )
    sim.set_joint_controls(init_q)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    d_q = solve_differential_ik(model=model, 
                          data=data, 
                          target_pos=target_pos, target_rot_matrix=target_rot_matrix, 
                          q0=init_q, 
                          ee_site_id=ee_site_id)
    print("Computed joint positions (d_q):", d_q)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.sync()
        while viewer.is_running():
            d_q = solve_differential_ik(model=model, 
                                  data=data, 
                                  target_pos=target_pos, target_rot_matrix=target_rot_matrix, 
                                  q0=init_q, 
                                  ee_site_id=ee_site_id)
            # print(f"robot_joint_indices: {sim.robot_joint_indices}")
            # robot_joint_indices is an integer array of qpos addresses; index with it (not slice).
            curr_q = data.qpos[sim.robot_joint_indices]

            # SimEngine.set_joint_controls expects 9 values (7 arm + 2 fingers).
            # Update only arm joints; keep finger joints as-is.
            next_q = curr_q.copy()
            next_q[:7] = curr_q[:7] + d_q * INTEGRATION_DT

            sim.set_joint_controls(next_q)
            sim.forward()
            viewer.sync()
            time.sleep(INTEGRATION_DT)

if __name__ == "__main__":
    main()