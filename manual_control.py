import mujoco
import mujoco.viewer
import numpy as np
import json
import time
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from sim_engine import RobotSim

def manual_control(model_config_path="configs/pandaconfig.json"):
    # Load configuration and simulation
    model_config = json.load(open(model_config_path))
    sim = RobotSim(model_path="panda_scene.xml", model_config=model_config)
    sim.reset()

    # Number of joints to control (Panda has 7 arm joints)
    n_joints = 7
    
    # Get initial joint positions
    init_qpos = sim.get_joint_positions()
    
    # Ensure we don't exceed the number of available joints/actuators
    n_joints = min(n_joints, sim.model.nu)
    
    # Create Matplotlib window for sliders
    fig, ax = plt.subplots(figsize=(6, 0.5 * n_joints + 1))
    plt.subplots_adjust(left=0.35, bottom=0.1, right=0.95, top=0.95)
    ax.axis('off')
    ax.set_title("Joint Control")

    # Text for End-Effector Position
    ee_text = fig.text(0.05, 0.05, "EE Pos: [0.00, 0.00, 0.00]", fontsize=10)
    # Text for End-Effector Orientation
    ee_quat_text = fig.text(0.05, 0.02, "EE Quat: [1.00, 0.00, 0.00, 0.00]", fontsize=10)

    sliders = []
    
    # Create a slider for each joint
    for i in range(n_joints):
        # Get joint limits directly from the joint definition
        # We assume the first n_joints in the model are the ones we want to control
        try:
            min_val, max_val = sim.model.jnt_range[i]
            # min_val, max_val = -3.14, 3.14
        except:
            # Fallback defaults
            min_val, max_val = -3.14, 3.14

        # Create slider axis
        ax_slider = plt.axes([0.35, 0.9 - i * (0.8/n_joints), 0.55, 0.03])
        
        slider = Slider(
            ax=ax_slider,
            label=f'Joint {i+1}',
            valmin=min_val,
            valmax=max_val,
            valinit=init_qpos[i] if i < len(init_qpos) else 0.0,
        )
        sliders.append(slider)

    print("Launching viewer... Use the sliders window to control joints.")
    print("Press Ctrl+C in terminal to exit if windows don't close.")

    # Launch MuJoCo viewer
    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        # Main loop
        while viewer.is_running():
            # Check if matplotlib figure is still open
            if not plt.fignum_exists(fig.number):
                print("Slider window closed. Exiting.")
                break

            # 1. Read slider values and set qpos directly (Kinematic Control)
            for i, slider in enumerate(sliders):
                # Get the address in qpos for this joint
                q_adr = sim.model.jnt_qposadr[i]
                sim.data.qpos[q_adr] = slider.val
            
            # 2. Forward kinematics (no physics stepping)
            mujoco.mj_forward(sim.model, sim.data)

            # 3. Sync viewer
            viewer.sync()
            
            # 4. Update End-Effector Position display
            try:
                ee_pos = sim.get_ee_position()
                ee_text.set_text(f"EE Pos: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]")
                
                ee_quat = sim.get_ee_orientation()
                ee_quat_text.set_text(f"EE Quat: [{ee_quat[0]:.3f}, {ee_quat[1]:.3f}, {ee_quat[2]:.3f}, {ee_quat[3]:.3f}]")
            except Exception:
                pass

            # 6. Update Matplotlib GUI
            # plt.pause handles the GUI event loop
            plt.pause(sim.dt)

if __name__ == "__main__":
    manual_control()
