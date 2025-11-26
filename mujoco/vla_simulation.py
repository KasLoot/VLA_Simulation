import mujoco
import mujoco.viewer
import numpy as np
import cv2
import math
import os
import argparse
import time
from panda_robot import PandaRobot
from scene_builder import SceneBuilder
from VLAs.communications import VLAClient
import concurrent.futures


def add_label(image, text):
    """Adds a text label to the image."""
    img_labeled = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    color = (255, 255, 255) # White
    bg_color = (0, 0, 0) # Black background for text
    
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    
    # Draw background rectangle for text
    cv2.rectangle(img_labeled, (0, 0), (text_w + 4, text_h + 6), bg_color, -1)
    # Draw text
    cv2.putText(img_labeled, text, (2, text_h + 2), font, font_scale, color, font_thickness)
    return img_labeled

def create_dashboard(static_info, robot_infos, sim_time=None, padding=10):
    """
    Creates a dashboard layout with static camera on top and robot cameras below.
    """
    if not static_info and not robot_infos:
        return None

    # Process static image
    static_name, static_img = static_info
    static_img = add_label(static_img, static_name)
    sh, sw, _ = static_img.shape
    
    # Process robot images
    robot_imgs = [add_label(img, name) for name, img in robot_infos]
    
    if not robot_imgs:
        return static_img

    rh, rw, _ = robot_imgs[0].shape
    n_robots = len(robot_imgs)
    
    # Calculate grid dimensions for robots
    cols = int(math.ceil(math.sqrt(n_robots)))
    rows = int(math.ceil(n_robots / cols))
    
    grid_w = cols * rw + (cols - 1) * padding
    grid_h = rows * rh + (rows - 1) * padding
    
    # Calculate total canvas size
    total_w = max(sw, grid_w) + 2 * padding
    total_h = sh + grid_h + 3 * padding # padding top, middle, bottom
    
    canvas = np.zeros((total_h, total_w, 3), dtype=np.uint8)
    
    # Place static image (centered horizontally)
    static_x = (total_w - sw) // 2
    static_y = padding
    canvas[static_y:static_y+sh, static_x:static_x+sw] = static_img
    
    # Place robot images
    grid_start_y = static_y + sh + padding
    grid_start_x = (total_w - grid_w) // 2
    
    for i, img in enumerate(robot_imgs):
        r = i // cols
        c = i % cols
        
        y = grid_start_y + r * (rh + padding)
        x = grid_start_x + c * (rw + padding)
        
        canvas[y:y+rh, x:x+rw] = img
        
    # Draw time if provided
    if sim_time is not None:
        time_text = f"Time: {sim_time:.2f} s"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 2
        color = (255, 255, 255)
        bg_color = (0, 0, 0)
        
        text_size, _ = cv2.getTextSize(time_text, font, font_scale, font_thickness)
        text_w, text_h = text_size
        
        # Position: Top right corner
        x = total_w - text_w - padding
        y = text_h + padding
        
        # Draw background
        cv2.rectangle(canvas, (x - 4, y - text_h - 4), (x + text_w + 4, y + 4), bg_color, -1)
        # Draw text
        cv2.putText(canvas, time_text, (x, y), font, font_scale, color, font_thickness)

    return canvas


def run_multi_robot_simulation(args):

    client = VLAClient(url="http://localhost:8000")

    num_robots = args.num_robots
    spacing = args.spacing
    
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    panda_xml = os.path.join(script_dir, 'franka_emika_panda', 'panda.xml')
    scene_xml = os.path.join(script_dir, 'franka_emika_panda', 'scene.xml')
    
    # Build Scene
    print("Building scene...")
    builder = SceneBuilder(panda_xml, scene_xml)
    
    for i in range(num_robots):
        builder.add_robot(name=f"robot{i+1}", pos=[0, i * spacing, 0])
        
    try:
        model = builder.build(save_built_scene_xml=False)
    except Exception as e:
        print(f"Error building scene: {e}")
        return

    data = mujoco.MjData(model)

    # Initialize robots
    robots = []
    for i in range(num_robots):
        robots.append(PandaRobot(model, data, i))
    
    print(f"Initialized {len(robots)} robots.")

    width, height = 200, 200 
    renderer = mujoco.Renderer(model, width, height)
    
    static_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "static_cam")
    
    # Initialize async executor and robot states
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_robots)
    # Initial pose: [0, -0.58, 0, -1.68, 0, 1.13, 0.8, 0.0, 0.0, 0.0]
    # ready_for_request: True means we can send a new request (previous was applied or it's the first)
    robot_states = [{"action": [0, -0.58, 0, -1.68, 0, 1.13, 0.8, 0.0, 0.0], "future": None, "ready_for_request": True} for _ in range(num_robots)]

    print("Starting simulation. Press 'Esc' in the viewer or Ctrl+C to exit.")

    # Timing variables for real-time simulation
    render_interval = 1.0 / 30.0  # Render at 30 FPS
    last_render_time = 0.0
    wall_clock_start = None

    with mujoco.viewer.launch_passive(model, data) as viewer:
        wall_clock_start = time.time()
        
        while viewer.is_running():
            # Calculate target simulation time based on wall clock
            wall_elapsed = time.time() - wall_clock_start
            
            # Step simulation until it catches up to real time
            while data.time < wall_elapsed:
                # Apply controls for all robots before stepping
                for i, robot in enumerate(robots):
                    state = robot_states[i]
                    
                    # Check if we have a pending request that's done
                    if state["future"] is not None and state["future"].done():
                        try:
                            new_action = state["future"].result()
                            if new_action is not None:
                                # Server returns (prediction_horizon, 9) array, take the first action
                                if isinstance(new_action, (list, np.ndarray)) and len(new_action) > 0:
                                    action = new_action[-1] if hasattr(new_action[0], '__len__') else new_action
                                    state["action"] = action
                                else:
                                    state["action"] = new_action
                            # If new_action is None (timeout/error), keep using previous action
                        except Exception as e:
                            print(f"VLA request failed for robot {i+1}: {e}")
                            # Keep using previous action on failure
                        state["future"] = None
                        state["ready_for_request"] = True  # Ready for next request (even if this one failed)
                    
                    # Apply current action
                    robot.set_control(state["action"])
                
                mujoco.mj_step(model, data)
            
            # Render at fixed interval (not every step)
            current_time = time.time()
            if current_time - last_render_time >= render_interval:
                last_render_time = current_time
                t = data.time
                
                # Render views
                # 1. Static camera
                renderer.update_scene(data, camera=static_id)
                rgb_static = renderer.render()
                static_img = cv2.cvtColor(rgb_static, cv2.COLOR_RGB2BGR)
                static_info = ("Static Camera", static_img)
                
                # 2. Robot cameras
                robot_infos = []
                robot_images = [None] * num_robots
                for i, robot in enumerate(robots):
                    rgb = robot.get_camera_view(renderer)
                    if rgb is not None:
                        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                        robot_infos.append((f"Robot {robot.index+1}", img))
                        robot_images[i] = img

                # Combine images into a dashboard
                dashboard = create_dashboard(static_info, robot_infos, sim_time=t)
                if dashboard is not None:
                    cv2.imshow("Multi-Robot Simulation", dashboard)

                # Add labels to the MuJoCo viewer
                if viewer.user_scn:
                    viewer.user_scn.ngeom = 0
                    for robot in robots:
                        pos = robot.get_base_position()
                        mujoco.mjv_initGeom(
                            viewer.user_scn.geoms[viewer.user_scn.ngeom],
                            type=mujoco.mjtGeom.mjGEOM_LABEL,
                            size=np.zeros(3),
                            pos=pos + np.array([0, 0, 0.8]),
                            mat=np.eye(3).flatten(),
                            rgba=np.array([1, 1, 1, 1])
                        )
                        viewer.user_scn.geoms[viewer.user_scn.ngeom].label = f"Robot {robot.index+1}"
                        viewer.user_scn.ngeom += 1

                # Submit new VLA requests (non-blocking)
                for i, robot in enumerate(robots):
                    state = robot_states[i]
                    img = robot_images[i]
                    
                    # Only send a new request if previous response was received and applied
                    if state["ready_for_request"] and state["future"] is None and img is not None:
                        ee_pos, ee_quat = robot.get_ee_pose()
                        joint_pos = robot.get_joint_positions()
                        img = pad_image(img, 256, 256)
                        # Submit new request
                        state["future"] = executor.submit(client.get_action, img, "Pick up the red block.", ee_pos=ee_pos, ee_quat=ee_quat, joint_pos=joint_pos)
                        state["ready_for_request"] = False  # Wait for this response before sending another

                if cv2.waitKey(1) == 27:  # Esc key
                    break
                
                viewer.sync()
    
    executor.shutdown(wait=False)

def pad_image(img, target_width, target_height):
    """Pads the image to the target width and height with black borders."""
    h, w, _ = img.shape
    pad_w = max(0, target_width - w)
    pad_h = max(0, target_height - h)
    
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    
    padded_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return padded_img


def main():
    # Configuration
    parser = argparse.ArgumentParser(description='Run multi-robot simulation.')
    parser.add_argument('--num_robots', type=int, default=1, help='Number of robots to simulate')
    parser.add_argument('--spacing', type=float, default=1.0, help='Spacing between robots')
    args = parser.parse_args()

    run_multi_robot_simulation(args)



if __name__ == '__main__':
    main()
