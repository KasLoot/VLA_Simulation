import mujoco
import mujoco.viewer
import numpy as np
import cv2
import math
import os
import argparse
from panda_robot import PandaRobot
from scene_builder import SceneBuilder

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

def create_dashboard(static_info, robot_infos, padding=10):
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
        
    return canvas


def run_multi_robot_simulation(args):
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
    
    print("Starting simulation. Press 'Esc' in the viewer or Ctrl+C to exit.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            t = data.time
            
            # Apply controls to all robots
            for i, robot in enumerate(robots):
                # Example control: wave based on index
                ctrl = [0, -0.58, 0, -1.68, 0, 1.13, 0.8, 0]
                # Add some variation
                ctrl[0] = 0.5 * np.sin(t + i) 
                robot.set_control(ctrl)

            mujoco.mj_step(model, data)

            # Render views
            # 1. Static camera
            renderer.update_scene(data, camera=static_id)
            rgb_static = renderer.render()
            static_img = cv2.cvtColor(rgb_static, cv2.COLOR_RGB2BGR)
            static_info = ("Static Camera", static_img)
            
            # 2. Robot cameras
            robot_infos = []
            for robot in robots:
                rgb = robot.get_camera_view(renderer)
                if rgb is not None:
                    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    robot_infos.append((f"Robot {robot.index+1}", img))

            # Combine images into a dashboard
            dashboard = create_dashboard(static_info, robot_infos)
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

            if cv2.waitKey(1) == 27: # Esc key
                break
            
            viewer.sync()


def main():
    # Configuration
    parser = argparse.ArgumentParser(description='Run multi-robot simulation.')
    parser.add_argument('--num_robots', type=int, default=4, help='Number of robots to simulate')
    parser.add_argument('--spacing', type=float, default=1.0, help='Spacing between robots')
    args = parser.parse_args()

    run_multi_robot_simulation(args)



if __name__ == '__main__':
    main()
