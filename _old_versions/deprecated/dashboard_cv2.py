import cv2
import numpy as np
import math
from scipy.spatial.transform import Rotation as R

class RobotDashboardCV2:
    def __init__(self, num_robots, toggle_names_callback, ui=True):
        self.num_robots = num_robots
        self.toggle_names_callback = toggle_names_callback
        self.show_names = False
        self.running = True
        self.ui = ui
        self.window_name = "Robot Dashboard"
        
        # Data storage
        self.robot_data = [{} for _ in range(num_robots)]
        
        # Layout config
        self.img_w = 160
        self.img_h = 120
        self.padding = 10
        self.text_area_h = 100 # Height for text below image
        
        # Calculate grid
        self.cols = int(math.ceil(math.sqrt(num_robots)))
        self.rows = int(math.ceil(num_robots / self.cols))
        
        self.cell_w = self.img_w
        self.cell_h = self.img_h + self.text_area_h
        
        self.total_w = self.cols * (self.cell_w + self.padding) + self.padding
        self.total_h = self.rows * (self.cell_h + self.padding) + self.padding + 40 # +40 for top bar

    def update_robot(self, index, cam_image, joints, ee_pos, ee_quat):
        self.robot_data[index] = {
            "image": cam_image,
            "joints": joints,
            "ee_pos": ee_pos,
            "ee_quat": ee_quat
        }

    def update(self):
        if not self.running: return
        if not self.ui: return

        # Create canvas
        canvas = np.zeros((self.total_h, self.total_w, 3), dtype=np.uint8)
        
        # Draw Top Bar
        cv2.putText(canvas, f"Show Names: {'ON' if self.show_names else 'OFF'} (Press 'n' to toggle)", 
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw Robots
        for i in range(self.num_robots):
            data = self.robot_data[i]
            if not data: continue
            
            r = i // self.cols
            c = i % self.cols
            
            x = self.padding + c * (self.cell_w + self.padding)
            y = 40 + self.padding + r * (self.cell_h + self.padding)
            
            # Draw Image
            img = data.get("image")
            if img is not None:
                # Resize if needed (though renderer should match)
                if img.shape[:2] != (self.img_h, self.img_w):
                    img = cv2.resize(img, (self.img_w, self.img_h))
                
                # Convert RGB to BGR for OpenCV
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                canvas[y:y+self.img_h, x:x+self.img_w] = img_bgr
            
            # Draw Text
            text_y_start = y + self.img_h + 15
            line_height = 15
            font_scale = 0.4
            color = (200, 200, 200)
            
            cv2.putText(canvas, f"Robot {i}", (x, text_y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            joints = data.get("joints", [])
            if len(joints) > 0:
                j_str = "J: " + " ".join([f"{j:.1f}" for j in joints[:4]]) # First 4
                cv2.putText(canvas, j_str, (x, text_y_start + line_height), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1)
                j_str2 = "   " + " ".join([f"{j:.1f}" for j in joints[4:7]]) # Next 3
                cv2.putText(canvas, j_str2, (x, text_y_start + line_height*2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1)

            ee_pos = data.get("ee_pos", [])
            if len(ee_pos) == 3:
                pos_str = f"Pos: {ee_pos[0]:.2f} {ee_pos[1]:.2f} {ee_pos[2]:.2f}"
                cv2.putText(canvas, pos_str, (x, text_y_start + line_height*3), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1)
            
            ee_quat = data.get("ee_quat", [])
            if len(ee_quat) == 4:
                # Quat to Euler
                r_obj = R.from_quat([ee_quat[1], ee_quat[2], ee_quat[3], ee_quat[0]])
                euler = r_obj.as_euler('xyz', degrees=True)
                rot_str = f"Rot: {euler[0]:.0f} {euler[1]:.0f} {euler[2]:.0f}"
                cv2.putText(canvas, rot_str, (x, text_y_start + line_height*4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1)

        try:
            cv2.imshow(self.window_name, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == 27: # Esc
                self.running = False
            elif key == ord('n'):
                self.show_names = not self.show_names
                if self.toggle_names_callback:
                    self.toggle_names_callback(self.show_names)
        except Exception as e:
            print(f"Warning: OpenCV imshow failed. Disabling dashboard. Error: {e}")
            self.ui = False
            cv2.destroyAllWindows()

    def close(self):
        self.running = False
        cv2.destroyAllWindows()
