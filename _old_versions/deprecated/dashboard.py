import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
from scipy.spatial.transform import Rotation as R
import threading
import time

class RobotDashboard:
    def __init__(self, num_robots, toggle_names_callback):
        self.root = tk.Tk()
        self.root.title("Robot Dashboard")
        self.root.geometry("600x800")
        
        self.num_robots = num_robots
        self.toggle_names_callback = toggle_names_callback
        self.robot_widgets = []
        
        # Top Control Panel
        control_frame = ttk.Frame(self.root)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.names_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(control_frame, text="Show Robot Names in Sim", 
                        variable=self.names_var, 
                        command=self._on_toggle_names).pack(side=tk.LEFT)
        
        # Scrollable Main Area
        self.canvas = tk.Canvas(self.root)
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )
        
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)
        
        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Create Robot Chunks
        for i in range(num_robots):
            self._create_robot_chunk(i)
            
        self.running = True
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _create_robot_chunk(self, index):
        frame = ttk.LabelFrame(self.scrollable_frame, text=f"Robot {index}")
        frame.pack(fill=tk.X, padx=5, pady=5, expand=True)
        
        # Layout: Left = Camera, Right = Data
        
        # Camera View
        cam_frame = ttk.Frame(frame)
        cam_frame.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Placeholder image
        self.img_label = ttk.Label(cam_frame, text="No Signal")
        self.img_label.pack()
        
        # Data View
        data_frame = ttk.Frame(frame)
        data_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        # Labels
        lbl_joints = ttk.Label(data_frame, text="Joints: --")
        lbl_joints.pack(anchor="w")
        
        lbl_ee_pos = ttk.Label(data_frame, text="EE Pos (x,y,z): --")
        lbl_ee_pos.pack(anchor="w")
        
        lbl_ee_rot = ttk.Label(data_frame, text="EE Rot (r,p,y): --")
        lbl_ee_rot.pack(anchor="w")
        
        lbl_ee_quat = ttk.Label(data_frame, text="EE Quat (w,x,y,z): --")
        lbl_ee_quat.pack(anchor="w")
        
        self.robot_widgets.append({
            "img_label": self.img_label,
            "lbl_joints": lbl_joints,
            "lbl_ee_pos": lbl_ee_pos,
            "lbl_ee_rot": lbl_ee_rot,
            "lbl_ee_quat": lbl_ee_quat,
            "image_ref": None # Keep reference to avoid GC
        })

    def update_robot(self, index, cam_image, joints, ee_pos, ee_quat):
        if not self.running: return
        
        widgets = self.robot_widgets[index]
        
        # Update Image
        if cam_image is not None:
            # Resize for dashboard
            img = Image.fromarray(cam_image)
            img = img.resize((160, 120)) 
            imgtk = ImageTk.PhotoImage(image=img)
            widgets["img_label"].configure(image=imgtk, text="")
            widgets["image_ref"] = imgtk
            
        # Update Text
        # Format joints (first 7 for panda)
        joints_str = ", ".join([f"{j:.2f}" for j in joints[:7]])
        widgets["lbl_joints"].configure(text=f"Joints: [{joints_str}]")
        
        widgets["lbl_ee_pos"].configure(text=f"EE Pos: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]")
        
        # Quat to Euler (scipy uses x,y,z,w but mujoco uses w,x,y,z)
        # Scipy: scalar_last=True (default) -> x,y,z,w
        # MuJoCo: w,x,y,z
        r = R.from_quat([ee_quat[1], ee_quat[2], ee_quat[3], ee_quat[0]])
        euler = r.as_euler('xyz', degrees=True)
        
        widgets["lbl_ee_rot"].configure(text=f"EE Rot (RPY): [{euler[0]:.1f}, {euler[1]:.1f}, {euler[2]:.1f}]")
        widgets["lbl_ee_quat"].configure(text=f"EE Quat: [{ee_quat[0]:.2f}, {ee_quat[1]:.2f}, {ee_quat[2]:.2f}, {ee_quat[3]:.2f}]")

    def _on_toggle_names(self):
        if self.toggle_names_callback:
            self.toggle_names_callback(self.names_var.get())

    def _on_close(self):
        self.running = False
        self.root.destroy()

    def update(self):
        if self.running:
            self.root.update()
