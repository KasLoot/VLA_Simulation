
import mujoco
import os

urdf_path = "models/panda_description/panda.urdf"
xml_path = "models/panda_description/panda.xml"

try:
    print(f"Loading URDF from {urdf_path}")
    model = mujoco.MjModel.from_xml_path(urdf_path)
    print("Loaded URDF successfully.")
    
    print(f"Saving to MJCF at {xml_path}")
    mujoco.mj_saveLastXML(xml_path, model)
    print("Saved MJCF successfully.")
    
except Exception as e:
    print(f"Error: {e}")
