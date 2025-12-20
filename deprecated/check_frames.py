import pinocchio as pin
import os

model_path = "/home/yuxin/VLA_Simulation/robot_models/franka_emika_panda/robot.xml"
model = pin.buildModelFromMJCF(model_path)

print("Frames:")
for f in model.frames:
    print(f.name)

if model.existFrame("tcp"):
    print("TCP frame exists!")
else:
    print("TCP frame does NOT exist.")
