import numpy as np



class State_Manager:

    def __init__(self, task_name):
        self.task_name = task_name
        self.state = self.build_state()

    def build_state(self):
        if self.task_name == "pick_and_place":
            self.state = ["APPROACH", "DESCEND", "GRASP", "MOVE"]
            return self.state
        else:
            raise ValueError(f"Unknown task name: {self.task_name}")
    
    def get_full_state(self):
        return self.state
    
    def get_state_target_qpos(self, current_state, target_pos):
        if current_state == "APPROACH":
            target_pos = target_pos + np.array([0, 0, 0.10])
            target_grip = 0.04  # OPEN
            return target_pos, target_grip

        elif current_state == "DESCEND":
            target_pos = target_pos + np.array([0, 0, 0.01])
            target_grip = 0.04  # KEEP OPEN
            return target_pos, target_grip

        elif current_state == "GRASP":
            target_pos = target_pos + np.array([0, 0, 0.01])
            target_grip = 0.0 # CLOSE
            return target_pos, target_grip

        elif current_state == "MOVE":
            target_pos = target_pos + np.array([-0.05, -0.05, 0.05])
            target_grip = 0.0  # Keep CLOSED
            return target_pos, target_grip

        else:
            raise ValueError(f"Unknown state: {current_state}")

        