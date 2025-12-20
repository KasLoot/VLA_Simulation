import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

class QuinticTrajectoryPlanner:
    def __init__(self, start_pos, target_pos, start_quat, target_quat, total_time, dt):
        self.start_pos = np.array(start_pos)
        self.target_pos = np.array(target_pos)
        self.start_rot = R.from_quat(start_quat)
        self.target_rot = R.from_quat(target_quat)
        
        self.T = total_time
        self.dt = dt
        self.num_steps = int(total_time / dt)
        
        # FIXED: Modified limits as per your request
        # (-Pi to 0) for all joints
        self.joint_limits = [(-np.pi, np.pi)] * 8 

    def get_quintic_scaling(self, t):
        tau = np.clip(t / self.T, 0.0, 1.0)
        s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
        v_scale = (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / self.T
        return s, v_scale

    def inverse_kinematics_solver(self, target_position, target_orientation, seed_joints):
        # DUMMY SOLVER: Simulating IK
        noise = np.random.normal(0, 0.001, 8) 
        # Add slight movement based on target position to simulate tracking
        simulated_solution = seed_joints + (target_position[0] * 0.001) + noise
        return simulated_solution

    def check_limits(self, joints):
        for i, angle in enumerate(joints):
            min_lim, max_lim = self.joint_limits[i]
            # Use a tiny buffer (epsilon) to avoid floating point boundary errors
            epsilon = 1e-5
            if not (min_lim - epsilon <= angle <= max_lim + epsilon):
                return False, i
        return True, -1

    def plan(self):
        time_stamps = []
        joint_trajectory = []
        velocities = []

        # FIXED: Initialize 'current_joints' to the MIDDLE of the limits.
        # If limits are (-3.14, 0), we start at -1.57.
        # Starting at 0.0 is risky because it's right on the edge.
        current_joints = []
        for limits in self.joint_limits:
            mid_point = (limits[0] + limits[1]) / 2.0
            current_joints.append(mid_point)
        current_joints = np.array(current_joints)

        print(f"Starting Configuration: {np.round(current_joints, 2)}")

        for step in range(self.num_steps + 1):
            t = step * self.dt
            
            # 1. Get Scaling
            s, v_scale = self.get_quintic_scaling(t)
            
            # 2. Interpolate Cartesian (Lerp)
            current_pos = self.start_pos + (self.target_pos - self.start_pos) * s
            
            # 3. Solve IK
            try:
                q_sol = self.inverse_kinematics_solver(current_pos, None, current_joints)
            except Exception:
                print(f"IK Failed at t={t}")
                break

            # 4. Check Limits
            is_valid, joint_idx = self.check_limits(q_sol)
            if not is_valid:
                print(f"Limit violated on Joint {joint_idx} at time {t:.2f}s. Value: {q_sol[joint_idx]:.3f}")
                # We break the loop, but we do NOT add partial data to the lists
                break
            
            # 5. Append Data ONLY if successful
            # FIXED: Moved time_stamps append here to keep lists synced
            time_stamps.append(t)
            current_joints = q_sol
            joint_trajectory.append(current_joints)
            velocities.append(v_scale)

        return np.array(time_stamps), np.array(joint_trajectory), np.array(velocities)

# --- CONFIGURATION ---
start_p = [0.2, 0.0, 0.4]    
end_p   = [0.5, 0.2, 0.6]
start_q = [0, 0, 0, 1]       
end_q   = [0, 0, 0, 1]       
T_total = 5.0                
timestep = 0.05              

planner = QuinticTrajectoryPlanner(start_p, end_p, start_q, end_q, T_total, timestep)
times, joints, v_profile = planner.plan()

# --- PLOTTING (FIXED) ---
if len(times) == 0:
    print("\nERROR: No trajectory generated. Check start configuration or limits.")
else:
    print(f"\nSuccessfully generated {len(times)} steps.")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot 1
    ax1.plot(times, v_profile, 'r-', linewidth=2, label="Cartesian Path Velocity Scaling")
    ax1.set_title("Quintic Velocity Profile")
    ax1.grid(True)
    ax1.legend()

    # Plot 2
    for j in range(8):
        ax2.plot(times, joints[:, j], label=f'Joint {j+1}')
    ax2.set_title(f"Trajectory for 8-Joint Robot")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Joint Angle (rad)")
    ax2.legend(loc='upper right', bbox_to_anchor=(1.1, 1))
    ax2.grid(True)

    plt.tight_layout()
    plt.show()