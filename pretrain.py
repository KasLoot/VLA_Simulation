import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

from utils.robots import FrankaPandaRobot

from test_1 import Model_2

# data2save = {
#             "trajectory_joint_positions": trajectory_joint_positions,
#             "target_pos_input": target_pos_input,
#             "target_orientations": target_orientations,
#             "start_ee_positions_local": start_ee_positions_local,
#             "start_ee_orientations": start_ee_orientations,
#             "start_joint_positions": start_joint_positions,
#             "init_joint_positions": init_joint_positions,
#         }

class PretrainDataset(Dataset):
    def __init__(self, data_path):
        self.data = np.load(data_path)
        self.trajectory_joint_positions = self.data['trajectory_joint_positions']
        self.target_pos_input = self.data['target_pos_input']
        self.target_orientations = self.data['target_orientations']
        self.start_ee_positions_local = self.data['start_ee_positions_local']
        self.start_ee_orientations = self.data['start_ee_orientations']
        self.start_joint_positions = self.data['start_joint_positions']
        self.init_joint_positions = self.data['init_joint_positions']

        print(f"trajectory_joint_positions shape: {self.trajectory_joint_positions.shape}")
        print(f"target_pos_input shape: {self.target_pos_input.shape}")
        print(f"target_orientations shape: {self.target_orientations.shape}")
        print(f"start_ee_positions_local shape: {self.start_ee_positions_local.shape}")
        print(f"start_ee_orientations shape: {self.start_ee_orientations.shape}")
        print(f"start_joint_positions shape: {self.start_joint_positions.shape}")
        print(f"init_joint_positions shape: {self.init_joint_positions.shape}")

        robot = FrankaPandaRobot(model_path="robot_models/franka_emika_panda/robot.xml")
        self.target_orientations_euler = robot.rotation_matrix_to_euler_angles(self.target_orientations)
        self.start_ee_orientations_euler = robot.rotation_matrix_to_euler_angles(self.start_ee_orientations)
        print(f"target_orientations_euler shape: {self.target_orientations_euler.shape}")
        print(f"start_ee_orientations_euler shape: {self.start_ee_orientations_euler.shape}")

        state = np.hstack([
            self.start_ee_positions_local,
            self.start_ee_orientations_euler,
            self.target_pos_input,
            self.target_orientations_euler
        ])
        print(f'state before adding joint positions shape: {state.shape}')

        traj = self.trajectory_joint_positions
        if traj.ndim != 3:
            raise ValueError(
                f"Expected trajectory_joint_positions to have shape (N, T, J), got {traj.shape}"
            )

        # Panda has 7 arm joints; some datasets may include extra dims (e.g., gripper).
        joint_dim = 7 if traj.shape[-1] >= 7 else traj.shape[-1]
        traj_joints = traj[:, :, :joint_dim]
        print(f"trajectory_joint_positions (used) shape: {traj_joints.shape}")

        # Expand per-trajectory state (N, S) -> per-timestep state (N, T, S)
        T = traj.shape[1]
        state_seq = np.broadcast_to(state[:, None, :], (state.shape[0], T, state.shape[1]))
        print(f"state_seq shape: {state_seq.shape}")

        # Concatenate along feature axis: (N, T, S+J)
        state = np.concatenate([state_seq, traj_joints], axis=-1)
        print(f"State shape: {state.shape}")

        self.input_states = state[:, :-1, :]  # All but last timestep
        self.target_states = self.trajectory_joint_positions[:, 1:, :]  # All but first timestep

        self.target_states = torch.tensor(self.trajectory_joint_positions.reshape(-1, self.trajectory_joint_positions.shape[-1]), dtype=torch.float32)[:, :7]
        self.input_states = torch.tensor(self.input_states.reshape(-1, self.input_states.shape[-1]), dtype=torch.float32)
        print(f"Input states shape: {self.input_states.shape}")
        print(f"Target states shape: {self.target_states.shape}")

        # print(temp[:5]== temp_input[:5])
        # print(temp_input[1:5+1]== temp_target[:5])
        
    def __len__(self):
        return self.input_states.shape[0]

    def __getitem__(self, idx):
        return self.input_states[idx], self.target_states[idx]
        


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    dataset = PretrainDataset("data/v_1/trajectory_data.npz")
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    model = Model_2(input_size=19, hidden_size=128, output_size=7).to(torch.float32).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    num_epochs = 100
    for epoch in range(num_epochs):
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs[0], targets)  # Only compare joint positions
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "checkpoints/pretrained_model.pth")
    print("Model saved as pretrained_model.pth")



if __name__ == "__main__":
    main()