import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F



class SimpleModel(nn.Module):
    def __init__(self, input_size=12, hidden_size=128, output_size=6):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)

        self.rms1 = nn.RMSNorm(hidden_size)
        self.rms2 = nn.RMSNorm(hidden_size)
        self.rms3 = nn.RMSNorm(hidden_size)

        self.log_std = nn.Parameter(torch.ones(output_size)*-1.0)


    def forward(self, x):
        x = F.silu(self.rms1(self.fc1(x)))
        x = F.silu(self.rms2(self.fc2(x)))
        x = F.silu(self.rms3(self.fc3(x)))
        x = self.fc4(x)

        mean = F.tanh(x)  # Assuming action space is between -1 and 1

        return mean, self.log_std.exp()


def get_reward(state: torch.Tensor, target: torch.Tensor):
    """
    state: (batch_size, seq_len, state_size)
    target: (batch_size, state_size)

    returns:
    reward: (batch_size, seq_len - 1, state_size)
    """

    # schedule = torch.linspace(0, 1, steps=state.size(1)).to(state.device).exp() - 1.0
    # schedule = schedule / schedule.max()
    

    # 1. Calculate distance for ALL states (from t=0 to t=200)
    # Shape: (batch, 201)
    all_dists = torch.norm(target.unsqueeze(1) - state, dim=-1)
    
    # 2. Define Previous and Next distances aligned by time step
    # dist_prev: Distances at t=0, 1, ..., 199 (Shape: batch, 200)
    dist_prev = all_dists[:, :-1]
    # dist_curr: Distances at t=1, 2, ..., 200 (Shape: batch, 200)
    dist_curr = all_dists[:, 1:]

    # 3. Calculate Improvement (Shape: batch, 200)
    # This now perfectly matches the shape of your other rewards
    improvement = dist_prev - dist_curr 
    progress_reward = improvement * 10.0

    # 4. Other Rewards (Using dist_prev to match your original logic of rewarding state_t)
    state_all = state[:, :-1, :]
    action_all = state[:, 1:, :]
    action_diff = torch.norm(action_all - state_all, dim=-1)
    
    target_reward = -torch.clamp(dist_prev, max=2.0)
    action_reward = -torch.abs(action_diff) * 0.01

    reward = target_reward + action_reward + progress_reward

    return reward

