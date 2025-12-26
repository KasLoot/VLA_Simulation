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

        self.log_std = nn.Parameter(torch.zeros(output_size))


    def forward(self, x):
        x = F.silu(self.rms1(self.fc1(x)))
        x = F.silu(self.rms2(self.fc2(x)))
        x = F.silu(self.rms3(self.fc3(x)))
        x = self.fc4(x)
        return x, self.log_std.exp()


def get_reward(state: torch.Tensor, target: torch.Tensor):
    """
    state: (batch_size, seq_len, state_size)
    target: (batch_size, state_size)

    returns:
    reward: (batch_size, seq_len - 1, state_size)
    """

    state_all = state[:, :-1, :]
    action_all = state[:, 1:, :]

    action_diff = torch.norm(action_all - state_all, dim=-1)
    target_diff = torch.norm(target.unsqueeze(1) - state_all, dim=-1)
    # print(f"state_all shape: {state_all.shape}")
    # print(f"action_all shape: {action_all.shape}")
    # print(f"action_diff shape: {action_diff.shape}")
    # print(f"target_diff shape: {target_diff.shape}")

    schedule = torch.linspace(0, 1, steps=state_all.size(1)).to(state.device).exp() - 1.0
    schedule = schedule / schedule.max()
    # print(f"schedule.shape: {schedule.shape}")

    target_reward = -target_diff * schedule
    # print(f"target_reward.shape: {target_reward.shape}")

    action_reward = -torch.abs(action_diff) * 0.1
    # print(f"action_reward.shape: {action_reward.shape}")
    
    reward = target_reward + action_reward

    return reward



def rl_train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model = SimpleModel().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001)



    for epoch in range(1):
        state = torch.randn(32, 6).to(device)
        target = torch.randn(32, 6).to(device)
        state_all = []
        log_probs_all = []
        reward_all = []
        state_all.append(state)

        for step in range(100):
            model.train()

            input_state = torch.cat([state, target], dim=-1)

            action_mean, action_std = model(input_state)
            dist = torch.distributions.Normal(action_mean, action_std)
            actions = dist.rsample()
            state_all.append(actions)
            state = actions.detach()

            log_probs = dist.log_prob(actions).sum(-1)
            log_probs_all.append(log_probs)

        state_all = torch.stack(state_all, dim=1)
        log_probs_all = torch.stack(log_probs_all, dim=1)
        print(f"state_all shape after stacking: {state_all.shape}")
        print(f"log_probs_all shape after stacking: {log_probs_all.shape}")

        reward = get_reward(state_all, target)
        print(f"reward shape: {reward.shape}")

        total_return = reward.sum(dim=1)
        print(f"total_return shape: {total_return.shape}")

        loss = - (log_probs_all * total_return.unsqueeze(-1).detach()).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")





if __name__ == "__main__":
    rl_train()