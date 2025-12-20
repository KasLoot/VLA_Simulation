import torch
import torch.nn as nn
import torch.nn.functional as F


class RLModel(nn.Module):
    """Gaussian policy network for continuous control.

    This model outputs the mean action and provides a Normal distribution
    (with learnable log-std) so callers can sample actions and compute
    log-probabilities for policy-gradient methods.
    """

    def __init__(
        self,
        state_dim: int = 20,
        action_dim: int = 7,
        init_log_std: float = -0.5,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Improved architecture with consistent hidden dimensions
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, action_dim)

        # Layer normalization for better training stability
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)

        # Diagonal Gaussian std; one parameter per action dimension.
        # Clamp log_std to prevent numerical issues
        self.log_std = nn.Parameter(torch.full((action_dim,), float(init_log_std)))
        self.log_std_min = -20.0
        self.log_std_max = 2.0

        # Orthogonal initialization for better gradient flow
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using orthogonal initialization."""
        for module in [self.fc1, self.fc2, self.fc3]:
            nn.init.orthogonal_(module.weight, gain=1.0)
            nn.init.zeros_(module.bias)
        # Smaller init for output layer
        nn.init.orthogonal_(self.fc4.weight, gain=0.01)
        nn.init.zeros_(self.fc4.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the mean action."""
        x = self.ln1(F.silu(self.fc1(x)))
        x = self.ln2(F.silu(self.fc2(x)))
        x = self.ln3(F.silu(self.fc3(x)))
        mean = self.fc4(x)
        return mean

    def get_dist(self, x: torch.Tensor) -> torch.distributions.Distribution:
        """Returns a factorized Normal distribution pi(a|s)."""
        mean = self.forward(x)
        # Clamp log_std to prevent numerical instability
        log_std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std).expand_as(mean)
        return torch.distributions.Normal(mean, std)
    


class RLModel_2(nn.Module):
    def __init__(self, state_dim: int = 6, action_dim: int = 3, hidden_dim: int = 256):
        super(RLModel_2, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

        self.rms1 = nn.RMSNorm(hidden_dim)
        self.rms2 = nn.RMSNorm(hidden_dim)

        self.log_std = nn.Parameter(torch.ones(action_dim))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.rms1(self.fc1(x)))
        x = F.silu(self.rms2(self.fc2(x)))
        action = self.fc3(x)
        return action
    
