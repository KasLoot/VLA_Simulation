import torch
import torch.nn as nn
import torch.nn.functional as F

class VLAModel_1(nn.Module):
    def __init__(self, in_features=9, hidden_features=128, out_features=6):
        super(VLAModel_1, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, hidden_features)
        self.fc3 = nn.Linear(hidden_features, out_features)

        self.rmsnorm1 = nn.RMSNorm(hidden_features)
        self.rmsnorm2 = nn.RMSNorm(hidden_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.rmsnorm1(x)
        x = F.relu(self.fc2(x))
        x = self.rmsnorm2(x)
        x = self.fc3(x)
        return x
