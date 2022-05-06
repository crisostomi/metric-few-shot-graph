import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_size: int, hidden_size: int, out_size: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_size),
        )
        self.downsample = nn.Linear(in_features=in_size, out_features=out_size)

    def forward(self, x):
        return F.relu(self.downsample(x) + self.block(x))


class TaskEmbeddingNetwork(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, beta_0, gamma_0):
        super().__init__()
        self.beta_0 = beta_0
        self.gamma_0 = gamma_0
        self.gamma_res_block = ResBlock(in_size=in_size, hidden_size=hidden_size, out_size=out_size)
        self.beta_res_block = ResBlock(in_size=in_size, hidden_size=hidden_size, out_size=out_size)

    def forward(self, task_embedding):
        # TODO: L2 penalty for gamma0 and beta0
        gammas = self.gamma_res_block(task_embedding) * self.gamma_0 + 1
        betas = self.beta_res_block(task_embedding) * self.beta_0

        return gammas, betas
