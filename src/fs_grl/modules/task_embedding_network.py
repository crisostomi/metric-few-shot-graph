from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_size: int, hidden_size: int, out_size: int):
        """
        Standard residual block.

        :param in_size:
        :param hidden_size:
        :param out_size:
        """
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
    """ """

    def __init__(self, hidden_size, embedding_dim, num_convs, beta_0_init, gamma_0_init):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_convs = num_convs

        self.register_parameter("beta_0", nn.Parameter(torch.tensor([beta_0_init] * num_convs)))
        self.register_parameter("gamma_0", nn.Parameter(torch.tensor([gamma_0_init] * num_convs)))

        self.gamma_res_block = ResBlock(
            in_size=embedding_dim, hidden_size=hidden_size, out_size=embedding_dim * num_convs
        )
        self.beta_res_block = ResBlock(
            in_size=embedding_dim, hidden_size=hidden_size, out_size=embedding_dim * num_convs
        )

    def forward(self, task_embedding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        :param task_embedding: tensor (num_episodes, embedding_dim) containing for each episode the mean of its prototypes
        :return:

        """

        # (num_episodes, embedding_dim, num_convs)
        gammas = self.gamma_res_block(task_embedding).reshape(-1, self.embedding_dim, self.num_convs)

        # (num_episodes, embedding_dim, num_convs)
        betas = self.beta_res_block(task_embedding).reshape(-1, self.embedding_dim, self.num_convs)

        gammas = gammas * self.gamma_0[None, None, :] + 1
        betas = betas * self.beta_0[None, None, :]

        return gammas, betas
