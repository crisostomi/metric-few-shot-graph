import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, num_layers: int, input_dim, output_dim, hidden_dim):
        """
        num_layers: number of layers in the neural networks
                    If num_layers=1, this reduces to linear model.
        input_dim: dimensionality of input features
        hidden_dim: dimensionality of hidden units at ALL layers
        output_dim: output dimensionality
        """

        super(MLP, self).__init__()

        self.num_layers = num_layers

        hidden_sizes = [hidden_dim for i in range(num_layers - 1)] + [output_dim]

        self.linears = torch.nn.ModuleList()
        self.linears.append(nn.Linear(input_dim, hidden_sizes[0]))
        for layer in range(1, num_layers):
            self.linears.append(nn.Linear(hidden_sizes[layer - 1], hidden_sizes[layer]))

    def forward(self, x):
        h = x
        for layer in range(self.num_layers - 1):
            h = self.linears[layer](h)
            h = F.relu(h)
        output = self.linears[-1](h)
        return output
