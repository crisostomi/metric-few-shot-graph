import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, num_layers: int, input_dim, output_dim, hidden_dim, non_linearity="relu"):
        """
        num_layers: number of layers in the neural networks
                    If num_layers=1, this reduces to linear model.
        input_dim: dimensionality of input features
        hidden_dim: dimensionality of hidden units at ALL layers
        output_dim: output dimensionality
        """

        super(MLP, self).__init__()

        self.num_layers = num_layers

        hidden_sizes = [hidden_dim for _ in range(num_layers - 1)] + [output_dim]

        self.linears = torch.nn.ModuleList()
        self.linears.append(nn.Linear(input_dim, hidden_sizes[0]))
        for layer in range(1, num_layers):
            self.linears.append(nn.Linear(hidden_sizes[layer - 1], hidden_sizes[layer]))

        self.non_linearity = self.get_non_linearity(non_linearity)

    def forward(self, x):
        h = x
        for layer in range(self.num_layers - 1):
            h = self.linears[layer](h)
            h = self.non_linearity(h)
        output = self.linears[-1](h)
        return output

    def get_non_linearity(self, non_linearity):
        if non_linearity == "relu":
            return nn.ReLU()
        elif non_linearity == "tanh":
            return nn.Tanh()
        else:
            raise NotImplementedError(f"No such activation {non_linearity}")
