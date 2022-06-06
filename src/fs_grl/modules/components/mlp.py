import torch
import torch.nn as nn
import torch_geometric  # noqa


class MLP(nn.Module):
    def __init__(self, num_layers: int, input_dim, output_dim, hidden_dim, non_linearity, norm=None):
        """

        :param num_layers:
        :param input_dim:
        :param output_dim:
        :param hidden_dim:
        :param non_linearity:
        :param norm:
        """

        super(MLP, self).__init__()

        self.num_layers = num_layers
        self.norm_class = eval(norm) if norm else None

        hidden_sizes = [hidden_dim for _ in range(num_layers - 1)] + [output_dim]

        self.linears = torch.nn.ModuleList()

        self.linears.append(nn.Linear(input_dim, hidden_sizes[0]))

        if self.norm_class is not None:
            self.norms = torch.nn.ModuleList()
            self.norms.append(self.norm_class(hidden_sizes[0]))

        for layer in range(1, num_layers):
            self.linears.append(nn.Linear(hidden_sizes[layer - 1], hidden_sizes[layer]))
            if self.norm_class is not None:
                self.norms.append(self.norm_class(hidden_sizes[layer]))

        self.non_linearity = non_linearity

    def forward(self, x):
        h = x

        for layer in range(self.num_layers - 1):
            h = self.linears[layer](h)
            if self.norm_class is not None:
                h = self.norms[layer](h)
            h = self.non_linearity(h)

        output = self.linears[-1](h)
        return output
