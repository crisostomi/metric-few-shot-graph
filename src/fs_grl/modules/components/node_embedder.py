import torch
import torch.nn
from hydra.utils import instantiate
from torch_geometric.nn import JumpingKnowledge

from fs_grl.modules.components.mlp import MLP


class NodeEmbedder(torch.nn.Module):
    def __init__(
        self,
        feature_dim,
        hidden_dim,
        embedding_dim,
        num_preproc_mlp_layers,
        num_postproc_mlp_layers,
        num_gin_mlp_layers,
        num_convs,
        dropout_rate,
        do_preprocess,
        conv_type,
        conv_norm,
        postproc_mlp_norm,
        jump_mode="cat",
        non_linearity=torch.nn.ReLU,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_convs = num_convs
        self.num_gin_mlp_layers = num_gin_mlp_layers
        self.jump_mode = jump_mode
        self.do_preprocess = do_preprocess
        self.non_linearity = instantiate(non_linearity)

        self.preprocess_mlp = (
            MLP(
                num_layers=num_preproc_mlp_layers,
                input_dim=self.feature_dim,
                output_dim=self.hidden_dim,
                hidden_dim=self.hidden_dim,
                norm=conv_norm,
                non_linearity=self.non_linearity,
            )
            if do_preprocess
            else None
        )

        self.convs = torch.nn.ModuleList()
        for conv in range(self.num_convs):
            input_dim = self.feature_dim if (conv == 0 and not self.do_preprocess) else self.hidden_dim

            if "nn" in conv_type:
                nn = instantiate(conv_type.nn, input_dim=input_dim)
                conv = instantiate(conv_type, _recursive_=False, nn=nn)
            else:
                conv = instantiate(conv_type, in_channels=input_dim)

            self.convs.append(conv)

        self.dropout = torch.nn.Dropout(p=dropout_rate)

        num_layers = (self.num_convs + 1) if self.do_preprocess else self.num_convs
        after_conv_dim = (
            (num_layers * self.hidden_dim) + self.feature_dim if self.jump_mode == "cat" else self.hidden_dim
        )

        self.postprocess_mlp = MLP(
            num_layers=num_postproc_mlp_layers,
            input_dim=after_conv_dim,
            output_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            norm=postproc_mlp_norm,
            non_linearity=self.non_linearity,
        )

        self.jumping_knowledge = JumpingKnowledge(mode=self.jump_mode) if self.jump_mode != "none" else None

    def forward(self, batch, gammas: torch.Tensor = None, betas: torch.Tensor = None) -> torch.Tensor:
        """
        Embeds a batch of graphs given as a single large graph

        :param batch: Batch containing graphs to embed
        :param gammas: (num_graphs_in_batch, embedding_dim, num_convs)
        :param betas: (num_graphs_in_batch, embedding_dim, num_convs)

        :return
        """

        # X ~ (num_nodes_in_batch, feature_dim)
        # edge_index ~ (2, num_edges_in_batch)
        X, edge_index = batch.x, batch.edge_index

        h = self.preprocess_mlp(X) if self.do_preprocess else X
        jump_xs = [X, h] if self.do_preprocess else [X]

        for ind, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            if gammas is not None and betas is not None:
                h = h * gammas[ind] + betas[ind]
            h = self.dropout(h)
            jump_xs.append(h)

        if self.jump_mode != "none":
            h = self.jumping_knowledge(jump_xs)

        # out ~ (num_nodes_in_batch, output_dim)
        node_out_features = self.postprocess_mlp(h)

        if gammas is not None and betas is not None:
            node_out_features = node_out_features * gammas[-1] + betas[-1]

        return node_out_features
