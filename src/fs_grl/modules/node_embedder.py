import torch
import torch.nn as nn
from torch_geometric.nn import GINConv, JumpingKnowledge

from fs_grl.modules.mlp import MLP


class NodeEmbedder(nn.Module):
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
        batch_norm="none",
        jump_mode="cat",
        non_linearity=nn.ReLU,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_convs = num_convs
        self.num_gin_mlp_layers = num_gin_mlp_layers
        self.jump_mode = jump_mode
        self.do_preprocess = do_preprocess
        self.batch_norm = batch_norm
        self.non_linearity = non_linearity

        self.preprocess_mlp = (
            MLP(
                num_layers=num_preproc_mlp_layers,
                input_dim=self.feature_dim,
                output_dim=self.hidden_dim,
                hidden_dim=self.hidden_dim,
                batch_norm=self.batch_norm,
                non_linearity=self.non_linearity,
            )
            if do_preprocess
            else None
        )

        self.convs = nn.ModuleList()
        for conv in range(self.num_convs):
            input_dim = self.feature_dim if (conv == 0 and not self.do_preprocess) else self.hidden_dim
            conv = GINConv(
                nn=MLP(
                    num_layers=num_gin_mlp_layers,
                    input_dim=input_dim,
                    output_dim=self.hidden_dim,
                    hidden_dim=self.hidden_dim,
                    batch_norm=self.batch_norm,
                    non_linearity=self.non_linearity,
                )
            )
            self.convs.append(conv)

        self.dropout = nn.Dropout(p=dropout_rate)

        num_layers = (self.num_convs + 1) if self.do_preprocess else self.num_convs
        after_conv_dim = (
            (num_layers * self.hidden_dim) + self.feature_dim if self.jump_mode == "cat" else self.hidden_dim
        )

        self.postprocess_mlp = MLP(
            num_layers=num_postproc_mlp_layers,
            input_dim=after_conv_dim,
            output_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            batch_norm="layer",
            non_linearity=self.non_linearity,
        )

        self.jumping_knowledge = JumpingKnowledge(mode=self.jump_mode) if self.jump_mode != "none" else None

    def forward(self, batch, gammas=None, betas=None):
        """
        Embeds a batch of graphs given as a single large graph

        :param batch: Batch containing graphs to embed
        :return:
        """

        # X ~ (num_nodes_in_batch, feature_dim)
        # edge_index ~ (2, num_edges_in_batch)
        X, edge_index = batch.x, batch.edge_index

        h = self.preprocess_mlp(X) if self.do_preprocess else X
        jump_xs = [X, h] if self.do_preprocess else [X]

        if gammas is not None and betas is not None:
            _, num_repetitions = torch.unique(batch.batch, return_counts=True)
            gammas = torch.repeat_interleave(gammas, num_repetitions, dim=0)
            betas = torch.repeat_interleave(betas, num_repetitions, dim=0)

        for ind, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            if gammas is not None and betas is not None:
                h = h * gammas[:, ind].unsqueeze(1) + betas[:, ind].unsqueeze(1)
            jump_xs.append(h)

        if self.jump_mode != "none":
            h = self.jumping_knowledge(jump_xs)

        h = self.dropout(h)
        # out ~ (num_nodes_in_batch, output_dim)
        node_out_features = self.postprocess_mlp(h)

        return node_out_features
