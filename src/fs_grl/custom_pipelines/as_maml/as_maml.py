from typing import Any, Sequence, Tuple, Union

import higher
import hydra
import torch
from hydra.utils import instantiate
from torch import nn
from torch.optim import Optimizer
from torch_geometric.data import Batch
from torchmetrics import Accuracy

from fs_grl.custom_pipelines.as_maml.graph_embedder import GraphEmbedder
from fs_grl.data.episode.episode_batch import EpisodeBatch
from fs_grl.data.utils import SampleType
from fs_grl.pl_modules.meta_learning import MetaLearningModel


class AS_MAML(MetaLearningModel):
    def __init__(self, cfg, metadata, num_inner_steps, *args, **kwargs) -> None:
        super().__init__(metadata=metadata, *args, **kwargs)
        self.save_hyperparameters(logger=False, ignore=("metadata",))
        self.cfg = cfg

        self.num_inner_steps = num_inner_steps

        self.gnn_mlp: GraphEmbedder = instantiate(
            cfg.model,
            feature_dim=metadata.feature_dim,
            num_classes=self.metadata.num_classes_per_episode,
            _recursive_=False,
        )

        self.inner_optimizer = instantiate(cfg.inner_optimizer, params=self.gnn_mlp.parameters())

        self.inner_loss_func = nn.CrossEntropyLoss()
        self.outer_loss_func = nn.CrossEntropyLoss()

    def forward(self, batch: EpisodeBatch) -> torch.Tensor:
        """ """

        raise NotImplementedError

    def step(self, metatrain: bool, batch: EpisodeBatch):

        self.gnn_mlp.zero_grad()

        outer_optimizer = self.optimizers()

        outer_loss = torch.tensor(0.0)
        inner_loss = torch.tensor(0.0)

        metric = Accuracy().to(self.device)
        outer_accuracy = metric.clone()
        inner_accuracy = metric.clone()

        supports_by_episode = batch.split_in_episodes(SampleType.SUPPORT)
        queries_by_episode = batch.split_in_episodes(SampleType.QUERY)

        for episode_idx, (episode_supports, episode_queries) in enumerate(zip(supports_by_episode, queries_by_episode)):

            episode_supports = Batch.from_data_list(episode_supports)
            episode_queries = Batch.from_data_list(episode_queries)

            track_higher_grads = True if metatrain else False

            with higher.innerloop_ctx(
                self.gnn_mlp, self.inner_optimizer, copy_initial_weights=False, track_higher_grads=track_higher_grads
            ) as (fmodel, diffopt):

                for k in range(self.num_inner_steps):
                    train_logits, _, _ = fmodel(episode_supports)
                    loss = self.inner_loss_func(train_logits, episode_supports.local_y)
                    diffopt.step(loss)

                with torch.no_grad():
                    train_logits, _, _ = fmodel(episode_supports)
                    train_preds = torch.softmax(train_logits, dim=-1)
                    inner_loss += self.inner_loss_func(train_logits, episode_supports.local_y).cpu()
                    inner_accuracy.update(train_preds, episode_supports.local_y)

                test_logits, _, _ = fmodel(episode_queries)
                outer_loss += self.outer_loss_func(test_logits, episode_queries.local_y).cpu()

                with torch.no_grad():
                    test_preds = torch.softmax(test_logits, dim=-1)
                    outer_accuracy.update(test_preds, episode_queries.local_y)

        if metatrain:
            self.manual_backward(outer_loss)
            outer_optimizer.step()

        outer_loss.div_(episode_idx + 1)
        inner_loss.div_(episode_idx + 1)

        return outer_loss, inner_loss, outer_accuracy, inner_accuracy

    def configure_optimizers(
        self,
    ) -> Union[Optimizer, Tuple[Sequence[Optimizer], Sequence[Any]]]:
        outer_optimizer = hydra.utils.instantiate(self.cfg.outer_optimizer, params=self.parameters())

        if self.cfg.use_lr_scheduler:
            scheduler = hydra.utils.instantiate(self.cfg.lr_scheduler, optimizer=outer_optimizer)
            return [outer_optimizer], [scheduler]

        return outer_optimizer
