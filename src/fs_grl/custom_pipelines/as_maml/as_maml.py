import math
from typing import Any, Sequence, Tuple, Union

import higher
import hydra
import numpy as np
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


class StopControl(nn.Module):
    def __init__(self, input_size, hidden_size):
        nn.Module.__init__(self)
        self.lstm = nn.LSTMCell(input_size=input_size, hidden_size=hidden_size)
        self.output_layer = nn.Linear(hidden_size, 1)
        self.output_layer.bias.data.fill_(0.0)
        self.h_0 = nn.Parameter(torch.randn((hidden_size,), requires_grad=True))
        self.c_0 = nn.Parameter(torch.randn((hidden_size,), requires_grad=True))

    def forward(self, inputs, hx):
        if hx is None:
            hx = (self.h_0.unsqueeze(0), self.c_0.unsqueeze(0))
        h, c = self.lstm(inputs, hx)
        return torch.sigmoid(self.output_layer(h).squeeze()), (h, c)


class AS_MAML(MetaLearningModel):
    def __init__(self, cfg, metadata, num_inner_steps, *args, **kwargs) -> None:
        super().__init__(metadata=metadata, *args, **kwargs)
        self.save_hyperparameters(logger=False, ignore=("metadata",))
        self.cfg = cfg
        self.num_inner_steps = num_inner_steps

        self.stop_prob = 0.5
        self.max_step = 15
        self.min_step = 5
        self.num_test_steps = 15
        self.step_penalty = 0.001
        self.use_loss = True
        self.use_score = True
        self.stop_lr = 0.0001
        self.outer_lr = 0.001
        self.inner_lr = 0.01
        self.weight_decay = 1e-5
        self.index = 0

        stop_input_size = 0
        if self.use_score:
            stop_input_size = stop_input_size + 1
        if self.use_loss:
            stop_input_size = stop_input_size + 1

        hidden_size = stop_input_size * 10
        self.stop_gate = StopControl(stop_input_size, hidden_size)

        self.gnn_mlp: GraphEmbedder = instantiate(
            cfg.model,
            feature_dim=metadata.feature_dim,
            num_classes=self.metadata.num_classes_per_episode,
            _recursive_=False,
        )

        self.inner_optimizer = instantiate(cfg.inner_optimizer, params=self.gnn_mlp.parameters())
        self.grad_clip_value = 5

        self.inner_loss_func = nn.CrossEntropyLoss()
        self.outer_loss_func = nn.CrossEntropyLoss()

    def forward(self, batch: EpisodeBatch) -> torch.Tensor:
        """ """

        raise NotImplementedError

    def step(self, metatrain: bool, batch: EpisodeBatch):

        outer_loss = torch.tensor(0.0)
        inner_loss = torch.tensor(0.0)

        metric = Accuracy().to(self.device)
        outer_accuracy = metric.clone()
        inner_accuracy = metric.clone()

        # stop_gates and scores contain respectively the stop gate and score for each inner step
        stop_gates, scores = [], []

        query_losses, query_accs = [], []
        # just for logging purposes
        adaptation_steps = []

        supports_by_episode = batch.split_in_episodes(SampleType.SUPPORT)
        queries_by_episode = batch.split_in_episodes(SampleType.QUERY)

        for episode_idx, (episode_supports, episode_queries) in enumerate(zip(supports_by_episode, queries_by_episode)):

            episode_supports = Batch.from_data_list(episode_supports)
            episode_queries = Batch.from_data_list(episode_queries)

            track_higher_grads = True if metatrain else False

            with higher.innerloop_ctx(
                self.gnn_mlp, self.inner_optimizer, copy_initial_weights=False, track_higher_grads=track_higher_grads
            ) as (fmodel, diffopt):

                self.stop_prob = 0.1 if self.stop_prob < 0.1 else self.stop_prob
                num_adaptive_steps = min(self.max_step, self.min_step + int(1.0 / self.stop_prob))

                adaptation_steps.append(num_adaptive_steps)

                for k in range(num_adaptive_steps):
                    train_logits, score, _ = fmodel(episode_supports)

                    supports_loss = self.inner_loss_func(train_logits, episode_supports.local_y)
                    inner_loss += supports_loss.cpu()
                    diffopt.step(supports_loss)

                    stop_prob = self.stop(k, supports_loss, score)
                    self.stop_prob = stop_prob

                    stop_gates.append(stop_prob)
                    scores.append(score.item())

                    with torch.no_grad():
                        train_preds = torch.softmax(train_logits, dim=-1)
                        inner_accuracy.update(train_preds, episode_supports.local_y)

                    test_logits, _, _ = fmodel(episode_queries)
                    query_loss = self.outer_loss_func(test_logits, episode_queries.local_y).cpu()
                    query_losses.append(query_loss)

                    with torch.no_grad():
                        test_preds = torch.softmax(test_logits, dim=-1)
                        outer_accuracy.update(test_preds, episode_queries.local_y)
                        query_acc = outer_accuracy.compute()
                        query_accs.append(query_acc)

        final_query_loss = query_losses[k]
        final_query_acc = query_accs[k]

        if metatrain:
            for step, (stop_gate, step_acc) in enumerate(
                zip(stop_gates[self.min_step - 1 :], query_accs[self.min_step - 1 :])
            ):
                assert stop_gate >= 0.0 and stop_gate <= 1.0, "stop_gate error value: {:.5f}".format(stop_gate)
                log_prob = torch.log(1 - stop_gate)
                tem_loss = -log_prob * (final_query_acc - step_acc - (np.exp(step) - 1) * self.step_penalty)

                outer_loss = outer_loss + tem_loss

        self.log("final_query_acc", final_query_acc)
        self.log("final_query_loss", final_query_loss)
        self.log("outer_loss", outer_loss)
        outer_loss = outer_loss + final_query_acc + final_query_loss
        self.manual_backward(outer_loss)

        if metatrain and self.index == 5:
            torch.nn.utils.clip_grad_norm_(self.gnn_mlp.parameters(), self.grad_clip_value)
            self.optimizers().optimizer.step()
            self.optimizers().optimizer.zero_grad()
            self.stop_gate.zero_grad()
            self.index = 0
        else:
            self.index += 1

        self.log("adaptation_steps", np.array(adaptation_steps).mean())
        self.log("stop_gates", np.array([stop_gate.item() for stop_gate in stop_gates]).mean())
        self.log("step_accuracies", np.array([query_acc.item() for query_acc in query_accs]).mean())

        inner_loss.div_(episode_idx + 1)

        return outer_loss, inner_loss, outer_accuracy, inner_accuracy

    def stop(self, step, loss, node_score):
        stop_hx = None
        if step < self.max_step:

            inputs = []
            if self.use_loss:
                inputs = inputs + [loss.detach()]

            if self.use_score:
                score = node_score.detach()
                inputs = inputs + [score]

            inputs = torch.stack(inputs, dim=0).unsqueeze(0)
            inputs = self.smooth(inputs)[0]
            assert torch.sum(torch.isnan(inputs)) == 0, "inputs has nan"

            stop_gate, stop_hx = self.stop_gate(inputs, stop_hx)
            assert torch.sum(torch.isnan(stop_gate)) == 0, "stop_gate has nan"

            return stop_gate

        return loss.new_zeros(1, dtype=torch.float)

    def smooth(self, weight, p=10, eps=1e-10):
        weight_abs = weight.abs()
        less = (weight_abs < math.exp(-p)).type(torch.float)
        noless = 1.0 - less
        log_weight = less * -1 + noless * torch.log(weight_abs + eps) / p
        sign = less * math.exp(p) * weight + noless * weight.sign()
        assert torch.sum(torch.isnan(log_weight)) == 0, "stop_gate input has nan"
        return log_weight, sign

    def configure_optimizers(
        self,
    ) -> Union[Optimizer, Tuple[Sequence[Optimizer], Sequence[Any]]]:
        outer_optimizer = torch.optim.Adam(
            [
                {"params": self.gnn_mlp.parameters(), "lr": self.outer_lr},
                {"params": self.stop_gate.parameters(), "lr": self.stop_lr},
            ],
            lr=0.0001,
            weight_decay=self.weight_decay,
        )

        if self.cfg.use_lr_scheduler:
            scheduler = hydra.utils.instantiate(self.cfg.lr_scheduler, optimizer=outer_optimizer)
            return [outer_optimizer], [scheduler]

        return outer_optimizer
