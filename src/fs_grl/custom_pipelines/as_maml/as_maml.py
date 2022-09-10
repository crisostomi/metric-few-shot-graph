import math
from typing import Any, Sequence, Tuple, Union

import higher
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


class Model(nn.Module):
    def __init__(self, stop_control, embedder):
        super().__init__()
        self.stop_control = stop_control
        self.embedder = embedder

    def forward(self, batch):
        return self.embedder(batch)


class AS_MAML(MetaLearningModel):
    def __init__(
        self,
        cfg,
        metadata,
        num_inner_steps,
        stop_prob,
        max_step,
        min_step,
        num_test_steps,
        step_penalty,
        stop_lr,
        outer_lr,
        inner_lr,
        weight_decay,
        grad_clip_value,
        *args,
        **kwargs
    ) -> None:
        super().__init__(metadata=metadata, *args, **kwargs)
        self.save_hyperparameters(logger=False, ignore=("metadata",))

        self.cfg = cfg

        self.num_inner_steps = num_inner_steps

        self.stop_prob = stop_prob
        self.max_step = max_step
        self.min_step = min_step
        self.num_test_steps = num_test_steps
        self.step_penalty = step_penalty
        self.stop_lr = stop_lr
        self.outer_lr = outer_lr
        self.inner_lr = inner_lr
        self.weight_decay = weight_decay
        self.grad_clip_value = grad_clip_value

        # loss and score
        stop_input_size = 2

        stop_control_hidden_size = stop_input_size * 10

        self.stop_gate = StopControl(stop_input_size, stop_control_hidden_size)

        self.gnn_mlp: GraphEmbedder = instantiate(
            cfg.model,
            feature_dim=metadata.feature_dim,
            num_classes=self.metadata.num_classes_per_episode,
            _recursive_=False,
        )

        self.model = Model(self.stop_gate, self.gnn_mlp)

        self.inner_optimizer = instantiate(cfg.inner_optimizer, params=self.gnn_mlp.parameters())

        self.inner_loss_func = nn.CrossEntropyLoss()
        self.outer_loss_func = nn.CrossEntropyLoss()

        self.train_query_losses = []

    def forward(self, batch: EpisodeBatch) -> torch.Tensor:
        """ """

        raise NotImplementedError

    def training_step(self, batch: Any, batch_idx: int):
        """
        A meta-training step

        :param batch:
        :param batch_idx:
        :return
        """
        torch.set_grad_enabled(True)
        self.gnn_mlp.train()
        self.stop_gate.train()

        outer_loss, inner_loss, outer_acc, inner_acc, adaptation_step_avg, stop_gates_avg = self.step(
            batch=batch, metatrain=True
        )

        self.log_dict(
            {"metatrain/inner_loss": inner_loss.item(), "metatrain/inner_accuracy": inner_acc.compute()},
            on_epoch=False,
            on_step=True,
            prog_bar=False,
        )

        self.log_dict(
            {"metatrain/adaptation_steps": adaptation_step_avg, "metatrain/stop_gates": stop_gates_avg},
            on_epoch=True,
            on_step=False,
            prog_bar=False,
        )

        self.log_dict(
            {"metatrain/outer_loss": outer_loss.item(), "metatrain/outer_accuracy": outer_acc},
            on_epoch=False,
            on_step=True,
            prog_bar=True,
        )

    def validation_step(self, batch: Any, batch_idx: int):
        """
        A meta-validation step

        :param batch:
        :param batch_idx:
        :return
        """

        # force training
        torch.set_grad_enabled(True)
        self.gnn_mlp.train()
        self.stop_gate.train()

        outer_loss, inner_loss, outer_acc, inner_acc, adaptation_step_avg, stop_gates_avg = self.step(
            batch=batch, metatrain=False
        )

        self.log_dict(
            {"metaval/inner_loss": inner_loss.item(), "metaval/inner_accuracy": inner_acc.compute()}, prog_bar=False
        )

        self.log_dict(
            {"metaval/adaptation_steps": adaptation_step_avg, "metaval/stop_gates": stop_gates_avg},
            on_epoch=True,
            on_step=False,
            prog_bar=False,
        )

        self.log_dict({"metaval/outer_loss": outer_loss.item(), "metaval/outer_accuracy": outer_acc}, prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        """
        A meta-testing step

        :param batch:
        :param batch_idx:
        :return
        """

        # force training
        torch.set_grad_enabled(True)
        self.gnn_mlp.train()
        self.stop_gate.train()

        outer_loss, inner_loss, outer_acc, inner_acc, adaptation_steps, stop_gates_avg = self.step(
            batch=batch, metatrain=False
        )

        self.log_dict(
            {
                "metatest/outer_loss": outer_loss.item(),
                "metatest/inner_loss": inner_loss.item(),
                "metatest/inner_accuracy": inner_acc.compute(),
                "metatest/outer_accuracy": outer_acc,
            },
            on_step=True,
        )

    def step(self, metatrain: bool, batch: EpisodeBatch):
        """
        :param metatrain:
        :param batch:
        :return:
        """

        outer_optimizer = self.optimizers()

        inner_loss = torch.tensor(0.0)

        inner_accuracy = Accuracy().to(self.device)

        supports_by_episode = batch.split_in_episodes(SampleType.SUPPORT)
        queries_by_episode = batch.split_in_episodes(SampleType.QUERY)

        # just for logging purposes
        adaptation_steps = []

        final_query_losses = []
        final_query_accs = []

        for episode_idx, (episode_supports, episode_queries) in enumerate(zip(supports_by_episode, queries_by_episode)):
            episode_supports = Batch.from_data_list(episode_supports)
            episode_queries = Batch.from_data_list(episode_queries)

            stop_gates = []
            query_accs = []

            track_higher_grads = True if metatrain else False
            # if true --> clone + detach, else just clone
            copy_initial_weights = False if metatrain else True

            with higher.innerloop_ctx(
                self.model,
                self.inner_optimizer,
                copy_initial_weights=copy_initial_weights,
                track_higher_grads=track_higher_grads,
            ) as (fmodel, diffopt):

                self.stop_prob = 0.1 if self.stop_prob < 0.1 else self.stop_prob

                numerator = 1.0 if metatrain else 2.0
                num_adaptive_steps = min(self.max_step, self.min_step + int(numerator / self.stop_prob))
                adaptation_steps.append(num_adaptive_steps)

                for k in range(num_adaptive_steps):
                    train_logits, score, _ = fmodel.embedder(episode_supports)

                    supports_loss = self.inner_loss_func(train_logits, episode_supports.local_y)

                    inner_loss += supports_loss.cpu()

                    with torch.set_grad_enabled(metatrain):
                        stop_prob = self.stop(k, supports_loss, score)

                    self.stop_prob = stop_prob
                    stop_gates.append(stop_prob)

                    diffopt.step(supports_loss)

                    with torch.no_grad():
                        train_preds = torch.softmax(train_logits, dim=-1)
                        inner_accuracy.update(train_preds, episode_supports.local_y)

                    with torch.set_grad_enabled(metatrain):
                        test_logits, _, _ = fmodel(episode_queries)
                        query_loss = self.outer_loss_func(test_logits, episode_queries.local_y)
                        self.train_query_losses.append(query_loss.cpu())

                    with torch.no_grad():
                        test_preds = torch.softmax(test_logits, dim=-1)
                        num_corrects = torch.sum(torch.argmax(test_preds, dim=1) == episode_queries.local_y)
                        query_acc = num_corrects / episode_queries.local_y.shape[0]
                        query_accs.append(query_acc)

            final_query_acc = query_accs[-1]
            final_query_accs.append(final_query_acc)

            final_query_loss = query_loss
            final_query_losses.append(final_query_loss)

            stop_loss = torch.tensor(0.0)
            if metatrain:
                for step, (stop_gate, step_acc) in enumerate(
                    zip(stop_gates[self.min_step - 1 :], query_accs[self.min_step - 1 :])
                ):
                    assert 0.0 <= stop_gate <= 1.0, "stop_gate error value: {:.5f}".format(stop_gate)
                    log_prob = torch.log(1 - stop_gate)
                    tem_loss = -log_prob * (final_query_acc - step_acc - (np.exp(step) - 1) * self.step_penalty)

                    stop_loss = stop_loss + tem_loss

            episode_outer_loss = stop_loss + final_query_acc + final_query_loss
            if metatrain:
                self.manual_backward(episode_outer_loss)

        if metatrain:
            torch.nn.utils.clip_grad_norm_(self.gnn_mlp.parameters(), self.grad_clip_value)
            outer_optimizer.step()
            outer_optimizer.optimizer.zero_grad()

        inner_loss = inner_loss / batch.num_episodes

        final_query_loss = torch.tensor(final_query_losses).mean().cpu()
        final_query_acc = torch.tensor(final_query_accs).mean()

        stop_gates_avg = np.array([stop_gate.item() for stop_gate in stop_gates]).mean()
        adaptation_steps = np.array(adaptation_steps).mean()

        return final_query_loss, inner_loss, final_query_acc, inner_accuracy, adaptation_steps, stop_gates_avg

    def on_train_epoch_end(self) -> None:
        mean_train_query_loss = torch.tensor(self.train_query_losses).mean()
        self.lr_schedulers().step(mean_train_query_loss)
        self.train_query_losses = []

    def stop(self, step, loss, node_score):
        """

        :param step:
        :param loss:
        :param node_score:
        :return:
        """
        stop_hx = None
        if step < self.max_step:

            inputs = []
            inputs = inputs + [loss.detach()]

            score = node_score.detach()
            inputs = inputs + [score]

            inputs = torch.stack(inputs, dim=0).unsqueeze(0)

            inputs = self.smooth(inputs)[0]
            assert torch.sum(torch.isnan(inputs)) == 0, "inputs has nan"

            stop_gate, stop_hx = self.stop_gate(inputs, stop_hx)
            assert torch.sum(torch.isnan(stop_gate)) == 0, "stop_gate has nan"

            return stop_gate

        return loss.new_zeros(1, dtype=torch.float)

    def smooth(self, weight, p: int = 10, eps: float = 1e-6):
        """

        :param weight:
        :param p:
        :param eps:
        :return:
        """

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
            scheduler = instantiate(self.cfg.lr_scheduler, optimizer=outer_optimizer)
            return [outer_optimizer], [scheduler]

        return outer_optimizer
