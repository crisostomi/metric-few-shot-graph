import logging
from abc import ABC

import numpy as np
import plotly.graph_objects as go
import wandb
from plotly.graph_objs import Annotation
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from torchmetrics import ConfusionMatrix

from fs_grl.pl_modules.pl_module import MyLightningModule

pylogger = logging.getLogger(__name__)


class TransferLearningBaseline(MyLightningModule, ABC):
    def __init__(self):
        super().__init__()

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self.log_metrics(split="val", on_step=False, on_epoch=True)

    def log_metrics(self, split: str, on_step: bool, on_epoch: bool):
        to_log = {}

        for metric_name, metric in getattr(self, f"{split}_metrics").items():
            if "none" in metric_name:
                self.handle_no_average_metric(metric_name, metric, to_log)
            elif "cm" in metric_name:
                self.handle_confusion_matrix(metric, split=split)
            else:
                to_log[metric_name] = metric

        self.log_dict(to_log, on_step=on_step, on_epoch=on_epoch)

    def handle_no_average_metric(self, metric_name, metric, to_log):
        for label, score in list(
            zip(
                self.classes,
                metric.compute(),
            )
        ):
            to_log[f"{metric_name}/{label}"] = score
        metric.reset()

    def handle_confusion_matrix(self, metric, split: str):
        fig: go.Figure = self.plot_cm(cm=metric)
        wandb.log(
            data={f"{split}/confusion_matrix": fig},
        )
        metric.reset()

    def plot_cm(self, cm: ConfusionMatrix) -> go.Figure:
        z: np.ndarray = cm.compute().cpu().numpy()
        # TODO
        x = y = list(self.classes)
        class2index = {c: i for i, c in enumerate(self.classes)}

        hover_text = [[str(y) for y in x] for x in z]

        z = (z / z.sum(axis=1)).round(2)
        z_text = [[str(y) for y in x] for x in z]

        fig = go.Figure(
            data=go.Heatmap(
                z=z,
                text=z_text,
                x=x,
                y=y,
                customdata=hover_text,
                colorscale="Blues",
                zmin=0,
                zmax=1,
                hovertemplate="<br>".join(
                    (
                        "<b>Predicted</b>: %{y}",
                        "<b>Label</b>: %{x}",
                        "",
                        "<b>Row-normalized</b>: %{z:.3f}",
                        "<b>Original</b>: %{customdata}",
                    )
                ),
            )
        )

        annotations = []
        for n, row in enumerate(z):
            for m, val in enumerate(row):
                annotations.append(
                    Annotation(
                        text=str(z[n][m]),
                        x=class2index[x[m]],
                        y=class2index[y[n]],
                        xref="x1",
                        yref="y1",
                        showarrow=False,
                    )
                )

        fig.update_yaxes(autorange="reversed", type="category")
        fig.update_xaxes(type="category")

        fig.update_layout(
            font=dict(family="Courier New, monospace", size=20, color="black"),
            annotations=annotations,
            xaxis_title="Prediction",
            yaxis_title="Target",
        )

        return fig
