import os
import tempfile

import pandas as pd
import plotly.express as px
import torch
from sklearn.manifold import TSNE

from fs_grl.custom_pipelines.gsm.graph_spectral_measures import GraphSpectralMeasures
from fs_grl.modules.architectures.tadam import TADAM


def prepare_data_for_tsne(model_out, batch):
    """
    Prepare tensors for tsne from the model's output.

    :param model_out: output of the model
    :param batch: episode batch
    """
    query_embeds = model_out["embedded_queries"].detach().cpu()
    query_labels = batch.queries.y.detach().cpu()

    support_embeds = model_out["embedded_supports"].detach().cpu()
    support_labels = batch.supports.y.detach().cpu()

    prototype_embeds, prototype_labels = prototypes_dict_to_tensor(model_out["prototypes_dicts"][0])

    embeds = torch.cat([support_embeds, query_embeds, prototype_embeds], dim=0)
    classes = torch.cat([support_labels, query_labels, prototype_labels], dim=0)
    lens = {"support": len(support_labels), "query": len(query_labels), "prototype": len(prototype_labels)}

    return embeds, classes, lens


def log_tsne_plot(embeds, classes, lens, batch_idx, model, hparams, logger):
    """
    Create and log the tsne plot of a test episode.

    :param embeds: concat of the supports, queries and, if present, prototypes embeddings (take care to the ordering)
    :param classes: concat of the supports, queries and, if present, prototypes classes
    :param lens: dict of the lens to recover the number of support samples, query samples and, if present, the number of prototypes
    :param batch_idx: number of current episode
    :param model:
    :param hparams:
    :param logger:
    """

    # temporary dir where to save the plot
    dir = tempfile.mkdtemp()

    plot = create_tsne_plot(embeds, classes, lens)

    file_name = get_file_name_from_model(dir, model, hparams, batch_idx)

    plot.write_image(file_name)
    logger.experiment.save(file_name)
    logger.experiment.log({f"episode/{batch_idx}": plot})


def create_tsne_plot(embeds, classes, lens):

    tsne_res = compute_tsne(n_components=2, embeds=embeds)
    tsne_res = torch.from_numpy(tsne_res)

    plot = plot_from_dataframe(tsne_res, classes, lens)

    return plot


def compute_tsne(n_components, embeds):
    tsne = TSNE(n_components=n_components, n_iter=1000)
    tsne_res = tsne.fit_transform(embeds.detach().cpu())

    return tsne_res


def plot_from_dataframe(tsne_res, classes, lens):
    if "prototype" in list(lens.keys()):
        sizes = [15] * lens["support"] + [8] * lens["query"] + [20] * lens["prototype"]
        marker = [1] * lens["support"] + [0] * lens["query"] + [2] * lens["prototype"]
        symbol_sequence = ["circle", "x", "star"]
    else:
        sizes = [15] * lens["support"] + [8] * lens["query"]
        marker = [1] * lens["support"] + [0] * lens["query"]
        symbol_sequence = ["circle", "x"]

    df = pd.DataFrame(
        {
            "x": tsne_res[:, 0],
            "y": tsne_res[:, 1],
            "class": [str(x.detach().cpu().item()) for x in classes],
            "sizes": sizes,
            "marker": marker,
        }
    )
    plot = px.scatter(
        df,
        x="x",
        y="y",
        color="class",
        symbol="marker",
        symbol_sequence=symbol_sequence,
        size="sizes",
        title="",
        labels={"x": "", "y": ""},
    )
    plot.update_layout(showlegend=False)
    plot.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )

    return plot


def get_file_name_from_model(dir, model, hparams, batch_idx):
    if isinstance(model, TADAM):
        if hparams.model.loss_weights.latent_mixup_reg > 0.0:
            file_name = os.path.join(dir, f"full_tsne_episode_{batch_idx}.pdf")
        else:
            file_name = os.path.join(dir, f"tadam_no_mixup_tsne_episode_{batch_idx}.pdf")
    elif isinstance(model, GraphSpectralMeasures):
        file_name = os.path.join(dir, f"gsm_tsne_episode_{batch_idx}.pdf")
    else:
        file_name = os.path.join(dir, f"protonet_tsne_episode_{batch_idx}.pdf")

    return file_name


def prototypes_dict_to_tensor(prototypes_dict):
    prototype_embeds = []
    prototype_labels = []

    for label, embed in prototypes_dict.items():
        prototype_embeds.append(embed.detach().cpu())
        prototype_labels.append(label)

    return torch.stack(prototype_embeds, dim=0), torch.tensor(prototype_labels)
