import logging
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch, Data

from fs_grl.data.dataset import VanillaGraphDataset

pylogger = logging.getLogger(__name__)


def compute_global_prototypes(
    model, data_list_by_label: Dict[int, List[Data]], label_to_class_dict: Dict[int, str]
) -> Dict[str, torch.Tensor]:
    """
    Computes the prototype for each label in the dataset by averaging the samples of that label.
    :param model: pretrained model
    :param data_list_by_label: samples grouped by label
    :param label_to_class_dict: mapping label -> class
    :return:
    """
    model.eval()
    model.model.eval()

    prototypes = {}

    for label, data_list in data_list_by_label.items():

        dataset = VanillaGraphDataset(data_list)
        dataloader = DataLoader(dataset=dataset, collate_fn=Batch.from_data_list, batch_size=64)

        all_label_embeddings = []
        for batch in dataloader:
            batch.to(model.device)
            batch_embeddings = model.model._embed(batch)
            all_label_embeddings.append(batch_embeddings)

        all_label_embeddings = torch.cat(all_label_embeddings, dim=0)
        label_prototype = torch.mean(all_label_embeddings, dim=0)

        cls = label_to_class_dict[label]
        prototypes[cls] = label_prototype

    return prototypes


def handle_fast_dev_run(cfg):
    pylogger.info(f"Debug mode <{cfg.train.trainer.fast_dev_run=}>. Forcing debugger friendly configuration!")
    # Debuggers don't like GPUs nor multiprocessing
    cfg.train.trainer.gpus = 0
    cfg.nn.data.num_workers.train = 0
    cfg.nn.data.num_workers.val = 0
    cfg.nn.data.num_workers.test = 0
