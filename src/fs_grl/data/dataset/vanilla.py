import logging
from typing import List

from torch.utils.data import Dataset
from torch_geometric.data import Data

pylogger = logging.getLogger(__name__)


class VanillaGraphDataset(Dataset):
    """
    Vanilla graph dataset. Used in the base training phase of transfer learning.
    """

    def __init__(self, samples: List[Data]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
