from functools import partial

from torch.utils.data import DataLoader, Dataset

from fs_grl.data.episode.episode import EpisodeHParams
from fs_grl.data.episode.episode_batch import EpisodeBatch, MolecularEpisodeBatch


class EpisodicDataLoader(DataLoader):
    def __init__(
        self,
        dataset: Dataset,
        episode_hparams: EpisodeHParams,
        **kwargs,
    ):
        collate_fn = partial(
            EpisodeBatch.from_episode_list,
            episode_hparams=episode_hparams,
        )
        super().__init__(dataset, collate_fn=collate_fn, **kwargs)


class MolecularEpisodicDataLoader(DataLoader):
    def __init__(
        self,
        dataset: Dataset,
        episode_hparams: EpisodeHParams,
        **kwargs,
    ):
        collate_fn = partial(
            MolecularEpisodeBatch.from_episode_list,
            episode_hparams=episode_hparams,
        )
        super().__init__(dataset, collate_fn=collate_fn, **kwargs)
