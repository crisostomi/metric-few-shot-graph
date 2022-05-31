from functools import partial

from torch.utils.data import Dataset

from fs_grl.data.episode.episode_batch import EpisodeBatch


class FullyGraphicalEpisodicDataLoader:
    def __init__(
        self,
        dataset: Dataset,
        episode_hparams,
        add_prototype_nodes: bool = False,
        plot_graphs: bool = False,
        artificial_node_features: str = "",
        **kwargs,
    ):
        collate_fn = partial(
            EpisodeBatch.from_episode_list,
            episode_hparams=episode_hparams,
            add_prototype_nodes=add_prototype_nodes,
            artificial_node_features=artificial_node_features,
            plot_graphs=plot_graphs,
        )
        super().__init__(dataset, collate_fn=collate_fn, **kwargs)
