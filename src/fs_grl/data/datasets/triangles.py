import pickle

from fs_grl.data.dataset import GraphDataset
from fs_grl.data.io_utils import load_data


class TrianglesDataset(GraphDataset):
    @property
    def processed_dir(self):
        directory = super(GraphDataset, self).processed_dir
        return directory

    @property
    def processed_file_names(self):
        return ["{}_complex_list.pt".format(self.name)]

    @property
    def raw_file_names(self):
        # The processed graph files are our raw files.
        # They are obtained when running the initial data conversion S2V_to_PyG.
        return ["{}_graph_list_degree_as_tag_{}.pkl".format(self.name, self.degree_as_tag)]

    def download(self):
        # This will process the raw data into a list of PyG Data objs.
        data_list, num_classes = load_data(self.raw_dir, self.name, self.degree_as_tag)
        self._num_classes = num_classes
        print("Converting graph data into PyG format...")
        with open(self.raw_paths[0], "wb") as handle:
            pickle.dump(data_list, handle)

    def process(self):
        pass
