from torch_geometric.data import Batch


# TODO: implement
class GraphBatch(Batch):
    def __init__(self, x, y):
        super(GraphBatch, self).__init__()
        self.x = x
        self.y = y
