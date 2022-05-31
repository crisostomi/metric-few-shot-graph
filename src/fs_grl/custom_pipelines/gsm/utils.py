import glob
import json

import networkx as nx


def get_graph_list_from_json(json_path):
    all_files = glob.glob(json_path)
    all_files.sort(key=lambda x: int((x.strip().split("/")[-1]).split(".")[0]))

    all_graphs = []

    for file in all_files:

        name = (file.strip().split("/")[-1]).split(".")[0]
        with open(file, "r") as f1:
            graph_json = json.load(f1)

        if len(graph_json["labels"]) == 1:
            raise ValueError("Only one node")

        G = nx.Graph()

        G.graph["name"] = name
        G.graph["target"] = graph_json["target"]
        G.add_edges_from(graph_json["edges"])

        all_graphs.append(G)

    return all_graphs


def create_gat_knn_params(model_cfg):
    knn_params = {"initial": model_cfg.knn_value}
    gat_layer_params = {}

    for i in range(model_cfg.gat_params.num_gat_layers):
        knn_params[i] = model_cfg.knn_value

        gat_layer_params[i] = {}
        if i == 0:
            gat_layer_params[i]["in_channels"] = model_cfg.hidden_dim * (model_cfg.num_layers - 1)
            gat_layer_params[i]["out_channels"] = model_cfg.gat_params.gat_out_dim

        else:
            if model_cfg.gat_params.gat_concat == 1:
                gat_layer_params[i]["in_channels"] = (
                    gat_layer_params[i - 1]["out_channels"] * gat_layer_params[i - 1]["heads"]
                )
            else:
                gat_layer_params[i]["in_channels"] = gat_layer_params[i - 1]["out_channels"]
            gat_layer_params[i]["out_channels"] = model_cfg.gat_params.gat_out_dim

        if model_cfg.gat_params.gat_concat == 1:
            gat_layer_params[i]["concat"] = True
        else:
            gat_layer_params[i]["concat"] = False
        gat_layer_params[i]["heads"] = model_cfg.gat_params.gat_heads
        gat_layer_params[i]["leaky_slope"] = model_cfg.gat_params.gat_leaky_slope
        gat_layer_params[i]["dropout"] = model_cfg.gat_params.gat_dropout

    return gat_layer_params, knn_params
