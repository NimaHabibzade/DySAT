#!/usr/bin/env python3
import os, pickle, numpy as np, networkx as nx

INPUT_DIR = "data/processed_data/enron_simple"  # path to your pickles
OUT_PATH = "data/EnronDySAT/graphs_nx.npz"

graphs = []
files = sorted([f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.pickle','.pkl'))],
               key=lambda x: int(os.path.splitext(x)[0]))

for fname in files:
    path = os.path.join(INPUT_DIR, fname)
    with open(path, "rb") as f:
        data = pickle.load(f)
    edge_index = np.asarray(data["edge_index"])
    edges = list(zip(edge_index[0], edge_index[1]))
    G = nx.Graph()
    G.add_edges_from(edges)
    # Optionally, only keep active nodes based on node_mask
    mask = np.asarray(data["node_mask"]).astype(bool)
    if mask.size == G.number_of_nodes():
        active_nodes = [i for i, m in enumerate(mask) if m]
        G = G.subgraph(active_nodes).copy()
    graphs.append(G)
    print(f"{fname}: nodes={G.number_of_nodes()} edges={G.number_of_edges()}")

graphs_arr = np.array(graphs, dtype=object)
np.savez_compressed(OUT_PATH, graph=graphs_arr)
print("Saved NetworkX graphs to", OUT_PATH)
