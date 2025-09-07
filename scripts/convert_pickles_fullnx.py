#!/usr/bin/env python3
import os, pickle, numpy as np, networkx as nx

INPUT_DIR = "data/processed_data/enron_simple" 
OUT_PATH = "data/EnronDySAT/graphs_full.npz"

graphs = []
files = sorted([f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.pickle','.pkl'))],
               key=lambda x: int(os.path.splitext(x)[0]))

for fname in files:
    data = pickle.load(open(os.path.join(INPUT_DIR, fname), "rb"))
    edge_index = np.asarray(data["edge_index"])
    edges = list(zip(edge_index[0], edge_index[1]))
    G = nx.Graph()
    G.add_nodes_from(range(len(data["node_mask"])))  # add all nodes
    G.add_edges_from(edges)
    print(f"{fname}: nodes={G.number_of_nodes()} edges={G.number_of_edges()}")
    graphs.append(G)

graphs_arr = np.array(graphs, dtype=object)
np.savez_compressed(OUT_PATH, graph=graphs_arr)
print("Saved full-node NetworkX graphs to", OUT_PATH)
