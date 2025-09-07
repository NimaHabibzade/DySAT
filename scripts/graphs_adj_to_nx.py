#!/usr/bin/env python3
import os, sys
import numpy as np
import networkx as nx
from scipy import sparse

INPATH = "data/EnronDySAT/graphs.npz"
OUTPATH = INPATH  # overwrite in-place

if not os.path.exists(INPATH):
    print("ERROR: graphs.npz not found at", INPATH)
    sys.exit(1)

z = np.load(INPATH, allow_pickle=True, encoding="latin1")
if 'graph' in z:
    arr = z['graph']
else:
    keys = list(z.keys())
    if len(keys) == 1:
        arr = z[keys[0]]
    else:
        raise RuntimeError("graphs.npz missing 'graph' key; found keys: " + ",".join(keys))

graphs_nx = []
for i, g in enumerate(arr):
    # scipy sparse
    if sparse.isspmatrix(g):
        G = nx.from_scipy_sparse_matrix(g.tocsr())
        graphs_nx.append(G)
        print("Index", i, "converted from CSR to nx.Graph; nodes:", G.number_of_nodes(), "edges:", G.number_of_edges())
    # numpy 2D adjacency
    elif isinstance(g, np.ndarray) and g.ndim == 2:
        G = nx.from_numpy_array(g)
        graphs_nx.append(G)
        print("Index", i, "converted from ndarray to nx.Graph; nodes:", G.number_of_nodes(), "edges:", G.number_of_edges())
    # already NetworkX Graph
    elif isinstance(g, nx.Graph):
        graphs_nx.append(g)
        print("Index", i, "already nx.Graph; nodes:", g.number_of_nodes(), "edges:", g.number_of_edges())
    else:
        # try to coerce if it has toarray()
        try:
            if hasattr(g, "toarray"):
                arr2 = np.asarray(g.toarray())
                G = nx.from_numpy_array(arr2)
                graphs_nx.append(G)
                print("Index", i, "coerced via toarray -> nx.Graph; nodes:", G.number_of_nodes(), "edges:", G.number_of_edges())
                continue
        except Exception:
            pass
        raise RuntimeError("Unsupported graph element type at index {}: {}".format(i, type(g)))

# Save back as object array of NetworkX graphs
graphs_obj = np.array(graphs_nx, dtype=object)
np.savez_compressed(OUTPATH, graph=graphs_obj)
print("Wrote back NetworkX graphs to", OUTPATH)
