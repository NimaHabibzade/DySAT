#!/usr/bin/env python3
import os, sys, pickle, numpy as np, networkx as nx, argparse

def safe_edge_index_to_edges(ei, N):
    if ei is None:
        return []
    ei = np.asarray(ei)
    if ei.size == 0:
        return []
    # support both (2, E) and (E, 2)
    if ei.ndim == 2 and ei.shape[0] == 2:
        u = ei[0].astype(int); v = ei[1].astype(int)
    elif ei.ndim == 2 and ei.shape[1] == 2:
        u = ei[:,0].astype(int); v = ei[:,1].astype(int)
    else:
        raise ValueError("Unexpected edge_index shape: {}".format(ei.shape))
    # heuristic: convert 1-based -> 0-based if max >= N and min >= 1
    try:
        if u.min() >= 1 and u.max() >= N:
            u = u - 1; v = v - 1
    except Exception:
        pass
    u = np.clip(u, 0, N-1); v = np.clip(v, 0, N-1)
    return list(zip(u.tolist(), v.tolist()))

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def infer_N_from_first(d):
    # d may be dict-like or object-like
    try:
        node_mask = d.get('node_mask') if isinstance(d, dict) else getattr(d, 'node_mask', None)
        if node_mask is not None:
            return int(np.asarray(node_mask).size)
    except Exception:
        pass
    try:
        x = d.get('x') if isinstance(d, dict) else getattr(d, 'x', None)
        if x is not None:
            return int(np.asarray(x).shape[0])
    except Exception:
        pass
    try:
        ei = d.get('edge_index') if isinstance(d, dict) else getattr(d, 'edge_index', None)
        if ei is not None:
            arr = np.asarray(ei)
            if arr.size:
                return int(arr.max() + 1)
    except Exception:
        pass
    return None

def main(processed_dir, out_dir, N=None, verbose=True):
    files = sorted([f for f in os.listdir(processed_dir) if f[0].isdigit()],
                   key=lambda x: int(os.path.splitext(x)[0]))
    if not files:
        print("No files in", processed_dir); return 2

    first = load_pickle(os.path.join(processed_dir, files[0]))
    if N is None:
        N = infer_N_from_first(first)
    if N is None:
        print("Could not infer N; provide --N"); return 3
    print("Using N =", N)

    graphs = []
    masks = []

    for idx, f in enumerate(files):
        p = os.path.join(processed_dir, f)
        print("=== Reading:", p)
        d = load_pickle(p)
        # print type & keys for debugging
        if isinstance(d, dict):
            print("  pickle type: dict, keys:", list(d.keys()))
        else:
            print("  pickle type:", type(d))
            # list attributes if not dict (limited)
            try:
                print("  attrs:", [a for a in dir(d) if not a.startswith('_')][:30])
            except Exception:
                pass

        # build NetworkX graph
        G = nx.Graph()
        G.add_nodes_from(range(N))
        # get edge_index either from dict or object
        ei = d.get('edge_index') if isinstance(d, dict) else getattr(d, 'edge_index', None)
        try:
            edges = safe_edge_index_to_edges(ei, N)
        except Exception as e:
            print("  ERROR parsing edge_index:", e)
            raise
        if edges:
            G.add_edges_from(edges)
        print("  built Graph: nodes={}, edges={}".format(G.number_of_nodes(), G.number_of_edges()))
        graphs.append(G)

        # node mask
        mask_raw = d.get('node_mask') if isinstance(d, dict) else getattr(d, 'node_mask', None)
        if mask_raw is None:
            mask = np.ones(N, dtype=bool)
        else:
            mask = np.asarray(mask_raw).astype(bool)
            if mask.size != N:
                mask = np.resize(mask, N)[:N].astype(bool)
        print("  node_mask shape:", mask.shape, "present_count:", int(mask.sum()))
        masks.append(mask)

    # Save
    os.makedirs(out_dir, exist_ok=True)
    graphs_arr = np.array(graphs, dtype=object)
    out_graphs = os.path.join(out_dir, 'graphs.npz')
    np.savez_compressed(out_graphs, graph=graphs_arr)
    np.save(os.path.join(out_dir, 'node_masks.npy'), np.vstack(masks).astype(bool))
    print("Wrote:", out_graphs)
    print("Wrote:", os.path.join(out_dir, 'node_masks.npy'))

    # verify saved file
    print("=== Verifying saved graphs.npz ...")
    z = np.load(out_graphs, allow_pickle=True)
    arr = z['graph']
    ok = True
    for i, item in enumerate(arr):
        if not isinstance(item, nx.Graph):
            print("  Index", i, "NOT a Graph:", type(item))
            ok = False
        else:
            print("  Index", i, "Graph nodes=", item.number_of_nodes(), "edges=", item.number_of_edges())
    print("All Graph objects? ->", ok)
    return 0 if ok else 4

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--processed_dir', required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--N', type=int, default=None)
    args = p.parse_args()
    sys.exit(main(args.processed_dir, args.out_dir, args.N))
