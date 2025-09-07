#!/usr/bin/env python3
import os, argparse, pickle, numpy as np
from scipy import sparse

def infer_N_from_mask_or_edges(example_d, default=None):
    # prefer node_mask length, then x.shape, then max edge id+1
    if isinstance(example_d, dict):
        nm = example_d.get('node_mask', None)
        if nm is not None:
            return int(np.asarray(nm).size)
        x = example_d.get('x', None)
        if x is not None:
            return int(np.asarray(x).shape[0])
        ei = example_d.get('edge_index', None)
        if ei is not None:
            arr = np.asarray(ei)
            if arr.size:
                return int(arr.max() + 1)
    else:
        # object with attributes
        try:
            nm = getattr(example_d, 'node_mask', None)
            if nm is not None:
                return int(np.asarray(nm).size)
        except Exception:
            pass
        try:
            x = getattr(example_d, 'x', None)
            if x is not None:
                return int(np.asarray(x).shape[0])
        except Exception:
            pass
        try:
            ei = getattr(example_d, 'edge_index', None)
            if ei is not None:
                arr = np.asarray(ei)
                if arr.size:
                    return int(arr.max() + 1)
        except Exception:
            pass
    return default

def build_adj_and_mask(d, N, detect_1based=True):
    # d: dict or object; returns (scipy csr adj, mask boolean length N)
    # get edge_index numpy
    if isinstance(d, dict):
        ei = d.get('edge_index', None)
        nm = d.get('node_mask', None)
    else:
        ei = getattr(d, 'edge_index', None)
        nm = getattr(d, 'node_mask', None)

    ei = None if ei is None else np.asarray(ei)
    if ei is None or ei.size == 0:
        adj = sparse.csr_matrix((N, N))
    else:
        # support (2,E) or (E,2)
        if ei.ndim == 2 and ei.shape[0] == 2:
            u = ei[0].astype(int); v = ei[1].astype(int)
        elif ei.ndim == 2 and ei.shape[1] == 2:
            u = ei[:,0].astype(int); v = ei[:,1].astype(int)
        else:
            raise ValueError("edge_index unexpected shape: {}".format(ei.shape))
        # detect 1-based -> 0-based if heuristic applies
        try:
            if detect_1based and u.min() >= 1 and u.max() >= N:
                u = u - 1
                v = v - 1
        except Exception:
            pass
        u = np.clip(u, 0, N-1); v = np.clip(v, 0, N-1)
        data = np.ones(len(u), dtype=np.int8)
        mat = sparse.coo_matrix((data, (u, v)), shape=(N, N))
        mat = mat + mat.transpose()
        mat.data = np.clip(mat.data, 0, 1)
        mat.setdiag(0)
        mat.eliminate_zeros()
        adj = mat.tocsr()

    # node mask
    if nm is None:
        # mark nodes with any neighbor present
        if adj.nnz:
            csr = adj.tocsr()
            deg = (csr.indptr[1:] - csr.indptr[:-1])
            mask = deg > 0
        else:
            mask = np.zeros(N, dtype=bool)
    else:
        mask = np.asarray(nm).astype(bool)
        if mask.size != N:
            mask = np.resize(mask, N)[:N].astype(bool)

    # zero out rows/cols for absent nodes
    if not mask.all():
        adj = adj.tolil()
        absent_idx = np.where(~mask)[0]
        if absent_idx.size:
            adj[absent_idx, :] = 0
            adj[:, absent_idx] = 0
        adj = adj.tocsr()
        adj.eliminate_zeros()

    return adj, mask

def convert(processed_dir, out_dir, N=None):
    files = sorted([f for f in os.listdir(processed_dir) if f[0].isdigit()],
                   key=lambda x: int(os.path.splitext(x)[0]))
    if not files:
        raise RuntimeError("No files in processed_dir: " + processed_dir)

    first = pickle.load(open(os.path.join(processed_dir, files[0]), 'rb'))
    if N is None:
        N = infer_N_from_mask_or_edges(first, default=None)
    if N is None:
        raise RuntimeError("Could not infer node count; please supply --N")
    print("Using N =", N)

    adjs = []
    masks = []
    for f in files:
        p = os.path.join(processed_dir, f)
        with open(p, 'rb') as handle:
            d = pickle.load(handle)
        adj, mask = build_adj_and_mask(d, N)
        adjs.append(adj)
        masks.append(mask)

    os.makedirs(out_dir, exist_ok=True)
    arr = np.empty(len(adjs), dtype=object)
    for i, a in enumerate(adjs):
        arr[i] = a
    outpath = os.path.join(out_dir, "graphs.npz")
    np.savez_compressed(outpath, graph=arr)
    masks_arr = np.vstack(masks).astype(bool)  # shape T x N
    np.save(os.path.join(out_dir, "node_masks.npy"), masks_arr)
    print("Wrote:", outpath)
    print("Wrote:", os.path.join(out_dir, "node_masks.npy"))
    print("Done. Dataset ready as:", os.path.basename(out_dir))
    return outpath

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--processed_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--N", type=int, default=None, help="Number of nodes")
    args = p.parse_args()
    convert(args.processed_dir, args.out_dir, args.N)
