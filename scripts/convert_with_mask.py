# scripts/convert_with_mask.py
import os, argparse, pickle
import numpy as np
from scipy import sparse
import torch

def to_numpy(ei):
    if isinstance(ei, torch.Tensor):
        return ei.cpu().numpy()
    return ei

def infer_N_from_mask_or_edges(example_d, default=None):
    # prefer node_mask length, then x.shape, then max edge id+1
    if hasattr(example_d, 'node_mask') and example_d.node_mask is not None:
        try:
            m = example_d.node_mask
            if isinstance(m, torch.Tensor):
                return int(m.size(0))
            return int(len(m))
        except:
            pass
    if hasattr(example_d, 'x') and example_d.x is not None:
        try:
            return int(example_d.x.size(0))
        except:
            try:
                return int(example_d.x.shape[0])
            except:
                pass
    if hasattr(example_d, 'edge_index'):
        ei = to_numpy(example_d.edge_index)
        if ei.size:
            return int(ei.max() + 1)
    return default

def build_adj_and_mask(d, N, detect_1based=True):
    # get edge_index numpy
    ei = to_numpy(d.edge_index) if hasattr(d,'edge_index') else np.empty((2,0), dtype=int)
    if ei.size == 0:
        # empty graph
        adj = sparse.csr_matrix((N, N))
    else:
        u = ei[0].astype(int)
        v = ei[1].astype(int)
        # detect 1-based indices and convert to 0-based
        if detect_1based and u.min() >= 1:
            # a heuristic: if min index >=1 and max == N, convert by -1
            if u.min() >= 1:
                u = u - 1
                v = v - 1
        # clip any out-of-range indices (safety)
        u = np.asarray(u); v = np.asarray(v)
        u = np.clip(u, 0, N-1)
        v = np.clip(v, 0, N-1)
        data = np.ones(len(u), dtype=np.int8)
        mat = sparse.coo_matrix((data, (u, v)), shape=(N, N))
        mat = mat + mat.transpose()
        mat.data = np.clip(mat.data, 0, 1)
        mat.setdiag(0)
        mat.eliminate_zeros()
        adj = mat.tocsr()
    # node mask
    if hasattr(d, 'node_mask') and d.node_mask is not None:
        nm = d.node_mask
        if isinstance(nm, torch.Tensor):
            mask = nm.cpu().numpy().astype(bool)
        else:
            mask = np.asarray(nm).astype(bool)
        if mask.size != N:
            # if mask is 1-based or mismatched, try to resize/clip
            mask = np.resize(mask, N)[:N].astype(bool)
    else:
        # if no mask, treat nodes with any edge as present (or all present)
        # here we choose 'present if node has any neighbor' to be conservative
        present = np.zeros(N, dtype=bool)
        if adj.nnz:
            csr = adj.tocsr()
            deg = (csr.indptr[1:] - csr.indptr[:-1])
            present = deg > 0
        else:
            present = np.zeros(N, dtype=bool)
        mask = present
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

def convert(processed_dir, out_dir):
    files = sorted(os.listdir(processed_dir), key=lambda x: int(os.path.splitext(x)[0]))
    if not files:
        raise RuntimeError("No files in processed_dir")
    # try to get N from first file
    first = pickle.load(open(os.path.join(processed_dir, files[0]), 'rb'))
    N = infer_N_from_mask_or_edges(first, default=None)
    if N is None:
        raise RuntimeError("Could not infer node count; please supply --N")
    print("Inferred global N =", N)
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
    for i,a in enumerate(adjs):
        arr[i] = a
    outpath = os.path.join(out_dir, "graphs.npz")
    np.savez_compressed(outpath, data=arr)
    masks_arr = np.vstack(masks).astype(bool)  # shape T x N
    np.save(os.path.join(out_dir, "node_masks.npy"), masks_arr)
    print("Wrote:", outpath)
    print("Wrote:", os.path.join(out_dir, "node_masks.npy"))
    print("Done. Dataset ready as:", os.path.basename(out_dir))
    return outpath

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--processed_dir", required=True)
    p.add_argument("--dataset_name", default=None)
    p.add_argument("--out_dir", default=None)
    args = p.parse_args()
    if args.out_dir:
        out = args.out_dir
    elif args.dataset_name:
        out = os.path.join("data", args.dataset_name)
    else:
        raise RuntimeError("Provide --out_dir or --dataset_name")
    convert(args.processed_dir, out)
