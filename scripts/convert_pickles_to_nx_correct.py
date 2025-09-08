import os, argparse, pickle, numpy as np
from scipy import sparse
import networkx as nx

def safe_edge_index_to_edges(ei, N):
    if ei is None:
        return []
    ei = np.asarray(ei)
    if ei.size == 0:
        return []
    if ei.ndim == 2 and ei.shape[0] == 2:
        u = ei[0].astype(int)
        v = ei[1].astype(int)
    elif ei.ndim == 2 and ei.shape[1] == 2:
        u = ei[:,0].astype(int)
        v = ei[:,1].astype(int)
    else:
        raise ValueError(f"edge_index shape unexpected: {ei.shape}")
    # Ensure node indices are within [0, N-1]
    u = np.clip(u, 0, N-1)
    v = np.clip(v, 0, N-1)
    return list(zip(u.tolist(), v.tolist()))

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--processed_dir', required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--N', type=int, required=True)
    args = p.parse_args()

    files = sorted([f for f in os.listdir(args.processed_dir) if f.endswith('.pickle')],
                   key=lambda x: int(x.split('.')[0]))
    
    graphs = []
    masks = []
    for f in files:
        path = os.path.join(args.processed_dir, f)
        with open(path, 'rb') as fd:
            data = pickle.load(fd)
        
        # Extract edge_index and node_mask
        edge_index = data['edge_index'] if isinstance(data, dict) else getattr(data, 'edge_index', None)
        node_mask = data['node_mask'] if isinstance(data, dict) else getattr(data, 'node_mask', None)
        
        # Create graph with N nodes
        G = nx.Graph()
        G.add_nodes_from(range(args.N))
        edges = safe_edge_index_to_edges(edge_index, args.N)
        G.add_edges_from(edges)
        
        # Handle node mask
        if node_mask is None:
            node_mask = np.ones(args.N, dtype=bool)
        else:
            node_mask = np.asarray(node_mask).astype(bool)
            if node_mask.size != args.N:
                node_mask = np.resize(node_mask, args.N)
        
        graphs.append(G)
        masks.append(node_mask)
        print(f"Read {f} => nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}, mask_sum: {node_mask.sum()}")

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, 'graphs.pkl'), 'wb') as f:
        pickle.dump(graphs, f)
    # Save graphs as NetworkX objects
    np.savez_compressed(os.path.join(args.out_dir, 'graphs.npz'), graph=np.array(graphs, dtype=object))
    np.save(os.path.join(args.out_dir, 'node_masks.npy'), np.vstack(masks).astype(bool))
    print("Conversion complete.")

if __name__ == '__main__':
    main()