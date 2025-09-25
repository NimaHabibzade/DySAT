from __future__ import print_function
import pickle
import numpy as np
import networkx as nx
import scipy.sparse as sp
import tensorflow as tf
from .utilities import run_random_walks_n2v
import dill
import os
import random

flags = tf.app.flags
FLAGS = flags.FLAGS
np.random.seed(42)

# Optional flags you can set from command line (safe to define here)
# (If your run_script parses arguments and sets FLAGS, those will take precedence;
#  otherwise you can use environment variables described below)
flags.DEFINE_string('eval_sampling_strategy', '', "One of: rand_pos_rand_neg, rand_pos_hist_neg, hist_pos_rand_neg, hist_pos_hist_neg. If empty, use original split logic.")
flags.DEFINE_float('eval_neg_ratio', 1.0, "Negative-to-positive sampling ratio used when sampling negatives.")
# seed flag typically already exists; we will fall back to FLAGS.seed if present.

def load_graphs(dataset_str):
    """
    Robust loader that returns (graphs, adjs)
    - graphs: list of NetworkX graphs
    - adjs: list of scipy.sparse.csr_matrix adjacency matrices
    """
    path = os.path.join("data", dataset_str)
    
    # Load graphs
    graphs_path = os.path.join(path, "graphs.pkl")
    if os.path.exists(graphs_path):
        with open(graphs_path, 'rb') as f:
            graphs = pickle.load(f)
    else:
        # Fallback to npz format
        npz_path = os.path.join(path, "graphs.npz")
        if not os.path.exists(npz_path):
            raise RuntimeError(f"No graph files found in {path}")
        z = np.load(npz_path, allow_pickle=True)
        graphs = z['graph']
    
    # Convert to NetworkX graphs if they are arrays
    graphs_out, adjs = [], []
    for i, g in enumerate(graphs):
        if isinstance(g, np.ndarray):
            if g.ndim == 2:
                # Convert adjacency matrix to NetworkX graph
                G = nx.from_numpy_array(g)
            else:
                raise ValueError(f"Unsupported array shape: {g.shape}")
        elif sp.isspmatrix(g):
            # Convert sparse matrix to NetworkX graph
            G = nx.from_scipy_sparse_matrix(g)
        else:
            G = g
        
        # Create adjacency matrix
        adj = nx.adjacency_matrix(G)
        adj.setdiag(0)
        adj.eliminate_zeros()
        
        graphs_out.append(G)
        adjs.append(adj)
    
    return graphs_out, adjs

def load_feats(dataset_str):
    """ Load node attribute snapshots given the name of dataset (not used in experiments)"""
    features = np.load("data/{}/{}".format(dataset_str, "features.npz"), allow_pickle=True)['feats']
    print("Loaded {} X matrices ".format(len(features)))
    return features

def sparse_to_tuple(sparse_mx):
    """Convert scipy sparse matrix to tuple representation (for tf feed dict)."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    def to_tuple_list(matrices):
        # Input is a list of matrices.
        coords = []
        values = []
        shape = [len(matrices)]
        for i in range(0, len(matrices)):
            mx = matrices[i]
            if not sp.isspmatrix_coo(mx):
                mx = mx.tocoo()
            # Create proper indices - coords is a numpy array of pairs of indices.
            coords_mx = np.vstack((mx.row, mx.col)).transpose()
            z = np.array([np.ones(coords_mx.shape[0]) * i]).T
            z = np.concatenate((z, coords_mx), axis=1)
            z = z.astype(int)
            coords.extend(z)
            values.extend(mx.data)

        shape.extend(matrices[0].shape)
        shape = np.array(shape).astype("int64")
        values = np.array(values).astype("float32")
        coords = np.array(coords)
        return coords, values, shape

    if isinstance(sparse_mx, list) and isinstance(sparse_mx[0], list):
        # Given a list of lists, convert it into a list of tuples.
        for i in range(0, len(sparse_mx)):
            sparse_mx[i] = to_tuple_list(sparse_mx[i])

    elif isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense(), sparse_to_tuple(features)

def normalize_graph_gcn(adj):
    """GCN-based normalization of adjacency matrix (scipy sparse format). Output is in tuple format"""
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)

def get_context_pairs_incremental(graph):
    return run_random_walks_n2v(graph, graph.nodes())

def get_context_pairs(graphs, num_time_steps):
    """ Load/generate context pairs for each snapshot through random walk sampling."""
    load_path = "data/{}/train_pairs_n2v_{}.pkl".format(FLAGS.dataset, str(num_time_steps - 2))
    try:
        context_pairs_train = dill.load(open(load_path, 'rb'))
        print("Loaded context pairs from pkl file directly")
    except (IOError, EOFError):
        print("Computing training pairs ...")
        context_pairs_train = []
        for i in range(0, num_time_steps):
            context_pairs_train.append(run_random_walks_n2v(graphs[i], graphs[i].nodes()))
        dill.dump(context_pairs_train, open(load_path, 'wb'))
        print ("Saved pairs")

    return context_pairs_train

#
# --- New sampling helpers for "future-snapshot" evaluation ---
#

def _edges_from_adj(adj):
    """Return set of sorted edge tuples from a scipy CSR/COO adjacency or a networkx Graph."""
    try:
        if hasattr(adj, "tocoo"):
            coo = adj.tocoo()
            u = coo.row.tolist()
            v = coo.col.tolist()
            edges = {tuple(sorted((int(a), int(b)))) for a,b in zip(u,v) if a!=b}
            return edges
    except Exception:
        pass
    try:
        # networkx Graph
        edges = {tuple(sorted((int(a), int(b)))) for a,b in adj.edges()}
        return edges
    except Exception:
        pass
    try:
        arr = np.asarray(adj)
        if arr.ndim == 2 and arr.shape[0] in (2,):
            u = arr[0].tolist()
            v = arr[1].tolist()
            edges = {tuple(sorted((int(a), int(b)))) for a,b in zip(u,v) if a!=b}
            return edges
    except Exception:
        pass
    return set()

def sample_rand_pos_rand_neg(pos_edge_set: set, rand_neg_edge_set: set):
    n = min(len(pos_edge_set), len(rand_neg_edge_set))
    if n == 0:
        return [], []
    pos = random.sample(list(pos_edge_set), n)
    neg = random.sample(list(rand_neg_edge_set), n)
    return pos, neg

def sample_rand_pos_hist_neg(pos_edge_set: set, past_edge_set: set):
    hist_neg = list(past_edge_set.difference(pos_edge_set))
    n = min(len(pos_edge_set), len(hist_neg))
    if n == 0:
        return [], []
    pos = random.sample(list(pos_edge_set), n)
    neg = random.sample(hist_neg, n)
    return pos, neg

def sample_hist_pos_rand_neg(pos_edge_set: set, rand_neg_edge_set: set, past_edge_set: set):
    hist_pos = list(pos_edge_set.intersection(past_edge_set))
    n = min(len(hist_pos), len(rand_neg_edge_set))
    if n == 0:
        return [], []
    pos = random.sample(hist_pos, n)
    neg = random.sample(list(rand_neg_edge_set), n)
    return pos, neg

def sample_hist_pos_hist_neg(pos_edge_set: set, past_edge_set: set):
    hist_pos = list(pos_edge_set.intersection(past_edge_set))
    hist_neg = list(past_edge_set.difference(pos_edge_set))
    n = min(len(hist_pos), len(hist_neg))
    if n == 0:
        return [], []
    pos = random.sample(hist_pos, n)
    neg = random.sample(hist_neg, n)
    return pos, neg

def _build_full_past_edge_set(adjs, upto_idx):
    """Union of edges in adjs[0:upto_idx+1] (inclusive)."""
    full = set()
    for k in range(0, upto_idx+1):
        full |= _edges_from_adj(adjs[k])
    return full

def _random_neg_pool(num_needed, nodes_pool, forbidden_set, max_attempts=200000):
    """Generate up to num_needed unique negative edges (sorted tuples) from nodes_pool avoiding forbidden_set."""
    nodes = list(nodes_pool)
    rand_neg = set()
    attempts = 0
    while len(rand_neg) < max(1, num_needed) and attempts < max_attempts:
        u = random.choice(nodes)
        v = random.choice(nodes)
        if u == v:
            attempts += 1
            continue
        pair = tuple(sorted((int(u), int(v))))
        if pair in forbidden_set:
            attempts += 1
            continue
        rand_neg.add(pair)
        attempts += 1
    return rand_neg

def create_data_splits_future(adjs, eval_idx, dataset,
                              val_mask_fraction=0.2,
                              test_mask_fraction=0.6,
                              strategy='rand_pos_rand_neg',
                              neg_ratio=1.0,
                              seed=None):
    """
    Create train/val/test splits for predicting links from adjs[eval_idx] -> adjs[eval_idx+1].
    This function uses user-selectable negative-sampling strategies.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    adj = adjs[eval_idx]
    next_adj = adjs[eval_idx + 1]
    # edges in next snapshot (positive candidates)
    edges_next = np.array(list(set(nx.from_scipy_sparse_matrix(next_adj).edges())))
    # filter edges to those within node range of current adj
    edges = []
    for e in edges_next:
        if e[0] < adj.shape[0] and e[1] < adj.shape[0]:
            edges.append(e)
    edges = np.array(edges)

    # Build 'edges_all' to avoid sampling true edges as negatives
    # edges_all is next_adj's edge list
    edges_all = edges.copy()

    # Build full past up to eval_idx (inclusive of current snapshot)
    full_past = _build_full_past_edge_set(adjs, eval_idx)

    # Node presence pool: try loading node_masks if available
    node_masks_path = os.path.join("data", dataset, "node_masks.npy")
    if os.path.exists(node_masks_path):
        masks = np.load(node_masks_path)
        if masks.ndim == 2:
            nodes_pool = np.where(masks[eval_idx])[0].tolist()
        else:
            nodes_pool = list(range(adj.shape[0]))
    else:
        # fallback: nodes with degree > 0 in current adj
        csr = adj.tocsr()
        deg = (csr.indptr[1:] - csr.indptr[:-1])
        nodes_pool = np.where(deg >= 0)[0].tolist()  # include all nodes
    if len(nodes_pool) == 0:
        nodes_pool = list(range(adj.shape[0]))

    # Shuffle edges and split to train/val/test
    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    num_test = int(np.floor(edges.shape[0] * test_mask_fraction))
    num_val = int(np.floor(edges.shape[0] * val_mask_fraction))
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx] if len(test_edge_idx) > 0 else np.empty((0,2), dtype=int)
    val_edges = edges[val_edge_idx] if len(val_edge_idx) > 0 else np.empty((0,2), dtype=int)
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0) if edges.shape[0]>0 else np.empty((0,2), dtype=int)

    # For each positive set, produce negatives based on chosen strategy
    def make_negatives_for_posset(posset):
        posset_set = {tuple(sorted((int(a), int(b)))) for a,b in posset} if len(posset)>0 else set()
        # forbidden edges: those in next_adj (edges_all)
        forbidden = {tuple(sorted((int(a), int(b)))) for a,b in edges_all}
        # Build random neg pool large enough
        target_neg_total = int(round(len(posset) * neg_ratio))
        rand_neg_pool = _random_neg_pool(target_neg_total*3 if target_neg_total>0 else 1, nodes_pool, forbidden)
        # Now choose sampler
        if strategy == 'rand_pos_rand_neg':
            pos, neg = sample_rand_pos_rand_neg(posset_set, rand_neg_pool)
        elif strategy == 'rand_pos_hist_neg':
            pos, neg = sample_rand_pos_hist_neg(posset_set, full_past)
        elif strategy == 'hist_pos_rand_neg':
            pos, neg = sample_hist_pos_rand_neg(posset_set, rand_neg_pool, full_past)
        elif strategy == 'hist_pos_hist_neg':
            pos, neg = sample_hist_pos_hist_neg(posset_set, full_past)
        else:
            raise ValueError("Unknown strategy: " + str(strategy))
        # convert to list of [u,v]
        pos_list = [list(p) for p in pos]
        neg_list = [list(n) for n in neg]
        return pos_list, neg_list

    # produce lists
    train_pos, train_neg = make_negatives_for_posset(train_edges)
    val_pos, val_neg = make_negatives_for_posset(val_edges)
    test_pos, test_neg = make_negatives_for_posset(test_edges)

    print("# future-sampling: train/val/test positives sizes:", len(train_pos), len(val_pos), len(test_pos))
    print("# future-sampling: train/val/test negatives sizes:", len(train_neg), len(val_neg), len(test_neg))
    return train_pos, train_neg, val_pos, val_neg, test_pos, test_neg

#
# --- Original evaluation code (kept as fallback) and modified get_evaluation_data that can call
#     the future-sampling variant if requested via FLAGS or environment variable.
#

def get_evaluation_data(adjs, num_time_steps, dataset):
    """ Load train/val/test examples to evaluate link prediction performance"""
    eval_idx = num_time_steps - 2
    eval_path = "data/{}/eval_{}.npz".format(dataset, str(eval_idx))

    # Allow controlling sampling via TF flag or environment var:
    strategy_flag = getattr(FLAGS, 'eval_sampling_strategy', '') or os.environ.get('EVAL_SAMPLING_STRATEGY', '')
    neg_ratio_flag = float(getattr(FLAGS, 'eval_neg_ratio', 1.0) or os.environ.get('EVAL_NEG_RATIO', 1.0))
    seed_flag = getattr(FLAGS, 'seed', None)
    if seed_flag is None:
        seed_flag = int(os.environ.get('SEED', '0'))

    try:
        train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = \
            np.load(eval_path, encoding='bytes', allow_pickle=True)['data']
        print("Loaded eval data")
    except IOError:
        print("Generating and saving eval data ....")
        if strategy_flag:
            # use future-snapshot sampling with chosen strategy
            te, tne, ve, vne, tse, tsne = create_data_splits_future(
                adjs, eval_idx, dataset,
                val_mask_fraction=0.2,
                test_mask_fraction=0.6,
                strategy=strategy_flag,
                neg_ratio=neg_ratio_flag,
                seed=seed_flag
            )
            train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = \
                te, tne, ve, vne, tse, tsne
        else:
            # fallback to original splitting code (classic DySAT behavior)
            next_adjs = adjs[eval_idx + 1]
            train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = \
                create_data_splits(adjs[eval_idx], next_adjs, val_mask_fraction=0.2, test_mask_fraction=0.6)
        # save for faster reuse
        np.savez(eval_path, data=np.array([train_edges, train_edges_false, val_edges, val_edges_false,
                                           test_edges, test_edges_false], dtype=object))

    return train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false

def create_data_splits(adj, next_adj, val_mask_fraction=0.2, test_mask_fraction=0.6):
    """In: (adj, next_adj) along with test and val fractions. For link prediction (on all links), all links in
    next_adj are considered positive examples.
    Out: list of positive and negative pairs for link prediction (train/val/test)"""
    edges_all = sparse_to_tuple(next_adj)[0]  # All edges in original adj.
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)  # Remove diagonal elements
    adj.eliminate_zeros()
    assert np.diag(adj.todense()).sum() == 0
    if next_adj is None:
        raise ValueError('Next adjacency matrix is None')

    edges_next = np.array(list(set(nx.from_scipy_sparse_matrix(next_adj).edges())))
    edges = []   # Constraint to restrict new links to existing nodes.
    for e in edges_next:
        if e[0] < adj.shape[0] and e[1] < adj.shape[0]:
            edges.append(e)
    edges = np.array(edges)

    def ismember(a, b, tol=1e-8):
        """
        Safe membership test: for each row in a (shape (m, k)) returns True if that row
        appears in b (shape (n, k)). Handles empty inputs gracefully.
        Returns: boolean array of length m.
        """
        a = np.asarray(a)
        b = np.asarray(b)

        # If 'a' is empty -> nothing to compare
        if a.size == 0:
            return np.zeros((0,), dtype=bool)

        # If 'b' is empty -> no matches for any 'a' rows
        if b.size == 0:
            a_rows = a.shape[0] if a.ndim > 1 else 1
            return np.zeros((a_rows,), dtype=bool)

        # Ensure 2D (rows x cols)
        a2 = a.reshape((a.shape[0], -1))
        b2 = b.reshape((b.shape[0], -1))

        # compute absolute difference and check equality within tol
        # diff shape: (a_rows, b_rows, cols)
        diff = np.abs(a2[:, None, :] - b2[None, :, :])
        eq = np.all(diff <= tol, axis=-1)   # shape (a_rows, b_rows)
        # return boolean for each a-row whether any b-row matches
        return eq.any(axis=1)

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    num_test = int(np.floor(edges.shape[0] * test_mask_fraction))
    num_val = int(np.floor(edges.shape[0] * val_mask_fraction))
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    # Create train edges.
    train_edges_false = []
    while len(train_edges_false) < len(train_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        # Check if edge exists in any direction
        if ismember(np.array([[idx_i, idx_j]]), edges_all).any() or \
           ismember(np.array([[idx_j, idx_i]]), edges_all).any():
            continue
        # Check if edge already exists in false edges (both directions)
        if train_edges_false:
            existing_false = np.array(train_edges_false)
            if ismember(np.array([[idx_i, idx_j]]), existing_false).any() or \
               ismember(np.array([[idx_j, idx_i]]), existing_false).any():
                continue
        train_edges_false.append([idx_i, idx_j])

    # Create test edges.
    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        # Check if edge exists in any direction
        if ismember(np.array([[idx_i, idx_j]]), edges_all).any() or \
           ismember(np.array([[idx_j, idx_i]]), edges_all).any():
            continue
        # Check if edge already exists in false edges (both directions)
        if test_edges_false:
            existing_false = np.array(test_edges_false)
            if ismember(np.array([[idx_i, idx_j]]), existing_false).any() or \
               ismember(np.array([[idx_j, idx_i]]), existing_false).any():
                continue
        test_edges_false.append([idx_i, idx_j])

    # Create val edges.
    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        # Check if edge exists in any direction
        if ismember(np.array([[idx_i, idx_j]]), edges_all).any() or \
           ismember(np.array([[idx_j, idx_i]]), edges_all).any():
            continue
        # Check if edge already exists in false edges (both directions)
        if val_edges_false:
            existing_false = np.array(val_edges_false)
            if ismember(np.array([[idx_i, idx_j]]), existing_false).any() or \
               ismember(np.array([[idx_j, idx_i]]), existing_false).any():
                continue
        val_edges_false.append([idx_i, idx_j])

    # Helper: safe overlap tester
    def any_overlap(a, b):
        a = np.asarray(a)
        b = np.asarray(b)
        if a.size == 0 or b.size == 0:
            return False
        return ismember(a, b).any()

    # Replace fragile asserts with explicit checks (robust to empty arrays)
    if any_overlap(np.array(test_edges_false), edges_all):
        raise RuntimeError("Generated negative test edges overlap with true edges (test_edges_false vs edges_all).")

    if any_overlap(np.array(val_edges_false), edges_all):
        raise RuntimeError("Generated negative val edges overlap with true edges (val_edges_false vs edges_all).")

    if any_overlap(np.array(val_edges), np.array(train_edges)):
        raise RuntimeError("Val positive edges overlap with training positive edges (val_edges vs train_edges).")

    if any_overlap(np.array(test_edges), np.array(train_edges)):
        raise RuntimeError("Test positive edges overlap with training positive edges (test_edges vs train_edges).")

    if any_overlap(np.array(val_edges), np.array(test_edges)):
        raise RuntimeError("Val positive edges overlap with test positive edges (val_edges vs test_edges).")

    print("# train examples: ", len(train_edges), len(train_edges_false))
    print("# val examples:", len(val_edges), len(val_edges_false))
    print("# test examples:", len(test_edges), len(test_edges_false))

    return list(train_edges), train_edges_false, list(val_edges), val_edges_false, list(test_edges), test_edges_false
