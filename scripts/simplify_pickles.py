import os, argparse
# prefer pickle5 to read protocol 5 if present
try:
    import pickle5 as p_read
except Exception:
    import pickle as p_read
import pickle as p_write
import numpy as np

# torch may be used when tensors are present
try:
    import torch
except Exception:
    torch = None

def to_numpy(obj):
    if obj is None:
        return None
    # torch tensor
    if torch is not None and isinstance(obj, torch.Tensor):
        return obj.cpu().numpy()
    # numpy already
    try:
        import numpy as _np
        if isinstance(obj, _np.ndarray):
            return obj
    except Exception:
        pass
    # lists -> numpy
    if isinstance(obj, (list, tuple)):
        try:
            return np.asarray(obj)
        except Exception:
            return np.array(obj, dtype=object)
    # fallback: try attribute access for sparse or tensors
    try:
        if hasattr(obj, 'numpy'):
            return obj.numpy()
    except Exception:
        pass
    # last resort: return as object array
    return np.array(obj, dtype=object)

import os, argparse
# ... your existing imports ...

# --- Add this class definition to resolve the AttributeError ---
class DataEdgeAttr:
    # This is a placeholder class. You might need to adjust it if any attributes are accessed.
    def __init__(self, *args, **kwargs):
        # If you know the original structure, you can add attributes here.
        # For example, if it had 'edge_attr' and 'size', you could do:
        # self.edge_attr = kwargs.get('edge_attr', None)
        # self.size = kwargs.get('size', None)
        pass

# ... the rest of your code, including safe_load and main ...

def safe_load(path):
    with open(path, "rb") as f:
        return p_read.load(f)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--processed_dir", required=True, help="folder with numbered pickle files")
    p.add_argument("--out_dir", required=True, help="folder to write simplified pickles")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    files = sorted(os.listdir(args.processed_dir), key=lambda x: int(os.path.splitext(x)[0]))
    if not files:
        raise SystemExit("No files found in processed_dir: " + args.processed_dir)

    for f in files:
        inp = os.path.join(args.processed_dir, f)
        outp = os.path.join(args.out_dir, f)
        print("Reading", inp)
        try:
            obj = safe_load(inp)
        except Exception as e:
            print("ERROR loading", inp, ":", e)
            raise

        # Build a plain dictionary with safe numpy objects
        simple = {}
        # edge_index -> numpy (shape [2, E])
        edge_index = getattr(obj, "edge_index", None)
        simple["edge_index"] = to_numpy(edge_index)

        # node_mask -> numpy boolean vector
        node_mask = getattr(obj, "node_mask", None)
        simple["node_mask"] = to_numpy(node_mask)

        # x -> numpy (features) (may be sparse-like; keep as numpy object if needed)
        x = getattr(obj, "x", None)
        simple["x"] = to_numpy(x)

        # optional metadata
        simple["timestep"] = getattr(obj, "timestep", None)
        simple["timestamp"] = getattr(obj, "timestamp", None)
        simple["edge_count"] = getattr(obj, "edge_count", None)

        # Save the plain dict with protocol 4
        with open(outp, "wb") as wf:
            p_write.dump(simple, wf, protocol=4)
        print("Wrote simplified:", outp)

    print("Done. Simplified pickles are in:", args.out_dir)

if __name__ == "__main__":
    main()
