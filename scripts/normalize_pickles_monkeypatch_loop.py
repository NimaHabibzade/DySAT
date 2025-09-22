#!/usr/bin/env python3
import os, argparse, sys, types, importlib, re
# prefer pickle5 for protocol5 support
try:
    import pickle5 as p_read
except Exception:
    import pickle as p_read
import pickle as p_write
import numpy as np
# optional torch for tensor -> numpy conversion
try:
    import torch
except Exception:
    torch = None

def to_numpy(obj):
    if obj is None:
        return None
    if torch is not None and isinstance(obj, torch.Tensor):
        return obj.cpu().numpy()
    import numpy as _np
    if isinstance(obj, _np.ndarray):
        return obj
    if isinstance(obj, (list, tuple)):
        try:
            return _np.asarray(obj)
        except Exception:
            return _np.array(obj, dtype=object)
    if hasattr(obj, 'numpy'):
        try:
            return obj.numpy()
        except Exception:
            pass
    return _np.array(obj, dtype=object)

def ensure_placeholder_class(class_name):
    modname = 'torch_geometric.data.data'
    try:
        mod = importlib.import_module(modname)
    except Exception:
        mod = types.ModuleType(modname)
        sys.modules[modname] = mod
    if not hasattr(mod, class_name):
        cls = type(class_name, (), {})
        setattr(mod, class_name, cls)
        try:
            mod.__dict__[class_name] = cls
        except Exception:
            pass

def detect_missing_class_name(exc):
    txt = str(exc)
    m = re.search(r"Can't get attribute '([^']+)'", txt)
    if m:
        return m.group(1)
    return None

def safe_load_with_placeholders(path):
    created = set()
    # try repeatedly: detect missing class -> create placeholder -> retry
    while True:
        try:
            with open(path, 'rb') as f:
                return p_read.load(f)
        except Exception as e:
            missing = detect_missing_class_name(e)
            if missing and missing not in created:
                print("Detected missing class in pickle:", missing, " — creating placeholder and retrying.")
                ensure_placeholder_class(missing)
                created.add(missing)
                continue
            # nothing helpful to auto-fix, re-raise
            raise

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed_dir', required=True, help='folder with numbered pickle files')
    parser.add_argument('--out_dir', required=True, help='folder to write simplified pickles')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    files = sorted(
        [f for f in os.listdir(args.processed_dir) if f.lower().endswith(('.pkl','.pickle','.pt','.pth','.pickle'))],
        key=lambda x: int(os.path.splitext(x)[0])
    )
    if not files:
        print("No files found in", args.processed_dir)
        return

    for f in files:
        inp = os.path.join(args.processed_dir, f)
        outp = os.path.join(args.out_dir, f)
        print("Reading", inp)
        obj = safe_load_with_placeholders(inp)
        simple = {}
        simple['edge_index'] = to_numpy(getattr(obj, 'edge_index', None))
        simple['node_mask']  = to_numpy(getattr(obj, 'node_mask', None))
        simple['x']          = to_numpy(getattr(obj, 'x', None))
        simple['timestep']   = getattr(obj, 'timestep', None)
        simple['timestamp']  = getattr(obj, 'timestamp', None)
        simple['edge_count'] = getattr(obj, 'edge_count', None)
        with open(outp, 'wb') as wf:
            p_write.dump(simple, wf, protocol=4)
        print("Wrote simplified:", outp)

    print("Done. Simplified pickles are in:", args.out_dir)

if __name__== '__main__':
    main()
