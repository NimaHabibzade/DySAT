import os, argparse
# prefer pickle5 to support protocol 5 input
try:
    import pickle5 as p_read
except Exception:
    import pickle as p_read
import pickle as p_write
import types

class SafeUnpickler(p_read.Unpickler):
    def find_class(self, module, name):
        # Try the normal resolution first
        try:
            return super().find_class(module, name)
        except Exception:
            # Unknown class: return a simple dynamic class so attributes are set on it
            return type(name, (), {})

def safe_load(path):
    with open(path, "rb") as f:
        return SafeUnpickler(f).load()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--processed_dir", required=True)
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    files = sorted(os.listdir(args.processed_dir), key=lambda x: int(os.path.splitext(x)[0]))
    for f in files:
        inp = os.path.join(args.processed_dir, f)
        outp = os.path.join(args.out_dir, f)
        try:
            obj = safe_load(inp)
        except Exception as e:
            print("ERROR loading", inp, ":", e)
            raise
        # re-save with protocol 4
        with open(outp, "wb") as wf:
            p_write.dump(obj, wf, protocol=4)
        print("resaved", f)
    print("Done. Resaved pickles to", args.out_dir)

if __name__ == "__main__":
    main()
