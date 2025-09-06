import pickle

path = "data/processed_data/enron_simple/0.pickle"

with open(path, "rb") as f:
    data = pickle.load(f)

# Print keys if it's a dictionary
if isinstance(data, dict):
    print("Keys:", data.keys())
    for k, v in data.items():
        # Print shape/type for arrays or tensors
        try:
            print(k, type(v), getattr(v, "shape", len(v) if hasattr(v, "__len__") else None))
        except:
            print(k, type(v))
