import os
from visualization_utils import load_and_visualize_graph

# Path to the most recent results directory
results_dir = "results_20250717_153322"

# List all .pkl files in the directory
for fname in os.listdir(results_dir):
    if fname.endswith(".pkl"):
        graph_path = os.path.join(results_dir, fname)
        # Infer algorithm name from filename
        if "causal_graph_" in fname and fname.endswith(".pkl"):
            algo = fname.replace("causal_graph_", "").replace(".pkl", "").upper()
            title = f"Causal Graph ({algo})"
        else:
            title = fname
        print(f"Visualizing: {graph_path}")
        load_and_visualize_graph(graph_path, title=title) 