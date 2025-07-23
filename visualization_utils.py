import matplotlib.pyplot as plt
import networkx as nx

def visualize_graph(G, title="Causal Graph", fig_size=(10, 10)):
    """
    Visualize a NetworkX graph with matplotlib, using a hierarchical layout if possible.
    """
    plt.figure(figsize=fig_size)
    # Try to use a hierarchical layout (dot), fallback to spring layout
    try:
        # Try pygraphviz first
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    except Exception:
        try:
            # Try pydot as a fallback
            pos = nx.nx_pydot.graphviz_layout(G, prog="dot")
        except Exception:
            pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, labels={n: n for n in G.nodes()}, node_color='lightblue', edge_color='gray', node_size=2000, font_size=12, font_weight='bold', arrowsize=20)
    plt.title(title)
    plt.axis('off')
    plt.show()

def visualize_effect_estimate(estimate, title="Effect Estimate"):
    """
    Visualize the effect estimate (placeholder for custom implementation).
    """
    print(f"{title}: {estimate}")

def visualize_refutation(refutation_result, title="Refutation Result"):
    """
    Visualize the refutation result (placeholder for custom implementation).
    """
    print(f"{title}: {refutation_result}")

def load_and_visualize_graph(pkl_path, title="Causal Graph from Pickle", fig_size=(10, 10)):
    """
    Load a pickled networkx graph from a .pkl file and visualize it.
    """
    import pickle
    import os
    if not os.path.exists(pkl_path):
        print(f"File not found: {pkl_path}")
        return
    with open(pkl_path, "rb") as f:
        G = pickle.load(f)
    visualize_graph(G, title=title, fig_size=fig_size) 