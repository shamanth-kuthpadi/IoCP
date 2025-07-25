import matplotlib.pyplot as plt
import networkx as nx

import logging

logging.basicConfig(
    filename="pipeline_debug_output.txt",
    filemode="w",  # Overwrite each run; use "a" to append
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s: %(message)s"
)

def visualize_graph(G, title="Causal Graph", fig_size=(10, 10)):
    """
    Visualize a NetworkX graph with matplotlib, using a hierarchical layout if possible.
    """
    plt.figure(figsize=fig_size)
    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    except Exception:
        try:
            pos = nx.nx_pydot.graphviz_layout(G, prog="dot")
        except Exception:
            pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, labels={n: n for n in G.nodes()}, node_color='lightblue', edge_color='gray', node_size=2000, font_size=12, font_weight='bold', arrowsize=20)
    plt.title(title)
    plt.axis('off')
    plt.show()
    
def save_graph_to_pickle(G, pkl_path):
    """
    Save a NetworkX graph to a pickle file.
    """
    import pickle
    with open(pkl_path, "wb") as f:
        pickle.dump(G, f)
    print(f"Graph saved to {pkl_path}")    

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

def extract_graph_refutation_metrics(graph_ref_str):
    import re
    if not isinstance(graph_ref_str, str):
        graph_ref_str = str(graph_ref_str)
        
    tpa_match = re.search(r"informative because (\d+) / (\d+).*?\(p-value: ([0-9.]+)\)", graph_ref_str, re.DOTALL)
    lmc_match = re.search(r"violates (\d+)/(\d+) LMCs.*?\(p-value: ([0-9.]+)\)", graph_ref_str, re.DOTALL)
    tpa_num, tpa_total, tpa_p = (tpa_match.group(1), tpa_match.group(2), tpa_match.group(3)) if tpa_match else (None, None, None)
    lmc_num, lmc_total, lmc_p = (lmc_match.group(1), lmc_match.group(2), lmc_match.group(3)) if lmc_match else (None, None, None)
    return tpa_num, tpa_total, tpa_p, lmc_num, lmc_total, lmc_p

def extract_refuter_metrics(refuter_result):
    import re
    if not refuter_result:
        return None, None
    if not isinstance(refuter_result, str):
        refuter_result = str(refuter_result)

    # p value
    pval_match = re.search(r"p value:([0-9.eE+-]+)", refuter_result)
    pval = pval_match.group(1).strip() if pval_match else None
    # new effect
    neweff_match = re.search(r"New effect:([0-9.eE+-]+)", refuter_result)
    neweff = neweff_match.group(1).strip() if neweff_match else None
    return pval, neweff