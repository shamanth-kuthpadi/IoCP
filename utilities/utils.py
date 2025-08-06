import pandas as pd
import networkx as nx
import numpy as np
import pickle
import dill
from causallearn.utils.GraphUtils import GraphUtils
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import io
from pydot import graph_from_dot_data
import dowhy.gcm.independence_test
from dowhy.gcm.independence_test import generalised_cov_measure as gvm
import dowhy.gcm.ml
from dowhy.gcm.ml import regression as rg
from sklearn.ensemble import GradientBoostingRegressor

from pgmpy.base import DAG

# loader from txt file to pandas dataframe
def load_from_txt(filepath):
    data = pd.read_csv(filepath, sep='\t')
    return data

# rename nodes in a networkx if they are integers
def rename_nodes(G, new_node_names):
    mapping = {old_name: new_name for old_name, new_name in zip(G.nodes(), new_node_names)}
    G = nx.relabel_nodes(G, mapping)

    return G

# create a networkx from a graphviz
def gviz_to_nx(dot):
    dot.save('temp.dot')
    G = nx.drawing.nx_agraph.read_dot('temp.dot')
    return G

# calculate SHD
def shd(actual, predicted):
    actual_adj_mat = nx.adjacency_matrix(actual).todense()
    predicted_adj_mat = nx.adjacency_matrix(predicted).todense()
    predicted_adj_mat = (predicted_adj_mat != 0).astype(int)
    diff_mat = actual_adj_mat - predicted_adj_mat
    diff_mat = np.abs(diff_mat)

    return np.sum(diff_mat)

# calculate the Forbenius norm
def fnorm(actual, predicted):
    actual_adj_mat = nx.adjacency_matrix(actual).todense()
    predicted_adj_mat = nx.adjacency_matrix(predicted).todense()

    diff_mat = actual_adj_mat - predicted_adj_mat
    dotted = np.dot(diff_mat.T, diff_mat)
    trace = np.trace(dotted)

    return np.sqrt(trace)

# display a GeneralGraph
def disp_graph(graph, labels=None):
    pyd = GraphUtils.to_pydot(graph, labels=labels)
    tmp_png = pyd.create_png(f="png")
    fp = io.BytesIO(tmp_png)
    img = mpimg.imread(fp, format='png')
    plt.axis('off')
    plt.imshow(img)
    plt.show()

def disp_graph_nx(G, fig_size=(10, 10)):
    """
    Displays a NetworkX graph using pydot for rendering.
    
    Parameters:
    - G (networkx.Graph): The graph to display.
    - fig_size (tuple): Size of the matplotlib figure, default is (10, 10).
    """
    # Convert NetworkX graph to a PyDot graph
    pydot_graph = nx.nx_pydot.to_pydot(G)
    # Render graph to PNG
    tmp_png = pydot_graph.create_png()
    # Read the PNG image
    fp = io.BytesIO(tmp_png)
    img = mpimg.imread(fp, format='png')
    # Plot with adjusted figure size
    plt.figure(figsize=fig_size)
    plt.axis('off')
    plt.imshow(img)
    plt.show()

# GeneralGraph to networkx
def genG_to_nx(G, labels):
    pyd = GraphUtils.to_pydot(G, labels=labels)
    dot_data = pyd.to_string()
    pydot_graph = graph_from_dot_data(dot_data)[0]
    predicted_graph = nx.drawing.nx_pydot.from_pydot(pydot_graph)
    predicted_graph = nx.DiGraph(predicted_graph)
    predicted_graph = rename_nodes(predicted_graph, labels)
    return predicted_graph

def create_gradient_boost_regressor(**kwargs):
    return rg.SklearnRegressionModel(GradientBoostingRegressor(**kwargs))
def gcm(X, Y, Z=None):
    return gvm.generalised_cov_based(X, Y, Z=Z, prediction_model_X=create_gradient_boost_regressor,
                                 prediction_model_Y=create_gradient_boost_regressor)

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
        
    # maybe add a converting statement to convert if the graph is not in the format of a networkx graph
    disp_graph_nx(G) 

def load_instance_from_pickle(file_path='outputs/causal_module_instance.pkl'):
    """
    Loads a CausalModule instance from a pickle file.
    :param file_path: The path to the pickle file.
    :return: The loaded CausalModule instance.
    """
    with open(file_path, 'rb') as f:
        instance = dill.load(f)
    print(f"CausalModule instance loaded from {file_path}")
    return instance

def save_instance_to_pickle(instance, file_path):
    """
    Saves a CausalModule instance to a pickle file.
    :param instance: The CausalModule instance to save.
    :param file_path: The path to save the pickle file.
    """
    with open(file_path, 'wb') as f:
        dill.dump(instance, f)
    print(f"CausalModule instance saved to {file_path}")
      
    