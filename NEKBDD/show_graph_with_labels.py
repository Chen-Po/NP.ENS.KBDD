import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def show_graph_with_labels(adjacency_matrix, input_label=""):
    """
    Displays a graph (network) represented by an adjacency matrix with optional labels.

    Parameters:
    - adjacency_matrix : the network structure.
    - input_label : labels for the nodes of the graph (network).
    """
    gr = nx.from_numpy_array(np.array(adjacency_matrix))
    graph_pos = nx.spring_layout(gr, k=0.50, iterations=50)

    nx.draw_networkx_nodes(gr, graph_pos, node_color='#1f78b4', node_size=220, alpha=0.6)
    nx.draw_networkx_edges(gr, graph_pos, width=2, alpha=0.3)
    if input_label:
        labels = {i: str(label) for i, label in enumerate(input_label)}
        nx.draw_networkx_labels(gr, graph_pos, labels)
    else:
        nx.draw_networkx_labels(gr, graph_pos)
    plt.show()