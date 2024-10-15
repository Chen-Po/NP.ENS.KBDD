import numpy as np
import networkx as nx

def calculate_network_properties(networks):
    ave_path_len = []
    max_degree_centrality = []
    transitivity = []

    for net in networks:
        G = nx.from_numpy_array(np.array(net))
        # Average path length
        dis_mat = nx.floyd_warshall_numpy(G)
        dis_mat_without_inf = np.where(np.isinf(dis_mat), len(dis_mat), dis_mat)
        ave_path_length = dis_mat_without_inf.sum()/len(dis_mat)/len(dis_mat)
        ave_path_len.append(ave_path_length)
        # Max degree centrality
        max_degree_centrality.append(max(nx.degree_centrality(G).values()))
        # Transitivity
        transitivity.append(nx.transitivity(G))

    return ave_path_len, max_degree_centrality, transitivity


def calculate_means(values):
    return np.mean(values)


def standardize_deviation(actual, mean):
    deviation = (actual - mean) / np.std(actual)
    deviation[np.isnan(deviation)] = 0
    return deviation
