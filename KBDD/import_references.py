import numpy as np
import pandas as pd

def import_references(pathway_ids, base_path):
    """
    This function can read  output of KGML_to_Matrix to builds undirected adjacency matrices for each pathway specified by pathway_ids,

    Parameters:
    - pathway_ids : The IDs of the pathways for which to construct networks.
    - base_path : The base directory path where the reference networks are stored.

    Returns:
    - reference_network : A list containing the undirected adjacency matrices for each pathway.
    - network_size : A list containing the size of each network.
    - reference_sparsity : A list containing the sparsity of each network.
    """

    reference_network = []
    network_size = []
    reference_sparsity = []
    
    for pathway_id in pathway_ids:
        file_name = f"{base_path}/hsa{pathway_id}(directed)"
        directed_adjmatrix = pd.read_pickle(file_name).to_numpy()
        undirected_adjmatrix = directed_adjmatrix + directed_adjmatrix.T
        np.fill_diagonal(undirected_adjmatrix, 0)
        undirected_adjmatrix[undirected_adjmatrix > 1] = 1
        reference_network.append(undirected_adjmatrix)

        one_counts = np.sum(undirected_adjmatrix[np.tril_indices_from(undirected_adjmatrix, k=-1)])
        size = undirected_adjmatrix.shape[0]
        network_size.append(size)
        sparsity = 2 * one_counts / (size * (size - 1))
        reference_sparsity.append(sparsity)

    return reference_network, network_size, reference_sparsity
