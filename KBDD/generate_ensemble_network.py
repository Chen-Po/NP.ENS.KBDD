import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def generate_ensemble_network(Structure_output, Gene_label_output, Real_data_genes, reference_sparsity):
    """
    Generates an ensemble network based on the sorted structure outputs and gene labels.
    
    Parameters:
    - Structure_output : List of network structures.
    - Gene_label_output : List of gene labels for each network.
    - Real_data_genes : List of all genes considered in the analysis.
    - reference_sparsity : reference sparsity values to compare against.
        
    Returns:
    - ensemble_network : NetworkX graph. The generated ensemble network based on the optimal threshold.
    - ensemble_structure: The network after thresholding
    - potential_proportion: The networks before threholding
    """
    
    thresholds = np.arange(0, 1.05, 0.05)

    
    thresholds = np.arange(0, 1.05, 0.05)
    Structure_shuffle_output = [None] * len(Structure_output)

    for i in range(len(Structure_output)):
        match_index = []
        tmp_gene_label = np.array(Gene_label_output[i])
        for j in range(len(Real_data_genes)):
            match_index.append(np.where(tmp_gene_label == Real_data_genes[j])[0][0])
        shuffle_structure = np.copy(Structure_output[i])
        shuffle_structure = shuffle_structure[:, match_index]
        shuffle_structure = shuffle_structure[match_index,:]
        Structure_shuffle_output[i] = shuffle_structure

    potential_proportion = np.sum(Structure_shuffle_output, axis=0)/len(Structure_shuffle_output)

    ensemble_sparsity = [
        np.count_nonzero(np.where(potential_proportion > threshold, 1, 0)) /
        (((np.where(potential_proportion > threshold, 1, 0).size ** 0.5) * ((np.where(potential_proportion > threshold, 1, 0).size ** 0.5) - 1)))
        for threshold in thresholds
    ]

    closest_index, min_distance = None, float('inf')
    for index, sparsity in enumerate(ensemble_sparsity):
        distance = abs(sparsity - np.mean(reference_sparsity))
        if distance < min_distance:
            min_distance = distance
            closest_index = index
    print("Closest threshold: {}".format(thresholds[closest_index]))
    print("Closest sparsity: {}".format(ensemble_sparsity[closest_index]))

    threshold = round(thresholds[closest_index], 2)
    ensemble_structure = np.where(potential_proportion > threshold, 1, 0)
    
    ensemble_network = nx.from_numpy_array(ensemble_structure)
    mapping = {i: Real_data_genes[i] for i in range(len(Real_data_genes))}
    ensemble_network = nx.relabel_nodes(ensemble_network, mapping)

    return ensemble_structure, ensemble_network, potential_proportion
