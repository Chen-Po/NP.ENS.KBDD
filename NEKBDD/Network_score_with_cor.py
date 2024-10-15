import numpy as np
import networkx as nx
import random



def Network_score_with_cor(Candidate_network, Structure_score, Confidence_mat, Gene_list, Network_ranking_size, Permute_times = 10000, Permute_size = 2, Update_count = 1000):
    """
    Calculates network scores based on network structure and correlation, applying permutations to get optimal gene labels.

    Parameters:
    - Candidate_network : potential network structures.
    - Structure_score : Deviation measures for the structures.
    - Confidence_mat : confidence matrix of data.
    - Gene_list : genes of interest.
    - Network_ranking_size : top n% potential networks for further analysis.
    - Permute_times : maximum permutation times for each structure.
    - Permute_size : number of genes to switch in each permutation.
    - Update_count : threshold for increasing permutation size.

    Returns:
    - A tuple containing various outputs related to the network scoring and permutation process.
    """
    tmp_structure_score = Structure_score.copy()
    tmp_structure_score.sort()
    final_gene_label = []
    final_net_score = []
    final_structure_list = []
    Graph_list = []
    unique_network_count = 0
    structure_count = 0
    structure_count_list = []
    conv_net_score = []
    update_net_score = []
    Permute_size_list = []
    Permute_gene_label = []
    
    # execution
    while unique_network_count < Network_ranking_size and structure_count < len(tmp_structure_score):
        net_index = np.where(Structure_score == tmp_structure_score[structure_count])[0][0]
        structure_count += 1
        current_net = np.array(Candidate_network[net_index])

        isomorphic_result = False
        if unique_network_count != 0:
            current_graph = nx.from_numpy_array(current_net)
            for iso_test in range(unique_network_count)[::-1]:
                isomorphic_result = nx.is_isomorphic(Graph_list[iso_test], current_graph)
                if isomorphic_result == True:
                    break

        if isomorphic_result == False:
            structure_count_list.append(structure_count)
            Graph_list.append(nx.from_numpy_array(current_net))
            final_structure_list.append(current_net)
            unique_network_count += 1
            current_degree_seq = sum(np.array(current_net))
            max_degree_node_index = np.where(current_degree_seq == max(current_degree_seq))[0][0]
            cor_sum = np.sum(Confidence_mat, axis=0)
            max_cor_sum_gene = Gene_list[np.where(cor_sum == max(cor_sum))[0][0]]

            gene_label = np.array(Gene_list.copy())
            permute_candidate = np.array(np.where(gene_label == max_cor_sum_gene)[0][0])
            permute_candidate = np.append(permute_candidate, max_degree_node_index)
            gene_label[permute_candidate] = gene_label[permute_candidate[::-1]]
            gene_cor = np.copy(Confidence_mat)
            gene_cor[:,permute_candidate] = gene_cor[:,permute_candidate[::-1]]
            gene_cor[permute_candidate,:] = gene_cor[permute_candidate[::-1],:]
            net_score = 0
            for jj in range(len(Confidence_mat)):
                net_score = net_score + sum(gene_cor[jj]*current_net[jj])

            tmp_conv_net_score = []
            tmp_Permute_size_list = []
            tmp_update_net_score = []
            tmp_Permute_gene_label = []

            no_update_count = 0
            total_iterations = 0
            tmp_permute_size = Permute_size
            while total_iterations < Permute_times and tmp_permute_size < len(Gene_list):
                tmp_gene_label = np.copy(gene_label)
                permute_candidate = random.sample(range(len(Gene_list)), tmp_permute_size)
                tmp_Permute_size_list.append(tmp_permute_size)
                after_permute = permute_candidate.copy()
                while after_permute == permute_candidate:
                    after_permute = random.sample(permute_candidate, tmp_permute_size)
                tmp_gene_label[permute_candidate] = tmp_gene_label[after_permute]
                tmp_gene_cor = np.copy(gene_cor)
                tmp_gene_cor[:, permute_candidate] = tmp_gene_cor[:, after_permute]
                tmp_gene_cor[permute_candidate,:] = tmp_gene_cor[after_permute,:]

                tmp_score = 0
                for jj in range(len(Confidence_mat)):
                    tmp_score = tmp_score + sum(tmp_gene_cor[jj]*current_net[jj])
                if (tmp_score > net_score):
                    gene_label = tmp_gene_label.copy()
                    net_score = tmp_score.copy()
                    no_update_count = 0
                    tmp_Permute_gene_label.append(gene_label)
                else:
                    no_update_count += 1

                tmp_conv_net_score.append(tmp_score)
                tmp_update_net_score.append(net_score)

                if no_update_count >= Update_count:
                    tmp_permute_size += 1
                    no_update_count = 0
                    i = 0

                total_iterations += 1

            conv_net_score.append(tmp_conv_net_score)
            update_net_score.append(tmp_update_net_score)
            Permute_gene_label.append(tmp_Permute_gene_label)
            final_gene_label.append(gene_label)
            final_net_score.append(net_score)
            Permute_size_list.append(tmp_Permute_size_list)
    
    # Sort by network score
    sorted_index = np.argsort(final_net_score)
    sorted_index = sorted_index[::-1]
    sorted_structure_output = []
    sorted_score_output = []
    sorted_gene_label_output = []

    for i in sorted_index:
        sorted_structure_output.append(final_structure_list[i])
        sorted_score_output.append(final_net_score[i])
        sorted_gene_label_output.append(final_gene_label[i])

    return sorted_structure_output, sorted_gene_label_output, sorted_score_output
    

