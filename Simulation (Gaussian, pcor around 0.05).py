#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import pandas as pd 
from itertools import combinations
import random 
from scipy.stats import powerlaw
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
import networkx as nx
import powerlaw
import scipy.stats as stats
from statistics import stdev 
import xml.etree.ElementTree as ET
import math 
from tqdm import tqdm
import time
import pingouin as pg
import itertools
from collections import Counter
from matplotlib.lines import Line2D


# In[2]:


def EGtest(DS_input):
    """
    Tests if a given degree sequence DS_input can form a simple graph according to the Erdős–Gallai theorem.
    
    Parameters:
    - DS_input : the degree sequence of a graph.
    
    Returns:
    - 'success' if the sequence can form a simple graph, 'failure' otherwise.
    """
    # Check if the sum of degrees is odd or if the sequence cannot form a simple graph
    if np.sum(DS_input) % 2 == 1 or np.sum(DS_input) > len(DS_input) * (len(DS_input) - 1):
        return 'failure'
    
    # Remove zeros for proper delete operation, if necessary
    if np.sum(DS_input) != 0 and 0 in DS_input:
        DS_input = np.delete(DS_input, np.where(DS_input == 0))
    
    # Sort in non-increasing order
    DS_input = np.sort(DS_input)[::-1]
    
    # Apply the Erdős–Gallai theorem
    for index in range(len(DS_input)):
        k = index + 1
        tmp_DS_input = np.array([
            k if DS_input[x] > k and x >= k else DS_input[x] for x in range(len(DS_input))
        ])
        if sum(tmp_DS_input[0:k]) > k * (k - 1) + np.sum(tmp_DS_input[k:]):
            return 'failure'
    
    return 'success'


def rightmost_adj(DS_input):
    """
    Identifies the rightmost adjacency set for a graph based on its degree sequence.
    
    Parameters:
    - DS_input : the degree sequence of a graph.
    
    Returns:
    - rightmost_adj_set : indices of the rightmost adjacency set.
    """
    
    non_zero_degree_node_index = np.array([i for i, x in enumerate(DS_input) if x > 0])
    
    if 0 in DS_input:
        DS_input = np.delete(DS_input, np.where(DS_input == 0))
    
    order_index = np.argsort(-DS_input)  # Descending order index
    DS_input = np.sort(DS_input)[::-1]  # Sort in descending order
    
    rightmost_adj_set = np.array([], dtype=int)
    for non_leading_node in range(1, len(DS_input))[::-1]:  # Iterate in reverse order
        tmp_DS = np.copy(DS_input)
        tmp_DS[0] -= 1  # Decrease the degree of the leading node
        tmp_DS[non_leading_node] -= 1  # Decrease the degree of the non-leading node
        
        # Prepare for Erdős–Gallai test
        if tmp_DS[0] != 0:
            DS_for_test = np.array([tmp_DS[i] - 1 if i <= tmp_DS[0] else tmp_DS[i] for i in range(1, len(tmp_DS))])
        else:
            DS_for_test = np.copy(tmp_DS)
        
        if EGtest(DS_for_test) == 'success':
            rightmost_adj_set = np.append(rightmost_adj_set, non_leading_node)
            if tmp_DS[0] == 0:
                break  # Stop if the leading node's degree reaches 0
            DS_input = np.copy(tmp_DS)
    
    # Reorder indices to match original sequence
    rightmost_adj_set = np.array(non_zero_degree_node_index[order_index[rightmost_adj_set]])
    return rightmost_adj_set

def left_adj(current_adj_mat, DS_input_original, DS_input_current, rightmost_adj, leading_node_index):
    """
    Determines the left adjacency set for the current graph.

    Parameters:
    - current_adj_mat : the current adjacency matrix.
    - DS_input_original : the original degree sequence.
    - DS_input_current : the current degree sequence after updates.
    - rightmost_adj : indices of the rightmost adjacency set.
    - leading_node_index : index of the leading node.

    Returns:
    - left_adj_set : indices of nodes forming the left adjacency set.
    """
    
    leading_node_degree = DS_input_current[leading_node_index]
    tmp_DS = np.copy(DS_input_current)
    tmp_DS[leading_node_index] = 0
    order_index = np.argsort(-tmp_DS)
    tmp_DS = np.sort(tmp_DS)[::-1]
    # remove node withh zero degree
    non_zero_DS = np.delete(tmp_DS, np.where(tmp_DS == 0))
    number_of_min = len(np.where(non_zero_DS == min(non_zero_DS))[0])
    non_min_node_index = np.where(non_zero_DS != min(non_zero_DS))[0]
    number_of_non_min = len(non_min_node_index)
    # parameters for for-loop
    tmp_left_adj_set = [[0]*leading_node_degree] 
    i_start = max(leading_node_degree - number_of_min, 0) # 0要檢查 因為python index 0 = R inedx 1
    i_end = max(min(number_of_non_min, leading_node_degree), 0) + 1
    # deal with duplicated structure
    duplicate_marker = [mat.copy() for mat in current_adj_mat]
    [duplicate_marker[i].append(DS_input_original[i]) for i in range(len(DS_input_original))]
    duplicated_index = [0]*len(duplicate_marker)
    for i in range(len(duplicate_marker)):
        for j in range(len(duplicate_marker)):
            if duplicate_marker[j] == duplicate_marker[i]:
                duplicated_index[i] = j
    duplicated_index = [duplicated_index[order_index[i]] for i in range(len(order_index))]
    for i in range(i_start, i_end):
        # first part (for non-min degree node)
        if i == 1 and number_of_non_min == 1:
            first_part = [list(non_min_node_index)]
        elif i != 0:
            first_part = [list(l) for l in combinations(non_min_node_index,i)]
            duplicated_mat = [0]*len(first_part)
            for j in range(len(first_part)):
                duplicated_mat[j] = [duplicated_index[first_part[j][k]] for k in range(len(first_part[j]))]
            unique_index = []
            unique_value = []
            for j in range(len(duplicated_mat))[::-1]:
                x = duplicated_mat[j]
                if x not in unique_value:
                    unique_value.append(x)
                    unique_index.append(j)
            first_part = [first_part[m] for m in range(len(first_part)) if m in unique_index]
        # second part (for min degree node)
        if i != leading_node_degree:
            min_degree_node = np.where(non_zero_DS == min(non_zero_DS))[0]
            if len(min_degree_node) == 1:
                second_part = [list(min_degree_node)]
            else:
                second_part = [list(l) for l in combinations(min_degree_node,leading_node_degree-i)]
            
            duplicated_mat = [0]*len(second_part)
            for j in range(len(second_part)):
                duplicated_mat[j] = [duplicated_index[second_part[j][k]] for k in range(len(second_part[j]))]
            unique_index = []
            unique_value = []
            for j in range(len(duplicated_mat))[::-1]:
                x = duplicated_mat[j]
                if x not in unique_value:
                    unique_value.append(x)
                    unique_index.append(j)
            second_part = [second_part[m] for m in range(len(second_part)) if m in unique_index]
        #combine first part & second part
        if i == 0:
            combine_two_part = second_part
        elif i == leading_node_degree:
            combine_two_part = first_part
        else:
            combine_two_part = [x + y for x in first_part for y in second_part]
        tmp_left_adj_set = tmp_left_adj_set + combine_two_part
    tmp_left_adj_set.remove(tmp_left_adj_set[0])   # delete offset
    ## check colex order
    # calculate colex score (ex: colex_order_rightmost = [0 2 3] -> clex_score_rightmost = [2^0 2^2 2^3] = [1 4 8])
    # ex: tmp_left_adj_set = [[1,2,3],[0,2,3]] -> score = 14, 13
    #     colex_order_rightmost = [0, 2, 3] -> score = 13
    # then exclude [1,2,3]
    mapping_index = [np.where(tmp_DS == tmp_DS[k])[0][0] for k in range(len(tmp_DS))]
    colex_order_rightmost = [np.where(order_index == rightmost_adj[i])[0][0] for i in range(len(rightmost_adj))]
    colex_score_rightmost = np.sum([2**(mapping_index[colex_order_rightmost[i]]) for i in range(len(colex_order_rightmost))])
    colex_score_left = []
    for i in range(len(tmp_left_adj_set)):
        colex_score_i = np.sum([2**(mapping_index[tmp_left_adj_set[i][j]]) for j in range(len(colex_order_rightmost))])
        colex_score_left.append(colex_score_i)
    check_to_the_left = np.array(colex_score_left <= colex_score_rightmost, dtype=bool)
    # filter
    tmp_left_adj_set = np.asarray(tmp_left_adj_set)
    left_adj_set = tmp_left_adj_set[check_to_the_left]
    left_adj_set = [list(order_index[k]) for k in left_adj_set]
    return left_adj_set

def connect_adj_set(leading_node_index, current_adj_mat, adj_set):
    """
    Connects nodes in the adjacency set to the leading node, updating the adjacency matrix.
    
    Parameters:
    - leading_node_index : index of the leading node in the adjacency matrix.
    - current_adj_mat : current adjacency matrix of the graph.
    - adj_set : set of nodes to be connected to the leading node.
    
    Returns:
    - output_mat : updated adjacency matrix with new connections.
    """
    
    output_mat = []
    for ii in range(len(adj_set)):
        tmp_mat = [row.copy() for row in current_adj_mat]  # Create a deep copy of the current adjacency matrix
        for jj in range(len(adj_set[0])):
            # Update adjacency for both the leading node and the current node in the set
            tmp_mat[leading_node_index][adj_set[ii][jj]] = tmp_mat[adj_set[ii][jj]][leading_node_index] = 1
        output_mat.append([row.copy() for row in tmp_mat])
    return output_mat

def net_gen(original_DS):
    """
    Generates potential network structures from a given degree sequence.
    
    Parameters:
    - original_DS : the original degree sequence for network generation.
    
    Returns:
    - complete_adj_mat : generated potential network structures.
    """
    
    sum_DS = np.sum(original_DS)
    rows, cols = len(original_DS), len(original_DS)
    incomplete_adj_mat = [[[0] * cols for _ in range(rows)]]
    complete_adj_mat = []
    max_complete_mat_count = 1000  # Limit for generated potential network structures of each degree sequence
    
    while incomplete_adj_mat and len(complete_adj_mat) < max_complete_mat_count:
        last_matrix = incomplete_adj_mat.pop()  # Get the last matrix to work on
        current_DS = original_DS - np.array([sum(row) for row in last_matrix])  # Update the degree sequence
        
        if np.sum(current_DS != 0) > 1:  # Ensure there's more than one node left to connect
            leading_node = np.argmax(current_DS)  # Find the node with the highest degree
            rightmost_adj_set = rightmost_adj(current_DS)  # Find rightmost adjacency set
            
            # Generate left adjacency sets
            left_adj_set = left_adj(
                current_adj_mat=last_matrix,
                DS_input_original=original_DS,
                DS_input_current=current_DS,
                rightmost_adj=rightmost_adj_set,
                leading_node_index=leading_node
            )
            
            new_matrices = connect_adj_set(
                leading_node_index=leading_node,
                current_adj_mat=last_matrix,
                adj_set=left_adj_set
            )
            
            for matrix in new_matrices:
                if sum([sum(row) for row in matrix]) == sum_DS:  # Check if the matrix is complete
                    complete_adj_mat.append(matrix)
                else:
                    incomplete_adj_mat.append(matrix)
    
    return complete_adj_mat

def show_graph_with_labels(adjacency_matrix, input_label=""):
    """
    Displays a graph (network) represented by an adjacency matrix with optional labels.

    Parameters:
    - adjacency_matrix : the network structure.
    - input_label : labels for the nodes of the graph (network).
    """
    gr = nx.from_numpy_matrix(np.array(adjacency_matrix))
    graph_pos = nx.spring_layout(gr, k=0.50, iterations=50)

    # Draw nodes, edges, and labels based on the provided parameters.
    nx.draw_networkx_nodes(gr, graph_pos, node_color='#1f78b4', node_size=220, alpha=0.6)
    nx.draw_networkx_edges(gr, graph_pos, width=2, alpha=0.3)
    if input_label:
        labels = {i: str(label) for i, label in enumerate(input_label)}
        nx.draw_networkx_labels(gr, graph_pos, labels)
    else:
        nx.draw_networkx_labels(gr, graph_pos)
    plt.show()

def Network_score_with_cor(Candidate_network, Structure_score, Confidence_mat, Gene_list, Network_ranking_size, Permute_size = 2):
    """
    Calculates network scores based on network structure and correlation, applying permutations to get optimal gene labels.

    Parameters:
    - Candidate_network : potential network structures.
    - Structure_score : Deviation measures for the structures.
    - Confidence_mat : confidence matrix of data.
    - Gene_list : genes of interest.
    - Network_ranking_size : top n% potential networks for further analysis.
    - Permute_size : number of genes to switch in each permutation.

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
    permutation_time = Permutation_time_fun(len(Gene_list), Permute_size)
    # execution
    # output first 500 network by default
    while unique_network_count < Network_ranking_size and structure_count < len(tmp_structure_score):
        # ignore duplicated networks 
        net_index = np.where(Structure_score == tmp_structure_score[structure_count])[0][0]
        structure_count += 1
        current_net = np.array(Candidate_network[net_index])
        # check isomorphism
        isomorphic_result = False
        if unique_network_count != 0:
            #backward checking
            current_graph = nx.from_numpy_matrix(current_net)
            for iso_test in range(unique_network_count)[::-1]:
                isomorphic_result = nx.is_isomorphic(Graph_list[iso_test], current_graph)
                if isomorphic_result == True: 
                    break
    
        # calculate network score if network is unique
        if isomorphic_result == False: 
            Graph_list.append(nx.from_numpy_matrix(current_net))
            final_structure_list.append(current_net)
            unique_network_count += 1
            current_degree_seq = sum(np.array(current_net))
            # labeling
            # start with max degree node
            max_degree_node_index = np.where(current_degree_seq == max(current_degree_seq))[0][0]
            cor_sum = sum(Confidence_mat)
            max_cor_sum_gene = Gene_list[np.where(cor_sum == max(cor_sum))[0][0]]
            
            # (1) assign gene with max sum of correlation to node with max degree
            gene_label = np.array(Gene_list.copy())
            permute_candidate = np.array(np.where(gene_label == max_cor_sum_gene)[0][0])
            permute_candidate = np.append(permute_candidate, max_degree_node_index)
            # permute candidate one and candidate two
            gene_label[permute_candidate] = gene_label[permute_candidate[::-1]]
            # new gene co-expression matrix
            gene_cor = np.copy(Confidence_mat)
            gene_cor[:,permute_candidate] = gene_cor[:,permute_candidate[::-1]]
            gene_cor[permute_candidate,:] = gene_cor[permute_candidate[::-1],:]
            # (2) calculate initial score
            net_score = 0
            for jj in range(len(Confidence_mat)):
                net_score = net_score + sum(gene_cor[jj]*current_net[jj])
            # (3) permutation (100 times by default)
            # select two index to execute permutation
            for i in range(permutation_time):
                tmp_gene_label = np.copy(gene_label)
                permute_candidate = random.sample(range(len(Gene_list)), Permute_size)
                after_permute = permute_candidate.copy() 
                while after_permute == permute_candidate:  
                    after_permute = random.sample(permute_candidate, Permute_size) 
                # permute label
                tmp_gene_label[permute_candidate] = tmp_gene_label[after_permute]
                # new gene co-expression matrix
                tmp_gene_cor = np.copy(gene_cor)
                tmp_gene_cor[:, permute_candidate] = tmp_gene_cor[:, after_permute]
                tmp_gene_cor[permute_candidate,:] = tmp_gene_cor[after_permute,:]           
                # calculate new score
                tmp_score = 0
                for jj in range(len(Confidence_mat)):
                    tmp_score = tmp_score + sum(tmp_gene_cor[jj]*current_net[jj])
                # if score is higher than updating
                if (tmp_score > net_score):
                    gene_label = tmp_gene_label.copy()
                    net_score = tmp_score.copy()
            # (4) save result to a list
            final_gene_label.append(gene_label)
            final_net_score.append(net_score)
    return final_structure_list, final_gene_label, final_net_score

def Permutation_time_fun(total_gene, candidate_number):
    comb = math.factorial(total_gene) // (math.factorial(candidate_number) * math.factorial(total_gene - candidate_number))
    return comb ** 2


def truncated_power_law(alpha, maximum_value):
    """
    Generates a discrete truncated power law distribution.

    Parameters:
    - alpha : the parameter of a power-law distribution.
    - maximum_value : the maximum value of the distribution.

    Returns:
    - A sample of the distribution.
    """
    x = np.arange(1, maximum_value+1, dtype='float')
    pmf = 1/x**alpha
    pmf /= pmf.sum()
    return stats.rv_discrete(values=(range(1, maximum_value+1), pmf))


def calculate_network_properties(networks):
    ave_path_len = []
    max_degree_centrality = []
    transitivity = []
    
    for net in networks:
        G = nx.from_numpy_matrix(np.array(net))
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


# In[3]:


total_networks = 100
sample_network = []
reference_sparsity = []
for i in range(0, total_networks):
    netname = "true_network" + str(i) + ".csv"
    hugenet = pd.read_csv(netname, header = None)
    sample_network.append(hugenet.values)
    reference_sparsity.append(np.count_nonzero(hugenet) / ((hugenet.size ** 0.5) * ((hugenet.size ** 0.5) - 1)))

mean_sparsity = np.mean(reference_sparsity)
print(round(mean_sparsity, 3))


# In[4]:


start_time = time.time()

# Initialize lists for ensemble network performance metrics
sensitivity_ensemble_network = []
specificity_ensemble_network = []
accuracy_ensemble_network = []
Thresholds_ensemble_network = []

# Temporary storage for comparison metrics
comparison_sen = []
comparison_spc = []
comparison_acc = []

for kk in tqdm(range(total_networks)):
    target_net = sample_network[kk]
    target_node_index = [p for p in range(0, len(sample_network[kk]))]
    
    library_network = [sample_network[index] for index in range(len(sample_network)) if index != kk]
    
    ############################################
    # Estimate Alpha for Power-law Distribution
    ############################################
    alpha = []
    for index in range(len(target_net)):
        current_mat = sample_network[index]
        degree_seq = [sum(row) for row in current_mat]
        degree_seq_non_zero = np.array([d for d in degree_seq if d != 0])
        fit = powerlaw.Fit(degree_seq_non_zero, xmin=1, discrete=True)
        alpha.append(fit.power_law.alpha)

    estimated_alpha = sum(alpha)/len(alpha)
    sd_alpha = stdev(alpha)
    
    
    ##########################################
    # Potential Network Structure Construction
    ##########################################
    potential_net_size = len(target_net)
    potential_net = []
    DS_index_for_sample_net = []
    
    for DS_count in range(100):
        ii = 0
        EG_result = "failure"
        while EG_result == "failure":
            ii += 1
            np.random.seed((20220113 + kk + DS_count) * ii)
            input_alpha = np.random.normal(loc=estimated_alpha, scale=sd_alpha, size=1)
            d = truncated_power_law(alpha=input_alpha, maximum_value=potential_net_size - 1)  # truncated_power_law needs definition
            sample_DS = d.rvs(size = potential_net_size)
            sample_DS.sort()
            sample_DS = sample_DS[::-1]
            if (sum(sample_DS) / 2) / math.comb(potential_net_size, 2) <= 0.3:
                EG_result = EGtest(sample_DS)  # EGtest function needs to be defined

        random.seed(kk * DS_count)
        sample_net = net_gen(np.array(sample_DS))  # net_gen function needs to be defined
        potential_net.extend(sample_net)
        DS_index_for_sample_net += [DS_count] * len(sample_net)
            
            
    ######################################
    # section 6 --- rank by deviations#### 
    ######################################
    # network properties for potential network
    potential_ave_path_len, potential_max_degree_centrality, potential_transitivity = calculate_network_properties(potential_net)
    library_ave_path_len, library_max_degree_centrality, library_transitivity = calculate_network_properties(library_network)

    # mean of network properties
    mean_library_ave_path_len = calculate_means(library_ave_path_len)
    mean_library_max_degree_centrality = calculate_means(library_max_degree_centrality)
    mean_library_transitivity = calculate_means(library_transitivity)

    ave_path_len_deviation = standardize_deviation(np.array(potential_ave_path_len), mean_library_ave_path_len)
    max_degree_centrality_deviation = standardize_deviation(np.array(potential_max_degree_centrality), mean_library_max_degree_centrality)
    transitivity_deviation = standardize_deviation(np.array(potential_transitivity), mean_library_transitivity)
    
    # dissimilarity
    deviation = abs(ave_path_len_deviation) + abs(max_degree_centrality_deviation) + abs(transitivity_deviation)
    

    ######################################
    # Labeling by permutation and Calculating network score
    ######################################    
    dataname = "label_gene_exp" + str(kk) + ".csv"
    label_gene_exp = pd.read_csv(dataname, header = None)
    
    label_cor_from_gene_exp = np.abs(label_gene_exp.pcorr().to_numpy())
    
    # consider top 5% network structure with lower deviation
    n = 0.05
    top_n_percent = int(len(potential_net) * n)
    
    np.random.seed(kk)
    Structure_output, Gene_label_output, Score_output = Network_score_with_cor(Candidate_network = potential_net, 
                                                                               Structure_score = deviation, 
                                                                               Confidence_mat = label_cor_from_gene_exp, 
                                                                               Gene_list = target_node_index,
                                                                               Network_ranking_size = top_n_percent)   

    sorted_indices = np.argsort(Score_output)[::-1]
    sorted_structure_output = [Structure_output[i] for i in sorted_indices]
    sorted_score_output = [Score_output[i] for i in sorted_indices]
    sorted_gene_label_output = [Gene_label_output[i] for i in sorted_indices]

    ######################################################
    # Compare target network with each potential network #
    ######################################################
    sensitivity = []
    specificity = []
    accuracy = []
    
    for i in range(len(sorted_structure_output)):
        match_index = []
        tmp_gene_label = np.array(sorted_gene_label_output[i])
        for j in range(len(target_node_index)):
            match_index.append(np.where(tmp_gene_label == target_node_index[j])[0][0])
        shuffle_structure = np.copy(sorted_structure_output[i])
        shuffle_structure = shuffle_structure[:, match_index]
        shuffle_structure = shuffle_structure[match_index,:]
        Structure_output[i] = shuffle_structure
        structure_diff = abs(target_net - sorted_structure_output[i])

        # sensitivity
        array_target = target_net[np.tril_indices(n=target_net.shape[0], k=-1)]
        array_potential = shuffle_structure[np.tril_indices(n=shuffle_structure.shape[0], k=-1)]
        edge_pos = np.where(array_target == 1)[0]
        edge_neg = np.where(array_target != 1)[0]
        TP = sum(array_potential[edge_pos])
        FP = sum(array_potential[edge_neg])
        FN = len(array_potential[edge_pos]) - TP
        TN = len(array_potential[edge_neg]) - FP
        
        sensitivity.append(TP/(TP + FN))
        specificity.append(TN/(TN + FP))
        accuracy.append((TP + TN)/(TP + TN + FN + FP))    
    
    window_size = 5
    weights = np.ones(window_size) / window_size
    sensitivity_mov = np.convolve(sensitivity, weights, mode='valid')
    specificity_mov = np.convolve(specificity, weights, mode='valid')
    accuracy_mov = np.convolve(accuracy, weights, mode='valid')

    # Performance values over different threshold
    sensitivity_ensemble_values = []
    specificity_ensemble_values = []
    accuracy_ensemble_values= []
    

    # Run threshold from 0 to 1
    thresholds = np.arange(0, 1.05, 0.05)
    
    potential_proportion = np.sum(Structure_output, axis=0)/len(Structure_output)
    
    proportion_results = pd.DataFrame({'proportion': potential_proportion[np.tril_indices(n=potential_proportion.shape[0], k=-1)], 
                                       'true_value': array_target})
    grouped = proportion_results.groupby('true_value')
    grouped_data = [grouped.get_group(x)['proportion'] for x in grouped.groups]

    
    ensemble_sparsity = [np.count_nonzero(np.where(potential_proportion > threshold, 1, 0)) / ((np.where(potential_proportion > threshold, 1, 0).size ** 0.5) * ((np.where(potential_proportion > threshold, 1, 0).size ** 0.5) - 1)) for threshold in thresholds]
    
    closest_index = None
    min_distance = float('inf')

    for index, sparsity in enumerate(ensemble_sparsity):
        distance = abs(sparsity - mean_sparsity)
        if distance < min_distance:
            min_distance = distance
            closest_index = index
    
    for threshold in thresholds:
        threshold = round(threshold, 2)
        # Use the selected threshold to integrate the top n% potential networks
        ensemble_structure = np.where(potential_proportion > threshold, 1, 0)
        
        # Performance evaluation of ensemble network
        array_target = target_net[np.tril_indices(n=target_net.shape[0], k=-1)]
        array_ensemble = ensemble_structure[np.tril_indices(n=ensemble_structure.shape[0], k=-1)]
        edge_pos = np.where(array_target == 1)[0]
        edge_neg = np.where(array_target != 1)[0]
        TP = sum(array_ensemble[edge_pos])
        FP = sum(array_ensemble[edge_neg])
        FN = len(array_ensemble[edge_pos]) - TP
        TN = len(array_ensemble[edge_neg]) - FP # pretty large

        sensitivity_ensemble_tmp = TP/(TP + FN)
        specificity_ensemble_tmp = TN/(TN + FP)
        accuracy_ensemble_tmp = (TP + TN)/(TP + TN + FN + FP)
        

        # append the sensitivity and specificity values to the lists
        sensitivity_ensemble_values.append(sensitivity_ensemble_tmp)
        specificity_ensemble_values.append(specificity_ensemble_tmp)
        accuracy_ensemble_values.append(accuracy_ensemble_tmp)
        
        if threshold == round(thresholds[closest_index], 2):
            sensitivity_ensemble_network.append(sensitivity_ensemble_tmp)
            specificity_ensemble_network.append(specificity_ensemble_tmp)
            accuracy_ensemble_network.append(accuracy_ensemble_tmp)
            Thresholds_ensemble_network.append(threshold)

            # calculate how many potential networks' performance lower than ensemble networks'
            
            comparison_sen.append(len([x for x in sensitivity if x < sensitivity_ensemble_tmp])/int(len(potential_net) * 0.05))
            comparison_spc.append(len([x for x in specificity if x < specificity_ensemble_tmp])/int(len(potential_net) * 0.05))
            comparison_acc.append(len([x for x in accuracy if x < accuracy_ensemble_tmp])/int(len(potential_net) * 0.05))
            
            
            n = 0.07
            top_n_percent = int(len(potential_net) * n)
            
            np.random.seed(kk)
            Structure_output, Gene_label_output, Score_output = Network_score_with_cor(Candidate_network = potential_net, 
                                                                                       Structure_score = deviation, 
                                                                                       Confidence_mat = label_cor_from_gene_exp, 
                                                                                       Gene_list = target_node_index,
                                                                                       Network_ranking_size = top_n_percent)   

            sorted_index = np.argsort(Score_output)
            sorted_index = sorted_index[::-1]
            sorted_structure_output = []
            sorted_score_output = []
            sorted_gene_label_output = []
            for i in sorted_index:
                sorted_structure_output.append(Structure_output[i])
                sorted_score_output.append(Score_output[i])
                sorted_gene_label_output.append(Gene_label_output[i])

            ######################################
            # Compare target network with potential networks
            ######################################
            sensitivity = []
            specificity = []
            accuracy = []

            for i in range(len(sorted_structure_output)):
                match_index = []
                tmp_gene_label = np.array(sorted_gene_label_output[i])
                for j in range(len(target_node_index)):
                    match_index.append(np.where(tmp_gene_label == target_node_index[j])[0][0])
                shuffle_structure = np.copy(sorted_structure_output[i])
                shuffle_structure = shuffle_structure[:, match_index]
                shuffle_structure = shuffle_structure[match_index,:]
                Structure_output[i] = shuffle_structure
                structure_diff = abs(target_net - sorted_structure_output[i])

                # sensitivity
                array_target = target_net[np.tril_indices(n=target_net.shape[0], k=-1)]
                array_potential = shuffle_structure[np.tril_indices(n=shuffle_structure.shape[0], k=-1)]
                edge_pos = np.where(array_target == 1)[0]
                edge_neg = np.where(array_target != 1)[0]
                TP = sum(array_potential[edge_pos])
                FP = sum(array_potential[edge_neg])
                FN = len(array_potential[edge_pos]) - TP
                TN = len(array_potential[edge_neg]) - FP

                sensitivity.append(TP/(TP + FN)) 
                specificity.append(TN/(TN + FP))
                accuracy.append((TP + TN)/(TP + TN + FN + FP))     

            window_size = 5
            weights = np.ones(window_size) / window_size
            sensitivity_mov_7 = np.convolve(sensitivity, weights, mode='valid')
            specificity_mov_7 = np.convolve(specificity, weights, mode='valid')
            accuracy_mov_7 = np.convolve(accuracy, weights, mode='valid')
            
            '''     
            fig, ax = plt.subplots(figsize=(6.5, 3), dpi=800)

            p1, = ax.plot(specificity_mov_7, marker='o', markersize=2.5, color="tab:blue")
            p2, = ax.plot(accuracy_mov_7, marker='^', markersize=2.5, color="tab:orange")
            p3, = ax.plot(sensitivity_mov_7, marker='s', markersize=2.5, color="tab:green")

            # Set labels
            ax.set_xlabel('Index')  
            ax.set_ylabel('Performance')

            # Add vertical line (5% potential networks)
            ax.axvline(x=len(sensitivity_mov), color='gray', linewidth=1.2, zorder=-1, linestyle='--')
            plt.ylim([0, 1.1])
            p4, = ax.plot(len(sensitivity_mov), specificity_ensemble_tmp, color="red", marker='o', zorder=5, markersize = 4, linestyle='')
            p5, = ax.plot(len(sensitivity_mov), accuracy_ensemble_tmp, color="red", marker='^', zorder=5, markersize = 4, linestyle='')
            p6, = ax.plot(len(sensitivity_mov), sensitivity_ensemble_tmp, color="red", marker='s', zorder=5, markersize = 4, linestyle='')
            l = ax.legend([p1, p2, p3, (p4, p5, p6)], ['Specificity', 'Accuracy', 'Sensitivity', 'Ensemble'], 
              numpoints=1, bbox_to_anchor=(1.25, 0.7), prop={"size": 7}, frameon=False,
              handler_map={tuple: HandlerTuple(ndivide=None)}, labelspacing = 1)

            for handle in l.legendHandles:
                handle._legmarker.set_markersize(4)

            plt.tight_layout()
            plt.show()
            '''
    ''' 
    # plot performance values through different thresholds
    fig, ax = plt.subplots(figsize=(7,3), dpi = 800)
    ax.plot(np.array(thresholds), specificity_ensemble_values, label='Specificity', marker='o', markersize=3, color="tab:blue")
    ax.plot(np.array(thresholds), accuracy_ensemble_values, label='Accuracy', marker='^', markersize=3, color="tab:orange")
    ax.plot(np.array(thresholds), sensitivity_ensemble_values, label='Sensitivity', marker='s', markersize=3, color="tab:green")
    ax.set_xlabel('Threshold', fontsize=10)
    ax.set_ylabel('Performance', fontsize=10)
    plt.xticks(np.arange(0, 1.05, 0.1))
    ax.axvline(x=thresholds[closest_index], color='gray', linewidth=1.2, zorder=-1, linestyle='--')
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.38), prop={"size": 9}, frameon=False)
    plt.show()
    '''

end_time = time.time()
run_time = end_time - start_time
print("Computational time：", run_time/60, "mins")


# In[5]:


plt.figure(figsize=(7, 3), dpi=800)
plt.plot(specificity_ensemble_network, linestyle='-', marker='o', markersize=3, label='Specificity', color="tab:blue")
plt.plot(accuracy_ensemble_network, linestyle='-', marker='^', markersize=3, label='Accuracy', color="tab:orange")
plt.plot(sensitivity_ensemble_network, linestyle='-', marker='s', markersize=3, label='Sensitivity', color="tab:green")
plt.plot(Thresholds_ensemble_network, linestyle='-', marker='d', markersize=3, label='Threshold', color="tab:red")
l = plt.legend(loc="upper right", bbox_to_anchor=(1.3, 0.7), prop={"size": 10}, frameon=False, labelspacing=1, handlelength=2)

plt.ylim([0, 1.05])

for handle in l.legendHandles:
    handle._legmarker.set_markersize(4)
    
plt.tight_layout()
plt.show()


# In[6]:


plt.figure(figsize=(7, 3), dpi=800)
plt.plot(comparison_sen, linestyle='-', linewidth = 1, marker='o', markersize=3, label='Specificity', color="tab:blue")
plt.plot(comparison_spc, linestyle='-', linewidth = 1, marker='^', markersize=3, label='Accuracy', color="tab:orange")
plt.plot(comparison_acc, linestyle='-', linewidth = 1, marker='s', markersize=2, label='Sensitivity', color="tab:green")
plt.plot(Thresholds_ensemble_network, linestyle='-', marker='d', markersize=3, label='Threshold', color="tab:red")
l = plt.legend(loc="upper right", bbox_to_anchor=(1.3, 0.7), prop={"size": 10}, frameon=False, 
               labelspacing=1, handlelength=2)

for handle in l.legendHandles:
    handle._legmarker.set_markersize(4)

plt.ylim([0, 1.05])
plt.tight_layout()

plt.show()


# In[7]:


results = {
    'Sensitivity': [np.max(sensitivity_ensemble_network), np.mean(sensitivity_ensemble_network), 
                    np.median(sensitivity_ensemble_network), np.std(sensitivity_ensemble_network), 
                    np.min(sensitivity_ensemble_network)],
    'Specificity': [np.max(specificity_ensemble_network), np.mean(specificity_ensemble_network), 
                    np.median(specificity_ensemble_network), np.std(specificity_ensemble_network), 
                    np.min(specificity_ensemble_network)],
    'Accuracy': [np.max(accuracy_ensemble_network), np.mean(accuracy_ensemble_network), 
                 np.median(accuracy_ensemble_network), np.std(accuracy_ensemble_network), 
                 np.min(accuracy_ensemble_network)],
    'Threshold': [np.max(Thresholds_ensemble_network), np.mean(Thresholds_ensemble_network), 
                  np.median(Thresholds_ensemble_network), np.std(Thresholds_ensemble_network), 
                  np.min(Thresholds_ensemble_network)],
    "Better Sensitivity": [np.max(comparison_sen), np.mean(comparison_sen), 
                           np.median(comparison_sen), np.std(comparison_sen), 
                           np.min(comparison_sen)],
    "Better Specificity": [np.max(comparison_spc), np.mean(comparison_spc), 
                           np.median(comparison_spc), np.std(comparison_spc), 
                           np.min(comparison_spc)],
    "Better Accuracy": [np.max(comparison_acc), np.mean(comparison_acc), 
                        np.median(comparison_acc), np.std(comparison_acc), 
                        np.min(comparison_acc)]
}

# Convert to DataFrame and round the results
results_df = pd.DataFrame(results).round(3)
results_df = results_df.rename(index={0: 'max', 1: 'mean', 2: 'median', 3: 'sd', 4: "min"})

results_df

