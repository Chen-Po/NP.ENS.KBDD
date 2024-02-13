#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 
from itertools import combinations
import random 
from scipy.stats import powerlaw
import matplotlib.pyplot as plt
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


# ## Function

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
            structure_count_list.append(structure_count)
            Graph_list.append(nx.from_numpy_matrix(current_net))
            final_structure_list.append(current_net)
            unique_network_count += 1
            current_degree_seq = sum(np.array(current_net))
            # labeling
            # start with max degree node
            max_degree_node_index = np.where(current_degree_seq == max(current_degree_seq))[0][0]
            # print('max degree node is located in "{}" index'.format(max_degree_node_index))
            cor_sum = sum(Confidence_mat)
            # print('sum of correlation for each gene: {}'.format(cor_sum))
            max_cor_sum_gene = Gene_list[np.where(cor_sum == max(cor_sum))[0][0]]
            # print('Labeling max degree node by gene ({}) with max sum of correlation.'.format(max_cor_sum_gene))
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
            # (3) permutation
            # select two index to execute permutation
            tmp_conv_net_score = []
            tmp_Permute_size_list = []
            tmp_update_net_score = []
            tmp_Permute_gene_label = []
            # Counter initialization to 0
            no_update_count = 0
            # Initialize total iterations
            total_iterations = 0
            tmp_permute_size = Permute_size
            while total_iterations < Permute_times and tmp_permute_size < len(Gene_list):
                tmp_gene_label = np.copy(gene_label)
                permute_candidate = random.sample(range(len(Gene_list)), tmp_permute_size)
                tmp_Permute_size_list.append(tmp_permute_size)
                after_permute = permute_candidate.copy() 
                while after_permute == permute_candidate:  
                    after_permute = random.sample(permute_candidate, tmp_permute_size) 
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
                    no_update_count = 0  # Reset counter to 0 after each update
                    tmp_Permute_gene_label.append(gene_label) # gene label after updating
                else:
                    no_update_count += 1  # If no update, increment counter

                tmp_conv_net_score.append(tmp_score)
                tmp_update_net_score.append(net_score)

                if no_update_count >= Update_count:
                    tmp_permute_size += 1
                    no_update_count = 0  # Reset counter to 0 when restarting the loop
                    i = 0

                total_iterations += 1  # Increment total iterations

            # (4) save result to a list
            conv_net_score.append(tmp_conv_net_score)
            update_net_score.append(tmp_update_net_score)
            Permute_gene_label.append(tmp_Permute_gene_label)
            final_gene_label.append(gene_label)
            final_net_score.append(net_score)
            Permute_size_list.append(tmp_Permute_size_list)
    return final_structure_list, final_gene_label, final_net_score, conv_net_score, update_net_score, Permute_size_list, structure_count_list, Permute_gene_label

def truncated_power_law(alpha, maximum_value):
    """
    Generates a discrete truncated power law distribution.

    Parameters:
    - alpha : the parameter of a power-law distribution.
    - maximum_value : the maximum value of the distribution.

    Returns:
    - A sample of the distribution.
    """
    x = np.arange(1, maximum_value + 1, dtype='float')
    pmf = 1 / x**alpha
    pmf /= pmf.sum()
    return stats.rv_discrete(values=(range(1, maximum_value + 1), pmf))


# In[3]:


def KGML_to_Matrix(pathway_name, KGML_file_path = "", save_path = ""):
    """
    Converts KGML file to adjacency matrix and node detail for KEGG human biological pathway analysis.
    
    Parameters:
    - pathway_name : the name of the pathway.
    - KGML_file_path : the file path of the KGML file.
    - save_path : the directory path where the output files will be saved.
    
    The function extracts node information and relationships from the KGML file,
    processes them, and outputs an adjacency matrix and node details as pandas DataFrames.
    """
    
    # First, import the KGML file and rearrange it into 2 parts (node information and relation)
    # The xml.etree.ElementTree package is used to read the KGML file.
    
    # Read KGML file
    tree = ET.parse(KGML_file_path)
    root = tree.getroot()
    
    # Extract information from KGML file
    data_raw = []
    for child in root:
        data = {child.tag: child.attrib, "graphics": [], "component": [], "subtype": []}
        for children in child:
            # Extracting third layer of the KGML file
            data[children.tag].append(children.attrib)
        data_raw.append(data)
    
    # Remove empty sets from extracted data
    for ii in range(len(data_raw)):
        data_raw[ii] = {k: v for k, v in data_raw[ii].items() if v}
        
    # Separate the data into node information ('entry') and relationships ('relation')
    data_entry_raw, data_relation_raw = [], []
    for item in data_raw:
        if "entry" in item:
            data_entry_raw.append(item)
        if "relation" in item:
            data_relation_raw.append(item)   
            
    # Process node information
    for item in data_entry_raw:
        graphics = item.get("graphics", [{}])[0]
        if "name" in graphics:
            graphics["node_name"] = graphics.pop("name")
        if "type" in graphics:
            graphics["type1"] = graphics.pop("type")
    data_entry = [{**item["entry"], **item["graphics"][0]} for item in data_entry_raw]
    data_entry = pd.DataFrame(data_entry)[["id", "name", "type", "node_name"]]
    
    # Add component information to data_entry
    # "component.id", a variable for the index of nodes inside this group.
    # "number", the variable for the number of nodes inside this group.
    # "data_entry_new", a data-frame with an additional variable, "component", 
    # saves as (id / name / type / node_name / component)
    number = []
    for ii in range(len(data_entry_raw)):
        if "component" in data_entry_raw[ii]:
            number.append(len(data_entry_raw[ii]["component"]))
        else:
            number.append(int(1))
    component_id = []
    for ii in range(len(number)):
        for jj in range(number[ii]):
            component_id.append(range(len(number))[ii])  
            
    data_entry_component = pd.DataFrame(data_entry.iloc[component_id ,:])
    
    component = []
    for ii in range(len(data_entry_raw)):
        if "component" not in data_entry_raw[ii]:
            component.append(float("NaN"))
        if "component" in data_entry_raw[ii]:
            for jj in data_entry_raw[ii]["component"]:
                component.append(jj["id"])
    
    component = pd.DataFrame(component, index = component_id)
    component.columns = ["component"]
    data_entry_new = pd.concat([data_entry_component, component], axis = 1)
    
    # Filter out non-gene and non-compound nodes
    data_entry = data_entry[data_entry["type"] != "map"].reset_index(drop=True)
    
    # Process repeated nodes
    unique_correspond = None
    unique_entryname = None
    unique_entryid = None

    unique_entryname = data_entry["name"].unique()
    unique_erntryid = data_entry["id"][data_entry["name"].duplicated() == False]
    
    match = lambda a, b: [ b.index(x) if x in b else None for x in a ]
    
    position_repeat = match(list(data_entry["name"][data_entry["name"].duplicated()]), list(data_entry["name"]))
    
    Var1 = list(data_entry["id"][data_entry["name"].duplicated()])
    Var2 = list(data_entry["id"][position_repeat])
    unique_correspond = {"Var1":Var1, "Var2":Var2}
    unique_correspond = pd.DataFrame(unique_correspond)
    
    # Process relationships between nodes
    data_relation = pd.DataFrame([item["relation"] for item in data_relation_raw])[["entry1", "entry2"]]
    
    # processing group
    # Break group into separate nodes and make each node connect with the others.
    # If some nodes inside the pathways do not have !§relation!‥, then remove them and return the 
    # pathway names. 
    if len(data_relation) == 0:
        return print("There are no relation in the ", pathway_name, "!")
    else:
        # separate_from records group id
        # separate_to records component id    
        separate_from = list(data_entry_new[data_entry_new["type"] == "group"]["id"])
        # transfer data type to numeric
        separate_to = list(data_entry_new["component"][data_entry_new["component"].isnull() == False])
    
        # Break group into separate nodes in the relationship part  (need to use "separate_from" & 
        # "separate_to" variable)
        # replace group with its component
        relation_new = []
        for ii in range(len(data_relation.index)):
            # for relation 1
            if [i in data_relation.iloc[ii,][0] for i in separate_from].count(True) > 0:
                relation1 = [separate_to[i] for i, v in enumerate([i in data_relation.iloc[ii,][0] for i in separate_from]) if v == True]
            else:
                relation1 = [data_relation.iloc[ii,][0]]
            # for relation 2
            if [i in data_relation.iloc[ii,][1] for i in separate_from].count(True) > 0:
                relation2 = [separate_to[i] for i, v in enumerate([i in data_relation.iloc[ii,][1] for i in separate_from]) if v == True]

            else:
                relation2 = [data_relation.iloc[ii,][1]]
            relation_new.append([(x, y) for x in relation1 for y in relation2])
        # transfer from list to dataframe    
        relation_new = pd.concat(list(map(pd.DataFrame, relation_new)))
        relation_new.columns = ["Var1", "Var2"]
        relation_new = relation_new.reset_index(drop = True)
        # make each node connected with the rest in the same group.
        # "relation_group_fn", a function combines the nodes within the same group.
        def relation_group_fn(xx):
            sub_group = [separate_to[i] for i, v in enumerate([i in xx for i in separate_from]) if v == True]
            return pd.DataFrame(list(itertools.combinations(sub_group, 2)))
        
        if len(separate_from) > 0:
            relation_group = pd.concat(list(map(pd.DataFrame, list(map(relation_group_fn, sorted(list(set(separate_from))))))))
            relation_group.columns = ["Var1", "Var2"]
            relation_group = relation_group.reset_index(drop = True)
            # reverse relation between nodes inside the same group because they should be undirected (A <-> B)
            Reverse_relation = relation_group[["Var2", "Var1"]]
            Reverse_relation.columns = ["Var1", "Var2"]
            relation_group = pd.concat([relation_group, Reverse_relation]).reset_index(drop = True)
            # Finally, combine "relation_new" and "relation_group"
            relation_new = pd.concat([relation_new, relation_group]).reset_index(drop = True)
            
        # Create an adjacency matrix
        # In part 2, the relationship part, it uses node id to record the relation.
        # In order to save the relationship into the matrix, both the row name and column name in this 
        # adjacency matrix should be node id.
        relationship = pd.DataFrame(np.zeros((len(data_entry["id"]), len(data_entry["id"])), dtype = int), index = list(data_entry["id"]), columns = list(data_entry["id"]))
        # Save the connection of nodes.
        position1 = match(relation_new["Var1"], list(data_entry["id"]))
        position2 = match(relation_new["Var2"], list(data_entry["id"]))
        relation_position = pd.DataFrame({"position1" :position1, "position2" :position2})
        for ii in range(len(relation_position.index)):
            x = relation_position.iloc[ii, 0]
            y = relation_position.iloc[ii, 1]
            relationship.iloc[x, y] = 1
            
        # Deal with repeated nodes
        # Find the position of repeated nodes in the matrix.
        pos1 = match(unique_correspond["Var1"], list(data_entry["id"])) #will be deleted the end
        pos2 = match(unique_correspond["Var2"], list(data_entry["id"])) #preserved part
        # In order to avoid that pos1 does not include anything, add this "if else" constraint
        if len(pos1) != 0:
            # marge the relationship of each node if they are the same node
            for ii in range(len(pos1)):
                relationship.iloc[pos2[ii], ] = relationship.iloc[pos2[ii], ] + relationship.iloc[pos1[ii], ]
                relationship.iloc[:, pos2[ii]] = relationship.iloc[:, pos2[ii]] + relationship.iloc[:, pos1[ii]]
        # And, delete the id in column 1 to preserve unique nodes
        if len(pos1) > 0:
            pos1_relationship = [relationship.columns[ii] for ii in pos1]
            relationship = relationship.drop(pos1_relationship, axis = 1)
            relationship = relationship.drop(pos1_relationship, axis = 0)
            # the other output (node_detail) -> record all information about unique nodes.
            data_entry.index = list(data_entry["id"])
            entry_pos1 = [data_entry["id"][ii] for ii in pos1]
            node_detail = data_entry.drop(entry_pos1, axis = 0)
            node_detail = node_detail[["name", "type", "node_name"]]
            # remove the column and row of group node in the matrix
            if list(data_entry["type"] == "group").count(True) != 0:
                delete_group = data_entry.drop(entry_pos1, axis = 0)
                delete_group_drop = [delete_group.index[ii] for ii, vv in enumerate(delete_group["type"] == "group") if vv == True]
                relationship = relationship.drop(delete_group_drop, axis = 0)
                relationship = relationship.drop(delete_group_drop, axis = 1)
                node_detail = node_detail.drop(delete_group_drop, axis = 0) 
        else:
            data_entry.index = list(data_entry["id"])
            node_detail = data_entry[["name", "type", "node_name"]]
        # Change the value which is larger than 1 to 1
        for ii in range(len(relationship.index)):
            for jj in range(len(relationship.columns)):
                if relationship.iloc[ii, jj] > 1:
                    relationship.iloc[ii, jj] = 1
        adj_matrix = relationship
            
        # Save outputs    
        file_name = save_path + pathway_name + "(directed)"
        adj_matrix.to_pickle(file_name)    
        file_name = save_path + pathway_name + "(node_detail)"    
        node_detail.to_pickle(file_name)
        return "Success"


# # Reference networks (KEGG pathways)

# In[4]:


#KEGG patheay identifier
pathway2024 = ['04010', "04012", "04014", "04015", "04310", "04330", "04340", "04350", "04390","04392", 
               "04370", "04371", "04630", "04064", "04668", "04066", "04068", "04020", "04070", "04072", 
               "04071", "04024", "04022", "04151", "04152", "04150", "04115", "04550", "04620", "04621", 
               "04622", "04623", "04625", "04660", "04657", "04662", "04664","04062", "04910", "04922", 
               "04920", "03320", "04912", "04915", "04917", "04921", "04926","04919", "04261", "04723", 
               "04722", "04933"]
print(len(pathway2024))


# In[5]:


# import KEGG pathway (KGML files)
for ii in range(len(pathway2024)):
    pathway_name = "hsa" + pathway2024[ii]
    KGML_directory = "KEGG signaling pathway/hsa" + pathway2024[ii] + ".xml"
    save_directory = "KEGG application/KEGG signaling pathway/"
    KGML_to_Matrix(pathway_name = pathway_name, KGML_file_path = KGML_directory, save_path = save_directory)


# In[6]:


library_network = []
library_sparsity = []
network_size = []
alpha = []

for ii in range(len(pathway2024)):  # Assuming pathway2024 is a list of pathway identifiers
    file_name = "KEGG signaling pathway/hsa" + pathway2024[ii] + "(directed)"
    directed_adjmatrix = pd.read_pickle(file_name).to_numpy()
    # Convert directed adjacency matrix to undirected
    undirected_adjmatrix = directed_adjmatrix + directed_adjmatrix.T
    np.fill_diagonal(undirected_adjmatrix, 0)
    undirected_adjmatrix[undirected_adjmatrix > 1] = 1  # Ensure binary adjacency matrix
    library_network.append(undirected_adjmatrix)
    
    # Calculate sparsity
    one_counts = np.sum(undirected_adjmatrix[np.tril_indices_from(undirected_adjmatrix, k=-1)])
    size = undirected_adjmatrix.shape[0]
    network_size.append(size)
    sparsity = 2 * one_counts / (size * (size - 1))
    library_sparsity.append(sparsity)
    
mean_sparsity = np.mean(library_sparsity)

for current_mat in library_network:
    degree_seq = current_mat.sum(axis=0)  # Sum of rows or columns for degree
    degree_seq_non_zero = degree_seq[degree_seq != 0]
    fit = powerlaw.Fit(degree_seq_non_zero, xmin=1, discrete=True)
    alpha.append(fit.power_law.alpha)

estimated_alpha = round(np.mean(alpha), 2)
sd_alpha = round(stdev(alpha), 2)
input_alpha = np.random.normal(loc=estimated_alpha, scale=sd_alpha, size=1)


# In[7]:


print("Total reference network: {}".format(len(library_network)))


# ### Potential network construction

# In[8]:


seed = 20211208
np.random.seed(seed)
potential_net_size = 31
potential_net = []
sample_DS_data = []
start_time = time.time()
for DS_count in range(100):
    # degree sequence generation
    EG_result = "failure"
    while(EG_result == "failure"):
        input_alpha = np.random.normal(loc = estimated_alpha, scale = sd_alpha, size = 1)
        d = truncated_power_law(alpha = input_alpha, maximum_value = potential_net_size - 1)
        sample_DS = d.rvs(size = potential_net_size)
        sample_DS.sort()
        sample_DS = sample_DS[::-1]
        EG_result = EGtest(sample_DS)
    # network construction
    sample_net = net_gen(np.array(sample_DS))
    sample_DS_data.append(sample_DS)
    potential_net.extend(sample_net)
        
print("Total potential networks:", len(potential_net))
end_time = time.time()
run_time = end_time - start_time
print("Computational time：", run_time, "s")


# ### Deviation measure

# In[9]:


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


# In[10]:


library_ave_path_len, library_max_degree_centrality, library_transitivity = calculate_network_properties(library_network)
potential_ave_path_len, potential_max_degree_centrality, potential_transitivity = calculate_network_properties(potential_net)

mean_library_ave_path_len = calculate_means(library_ave_path_len)
mean_library_max_degree_centrality = calculate_means(library_max_degree_centrality)
mean_library_transitivity = calculate_means(library_transitivity)

ave_path_len_deviation = standardize_deviation(np.array(potential_ave_path_len), mean_library_ave_path_len)
max_degree_centrality_deviation = standardize_deviation(np.array(potential_max_degree_centrality), mean_library_max_degree_centrality)
transitivity_deviation = standardize_deviation(np.array(potential_transitivity), mean_library_transitivity)

deviation = abs(ave_path_len_deviation) + abs(max_degree_centrality_deviation) + abs(transitivity_deviation)


# ## Data input

# In[11]:


Ovarian_data = pd.read_csv("KEGG application/Ovarian_data.csv")

# Calculate the correlation matrix
Ovarian_data_cor = Ovarian_data.corr()

# Get gene labels
Ovarian_genes = Ovarian_data_cor.index

# Calculate partial correlations
Ovarian_data_pcor = Ovarian_data.pcorr() 

# Sorting indices based on the sum of absolute partial correlations
pcor_sort_index = np.argsort(np.sum(np.abs(Ovarian_data_pcor.values), axis=0))[::-1]

# Sort the partial correlation matrix and genes
Ovarian_pcor_sort = Ovarian_data_pcor.values[pcor_sort_index, :]
Ovarian_pcor_sort = Ovarian_pcor_sort[:, pcor_sort_index]
Ovarian_genes_pcor_sort = Ovarian_genes[pcor_sort_index]


# ## Sorted partial correlation

# In[12]:


######################################
# Labeling by permutation and Calculating network score
######################################
seed = 20211208
np.random.seed(seed)
start_time = time.time()
n = 0.05
top_n_percent = int(len(potential_net) * n)

Sturcture_output, Gene_label_output, Score_output, conv_net_score, update_net_score, permute_size_list, structure_index, permute_gene = Network_score_with_cor(Candidate_network = potential_net, 
                                                                                                                                                               Structure_score = deviation, Confidence_mat = np.abs(Ovarian_pcor_sort), 
                                                                                                                                                               Gene_list = Ovarian_genes_pcor_sort, Network_ranking_size = top_n_percent, 
                                                                                                                                                               Permute_times = 16000, Permute_size = 2, Update_count = 2000) 
results_df = pd.DataFrame({
    'StructureOutput': Sturcture_output,
    'ScoreOutput': Score_output,
    'GeneLabelOutput': Gene_label_output,
    'UpdateNetScore': update_net_score,
    'PermuteSizeList': permute_size_list,
    'StructureIndex': structure_index,
    'PermuteGene': permute_gene
})

# Sort by 'ScoreOutput' in descending order
sorted_results_df = results_df.sort_values(by='ScoreOutput', ascending=False).reset_index(drop=True)

# Access sorted columns as needed
sorted_sturcture_output = sorted_results_df['StructureOutput'].tolist()
sorted_score_output = sorted_results_df['ScoreOutput'].tolist()
sorted_gene_label_output = sorted_results_df['GeneLabelOutput'].tolist()
sorted_update_net_score = sorted_results_df['UpdateNetScore'].tolist()
sorted_permute_size_list = sorted_results_df['PermuteSizeList'].tolist()
sorted_structure_index = sorted_results_df['StructureIndex'].tolist()
sorted_permute_gene = sorted_results_df['PermuteGene'].tolist()

end_time = time.time()
run_time = end_time - start_time
print("Computational time：", run_time/60, "mins")


# In[13]:


plt.figure(figsize=(5, 4), dpi=350)

network_output = nx.from_numpy_matrix(sorted_sturcture_output[0])
node_names = sorted_permute_gene[0][0]
mapping = {i: node_names[i] for i in range(len(node_names))}
network_output = nx.relabel_nodes(network_output, mapping)

node_colors = ['cyan' if i in [0, 1] else 'white' for i in range(len(node_names))]

pos = nx.spring_layout(network_output, k=50, iterations=600, seed=100)
nx.draw(network_output, pos, with_labels=True, node_color=node_colors, node_size=350, width=1,
        edge_color='black', cmap=plt.cm.Blues, edgecolors="black", font_size=4)

plt.show()

print("Network score: {}\n".format(sorted_score_output[0]))
print("Network nodes:\n {}\n".format(sorted_gene_label_output[0]))
print("Degree_sequence:\n {}".format(np.sum(sorted_sturcture_output[0], axis = 0)))


# In[14]:


# 1st potential network
plt.figure(figsize=(5, 4), dpi=350)

network_output = nx.from_numpy_matrix(sorted_sturcture_output[0])
node_names = sorted_permute_gene[0][len(sorted_permute_gene[0])-1]
mapping = {i: node_names[i] for i in range(len(node_names))}
network_output = nx.relabel_nodes(network_output, mapping)

node_colors = ['cyan' if i in [0, 1] else 'white' for i in range(len(node_names))]

# 繪製圖形
node_colors = ['cyan' if i in [0, 1] else 'white' for i in range(len(node_names))]

pos = nx.spring_layout(network_output, k=50, iterations=600, seed=100)
nx.draw(network_output, pos, with_labels=True, node_color=node_colors, node_size=350, width=1,
        edge_color='black', cmap=plt.cm.Blues, edgecolors="black", font_size=4)

plt.show()
print("Network score: {}\n".format(sorted_score_output[0]))
print("Network nodes:\n {}\n".format(sorted_gene_label_output[0]))
print("Degree_sequence:\n {}".format(np.sum(sorted_sturcture_output[0], axis = 0)))


# In[15]:


# Thresholding and Ensemble network
thresholds = np.arange(0, 1.05, 0.05)

for i in range(len(sorted_sturcture_output)):
    match_index = []
    tmp_gene_label = np.array(sorted_gene_label_output[i])
    for j in range(len(Ovarian_genes_pcor_sort)):
        match_index.append(np.where(tmp_gene_label == Ovarian_genes_pcor_sort[j])[0][0])
    shuffle_structure = np.copy(sorted_sturcture_output[i])
    shuffle_structure = shuffle_structure[:, match_index]
    shuffle_structure = shuffle_structure[match_index,:]
    Sturcture_output[i] = shuffle_structure

potential_proportion = np.sum(Sturcture_output, axis=0)/len(Sturcture_output)

ensemble_sparsity = [np.count_nonzero(np.where(potential_proportion > threshold, 1, 0)) / ((np.where(potential_proportion > threshold, 1, 0).size ** 0.5) * ((np.where(potential_proportion > threshold, 1, 0).size ** 0.5) - 1)) for threshold in thresholds]

closest_index = None
min_distance = float('inf')

for index, sparsity in enumerate(ensemble_sparsity):
    distance = abs(sparsity - np.mean(library_sparsity))
    if distance < min_distance:
        min_distance = distance
        closest_index = index
print("Closest threshold: {}".format(thresholds[closest_index]))
print("Closest sparsity: {}".format(ensemble_sparsity[closest_index]))

for threshold in thresholds:
    threshold = round(threshold, 2)
    # Use the selected threshold to integrate the top n% potential networks
    ensemble_structure = np.where(potential_proportion > threshold, 1, 0)
    
    if threshold == round(thresholds[closest_index], 2):
        plt.figure(figsize=(5, 4), dpi=350)

        ensemble_network = nx.from_numpy_matrix(ensemble_structure)
        node_names = Ovarian_genes_pcor_sort
        mapping = {i: node_names[i] for i in range(len(node_names))}
        ensemble_network = nx.relabel_nodes(ensemble_network, mapping)


        node_colors = ['cyan' if i in [0, 1] else 'white' for i in range(len(node_names))]

        pos = nx.spring_layout(ensemble_network, k=50, iterations=600, seed=100)
        nx.draw(ensemble_network, pos, with_labels=True, node_color=node_colors, node_size=350, width=1,
                edge_color='black', cmap=plt.cm.Blues, edgecolors="black", font_size=4)
        
        plt.show()
        
        ensemble_structure_save = pd.DataFrame(ensemble_structure)
        ensemble_structure_save.index = Ovarian_genes_pcor_sort
        ensemble_structure_save.columns = Ovarian_genes_pcor_sort
        
        print(Ovarian_genes_pcor_sort)
        print(np.sum(ensemble_structure, axis = 0))


# In[16]:


# Proportion_network
G = nx.Graph()

G.add_nodes_from(Ovarian_genes_pcor_sort)

for i in range(len(potential_proportion)):
    for j in range(i + 1, len(potential_proportion)):
        weight = potential_proportion[i, j]
        G.add_edge(Ovarian_genes_pcor_sort[i], Ovarian_genes_pcor_sort[j], weight=weight)

edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
edge_weights_normalized = [2.5 * w for w in edge_weights]

plt.figure(figsize=(5, 4), dpi=350)

node_colors = ['cyan' if i in [0, 1] else 'white' for i in range(len(Ovarian_genes_pcor_sort))]

pos = nx.spring_layout(G, k=50, iterations=600, seed=100)
nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=350, width=edge_weights_normalized,
        edge_color='black', cmap=plt.cm.Blues, edgecolors="black", font_size=4)

plt.show()

