import numpy as np
from itertools import combinations
import math
from scipy import stats

def EGtest(DS_input):
    """
    Tests if a given degree sequence DS_input can form a simple graph according to the Erdős–Gallai theorem.

    Parameters:
    - DS_input : the degree sequence of a graph.

    Returns:
    - 'success' if the sequence can form a simple graph, 'failure' otherwise.
    """
    if np.sum(DS_input) % 2 == 1 or np.sum(DS_input) > len(DS_input) * (len(DS_input) - 1):
        return 'failure'
    if np.sum(DS_input) != 0 and 0 in DS_input:
        DS_input = np.delete(DS_input, np.where(DS_input == 0))

    DS_input = np.sort(DS_input)[::-1]
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

    order_index = np.argsort(-DS_input)
    DS_input = np.sort(DS_input)[::-1]

    rightmost_adj_set = np.array([], dtype=int)
    for non_leading_node in range(1, len(DS_input))[::-1]:
        tmp_DS = np.copy(DS_input)
        tmp_DS[0] -= 1
        tmp_DS[non_leading_node] -= 1

        if tmp_DS[0] != 0:
            DS_for_test = np.array([tmp_DS[i] - 1 if i <= tmp_DS[0] else tmp_DS[i] for i in range(1, len(tmp_DS))])
        else:
            DS_for_test = np.copy(tmp_DS)

        if EGtest(DS_for_test) == 'success':
            rightmost_adj_set = np.append(rightmost_adj_set, non_leading_node)
            if tmp_DS[0] == 0:
                break
            DS_input = np.copy(tmp_DS)

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

    non_zero_DS = np.delete(tmp_DS, np.where(tmp_DS == 0))
    number_of_min = len(np.where(non_zero_DS == min(non_zero_DS))[0])
    non_min_node_index = np.where(non_zero_DS != min(non_zero_DS))[0]
    number_of_non_min = len(non_min_node_index)

    tmp_left_adj_set = [[0]*leading_node_degree]
    i_start = max(leading_node_degree - number_of_min, 0)
    i_end = max(min(number_of_non_min, leading_node_degree), 0) + 1

    duplicate_marker = [mat.copy() for mat in current_adj_mat]
    [duplicate_marker[i].append(DS_input_original[i]) for i in range(len(DS_input_original))]
    duplicated_index = [0]*len(duplicate_marker)
    for i in range(len(duplicate_marker)):
        for j in range(len(duplicate_marker)):
            if duplicate_marker[j] == duplicate_marker[i]:
                duplicated_index[i] = j
    duplicated_index = [duplicated_index[order_index[i]] for i in range(len(order_index))]
    for i in range(i_start, i_end):
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

        if i == 0:
            combine_two_part = second_part
        elif i == leading_node_degree:
            combine_two_part = first_part
        else:
            combine_two_part = [x + y for x in first_part for y in second_part]
        tmp_left_adj_set = tmp_left_adj_set + combine_two_part
    tmp_left_adj_set.remove(tmp_left_adj_set[0])

    mapping_index = [np.where(tmp_DS == tmp_DS[k])[0][0] for k in range(len(tmp_DS))]
    colex_order_rightmost = [np.where(order_index == rightmost_adj[i])[0][0] for i in range(len(rightmost_adj))]
    colex_score_rightmost = np.sum([2**(mapping_index[colex_order_rightmost[i]]) for i in range(len(colex_order_rightmost))])
    colex_score_left = []
    for i in range(len(tmp_left_adj_set)):
        colex_score_i = np.sum([2**(mapping_index[tmp_left_adj_set[i][j]]) for j in range(len(colex_order_rightmost))])
        colex_score_left.append(colex_score_i)
    check_to_the_left = np.array(colex_score_left <= colex_score_rightmost, dtype=bool)

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
        tmp_mat = [row.copy() for row in current_adj_mat]
        for jj in range(len(adj_set[0])):
            tmp_mat[leading_node_index][adj_set[ii][jj]] = tmp_mat[adj_set[ii][jj]][leading_node_index] = 1
        output_mat.append([row.copy() for row in tmp_mat])
    return output_mat


def net_gen(original_DS, max_pns=1000):
    """
    Generates potential network structures from a given degree sequence.

    Parameters:
    - original_DS : the original degree sequence for network generation.
    - max_pns: the maximum for generating potential network structures per degree sequence.

    Returns:
    - complete_adj_mat : generated potential network structures.
    """

    sum_DS = np.sum(original_DS)
    rows, cols = len(original_DS), len(original_DS)
    incomplete_adj_mat = [[[0] * cols for _ in range(rows)]]
    complete_adj_mat = []

    while (len(incomplete_adj_mat) != 0) and (len(complete_adj_mat) < max_pns):
        last_matrix = incomplete_adj_mat.pop()
        current_DS = original_DS - np.array([sum(row) for row in last_matrix])

        if np.sum(current_DS != 0) > 1:
            leading_node = np.argmax(current_DS)
            rightmost_adj_set = rightmost_adj(current_DS)

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
                if sum([sum(row) for row in matrix]) == sum_DS:
                    complete_adj_mat.append(matrix)
                else:
                    incomplete_adj_mat.append(matrix)

    return complete_adj_mat


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


def pns_generation(Real_data_genes, input_alpha, DS_number=10):
    """
    Generates potential network structure based on degree sequences drawn from truncated power law distribution
    
    Parameters:
    - Real_data_genes: List of real data genes to determine the size of potential network structures.
    - input_alpha: The alpha parameter for the power law distribution.
    - DS_number: totoal number of degree sequences

    Returns:
    - potential_net: List of generated potential network structures.
    - sample_DS_data: List of sampled degree sequences that passed EG test.
    """
    
    potential_net_size = len(Real_data_genes)
    potential_net = []
    sample_DS_data = []

    for DS_count in range(DS_number):
        EG_result = "failure"
        while EG_result == "failure":
            d = truncated_power_law(alpha=input_alpha, maximum_value=potential_net_size - 1)
            sample_DS = d.rvs(size=potential_net_size)
            sample_DS.sort()
            sample_DS = sample_DS[::-1]
            EG_result = EGtest(sample_DS)

        sample_net = net_gen(np.array(sample_DS))
        sample_DS_data.append(sample_DS)
        for net_index in range(len(sample_net)):
            potential_net.append(sample_net[net_index])

    print("Total potential network structures:", len(potential_net))
    
    return potential_net, sample_DS_data

