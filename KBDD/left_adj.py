import numpy as np
from itertools import combinations


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