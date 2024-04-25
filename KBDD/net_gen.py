import numpy as np
from itertools import combinations


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
