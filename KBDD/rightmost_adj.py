import numpy as np


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