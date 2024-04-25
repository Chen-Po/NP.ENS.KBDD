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