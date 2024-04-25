import numpy as np


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