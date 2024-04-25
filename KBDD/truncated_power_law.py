import numpy as np
from scipy import stats


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