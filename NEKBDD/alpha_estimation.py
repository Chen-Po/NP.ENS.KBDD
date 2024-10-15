import powerlaw
from statistics import stdev
import numpy as np

def alpha_estimation(reference_network):
    '''
    Estimate the parameter of power-law ditribution
    '''
    """
    Fit a power law distribution to the degree sequences and
    estimate the parameters of the power-law distribution across networks.

    Parameters:
    - reference_network: A list of adjacency matrices for each network.

    Returns:
    - estimated_alpha : The mean alpha parameter of the power law distribution fitted to the degree sequences.
    - sd_alpha : The standard deviation of the alpha parameters across networks.
    - input_alpha : A random alpha value sampled from a normal distribution centered around estimated_alpha
        with a standard deviation of sd_alpha.
    """
    
    alpha = []
    for current_mat in reference_network:
        degree_seq = current_mat.sum(axis=0)
        degree_seq_non_zero = degree_seq[degree_seq != 0]
        fit = powerlaw.Fit(degree_seq_non_zero, xmin=1, discrete=True)
        alpha.append(fit.power_law.alpha)

    estimated_alpha = round(np.mean(alpha), 2)
    sd_alpha = round(stdev(alpha), 2)
    input_alpha = np.random.normal(loc=estimated_alpha, scale=sd_alpha, size=1)
    
    return estimated_alpha, sd_alpha, input_alpha
