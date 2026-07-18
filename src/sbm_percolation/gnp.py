"""
gnp.py

Utility functions for generating graphs from the
Erdos-Renyi model G(n, p) and for working with these
graphs.

Laber, M - 2026/07/18
"""

from .percolation import percolate
from .utils import _log_choose

import math
import numpy as np
from scipy import optimize, special
from typing import List, Tuple

def _unrank(k:int, n:int) -> Tuple[int, int]:
    """Gets the kth 2-combination of n elements in lexicographic order.

    Input
    k : the number of combinations to skip.
    n : the number of elements.

    Output
    i, j : the indices of the nodes connected by the kth edges in lexicographic order.
    """

    # solve the quadratic equation that determines the row index if combinations
    # are aranged in an upper triangular matrix.
    disc = (2 * n - 1) ** 2 - 8 * k
    i = ((2 * n - 1) - math.isqrt(disc)) // 2

    # correct possible off-by-one from integer sqrt/flooring.
    s = i * (2 * n - i - 1) // 2
    if s > k:
        i -= 1
        s = i * (2 * n - i - 1) // 2
    elif (i + 1) * (2 * n - i - 2) // 2 <= k:
        i += 1
        s = i * (2 * n - i - 1) // 2

    # compute the column.
    offset = k - s
    j = i + 1 + offset

    return i, j

def generate_gnp(n:int, p:float, rng:np.random.Generator)->List[Tuple[int, int]]:
    """Generates a graph from the model G(n,p).
    
    Input
    n : the number of nodes
    p : the probability that two nodes are connected
    rng : a numpy random generator object.

    Output
    edges : edge list of the sampled graph.
    """
    
    mmax = n * (n - 1) // 2
    edges = []
    idx = -1
    while True:

        # sample a step in the adjacency matrix 
        k = rng.geometric(p)
        idx += k
        
        if idx >= mmax:
            break
        else:
            edges.append(_unrank(idx, n))
    
    return edges

def _random_vertex_component_size_distribution_gnp(n, p, rng, n_graphs=1000):
    """Experimentally determine the distribution of the size of a component a random vertex blongs to.
    
    Input
    n : number of nodes.
    p : probability that two nodes are connected.
    rng : a numpy random generator object.
    n_graphs : the number of random graphs from which to estimate the distribution.

    Output
    pT : the component size distribution, where pT[k] is the probability that a random component has size k. 
    """

    # initialize counts to zero
    T = np.arange(n + 1)
    pT = np.zeros(n + 1)

    for _ in range(n_graphs):
        
        # sample a random graph
        edges = generate_gnp(n, p, rng)
        components = percolate(edges, n)
        
        # count component sizes, squaring accounts for the fact that a random vertex is sampled.
        counts = np.bincount(components, minlength=n + 1)
        pT += T * counts
    
    # normalize distribution
    pT /= n * n_graphs

    return pT

def _expected_random_vertex_component_size_gnp_recursive(n:int, p:float) -> float:
    """Recursively compute the expected size of the component of a random vertex in G(n,p), including the giant component if it exists.
    
    Input
    n - the number of nodes in the graph
    p - the edge probability

    Output
    expected_T - the expected size of a small connected component
    """

    # sanity check
    assert n > 0, "n must be a positive integer."
    assert 0.0 <= p <= 1.0, "p must be in [0, 1]."

    # compute the auxiliary quantity f_k, the probability that k vertices chosen from n vertices are connected, recursively
    # we use that exp(j * (k - j) * log(1+(-p))) = (1-p)^(j*(k-j))
    j = np.arange(n + 1).astype(np.float64)
    f = np.zeros(n + 1).astype(np.float64)

    f[1] = 1.0
    for k in range(2, n + 1):
        log_terms = _log_choose(k - 1.0, j[1:k] - 1.0) + np.log(f[1:k]) + (j[1:k] * (k - j[1:k]) * np.log1p(-p))
        f[k] = -np.expm1(special.logsumexp(log_terms))
        f[k] = np.clip(f[k], np.finfo(np.float64).tiny, 1.0)

    # compute the probabilities, Pr[T=k]
    k = np.arange(1, n + 1).astype(np.float64)
    log_PT = _log_choose(n - 1.0, k - 1.0) + np.log(f[1:]) + (k * (n - k) * np.log1p(-p))

    # compute the expected size
    expected_T = np.exp(special.logsumexp(np.log(k) + log_PT))

    return expected_T

def _expected_random_vertex_component_size_gnp_asymptotic(n : int, p: float) -> float:
    """Computes the expected size of the component of a random vertex in the model G(n, p) asymptotically for large n.

    Input
    n : the number of nodes
    p : the connection probability

    Output
    expected_T : the expected size of a random component in the G(n, p) model in the thermodynamic limit.
    """

    # sanity check
    assert n > 0, "n must be a positive integer."
    assert 0.0 <= p <= 1.0, "p must be in [0, 1]."
    
    # compute the expected degree
    kbar = (n - 1) * p

    # for kbar < 1 no giant component exists.
    if kbar < 1.0:
        alpha = 0.0
    
    # solve the self-consistent equation for the size of the giant component if one exists
    else:

        # define objective function alpha = 1 - exp(-alpha * k) for root finding
        def _objective(alpha):
            return 1.0 - np.exp(-alpha * kbar) - alpha

        # find valid initial bracket
        alpha_lower = 1.0
        alpha_upper = 1.0
        while np.sign(_objective(alpha_lower)) == np.sign(_objective(alpha_upper)):
            alpha_lower -= 1e-6
            if alpha_lower < 0.0:
                raise RuntimeError("Failed to find a good initial bracket found for alpha.")

        # solve the root finding problem
        solution = optimize.root_scalar(_objective, method='brentq', bracket=[alpha_lower, alpha_upper])

        # extract valid solution
        if not solution.converged:
            raise RuntimeError("Failed to solve the self-consistent equation for alpha.")
        else:
            alpha = solution.root
    
    # compute the susceptibility
    expected_T = alpha**2 * n  + (1.0 - alpha) / (1.0 - kbar * (1.0 - alpha))

    return expected_T

def expected_random_vertex_component_size_gnp(n:int, p:float, method:str='numeric', rng:np.random.Generator=None, n_graphs:int=1000)->float:
    """Compute the expected size of the component of a random vertex in G(n,p) using either a numerical approximation, an exact recursive formula, or asymptotic expression.

    Input
    n : the number of nodes in the graph
    p : the edge probability
    method : the method to use for computing the expected component size: 'numeric' via Monte Carlo sampling of graphs, 'recursive' via an exact recursive formula, 'asymptotic' via an analytic asymptotic formula.
    rng : a numpy random generator object, required if method='numeric'.
    n_graphs : the number of random graphs to sample if method='numeric'.

    Output
    expected_T : the expected size of the component of a random vertex in G(n,p).
    """

    # sanity check
    assert n >= 1, "n must be a positive integer."
    assert 0.0 <= p <= 1.0, "p must be in [0, 1]."
    if method == 'numeric':
        assert rng is not None, "rng must be provided if method='numeric'."
        assert n_graphs > 0, "n_graphs must be a positive integer if method='numeric'." 
        n_graphs = int(n_graphs)

    if method == 'numeric':
        pT = _random_vertex_component_size_distribution_gnp(n, p, rng, n_graphs)
        T = np.arange(n + 1, dtype=np.float64)
        expected_T = np.sum(T * pT)
    elif method == 'recursive':
        expected_T = _expected_random_vertex_component_size_gnp_recursive(n, p)
    elif method == 'asymptotic':
        expected_T = _expected_random_vertex_component_size_gnp_asymptotic(n, p)
    else:
        raise ValueError("Invalid method. Must be one of 'numeric', 'recursive', or 'asymptotic'.")

    return expected_T
