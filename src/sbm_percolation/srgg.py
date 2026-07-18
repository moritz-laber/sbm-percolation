"""
srgg.py

Utility functions for working with soft random geometric graphs (SRGGs)
and their discretized SBM approximations.

Laber, M - 2026/07/18
"""

from .gnp import expected_random_vertex_component_size_gnp, _random_vertex_component_size_distribution_gnp

import numpy as np
import scipy.optimize as optimize
from typing import Tuple, Optional

def generate_srgg(n:int, c:float, beta:float, rng:np.random.Generator)->Tuple[list[Tuple[int, int]], np.ndarray]:
    """Generates a soft random geometric graph (SRGG) in 1D.
    
    Input
    n : the number of nodes
    c : length scale
    beta : inverse temperature
    rng : random number generator

    Output
    edges : list of edge in the SRGG
    """

    # type check and sanity check
    n = int(n)
    c = float(c)
    beta = float(beta)

    assert n > 0, "n must be a positive integer."
    assert beta > 0, "beta must be a positive number."

    # generate node positions uniformly in [0, n)
    x = rng.uniform(0, n, size=n)

    # compute pairwise distances on the circle
    q = rng.uniform(0, 1., size=n*(n - 1)//2)
    idx = 0
    edges = []
    for i in range(n):
        
        # compute the distance on the circle from i to all j > i .
        dij = n/2. - np.abs(n/2. - np.abs(x[i] - x[i+1:]))

        # compute the connection probability
        pij = np.minimum(np.power(c / dij, beta), 1.0)

        # determine which edges to create
        js = np.where(q[idx:idx + n - i - 1] < pij)[0] + i + 1
        idx += n - i - 1

        # add edges to the list
        if len(js) > 0:
            edges.extend(list(zip(len(js)*[i], js.tolist())))

    return edges

def _connection_probability_discretized_srgg_numerical(c:float, beta:float, s:float|int, m:float|int, N:int)->float:
    """Compute the connection probability in the SBM approximation of the SRGG model using numerical integration.
    
    Input
    c : the length scale in the SRGG model.
    beta : inverse temperature in the SRGG model.
    s : the number of nodes in each group of the SBM.
    m : the distance between groups.
    N : the number of discrete points on [0, s) used for numerical integration.

    Output
    p : the connection probability in the discretized SRGG.
    """

    # define a small quantity to avoid division by zero
    eps = np.finfo(float).eps

    # create a grid on [0, s) for numerical integration
    xi = np.linspace(0.0, s, N)
    xj = np.linspace(m*s, (m+1)*s, N)

    # compute the distances at resolution 1/N between two groups of size s at distance m
    dij = np.abs(xi[:, None] - xj[None, :])
    
    # compute the connection probability for this group pair. Add small eps to avoid division by zero.
    pij = np.minimum(1.0, np.power(c / (dij + eps), beta))

    # integrate the connection probability over the block.
    p = np.sum(pij) * (1.0 / N**2)

    return p

def _connection_probability_discretized_srgg_analytical(c:float, beta:float, s:float|int, m:float|int)->float:
    """Compute the connection probability in the SBM approximation of the SRGG model using the analytical formula.
    
    Input
    c : the length scale in the SRGG model.
    beta : inverse temperature in the SRGG model.
    s : the number of nodes in each group of the SBM.
    m : the distance between groups.

    Output
    p : the connection probability in the discretized SRGG.
    """

    # compute the numerator
    if m == 0:
        p = 2.0 * beta * c * (2.0 - beta) * s**(beta - 1.0) + beta * (beta - 1.0) * c**2.0 * s**(beta - 2.0) - 2.0 * c**beta
    elif m == 1:
        p = 2.0 * (1.0 - 2**(1.0 - beta)) * c**beta - 0.5 * beta * (beta - 1.0) * c**2.0 * s**(beta - 2.0)
    elif m > 1:
        p =  c**beta * (2.0 * m**(2.0 - beta) - (m - 1.0)**(2.0 - beta) - (m + 1.0)**(2.0 - beta))
    else:
        raise ValueError(f"Invalid value of m, {m}. m needs to be non-negative.")
    
    # normalize by shared denominator
    p /= (beta - 1.0) * (2.0 - beta) * s**beta

    return p

def connection_probability_discretized_srgg(c:float, beta:float, s:float|int, m:float|int, average:str='analytical', N:int=None)->float:
    """Compute the connection probability in the SBM approximation of the SRGG model.
    
    Input
    c : the length scale in the SRGG model.
    beta : inverse temperature in the SRGG model.
    s : the number of nodes in each group of the SBM.
    m : the distance between groups in terms of the number of groups.
    average : whether to compute the connection probability using the analytical formula ('analytical') or numerical integration ('numerical').
    N : the number of discrete points used for numerical integration. Only used if average='numerical'.
    
    Output
    p : the connection probability in the discretized SRGG.
    """
    
    # compute the connection probability analytically or numerically
    if average == 'analytical':
        p = _connection_probability_discretized_srgg_analytical(c, beta, s, m)
    elif average == 'numerical':
        p = _connection_probability_discretized_srgg_numerical(c, beta, s, m, N)
    else:
        raise ValueError(f"Unknown average: {average}. average must be one of: 'analytical' or 'numerical'.")
    
    return p

def connection_probability_matrix_discretized_srgg(n:int, c:float, beta:float, s:int, average:str='analytical', N:int=None) -> np.ndarray:
    """Compute the connection probability matrix of the SBM approximation of the SRGG model.
    
    Input
    n : the number of nodes in the graph.
    c : the length scale in the SRGG model.
    beta : inverse temperature in the SRGG model.
    s : the number of nodes in each group of the SBM.
    average : whether to compute the connection probabilities using the analytical formula ('analytical') or numerical integration ('numerical').
    N : the number of discrete points used for numerical integration. Only used if average='numerical'.    
    """

    # sanity check
    n = int(n)
    c = float(c)
    beta = float(beta)
    s = int(s)
    assert n > 0, "the number of nodes n must be a positive integer."
    assert 1.0 < beta  < 2.0, "the inverse temperature beta must be in (1.0, 2.0)."
    assert s > 0, "the group size s must be a positive integer."
    assert n % s == 0, "the number of nodes n must be divisible by the group size s."
    assert (n // s) % 2 == 0, "the number of groups n/s must be even."
    assert average in ['analytical', 'numerical'], "average must be one of: 'analytical' or 'numerical'."
    if average == 'numerical':
        assert N is not None, "N must be specified if average='numerical'."
        assert N > 1., "N must be a positive integer larger than 2. Ideally larger than 1000 for good accuracy."

    # compute the number of groups
    B = n // s

    # prepare the connection probability matrix
    P = np.zeros((B, B), dtype=np.float32)

    # fill upper triangle
    for m in range(1, B//2):
        pm = connection_probability_discretized_srgg(c, beta, s, m=m, average=average, N=N)
        P += np.diag(pm * np.ones(B - m), m)
        P += np.diag(pm * np.ones(m), B - m)    
    
    # avoid double filling the B//2 diagonal
    pBhalf = connection_probability_discretized_srgg(c, beta, s, m=B//2, average=average, N=N)
    P += np.diag(pBhalf * np.ones(B//2), B//2)
    
    # fill lower triangle
    P += P.T

    # fill the diagonal
    p0 = connection_probability_discretized_srgg(c, beta, s, m=0, average=average, N=N)
    P += np.diag(p0 * np.ones(B), 0)

    return P

def average_degree_srgg(c:float, beta:float, n:int=None) -> float:
    """Compute the average degree in the SRGG model in the thermodynamic limit.
    
    Input
    c : the length scale in the SRGG model.
    beta : inverse temperature in the SRGG model.
    n : the number of nodes if finite-size correction is desired (set None otherwise).

    Output
    kbar : the average degree in the SRGG model in the thermodynamic limit.
    """

    if n is None:
        kbar = 2.0 * c * (beta / (beta - 1.0))
    else:
        kbar = (2.0 * (n - 1.0) * c) * (beta - (2.0 * c/ n)**(beta - 1.0)) / (n * (beta - 1.0))

    return kbar

def average_degree_discretized_srgg(c:float, beta:float, s:int, n:Optional[int]=None, degree_type:str='total') -> float:
    """Compute the average degree in the discretized SRGG model in the thermodynamics.
    
    Input
    c : the length scale in the SRGG model.
    beta : inverse temperature in the SRGG model.
    s : the number of nodes in each group of the SBM.
    n : (optional, default None), if provided computes the average degree at a finite number of nodes n. 
    degree_type : whether to count all ('total'), only in-group ('in') or only out-group ('out') connections.

    Output
    kbar : the average degree in the SRGG model.
    """

    # uses thermodynamic limit formulas
    if n is None:
        # compute the total average degree
        numerator_tot = 2 * beta * (2.0 - beta) * (s - 1.0) * c * s**(beta - 1.0) - beta * (beta - 1.0) * c**2 * s**(beta - 2.0) + 2 * c**beta
        denominator_tot = (beta - 1.0) * (2.0 - beta) * s**beta
        kbar_tot = numerator_tot / denominator_tot

        # compute the average out-group degree
        numerator_out = 2 * c**beta - beta * (beta - 1.0) * c**2 * s**(beta - 2.0)
        denominator_out = (beta - 1.0) * (2.0 - beta) * s**(beta - 1.0)
        kbar_out = numerator_out / denominator_out
    
    elif n > 0:
        B = n//s
        kbar_in = (s - 1.0) * connection_probability_discretized_srgg(c, beta, s, m=0)
        kbar_out = s * (np.sum([2.0 * connection_probability_discretized_srgg(c, beta, s, m) for m in range(1, B//2)])
                    + s * connection_probability_discretized_srgg(c, beta, s, m=B//2))

        kbar_tot = kbar_in + kbar_out

    else:
        raise ValueError("n needs to be None or an positive number.")


    if degree_type == 'total':
        kbar = kbar_tot
    elif degree_type == 'out':
        kbar = kbar_out
    elif degree_type == 'in':
        kbar = kbar_tot - kbar_out
    else:
        raise ValueError(f"Unknown degree_type {degree_type}. Needs to be one of 'total', 'in', or 'out'. ")

    return kbar

def _critical_c_linearized(beta:float, s:int|float, method:str='numeric', initial_bracket:Tuple[float, float]=(1e-15, 1.0), rng:np.random.Generator=np.random.default_rng(), n_graphs:int=1000, n:Optional[int]=None):
    """Compute the critical value of the connection scale c in the discretized SRGG model using
       the linearized percolation condition E[T] * k_out = 1.0, where E[T] is the expected size
       of the component of a random vertex in G(s, p_in) and k_out is the average number of 
       out-group connections.

    Input
    beta : inverse temperature in the SRGG model.
    s : the number of nodes in each group of the SBM.
    method : method for computing E[T], either 'numeric' or 'recursive'.
    initial_bracket : initial bracket for finding the root of E[T] * k_out - 1.0 = 0.0.
    rng : a numpy random generator only used when method is 'numeric'.
    n_graphs : number of graphs to sample when method is 'numeric'.
    n : if provided k_out is computed at a finite number of nodes n, defaults to None.

    Output
    c_c : the critical value of the connection scale c in the SRGG model.
    """

    # sanity check
    assert 1. < beta < 2.0, "beta must be between 1 and 2."
    assert s > 2., "s must be larger than 2."
    assert initial_bracket[0] < initial_bracket[1], "initial_bracket must be increasing."

    # determine the method for computing E[T]
    if method == 'numeric':
        expected_T_func = lambda s, p_in: expected_random_vertex_component_size_gnp(s, p_in, method='numeric', rng=rng, n_graphs=n_graphs)
    elif method == 'recursive':
        expected_T_func = lambda s, p_in: expected_random_vertex_component_size_gnp(s, p_in, method='recursive')
    elif method == 'asymptotic':
        expected_T_func = lambda s, p_in: expected_random_vertex_component_size_gnp(s, p_in, method='asymptotic')
    else:
        raise ValueError(f"Unknown method: {method}. Must be 'numeric', ''asymptotic', or 'recursive'.")

    # define the objective function for root finding E[T] * k_out = 1.0.
    def _objective_percolation(c):

        # determine E[T]
        p_in = connection_probability_discretized_srgg(c=c, beta=beta, s=s, m=0, average='analytical')
        expected_T = expected_T_func(s, p_in)

        # determine k_out
        k_out = average_degree_discretized_srgg(c, beta, s, degree_type='out', n=n)

        return expected_T * k_out - 1.0

    # find the root of the objective function
    solution = optimize.root_scalar(_objective_percolation, method='brentq', bracket=initial_bracket)

    if not solution.converged:
        c_c = np.nan
    else:
        c_c = solution.root
    
    return c_c

def _critical_c_nonlinearized(beta:float, s:int|float, initial_bracket:Tuple[float, float]=(1e-15, 1.0), rng:np.random.Generator=np.random.default_rng(), eps:float=1e-6, n_graphs:int=1000, n:Optional[int]=None):
    """Compute the critical value of the connection scale c in the discretized SRGG model using
       the percolation condition 1 - eps = E[exp(k_out * eps T)], where E is the expectation over
       the size of the component of a random vertex in G(s, p_in) and k_out is the average number of 
       out-group connections.

    Input
    beta : inverse temperature in the SRGG model.
    s : the number of nodes in each group of the SBM.
    initial_bracket : initial bracket for finding the root that determines the critical point.
    rng : a numpy random generator only used when method is 'numeric'.
    n_graphs : number of graphs to sample when method is 'numeric'.
    n : optional number of nodes n used to compute k_out at finite n, defaults to None.

    Output
    c_c : the critical value of the connection scale c in the SRGG model.
    """

    # sanity check
    assert 1.0 < beta < 2.0, "beta must be between 1 and 2."
    assert s > 2, "s must be larger than 2."
    assert initial_bracket[0] < initial_bracket[1], "initial_bracket must be increasing."
    assert eps > 0.0, "eps must be positive."

    # define the array of possible component sizes
    T = np.arange(0, s+1)

    # define the objective function for percolation
    def _objective_percolation(c):

        # compute the distribution of the size of a component a random vertex belongs to in G(s, p_in)
        p_in = connection_probability_discretized_srgg(c, beta, s, m=0, average='analytical')
        pT = _random_vertex_component_size_distribution_gnp(s, p_in, rng, n_graphs=n_graphs)

        # compute the average number of connections to other groups
        k_out = average_degree_discretized_srgg(c, beta, s, degree_type='out', n=n)

        return (1.0 - eps) - np.sum(np.exp(-k_out * eps * T) * pT)

    # find the root of the implicit equation for the critical point
    solution = optimize.root_scalar(_objective_percolation, bracket=initial_bracket, method='brentq')

    if solution.converged:
        c_c = solution.root
    else:
        c_c = np.nan
    
    return c_c

def critical_c(beta:float, s:int|float, method:str='numeric', linearized:bool=True, initial_bracket:Tuple[float, float]=(1e-15, 1.0), eps:float=1e-6, rng:np.random.Generator=np.random.default_rng(), n_graphs:int=1000, n:Optional[int]=None):
    """Compute the critical value of the connection scale c in the discretized SRGG model using
       the percolation condition 1 - eps = E[exp(k_out * eps T)] or its linearization E[T]*k_out = 1,
       where E is the expectation over the size of the component of a random vertex in G(s, p_in) and
       k_out is the average number of out-group connections.

    Input
    beta : inverse temperature in the SRGG model.
    s : the number of nodes in each group of the SBM.
    method : the method used to determine the expectation over the size of the component of a random vertex, 'numeric', 'asymptotic', or 'recursive'.
    linearized : whether to use the linearized or the full percolation condition.
    initial_bracket : initial bracket for finding the root of E[T] * k_out - 1.0 = 0.0.
    eps : free parameter in the nonlinear percolation condition, only used when linearized=False.
    rng : a numpy random generator only used when method is 'numeric'.
    n_graphs : number of graphs to sample when method is 'numeric'.
    n : number of nodes to compute k_out at finite n, defaults to None and uses the thermodynamic limit expression.

    Output
    c_c : the critical value of the connection scale c in the SRGG model.
    """

    # sanity checks
    assert 1.0 < beta < 2.0, "beta must be between 1 and 2."
    assert s > 2, "s must be larger than 2."
    assert initial_bracket[0] < initial_bracket[1], "initial_bracket must be increasing."
    assert eps > 0.0, "eps must be positive."
    assert method in ["numeric", "asymptotic", "recursive"], "method must be one of 'numeric', 'asymptotic', or 'recursive'."

    # find the critical point.
    if linearized:
        c_c = _critical_c_linearized(beta, s, method=method, initial_bracket=initial_bracket, rng=rng, n_graphs=n_graphs, n=n)
    else:
        if method == 'numeric':
            c_c = _critical_c_nonlinearized(beta, s, initial_bracket=initial_bracket, rng=rng, eps=eps, n_graphs=n_graphs, n=n)
        else:
            raise ValueError("Only method='numeric' is supported if linearized=False.")

    return c_c