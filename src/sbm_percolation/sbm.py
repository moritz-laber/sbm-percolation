"""
sbm.py

Utility functions for generating graphs from the
Stochastic Block Model and for working with these
graphs.

Laber, M - 2026/07/18
"""

from .gnp import expected_random_vertex_component_size_gnp, _random_vertex_component_size_distribution_gnp

import itertools
import math
import more_itertools
import numpy as np
from numpy.typing import ArrayLike
from scipy import optimize
from typing import List, Tuple

def connection_probability_matrix_planted_partition_sbm(p_ins:List | np.ndarray, p_out:float, B:int)->np.ndarray:
    """Create the connection probability matrix P for a planted partition model.
    
    Input
    p_in - the probability of edges within groups as an array of shape (B,)
    p_out - the probability of edges between groups as a scalar
    B - the number of groups
    
    Output
    P - the connection probability matrix, shape (B, B)
    """
    
    # check input and typing
    assert np.all([0.<= p_in <= 1. for p_in in p_ins]), "All p_ins must be in [0, 1]"
    assert 0.<= p_out <= 1., "p_out must be in [0, 1]"
    assert B > 0, "B must be positive."
    B = int(B)

    # set the off-diagonal entries p_ij for i != j
    P = np.full((B, B), p_out, dtype=float)

    # set the diagonal entries p_ii
    np.fill_diagonal(P, p_ins)

    return P

def connection_probability_matrix_rootedtree_sbm(a:int, c_out:float, B:int) -> np.ndarray:
    """Creates the connection probability matrix for a rooted tree SBM.
    
    Input
    a - the branching factor
    c_out - the between group connectivity
    B - the number of communities

    Output
    P - the connection probability matrix, shape (B, B)
    """

    # check input and typing
    assert a >= 2, "Branching factor must be at least 2."
    assert 0. <= c_out/a <= 1., "q/a must be in [0, 1]."
    B = int(B)
    c_out = float(c_out)

    # compute the between layer connection probability
    p_outs = c_out / np.power(a, np.arange(1, B + 1))

    # all entries are zero apart from the ones on the first diagonal
    P = np.zeros((B, B), dtype=float)
    np.fill_diagonal(P[1:, :-1], p_outs)
    np.fill_diagonal(P[:-1, 1:], p_outs)

    return P

def connection_probability_matrix_group_to_node(P_B:np.ndarray, b:np.ndarray) -> np.ndarray:
    """Expand a B x B group-level connection probability matrix into a n x n node-level connection probability matrix given the group assignments b.
    
    Input
    P_B - the group-level connection probability matrix, shape (B, B)
    b - the group assignments of the nodes, shape (n,)

    Output
    P_n - the node-level connection probability matrix, shape (n, n)
    """

    P_n = P_B[b[:, None], b[None, :]]

    return P_n

def _advance_block_pair(idx:int, ij:int, sorted_ms_cum:np.ndarray) -> int:
    """Advance the block pair (i,j) given the current index into the list of possible edges idx.
    
    Inputx``
    idx - the current index into the sorted list of possible edges
    ij - the current index into the sorted list of group pairs
    sorted_ms_cum - the cumulative sum of the sorted number of possible edges between each block pair

    Output
    new_ij - the updated index into the sorted list of block pairs
    """

    # advance to the next group pair
    new_ij = ij
    while new_ij < sorted_ms_cum.shape[0] - 2 and idx >= sorted_ms_cum[new_ij + 1]:
        new_ij += 1

    return new_ij

def _pairs_up_to_row(i:int, n:int)->int:
    """Calculate the number of rows in the triangular matrix up to row a.
    
    Input
    ii - the row index
    n - the size of the triangular matrix
    
    Output
    S(i) - the number of columns up to row ii
    """

    #  derivation: sum_{k=0}^{i-1} (n - k - 1) = i*(2*n - i - 1) / 2
    
    return i * (2 * n - i  - 1) // 2

def _nth_comb_to_local(nb_i:int, local_idx:int) -> Tuple[int, int]:
    """Map the local index local_idx in the combinations of nb_i elements to the corresponding\
        pair (ii, jj).

    Input
    nb_i - the number of elements to choose from
    local_idx - the local index in the combinations of nb_i elements

    Output
    (ii, jj) - the corresponding pair of indices
    """

    # ensure valid input
    nb_i = int(nb_i)
    assert 0 <= local_idx < nb_i * (nb_i - 1) // 2, "local_idx out of bounds."

    
    # get the row index based on inverting the triangular number mapping:
    # local_idx = i * (2*nb_i - i - 1)/2 
    ii = ((2 * nb_i - 1) - math.isqrt((2 * nb_i - 1)**2 - 8 * local_idx)) // 2

    # safety check to account for rounding
    while ii > 0 and local_idx < _pairs_up_to_row(ii, nb_i):
        ii -= 1
    while ii + 1 < nb_i and local_idx >= _pairs_up_to_row(ii + 1, nb_i):
        ii += 1

    # get the column index
    jj = ii + 1 + local_idx - _pairs_up_to_row(ii, nb_i)

    return ii, jj

def _get_edge(ij:int, idx:int, sorted_group_pairs:List, sorted_ms_cum:np.ndarray, nb:np.ndarray, nb_cum:np.ndarray) -> Tuple[int, int]:
    """Get an edge at global index idx in block pair ij.
    
    Input
    ij - the index into the sorted list of group pairs
    idx - the global index into the list of possible edges
    sorted_group_pairs - the sorted list of group pairs
    sorted_ms_cum - the cumulative sum of the sorted number of possible edges between each block pair
    nb - the number of nodes in each group
    nb_cum - the cumulative sum of the number of nodes in each group

    Output
    (u, v) - the corresponding edge as a tuple of node indices
    """

    # get the current group pair
    i,j = sorted_group_pairs[ij]

    # get the local edge index within the group pair
    local_idx = idx - sorted_ms_cum[ij]

    # get the offsets to the node indices
    if i == j:
        # you can think of the block (i,i) as a triangular
        # matrix of size (nb[i], nb[i]) without the diagonal,
        # so nth_comb_to_local inverts this mapping.
        ii, jj = _nth_comb_to_local(nb[i], local_idx)
    else:
        # you can think of the block (i,j) as a matrix
        # of size (nb[i], nb[j]) so to get the element 
        # (ii, jj)corresponding to the local_idx by
        # modular division:
        # - row index: ii = floor(local_idx / nb[j])
        # - col index: jj = local_idx mod nb[j]
        ii, jj = divmod(int(local_idx), int(nb[j]))

    
    # calcuate
    u = int(nb_cum[i] + ii)
    v = int(nb_cum[j] + jj)
    
    return (int(u), int(v))

def _generate_sbm_mh(b:ArrayLike, P:ArrayLike, rng:np.random.Generator) -> List[Tuple[int, int]]:
    """Generate a graph from the Stochastic Block Model with group assignments b and connection\
        proability matrix P using a generalization of Batagelj and Brandes (2005) fast ER algorithm.

    Input
    b - the assignment of nodes to blocks, shape (n,)
    P - the matrix of edge probabilities, shape (B, B)
    rng - a numpy random number generator
    
    Output
    edges - the edge list of the generated graph as a list of tuples (i,j), shape (m,)
    """

    # check input
    b = np.asarray(b)
    P = np.asarray(P)

    # get the number of nodes and groups
    n = b.shape[0]
    B = P.shape[0]

    # get the number of nodes in each group, and their cumulative sums
    nb = np.bincount(b.astype(int), minlength=B).astype(int)
    nb_cum = np.cumsum([0, *nb])

    # get an index into group pairs and the number of edges between each block pair
    group_pairs = list(itertools.combinations_with_replacement(range(B), 2))
    ms = np.asarray([nb[i]*(nb[i] - 1)//2 if i==j else nb[i]*nb[j] for (i,j) in group_pairs], dtype=int)
    ps = np.asarray([P[i,j] for (i,j) in group_pairs], dtype=float)

    # sort the pairs by decreasing probability
    sort_idx = np.argsort(ps)[::-1]
    sorted_ps = ps[sort_idx]
    sorted_ms = ms[sort_idx]
    sorted_ms_cum = np.cumsum([0, *sorted_ms])
    sorted_group_pairs = [group_pairs[i] for i in sort_idx]
    m_max = sorted_ms_cum[-1]

    # sample edges
    edges = []
    idx = -1
    ij = 0
    while idx < m_max:

        # determine the current block pair
        p = sorted_ps[ij]
        if p == 0:
            break

        # sample the geometric distribution with success p and take a step of length k
        k = rng.geometric(p)
        idx += k

        # break if all possible edges have been considered
        if idx >= m_max:
            break

        # determine the block pair after the step
        ij = _advance_block_pair(idx, ij, sorted_ms_cum)
        q = sorted_ps[ij]

        # rejection sampling to account for p being too high
        r = rng.random()
        if r < q/p:
            edges.append(_get_edge(ij, idx, sorted_group_pairs, sorted_ms_cum, nb, nb_cum))
    
    return edges

def _generate_sbm_bb(b:ArrayLike, P:ArrayLike, rng:np.random.Generator) -> List[Tuple[int, int]]:
    """Generate a graph from the Stochastic Block Model with group assignments b and connection\
        proability matrix P using a generalization of Batagelj and Brandes (2005) fast ER algorithm.

    Input
    b - the assignment of nodes to blocks, shape (n,)
    P - the matrix of edge probabilities, shape (B, B)
    rng - a numpy random number generator
    
    Output
    edges - the edge list of the generated graph as a list of tuples (i,j), shape (m,)
    """
    
    # check input
    b = np.asarray(b)
    P = np.asarray(P)
    
    # get the number of groups
    B = P.shape[0]

    # get the number of nodes in each group, and their cumulative sums
    nb = np.bincount(b.astype(int), minlength=B).astype(int)
    nb_cum = np.cumsum([0, *nb])

    # generate fast ER graphs for each block pair
    edges = []
    for i in range(B):
        for j in range(i, B):

            # get the nodes in each block
            p = P[i, j]

            # if p = 0, skip this block pair
            if p == 0:
                continue

            if i == j:
                m_max = nb[i]*(nb[i]-1)//2
            else:
                m_max = nb[i]*nb[j]

            idx = -1
            while idx < m_max:

                # sample the geometric distribution with success p
                k = rng.geometric(p)

                # make a step
                idx += k
                if idx < m_max:
                    if i == j:
                        edges.append(more_itertools.nth_combination(range(nb_cum[i], nb_cum[i+1]), 2, idx))
                    else:
                        edges.append(more_itertools.nth_product(idx, range(nb_cum[i], nb_cum[i+1]), range(nb_cum[j], nb_cum[j+1])))
                else:
                    break
                
    return edges

def generate_sbm(b:ArrayLike, P:ArrayLike, rng:np.random.Generator, algorithm:str='bb') -> List[Tuple[int, int]]:
    """Generate a graph from the Stochastic Block Model with group assignments b and connection\
        proability matrix P.

    Input
    b - the assignment of nodes to blocks, shape (n,). Nodes need to sorted s.t. b is non-decreasing.
    P - the matrix of edge probabilities, shape (B, B)
    rng - a numpy random number generator
    algorithm - the algorithm to use 'bb' for Batagelj and Brandes (default), 'mh' for Miller-Hagberg
    
    Output
    edges - the edge list of the generated graph as a list of tuples (i,j), shape (m,)
    """

    if algorithm == 'bb':
        edges = _generate_sbm_bb(b, P, rng)
    elif algorithm == 'mh':
        edges = _generate_sbm_mh(b, P, rng)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. Supported algorithms are 'bb' and 'mh'.")

    return edges

def average_degree_sbm(nb:ArrayLike, P:ArrayLike, degree_type:str='total')->float:
    """Compute the expected degree, in-group degree, or out-group degree of a random node under the Stochastic Block Model. 
    
    Input
    nb : the number of nodes in each group, shape (B,)
    P : the connection probability matrix, shape (B, B)
    degree_type : outputs the average degree ('total'), in-group ('in'), or out-group ('out') average degree.
    
    Output
    kbar : the average degree
    """

    # ensure correct types
    nb = np.asarray(nb)
    P = np.asarray(P)

    # get the number of groups
    B = P.shape[0]

    # get the total number of nodes
    n = np.sum(nb)

    # count the number of edges
    m_in = 0.0
    m_out = 0.0
    for i in range(B):
        for j in range(i, B):
            if i == j:
                m_in += (nb[i] * (nb[j] - 1) / 2.0) * P[i, j]
            else:
                m_out += (nb[i] * nb[j]) * P[i, j]
    
    # compute the avarge degree
    if degree_type == 'total':
        kbar = 2.0 * (m_in + m_out) / n
    elif degree_type == 'out':
        kbar = 2.0 * m_out / n
    elif degree_type == 'in':
        kbar = 2.0 * m_in / n
    else:
        raise ValueError(f"Unknown degree_type {degree_type}. Must be one of 'total', 'in', or 'out'.")

    return kbar

## PERCOLATION: The case : p_in = const. p_out = const., s=B*n
def build_matrix_M(nb:ArrayLike, P:ArrayLike)->np.ndarray:
    """Build the matrix M governing percolation in SBM.
    
    Input
    nb - the number of nodes in each group, shape (B,)
    P - the connection probability matrix, shape (B, B)
    
    Output
    M - the matrix whose largest eigenvalue governs percolation, shape (B,B)
    """

    # check types
    nb = np.asarray(nb)
    P = np.asarray(P)

    # derive the matrix M whose eigenvalues govern percolation
    M = P * nb                       # M_ij = P_ij * n_j (of diagonal)
    diagM = np.diag(P) * (nb - 1)    # M_ii = P_ii * (n_i - 1) (diagonal)
    np.fill_diagonal(M, diagM)

    return M


## PERCOLATION: The case : p_in = const., p_out = q/n, s=const.
def _critical_q_linearized(p_in:float, s:int|float, method:str='numeric', initial_bracket:Tuple[float, float]=(1e-15, 2.0), rng:np.random.Generator=np.random.default_rng(), n_graphs:int=1000):
    """Compute the critical out-group degree q_c in the SBM with growing number of groups using
       the linearized percolation condition E[T] * q = 1.0, where E[T] is the expected size
       of the component of a random vertex in G(s, p_in) and q is the average number of 
       out-group connections.

    Input
    p_in : the within group connection probability.
    s : the number of nodes in each group of the SBM.
    method : method for computing E[T], either 'numeric' or 'recursive'.
    initial_bracket : initial bracket for finding the root of E[T] * k_out - 1.0 = 0.0.
    rng : a numpy random generator only used when method is 'numeric'.
    n_graphs : number of graphs to sample when method is 'numeric'.

    Output
    q_c : the critical number of out-group connections.
    """

    # sanity check
    assert 0. < p_in <= 1.0, "p_in must be in [0., 1.]"
    assert s > 2., "s must be larger than 2."
    assert initial_bracket[0] < initial_bracket[1], "initial_bracket must be increasing."

    # determine the expected size of the component of a random vertex in G(s, p_in).
    if method == 'numeric':
        expected_T = expected_random_vertex_component_size_gnp(s, p_in, method='numeric', rng=rng, n_graphs=n_graphs)
    elif method == 'recursive':
        expected_T = expected_random_vertex_component_size_gnp(s, p_in, method='recursive')
    elif method == 'asymptotic':
        expected_T = expected_random_vertex_component_size_gnp(s, p_in, method='asymptotic')
    else:
        raise ValueError(f"Unknown method: {method}. Must be 'numeric' or 'recursive'.")

    # define the objective function for root finding E[T] * k_out = 1.0 .
    def _objective_percolation(q):

        return expected_T * q - 1.0

    # find the root of the objective function
    solution = optimize.root_scalar(_objective_percolation, method='brentq', bracket=initial_bracket)

    if not solution.converged:
        q_c = np.nan
    else:
        q_c = solution.root
    
    return q_c

def _critical_q_nonlinearized(p_in, s, initial_bracket:Tuple[float, float]=(1e-15, 2.0), rng=np.random.default_rng(), eps=1e-6, n_graphs=1000):
    """Compute the critical out-group degree q_c in the SBM with growing number of groups using
       the percolation condition 1 - eps = E[exp(q * eps T)], where E is the expectation over
       the size of the component of a random vertex in G(s, p_in) and q is the average number of 
       out-group connections.

    Input
    p_in : the within group connection probability.
    s : the number of nodes in each group of the SBM.
    initial_bracket : initial bracket for finding the root that determines the critical point.
    rng : a numpy random generator only used when method is 'numeric'.
    n_graphs : number of graphs to sample when method is 'numeric'.

    Output
    q_c : the critical number of out-group connections.
    """

    # sanity check
    assert 0.0 < p_in <= 1.0, "beta must be between 1 and 2."
    assert s > 2, "s must be larger than 2."
    assert initial_bracket[0] < initial_bracket[1], "initial_bracket must be increasing."
    assert eps > 0.0, "eps must be positive."

    # define the array of possible component sizes
    T = np.arange(0, s+1)

    # compute the distribution of the size of a component a random vertex belongs to in G(s, p_in)
    pT = _random_vertex_component_size_distribution_gnp(s, p_in, rng, n_graphs=n_graphs)

    # define the objective function for percolation
    def _objective_percolation(q):        

        return (1.0 - eps) - np.sum(np.exp(-q * eps * T) * pT)

    # find the root of the implicit equation for the critical point
    solution = optimize.root_scalar(_objective_percolation, bracket=initial_bracket, method='brentq')

    if solution.converged:
        c_c = solution.root
    else:
        c_c = np.nan
    
    return c_c

def critical_q(p_in:float, s:int|float, method:str='numeric', linearized:bool=True, initial_bracket:Tuple[float, float]=(1e-15, 1.0), eps:float=1e-6, rng:np.random.Generator=np.random.default_rng(), n_graphs:int=1000):
    """Compute the critical value of the out-group connectivity q in the SBM with p_in = const., 
       p_out = q/n, and growing number of groups B=n/s using either the percolation condition 
       1 - eps = E[exp(q * eps T)] or its linearization E[T]*q = 1, where E is the expectation
       over the size of the component of a random vertex in G(s, p_in) and q is the average number
       of out-group connections.

    Input
    p_in : the within group connection probability
    s : the number of nodes in each group of the SBM.
    method : the method used to determine the expectation over the size of the component of a random vertex, 'numeric', 'analytical', or 'recursive'.
    linearized : whether to use the linearized or the full percolation condition.
    initial_bracket : initial bracket for finding the root of E[T] * k_out - 1.0 = 0.0.
    eps : free parameter in the nonlinear percolation condition, only used when linearized=False.
    rng : a numpy random generator only used when method is 'numeric'.
    n_graphs : number of graphs to sample when method is 'numeric'.

    Output
    q_c : the critical number of out-group connections.
    """

    # sanity checks
    assert 0.0 < p_in <= 1.0, "p_in must be in [0.0, 1.0]."
    assert s > 2, "s must be larger than 2."
    assert initial_bracket[0] < initial_bracket[1], "initial_bracket must be increasing."
    assert eps > 0.0, "eps must be positive."
    assert method in ["numeric", "analytical", "recursive"], "method must be one of 'numeric', 'analytical', or 'recursive'."

    # find the critical point.
    if linearized:
        q_c = _critical_q_linearized(p_in, s, method=method, initial_bracket=initial_bracket, rng=rng, n_graphs=n_graphs)
    else:
        if method == 'numeric':
            q_c = _critical_q_nonlinearized(p_in, s, initial_bracket=initial_bracket, rng=rng, eps=eps, n_graphs=n_graphs)
        else:
            raise ValueError("Only method='numeric' is supported if linearized=False.")

    return q_c