"""
utils.py

Miscellaneous utility functions for SBM percolation.

Laber, M - 2026/07/18
"""

import numpy as np
from numpy.typing import ArrayLike
from typing import Tuple, List
from scipy import special, sparse

def _log_choose(n:float, k: np.ndarray)->float | np.ndarray:
    """Compute the logarithm of the binomial coefficient (n choose k).
    
    Input
    n : the total number of items
    k : the number of items to choose

    Output
    log_choose : the logarithm of the binomial coefficient (n choose k)    
    """

    return special.gammaln(n + 1.0) - special.gammaln(k + 1.0) - special.gammaln(n - k + 1.0)

def edgelist_to_adjacency(n:int, edges:np.ndarray | List[Tuple[int, int]])->sparse.csr_matrix:
    """Convert an edgelist to an adjacency matrix in sparse format.
    
    Input
    n  : the number of nodes
    edges : the edgelist, shape (m, 2) or list of tuples where m is the number of edges

    Output
    A : sparse adjacency matrix in csr format, shape (n, n)
    """

    # check types
    n = int(n)
    edges = np.asarray(edges)

    # create the adjacency matrix in sparse format
    A = sparse.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(n, n))
    A = A + A.T  # make symmetric

    return A.tocsr()

def clustering_coefficient(A:sparse.spmatrix)->float:
    """Compute the clustering coefficient of a graph given its adjacency matrix A.
    
    Input
    A : the adjacency matrix in sparse format, shape (n, n)

    Output
    c : the average local clustering coefficient of the graph
    """

    # compute node degrees
    k = np.asarray(A.sum(axis=1)).flatten()
    # compute the number of triangles at each node (undirected)
    A3 = A @ A @ A
    t = A3.diagonal() / 2.0

    # select only nodes with at least 2 neighbors, since clustering is undefined otherwise
    valid = k > 1
    k = k[valid]
    t = t[valid]

    # compute the local clustering coefficient at each node
    cbar = np.mean(t / (k * (k - 1) / 2.0))

    return cbar

def transitivity(A:sparse.spmatrix)->float:
    """Compute the transitivity of a graph given its adjacency matrix A.
        
    Input
    A : the adjacency matrix in sparse format, shape (n, n)

    Output
    c : the transitivity of the graph
    """

    # compute the total number of triangles in the graph
    A3 = A @ A @ A
    triangles = A3.diagonal().sum() / 6.0  # factor of 1/2 for direction and 1/3 for each node.

    # compute the total number of connected triples in the graph
    k = np.asarray(A.sum(axis=1)).flatten()
    triples = np.sum(k * (k - 1) / 2.0)

    # compute the transitivity
    c = 3 * triangles / triples

    return c

def power_iteration(A:ArrayLike, max_iters:int=1000, tol:float=1e-6)->Tuple[float, np.ndarray, bool]:
    """Compute the largest eigenvalue and corresponding eigenvector of a matrix A via power iteration.
    
    Input
    A - the input array shape (n, n)
    max_iters - maximum number of iterations
    tol - tolerance for convergence

    Output
    lam_max - the largest eigenvalue
    x - the corresponding eigenvector
    converged - boolean indicating whether the method converged
    """

    # check assumptions
    assert tol > 0, "Tolerance must be positive."
    assert max_iters > 0, "Maximum number of iterations must be positive."

    # ensure correct types
    max_iters = int(max_iters)
    A = np.asarray(A)
    
    # initialization
    n = A.shape[0]
    x = np.random.rand(n)
    x = x / np.linalg.norm(x)

    # iteration
    converged = False
    for i in range(max_iters):

        x_new = A @ x
        x_new = x_new / np.linalg.norm(x_new)

        # check convergence
        if np.linalg.norm(x_new - x) < tol:
            converged = True
            x = x_new
            break
        else:
            x = x_new

    # compute the eigenvalue
    lam_max = x.T @ A @ x / (x.T @ x)
    
    return lam_max, x, converged

def balanced_tree_nodes(a:int, L:int) -> int:
    """Compute the number of nodes in a balanced a-ary tree of depth L.
    
    Input
    a - the branching factor of the tree
    L - the depth of the tree

    Output
    n - the number of nodes in the tree
    """

    a = int(a)
    L = int(L)

    return int(sum([a ** l for l in range(L + 1)]))

def lam_theo(pars:Tuple, sbm_type:str)->float:
    """Compute the largest eigenvalue of the connectivity matrix for different types of SBMs.
    
    Input
    pars - tuple of parameters for the relevant SBM type:
                - if sbm_type = const then (p_ins, p_out, n)
                - if sbm_type = sqrt then (p_ins, p_out, n)
                - if sbm_type = tree then (a, c_out, B)
    sbm_type - type of SBM, one of 'const', 'sqrt', 'tree'.

    Output
    lam_max - the largest eigenvalue of the connectivity matrix
    """

    if sbm_type == 'const':

        p_ins, p_out, n = pars
        nb = np.array([n//2, n//2])

        if (p_ins.shape[0] != 2):
            raise NotImplementedError("For sbm_type 'const', lam_theo is only implemented for two equally sized groups.") 

        t1 = np.sum((nb - 1.0) * p_ins) / 2.0
        t2 = np.diff((nb - 1.0) * p_ins)[0] / 2.0
        t3 = nb[0] * p_out
        lam_max = t1 + np.sqrt(t2**2 + t3**2)
    
    elif sbm_type == 'sqrt':

        p_ins, p_out, n = pars

        if np.all(p_ins[0] != p_ins):
            raise NotImplementedError("All p_ins need to be equal.") 

        c_in = p_ins[0] * np.sqrt(n)
        c_out = p_out * n

        lam_max = c_in + c_out + ((c_in + c_out) / np.sqrt(n))
    
    elif sbm_type == 'tree':

        a, c_out, B = pars

        lam_max = 2.0 * c_out * np.cos(np.pi / (B + 1)) / np.sqrt(a)

    else:
        raise ValueError(f"Unknown sbm_type: {sbm_type}")


    return lam_max

def clustering_coefficient_theoretical(P:np.ndarray, nb:np.ndarray, num_samples:int=int(1e4), rng:np.random.Generator=None, verbose:bool=True, exchangeable_groups:bool=False)->float:
    """Compute the theoretical clustering coefficient of a graph generated from a stochastic block model (SBM).
    
    Input
    P : the connection probability matrix, shape (B, B)
    nb : the number of nodes in each group, shape (B,)
    num_samples : the number of samples per root community used to estimate the clustering coefficient
    rng : a numpy random generator object. If None, a new generator will be created.
    verbose : whether to print progress information
    exchangeable_groups : whether the groups are exchangeable, then only one root community will be used.
    
    Output
    cbar : the theoretical clustering coefficient of the SBM
    """

    if rng is None:
        rng = np.random.default_rng()

    B = P.shape[0]

    if exchangeable_groups:
        if not np.all(nb == nb[0]):
            raise ValueError("For exchangeable groups, all group sizes must be equal.")
        B_root = 1
    else:
        B_root = B

    cs = np.zeros(B_root)
    for r in range(B_root):

        if verbose: print(f"clustering contribution: {r + 1}/{B_root}", end='\r')

        # sample neighbors for root nodes in a given group r 
        x = rng.binomial(n=nb, p=P[r, :], size=(num_samples, B))
        xtot = x.sum(axis=1)
        
        # diagonal contribution, neighbors in the same group
        diag = np.sum(np.diag(P) * special.comb(x, 2), axis=1)
        
        # of diagonal contribution, neighbors in different groups
        P_up = np.triu(P, 1)
        P_up_x = np.einsum('st, bt->bs', P_up, x)
        x_P_up_x = np.einsum('bs, bs->b', x, P_up_x)
        
        # all triagnles
        t = diag + x_P_up_x

        # normalize
        gtr1 = xtot > 1
        cs[r] = np.mean(t[gtr1] / special.comb(xtot, 2)[gtr1])

    if exchangeable_groups:
        cs = np.repeat(cs, B)

    cbar = np.sum(nb * cs) / np.sum(nb)
    if verbose: print("\n")
    
    return cbar