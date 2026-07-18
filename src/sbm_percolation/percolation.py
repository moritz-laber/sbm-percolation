"""
percolation.py

Utility functions for percolation analysis.
In particular, an implementation of the 
union find datastructure of the Newman-Ziff
algorithm.

Laber, M - 2026/07/18
"""

import numpy as np
from typing import Tuple, List

### NEWMAN-ZIFF DATA STRUCTURE ###
def _findroot(i:int, ptr:np.ndarray) -> int:
    """Find the root of a tree starting from an arbitrary node.
    
    Input
    i - the starting node
    ptr - the pointer array representing the tree structure

    Output
    root - the root of the tree
    """
    
    if ptr[i] < 0:
        return i
    else:
        return _findroot(ptr[i], ptr)

def percolate(edges:List[Tuple[int,int]], n:int) -> np.ndarray[int]:
    """Adds edges and outputs the final component sizes using
       the tree-datastructure of the Newman-Ziff algorithm.

       Input
       edges - edgelist of the graph
       n - number of nodes in the graph

       Output
       sizes - list of component sizes.
    """

    # check types
    n = int(n)

    # pointer array that keeps track of the cluster structur:
    # - positive entries indicate the parents of a node
    # - negatie entries indicate a root node and the absolute 
    #   value of the entry equals the cluster size.
    ptr = -1 * np.ones(n, dtype=np.int32)
    
    # keeps track of the size of the largest cluster as edges
    # are added.
    max_size = 1

    # add edges one by one
    for (i,j) in edges:

        # determine which cluster the nodes belong to
        root_i = _findroot(i, ptr)
        root_j = _findroot(j, ptr)

        # determine cluster sizes
        size_i = np.abs(ptr[root_i])
        size_j = np.abs(ptr[root_j])

        # if the nodes are in different clusters
        # merge the clusters by making the root of 
        # the smaller clusters a child of the root 
        # of the largest cluster.
        if root_i != root_j:
            if size_i >= size_j:
                ptr[root_i] -= size_j    # updating cluster size
                ptr[root_j] = root_i     # updating cluster assignment
                root_max = root_i
            else:
                ptr[root_j] -= size_i    # updating cluster size
                ptr[root_i] = root_j     # updating cluster assignment
                root_max = root_j
        else:
            continue
        
        # update the size of the largest cluster
        max_size = max(max_size, np.abs(ptr[root_max]))

        # end if the graph is fully connected
        if max_size == n:
            break
    
    # compute final cluster sizes
    sizes = np.asarray([np.abs(ptr[root]) for root in set([_findroot(v, ptr) for v in range(n)])])
        
    return sizes