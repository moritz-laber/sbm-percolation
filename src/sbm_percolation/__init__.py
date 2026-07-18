"""
sbm-percolation: Percolation in Stochastic Block Models.

Laber, M - 2026/07/18
"""

from .percolation import percolate
from .gnp import generate_gnp
from .sbm import connection_probability_matrix_planted_partition_sbm, connection_probability_matrix_rootedtree_sbm, connection_probability_matrix_group_to_node, generate_sbm, average_degree_sbm, build_matrix_M, critical_q
from .srgg import generate_srgg, connection_probability_discretized_srgg, connection_probability_matrix_discretized_srgg, average_degree_srgg, average_degree_discretized_srgg, critical_c
from .utils import edgelist_to_adjacency, clustering_coefficient, transitivity, power_iteration, balanced_tree_nodes, lam_theo, clustering_coefficient_theoretical

__all__ = ["percolate",
           "generate_gnp",
           "connection_probability_matrix_planted_partition_sbm",
           "connection_probability_matrix_rootedtree_sbm",
           "connection_probability_matrix_group_to_node",
           "generate_sbm",
           "average_degree_sbm",
           "build_matrix_M",
           "critical_q",
           "generate_srgg",
           "connection_probability_discretized_srgg",
           "connection_probability_matrix_discretized_srgg",
           "average_degree_srgg",
           "average_degree_discretized_srgg",
           "critical_c",
           "edgelist_to_adjacency",
           "clustering_coefficient",
           "transitivity",
           "power_iteration",
           "balanced_tree_nodes",
           "lam_theo",
           "clustering_coefficient_theoretical"]