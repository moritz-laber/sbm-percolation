# sbm-percolation

Code for the paper **Percolation in the Stochastic Block Model** by L. Pieleanu, M. Laber, N.G. Sabhahit, and D. Krioukov.

<!-- If you use this software, please cite:
```LaTeX
@misc{pieleanu2026_sbmpercolation,
  title = {Percolation in the Stochastic Block Model},
  author = {Pieleanu, Luca and Laber, Moritz and Sabhahit, Narayan G. and Krioukov, Dmitri},
  year = {2026},
  month = sep,
  doi = {doi/????},
  eprint = {???},
  arxivprefix = {???}
}
``` -->

## Installation

You can install this package directly from GitHub using the following command:

```bash
python -m pip install "git+https://github.com/moritz-laber/sbm-percolation"
```

If you want to try the tutorial (`tutorial/tutorial.ipynb`), its best to clone the repositoty and install additional dependencies using:

```bash
git clone https://github.com/moritz-laber/sbm-percolation
cd sbm-percolation
pip install -e ".[tutorial]"
```

## Usage

Here, we outline a minimal percolation experiment using this package. For worked examples 
refere to the tutorial (`tutorial/tutorial.ipynb`).

```python

import numpy as np
from sbm_percolation import plantet_partition_connection_probability, generate_sbm, percolate

# parameters
n = 10_000      # number of nodes
B = 2           # number of communities
p_in = 0.0025   # within community connection probability
p_out = 0.0015  # between community connection probability   
seed = 42       # random number generator seed

# initialize random number generator
rng = np.random.default_rng(seed=seed)

# compute the connection probability matrix
P = connection_probability_matrix_planted_partition_sbm(
        p_in * np.ones(B),
        p_out,
        B
    )

# assign nodes to communities (needs to be non-decreasing)
b = np.array([b for b in range(B) for _ in range(n // B)])

# sample a graph from the stochastic block model
edges = generate_sbm(b, P, rng)

# determine the size of all connected components
component_sizes = percolate(edges, n)

# print the size of the largest connected component
nlcc = np.max(component_sizes) 
print(f"xi = {nlcc}/{n}")

```