# CoCluster SVD

A Python implementation of biclustering algorithms using Singular Value Decomposition (SVD) for analyzing co-clustered data matrices.

## Overview

This repository contains tools for biclustering analysis, specifically designed to identify and evaluate biclusters (submatrices) within larger data matrices. The implementation includes:

- A compatibility scoring function for evaluating bicluster quality
- Synthetic data generation for testing biclustering algorithms
- Utilities for creating matrices with embedded biclusters

## Features

- **Bicluster Quality Assessment**: Compatibility scoring based on correlation analysis
- **Synthetic Data Generation**: Create test matrices with known bicluster structures
- **Flexible Configuration**: Customizable parameters for bicluster generation
- **Visualization Support**: Built-in plotting capabilities for data exploration

## Installation

### Prerequisites

This project requires Python 3.11+ and uses conda for environment management. Make sure you have [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/) installed.

### Setup Environment

1. Clone the repository:
```bash
git clone <repository-url>
cd cocluster
```

2. Create and activate the conda environment:
```bash
conda env create -f cocluster.yml
conda activate cocluster
```

The environment includes all necessary dependencies:
- NumPy for numerical computations
- Matplotlib for visualization
- SciPy for scientific computing
- Scikit-learn for machine learning utilities
- Jupyter for interactive development

## Usage

### Basic Bicluster Scoring

```python
from coclusterSVD import score
import numpy as np

# Create a sample matrix
X = np.random.rand(100, 50)

# Define row and column indices for a submatrix
subrow_indices = [True] * 20 + [False] * 80  # First 20 rows
subcol_indices = [True] * 10 + [False] * 40  # First 10 columns

# Calculate compatibility score
compatibility_score = score(X, subrow_indices, subcol_indices)
print(f"Bicluster compatibility score: {compatibility_score}")
```

### Generating Synthetic Data

```python
import numpy as np
import matplotlib.pyplot as plt
from main import generate  # From the notebook implementation

# Generate synthetic data with embedded biclusters
seed = 42
num_pool = 200        # Size of random pool for bicluster bases
num_bicluster = 15    # Number of biclusters to embed

# Create synthetic matrix
B = generate(seed=seed, num_pool=num_pool, num_bicluster=num_bicluster)

# Visualize the result
plt.figure(figsize=(10, 8))
plt.imshow(B, cmap='hot', interpolation='nearest')
plt.title('Synthetic Data Matrix with Embedded Biclusters')
plt.colorbar()
plt.show()
```

### Interactive Analysis

The repository includes a Jupyter notebook (`main.ipynb`) for interactive exploration:

```bash
jupyter notebook main.ipynb
```

This notebook demonstrates:
- Data generation with various parameters
- Visualization of bicluster structures
- Parameter sensitivity analysis

## Algorithm Details

### Compatibility Scoring

The core scoring function evaluates bicluster quality using correlation analysis:

1. **Input**: Data matrix X and boolean arrays indicating row/column subsets
2. **Process**: 
   - Extract the specified submatrix X_IJ
   - Compute correlation matrices for both row and column perspectives
   - Calculate deviations from identity matrices
   - Return the minimum compatibility score

3. **Output**: Compatibility score (lower values indicate better biclusters)

The mathematical formulation:
```
score = min(scoreHelper(lenJ, |corr(X_IJ) - I_J|), 
            scoreHelper(lenI, |corr(X_IJ^T) - I_I|))
```

Where `scoreHelper(length, C) = 1 - 1/(length-1) * sum(C, axis=1)`

### Synthetic Data Generation

The data generation process creates matrices with embedded bicluster structures:

1. **Base Generation**: Create random base vectors for each bicluster
2. **Matrix Construction**: Build a structured matrix using these bases
3. **Permutation**: Randomly permute rows and columns to hide structure
4. **Noise Addition**: Add controlled noise to make the problem realistic

## File Structure

```
cocluster/
├── cocluster.yml           # Conda environment specification
├── coclusterSVD.py        # Core scoring algorithms
├── main.py                # Basic data generation utilities
├── main.ipynb             # Interactive analysis notebook
├── test_coclusterSVD.py   # Unit tests
└── README.md              # This file
```

## Testing

Run the included unit tests to verify functionality:

```python
python -m pytest test_coclusterSVD.py -v
```

The test suite includes:
- Validation of scoring function with known inputs
- Edge case handling
- Numerical stability checks

## Development

### Key Functions

- `score(X, subrowI, subcolJ)`: Main compatibility scoring function
- `scoreHelper(length, C)`: Helper function for correlation-based scoring
- `generate(seed, num_bicluster, num_pool)`: Synthetic data generation

### Dependencies

Core computational dependencies:
- **NumPy** (1.24.2): Numerical computing
- **SciPy** (1.10.1): Scientific computing utilities
- **Scikit-learn** (1.2.2): Machine learning tools
- **Matplotlib** (3.7.1): Plotting and visualization
- **Pandas** (2.0.3): Data manipulation

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Make your changes and add tests
4. Ensure all tests pass (`pytest`)
5. Commit your changes (`git commit -am 'Add new feature'`)
6. Push to the branch (`git push origin feature/new-feature`)
7. Create a Pull Request

## Research Context

This implementation is designed for research in biclustering algorithms, particularly:
- Gene expression data analysis
- Market basket analysis
- Collaborative filtering
- Any domain requiring identification of coherent submatrices

For questions, issues, or contributions, please use the GitHub issue tracker or contact the maintainers directly.
