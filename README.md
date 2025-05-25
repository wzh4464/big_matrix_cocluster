# Big Matrix CoCluster SVD

A comprehensive Python implementation of biclustering algorithms using Singular Value Decomposition (SVD) for analyzing large-scale co-clustered data matrices.

## Overview

This repository provides a complete framework for biclustering analysis, designed to identify and evaluate biclusters (coherent submatrices) within large data matrices. The implementation combines SVD-based clustering with statistical validation and visualization tools.

## Key Features

### Core Biclustering Engine

- **SVD-based Clustering**: Uses K-means clustering on SVD-transformed matrices (U and V components)
- **Bicluster Quality Assessment**: Multiple scoring functions including compatibility scoring and rank estimation
- **Automated Bicluster Detection**: Identifies biclusters based on singular value ratio thresholds
- **Intersection Detection**: Algorithms to detect and merge overlapping biclusters

### Data Generation & Testing

- **Synthetic Data Generation**: Create test matrices with embedded bicluster structures
- **Configurable Bicluster Properties**: Control size, number, and distribution of biclusters
- **Ground Truth Validation**: Compare detected biclusters against known structures

### Statistical Analysis

- **Theoretical Analysis**: Tail probability estimation for bicluster detection
- **Experimental Validation**: Monte Carlo simulations for parameter optimization
- **Performance Metrics**: Comprehensive evaluation of detection accuracy

### Visualization & I/O

- **Matrix Visualization**: Heat map plotting of original and clustered matrices
- **Result Export**: Save bicluster lists, matrices, and visualizations
- **Progress Tracking**: Built-in progress bars for long-running analyses

### Parallel Processing

- **Multiprocessing Support**: Parallel computation of score matrices
- **Batch Processing**: Handle multiple parameter configurations simultaneously
- **HPC Integration**: SLURM job submission scripts included

## Installation

### Prerequisites

This project requires Python 3.11+ and uses conda for environment management.

### Setup Environment

1. Clone the repository:

```bash
git clone <repository-url>
cd big_matrix_cocluster
```

2. Create and activate the conda environment:

```bash
conda env create -f cocluster.yaml
conda activate cocluster
```

## Usage

### Basic Biclustering Analysis

```python
import numpy as np
from coclusterSVD import coclusterer
from expSetting import generate

# Generate synthetic data with biclusters
M, N = 1000, 1000  # Matrix dimensions
K = 5              # Number of biclusters
bicluster_size = 100

B, permx, permy, A = generate(
    num_bicluster=K, 
    M=M, N=N, 
    m=[bicluster_size]*K, 
    n=[bicluster_size]*K, 
    seed=42
)

# Initialize coclustering algorithm
clusterer = coclusterer(matrix=B, M=M, N=N, debug=True)

# Perform biclustering
clusterer.cocluster(tor=0.02, k1=10, k2=10)

# Print results
clusterer.printBiclusterList(save=True, path="results/biclusters.txt")

# Visualize results
clusterer.imageShowBicluster(save=True, filename="results/visualization.png")
```

### Advanced Scoring and Analysis

```python
from coclusterSVD import score, estimateRank

# Create sample submatrix indices
row_indices = np.random.choice([True, False], size=M, p=[0.1, 0.9])
col_indices = np.random.choice([True, False], size=N, p=[0.1, 0.9])

# Calculate compatibility score
compatibility_score = score(B, row_indices, col_indices)
print(f"Compatibility score: {compatibility_score}")

# Estimate rank of submatrix
r1, r2 = estimateRank(B, row_indices, col_indices, tor1=0.95, tor2=0.99)
print(f"Estimated ranks: {r1}, {r2}")
```

### Parameter Optimization

```python
from coclusterSVD import Tp, find_bicluster_count
import multiprocessing

# Theoretical analysis of search parameters
ranges = range(50, 200, 10)
Tp_values = Tp(ranges=ranges, phi=100, Tm=4, M=1000)

# Experimental validation
pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
results = pool.starmap(find_bicluster_count, [
    (test_matrix, tp_val, 4, 4, 100, 100, 1000) 
    for tp_val in Tp_values
])
```

### Interactive Analysis

Use the provided Jupyter notebooks for interactive exploration:

```bash
jupyter notebook main_ipy.py  # Main analysis notebook
```

## Algorithm Details

### SVD-based Biclustering

The core algorithm follows these steps:

1. **SVD Decomposition**: Compute `X = U Σ V^T`
2. **Feature Transformation**:
   - Row features: `U @ diag(Σ)`
   - Column features: `(diag(Σ) @ V^T)^T`
3. **K-means Clustering**: Cluster rows and columns independently
4. **Bicluster Validation**: Check each cluster pair using singular value ratios
5. **Quality Scoring**: Apply compatibility scoring to validate biclusters

### Compatibility Scoring

The scoring function evaluates bicluster coherence:

```python
def score(X, subrowI, subcolJ):
    # Extract submatrix
    subX = X[np.ix_(subrowI, subcolJ)]
    
    # Compute correlation matrices (or exponential similarity)
    SS1 = correlation_matrix(subX, axis=0)  # Column correlations
    SS2 = correlation_matrix(subX, axis=1)  # Row correlations
    
    # Calculate compatibility scores
    s1 = scoreHelper(lenJ, SS1)
    s2 = scoreHelper(lenI, SS2)
    
    return min(np.concatenate([s1, s2]))
```

### Theoretical Framework

The implementation includes theoretical analysis based on:

- **Hypergeometric Distribution**: Model probability of bicluster detection
- **Tail Probability Bounds**: Estimate required search iterations
- **Statistical Validation**: Compare empirical vs. theoretical performance

## File Structure

```
big_matrix_cocluster/
├── __init__.py                 # Package initialization
├── bicluster.py               # Bicluster data structure
├── coclusterSVD.py           # Main biclustering algorithms
├── submatrix.py              # Submatrix utilities
├── expSetting.py             # Synthetic data generation
├── main_ipy.py               # Interactive analysis notebook
├── notebook.py               # Batch processing script
├── test_coclusterSVD.py      # Unit tests and benchmarks
├── m_Tp_p.py                 # Parameter optimization analysis
├── run_hpc.sh                # HPC job submission script
├── cocluster.yaml            # Conda environment specification
└── README.md                 # This file
```

## Performance & Scalability

### Computational Complexity

- **Matrix Size**: Tested on matrices up to 10,000 × 10,000
- **Bicluster Detection**: Handles 10+ biclusters simultaneously
- **Memory Efficiency**: Uses TruncatedSVD for large matrices

### Benchmark Results

Based on Reuters-21578 dataset experiments:

- **5% dataset**: 41 minutes (our method) vs 85 minutes (CoCC)
- **20% dataset**: 80 minutes (our method) vs 1007 minutes (CoCC)
- **100% dataset**: 1750 minutes (266 minutes with Rust-optimized SVD)

### Parallel Processing

- **Multi-core Support**: Automatic CPU detection and utilization
- **HPC Integration**: SLURM job scripts for cluster computing
- **Memory Management**: Efficient handling of large matrices

## Research Applications

This implementation is designed for:

- **Gene Expression Analysis**: Identify co-expressed gene groups
- **Market Basket Analysis**: Find item co-occurrence patterns
- **Social Network Analysis**: Detect community structures
- **Recommendation Systems**: Collaborative filtering applications
- **Text Mining**: Document-term co-clustering

## Testing & Validation

Run the comprehensive test suite:

```bash
python test_coclusterSVD.py  # Performance benchmarks
pytest -v                    # Unit tests (if pytest configured)
```

The test suite includes:

- **Synthetic Data Validation**: Known ground truth verification
- **Performance Benchmarks**: Speed and memory usage tests
- **Edge Case Handling**: Robustness testing
- **Statistical Validation**: Theoretical vs. empirical comparison

## Development

### Key Classes and Functions

#### Core Classes

- `coclusterer`: Main biclustering engine
- `bicluster`: Bicluster data structure with row/column indices
- `submatrix`: Submatrix representation with coordinates

#### Key Functions

- `score()`: Compatibility scoring for bicluster quality
- `estimateRank()`: SVD-based rank estimation
- `Tp()`: Theoretical search parameter calculation
- `generate()`: Synthetic data generation with embedded biclusters

### Dependencies

- **NumPy** (1.25.2): Core numerical computations
- **SciPy** (1.10.1): Statistical functions and sparse matrices
- **Scikit-learn** (1.2.2): K-means clustering and TruncatedSVD
- **Matplotlib** (3.7.1): Visualization and plotting
- **Pandas** (2.0.3): Data manipulation (optional)
- **Joblib**: Parallel processing utilities
- **TQDM**: Progress bar displays

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-algorithm`)
3. Implement your changes with tests
4. Run the test suite and benchmarks
5. Update documentation as needed
6. Submit a pull request

## Citation

If you use this implementation in your research, please cite:

```bibtex
@software{big_matrix_cocluster,
  title={Big Matrix CoCluster SVD: SVD-based Biclustering for Large-Scale Data Analysis},
  author={Wu, Zihan},
  year={2023-2025},
  url={https://github.com/wzh4464/big_matrix_cocluster}
}
```

## License

[Specify your license here]

## Support

For questions, bug reports, or feature requests:

- Open an issue on GitHub
- Contact the maintainers at <wzh4464@gmail.com>

---

**Note**: This implementation is actively maintained and optimized for research applications. Performance improvements and new features are regularly added based on user feedback and research needs.
