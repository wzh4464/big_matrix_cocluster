# Big Matrix CoCluster SVD

A modern, comprehensive Python framework for biclustering analysis using Singular Value Decomposition (SVD). This package provides type-safe, high-performance tools for identifying and analyzing coherent submatrices (biclusters) within large-scale data matrices.

## Overview

This framework combines advanced SVD-based algorithms with modern Python architecture to deliver a complete solution for biclustering analysis. The implementation emphasizes code quality, maintainability, and ease of use while providing powerful analytical capabilities for researchers and practitioners.

## Key Features

### Modern Architecture

- **Type-Safe Design**: Full type hints and validation throughout the codebase
- **Modular Components**: Clean separation of concerns with pluggable algorithms
- **Configuration Management**: Centralized, validated configuration system
- **Pipeline Automation**: End-to-end workflow management with comprehensive reporting

### Core Analysis Engine

- **SVD-Based Detection**: Advanced singular value decomposition with K-means clustering
- **Multiple Scoring Methods**: Exponential similarity, Pearson correlation, and compatibility scoring
- **Quality Assessment**: Automated bicluster validation with configurable thresholds
- **Overlap Management**: Intelligent detection and merging of overlapping biclusters

### Data Generation and Testing

- **Synthetic Data Creation**: Configurable generation of matrices with embedded biclusters
- **Ground Truth Validation**: Compare detected results against known bicluster structures
- **Performance Benchmarking**: Built-in tools for algorithm evaluation and comparison

### Visualization and Reporting

- **Publication-Quality Plots**: Matrix heatmaps, statistical distributions, and overlay visualizations  
- **Comprehensive Reports**: Automated generation of detailed analysis summaries
- **Interactive Analysis**: Jupyter notebook integration with rich visual outputs

### Performance and Scalability

- **Efficient Processing**: Optimized algorithms with intelligent caching
- **Large Matrix Support**: Handles matrices up to 10,000 × 10,000 elements
- **Parallel Processing**: Multi-core utilization for computational intensive tasks

## Installation

### Prerequisites

This package requires Python 3.12+ and uses modern dependency management.

### Setup Environment

1. Clone the repository:

```bash
git clone <repository-url>
cd big-matrix-cocluster
```

2. Install dependencies:

```bash
pip install -e .
```

## Usage

### Quick Start

The simplest way to perform biclustering analysis:

```python
import numpy as np
from big_matrix_cocluster import analyze_matrix, create_synthetic_data

# Generate test data
matrix, _, ground_truth, _ = create_synthetic_data(
    n_biclusters=3, 
    matrix_shape=(200, 150)
)

# Perform analysis
biclusters = analyze_matrix(
    matrix, 
    k1=5, k2=5, 
    tolerance=0.05,
    scoring_method="exponential"
)

print(f"Found {len(biclusters)} biclusters")
for i, bc in enumerate(biclusters):
    print(f"Bicluster {i+1}: {bc.shape} (score: {bc.score:.4f})")
```

### Complete Pipeline Analysis

For comprehensive analysis with visualization and reporting:

```python
from big_matrix_cocluster import create_pipeline

# Configure and run complete analysis
pipeline = create_pipeline(
    k1=8, k2=8,
    tolerance=0.03,
    scoring_method="exponential",
    output_directory="results"
)

# Generate synthetic data and analyze
pipeline.generate_synthetic_data(
    n_biclusters=4,
    matrix_shape=(300, 250),
    bicluster_size_range=(20, 40)
)

pipeline.fit()

# View results
results = pipeline.get_results()
pipeline.print_summary()

# Results automatically saved to "results/" directory
```

### Advanced Configuration

For fine-tuned analysis with custom parameters:

```python
from big_matrix_cocluster import (
    create_analyzer, BiclusterConfig, 
    ScoringMethod, ClusteringMethod
)

# Create custom configuration
config = BiclusterConfig(
    k1=10, k2=10,
    tolerance=0.02,
    scoring_method=ScoringMethod.PEARSON,
    clustering_method=ClusteringMethod.SPECTRAL,
    min_rows=3, min_cols=3,
    max_overlap=0.3,
    random_state=42
)

# Initialize analyzer
analyzer = create_analyzer(config=config)

# Load your data matrix
your_matrix = np.loadtxt("your_data.csv", delimiter=",")

# Perform analysis
analyzer.fit(your_matrix)
biclusters = analyzer.get_biclusters()

# Apply post-processing
filtered_biclusters = analyzer.filter_biclusters(min_score=0.1, min_size=50)
merged_biclusters = analyzer.merge_overlapping(threshold=0.3)
```

### Visualization and Analysis

Generate comprehensive visualizations and reports:

```python
from big_matrix_cocluster import BiclusterVisualizer

# Create visualizer
visualizer = BiclusterVisualizer(figsize=(12, 8), dpi=300)

# Individual plots
visualizer.plot_matrix_comparison(your_matrix, biclusters)
visualizer.plot_bicluster_statistics(biclusters)
visualizer.plot_individual_biclusters(your_matrix, biclusters)

# Complete report generation
visualizer.create_report_visualizations(
    your_matrix, 
    biclusters, 
    output_dir="visualization_report"
)
```

### Synthetic Data Generation

Create test datasets with known bicluster structures:

```python
from big_matrix_cocluster import create_synthetic_data_with_generator

# Generate complex synthetic data
permuted_matrix, original_matrix, ground_truth_biclusters, generator = (
    create_synthetic_data_with_generator(
        n_biclusters=5,
        matrix_shape=(500, 400),
        bicluster_size_range=(30, 60),
        noise_level_spec=0.2,
        background_noise_config=0.1,
        random_state=42
    )
)

# Access generation metadata
ground_truth_info = generator.get_ground_truth_bicluster_info_dicts()
permutation_indices = generator.get_permutation_indices()
```

## Algorithm Details

### SVD-Based Biclustering Process

The core algorithm follows a sophisticated multi-step process:

1. **Matrix Decomposition**: Compute SVD decomposition X = UΣV^T
2. **Feature Engineering**: Transform row features as U@diag(Σ) and column features as (diag(Σ)@V^T)^T  
3. **Clustering**: Apply K-means or spectral clustering to row and column features independently
4. **Candidate Generation**: Form bicluster candidates from all cluster pair combinations
5. **Validation**: Apply scoring functions and quality thresholds to validate candidates
6. **Post-Processing**: Merge overlapping biclusters and apply final filters

### Scoring Methods

The framework provides multiple scoring approaches:

**Exponential Similarity Scoring**: Computes exponential similarity between rows and columns based on Euclidean distances, providing robust performance across diverse data types.

**Pearson Correlation Scoring**: Evaluates linear relationships between bicluster elements, effective for data with strong correlation patterns.

**Compatibility Scoring**: Combines multiple similarity measures for comprehensive bicluster quality assessment.

### Quality Validation

Each detected bicluster undergoes rigorous validation:

- Minimum size requirements (configurable row and column thresholds)
- Score-based quality filtering with adjustable tolerance levels
- Overlap detection and intelligent merging strategies
- Statistical significance assessment against random baselines

## Architecture

### File Structure

```
src/
├── __init__.py              # Modern package interface with legacy compatibility
├── bicluster.py            # Enhanced bicluster data structure with rich methods
├── core.py                 # Main analyzer and configuration management
├── detection.py            # SVD-based detection algorithms
├── scoring.py              # Multiple scoring strategy implementations  
├── visualization.py        # Comprehensive visualization and data generation
├── pipeline.py             # End-to-end analysis workflow automation
└── legacy/                 # Backward compatibility layer (if preserved)

test/
├── test_bicluster.py       # Comprehensive bicluster class testing
├── test_core.py            # Core analyzer functionality tests
├── test_detection.py       # Detection algorithm validation
├── test_scoring.py         # Scoring method verification
├── test_pipeline.py        # Pipeline integration testing
└── test_visualization.py   # Visualization component testing
```

### Key Classes

**BiclusterAnalyzer**: High-level interface providing complete analysis capabilities with configurable parameters and post-processing options.

**BiclusteringPipeline**: Comprehensive workflow manager supporting synthetic data generation, analysis execution, and automated reporting.

**Bicluster**: Modern data structure with enhanced functionality including overlap detection, Jaccard index calculation, and flexible serialization.

**BiclusterVisualizer**: Publication-ready visualization generator with support for matrix comparisons, statistical plots, and comprehensive reports.

## Performance Characteristics

### Computational Complexity

- **Matrix Processing**: Efficiently handles matrices up to 10,000 × 10,000 elements
- **Memory Management**: Optimized SVD computation with intelligent caching
- **Scalability**: Linear scaling with matrix size for most operations

### Benchmark Results

Performance testing on Reuters-21578 dataset demonstrates significant improvements:

- **Small datasets (5%)**: 2.1x faster than baseline methods
- **Medium datasets (20%)**: 12.6x performance improvement  
- **Large datasets (100%)**: 6.6x faster with standard implementation

### Resource Requirements

- **Memory Usage**: Approximately 8 × matrix_size in bytes for SVD operations
- **CPU Utilization**: Automatic multi-core detection and utilization
- **Storage**: Minimal disk usage with optional intermediate result caching

## Research Applications

This framework supports diverse analytical domains:

**Bioinformatics**: Gene expression analysis for identifying co-expressed gene modules and pathway discovery.

**Market Research**: Customer segmentation and product recommendation through purchase pattern analysis.

**Social Network Analysis**: Community detection and influence pattern identification in large networks.

**Text Mining**: Document clustering and topic modeling through term co-occurrence analysis.

**Financial Analytics**: Portfolio optimization and risk factor identification through correlation analysis.

## Testing and Validation

### Comprehensive Test Suite

The package includes extensive testing coverage:

```bash
# Run all tests
python -m pytest test/ -v

# Run specific test categories  
python -m pytest test/test_core.py -v        # Core functionality
python -m pytest test/test_detection.py -v  # Detection algorithms
python -m pytest test/test_pipeline.py -v   # Pipeline integration
```

### Test Categories

- **Unit Tests**: Validate individual component functionality and edge cases
- **Integration Tests**: Verify component interaction and data flow
- **Performance Tests**: Benchmark computational efficiency and memory usage
- **Synthetic Data Tests**: Validate detection accuracy against known ground truth

## Development and Extension

### Adding Custom Scoring Methods

Implement the `ScoringStrategy` protocol:

```python
from big_matrix_cocluster import ScoringStrategy, Matrix, Bicluster

class CustomScorer(ScoringStrategy):
    def score(self, matrix: Matrix, bicluster: Bicluster) -> float:
        submatrix = bicluster.extract_submatrix(matrix)
        # Implement your scoring logic
        return your_score_calculation(submatrix)
```

### Custom Detection Algorithms

Extend the `BiclusterDetector` base class:

```python
from big_matrix_cocluster import BiclusterDetector, BiclusterConfig

class CustomDetector(BiclusterDetector):
    def detect(self, matrix: Matrix) -> List[Bicluster]:
        # Implement your detection algorithm
        return detected_biclusters
```

## Migration from Legacy Code

For users transitioning from older versions, the package provides compatibility layers:

```python
# Legacy compatibility (deprecated but functional)
from big_matrix_cocluster import coclusterer, score

# Modern equivalent (recommended)
from big_matrix_cocluster import create_analyzer, CompatibilityScorer
```

## Contributing

We welcome contributions to enhance the framework:

1. Fork the repository and create a feature branch
2. Implement changes with comprehensive test coverage
3. Ensure all existing tests continue to pass
4. Update documentation to reflect new functionality
5. Submit a pull request with detailed description of changes

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{big_matrix_cocluster_modern,
  title={Big Matrix CoCluster SVD: Modern SVD-based Biclustering Framework},
  author={Wu, Zihan},
  year={2024-2025},
  version={2.0.0},
  url={https://github.com/wzh4464/big-matrix-cocluster}
}
```

## License

This project is licensed under the MIT License. See the LICENSE file for complete terms.

## Support and Contact

For questions, bug reports, or feature requests:

- **GitHub Issues**: Open an issue for bug reports or feature requests
- **Email**: <wzh4464@gmail.com> for direct inquiries
- **Documentation**: Comprehensive API documentation available in the repository

---

**Note**: This framework represents a complete modernization of biclustering analysis tools, emphasizing code quality, performance, and usability. The architecture supports extensibility while maintaining backward compatibility for existing workflows.
