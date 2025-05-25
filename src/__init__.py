"""
Modern Biclustering Analysis Package

A comprehensive Python framework for biclustering analysis using SVD-based algorithms.
Provides modern, type-safe interfaces for bicluster detection, evaluation, and visualization.

Key Components:
    - BiclusterAnalyzer: High-level analysis interface
    - Bicluster: Modern data structure for bicluster representation
    - BiclusteringPipeline: Complete end-to-end analysis workflow
    - BiclusterVisualizer: Comprehensive visualization tools
    - SyntheticDataGenerator: Advanced synthetic data creation

Example Usage:
    >>> from big_matrix_cocluster import create_analyzer, create_synthetic_data
    >>> analyzer = create_analyzer(k1=5, k2=5, tolerance=0.05)
    >>> matrix, _, _, _ = create_synthetic_data(n_biclusters=3)
    >>> analyzer.fit(matrix)
    >>> biclusters = analyzer.get_biclusters()
"""

from __future__ import annotations

# Import version information
__version__ = "2.0.0"
__author__ = "Zihan Wu"
__email__ = "wzh4464@gmail.com"

# Core data structures and configuration
from .bicluster import Bicluster
from .core import (
    BiclusterConfig,
    BiclusterAnalyzer,
    ScoringMethod,
    ClusteringMethod,
    create_analyzer,
)

# Detection and scoring algorithms
from .detection import BiclusterDetector, SVDBiclusterDetector
from .scoring import ScoringStrategy, CompatibilityScorer

# Visualization and data generation
from .visualization import (
    BiclusterVisualizer,
    SyntheticDataGenerator,
    BiclusterSpec,
    SyntheticDataConfig,
    create_synthetic_data_with_generator,
)

# High-level pipeline interface
from .pipeline import BiclusteringPipeline, PipelineConfig, create_pipeline


# Convenience factory functions
def create_synthetic_data(
    n_biclusters: int = 3,
    matrix_shape: tuple[int, int] = (200, 150),
    bicluster_size_range: tuple[int, int] = (20, 50),
    random_state: int | None = 42,
) -> tuple:
    """
    Create synthetic data with embedded biclusters using default parameters.

    Args:
        n_biclusters: Number of biclusters to embed
        matrix_shape: Shape of the output matrix (rows, cols)
        bicluster_size_range: Range for bicluster dimensions (min, max)
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (permuted_matrix, original_matrix, ground_truth_biclusters, generator)
    """
    return create_synthetic_data_with_generator(
        n_biclusters=n_biclusters,
        matrix_shape=matrix_shape,
        bicluster_size_range=bicluster_size_range,
        random_state=random_state,
    )


def analyze_matrix(
    matrix,
    k1: int = 10,
    k2: int = 10,
    tolerance: float = 0.05,
    scoring_method: str = "exponential",
    return_pipeline: bool = False,
):
    """
    Perform complete biclustering analysis on a matrix using default settings.

    Args:
        matrix: Input data matrix
        k1: Number of row clusters
        k2: Number of column clusters
        tolerance: Score threshold for bicluster validation
        scoring_method: Scoring algorithm ("exponential", "pearson")
        return_pipeline: Whether to return the pipeline object

    Returns:
        List of detected biclusters, optionally with pipeline object
    """
    pipeline = create_pipeline(
        k1=k1, k2=k2, tolerance=tolerance, scoring_method=scoring_method
    )

    pipeline.fit(matrix)
    biclusters = pipeline.get_biclusters()

    if return_pipeline:
        return biclusters, pipeline
    return biclusters


# Legacy compatibility layer for existing code
class LegacyCompatibilityLayer:
    """Provides backward compatibility with legacy coclusterSVD interface."""

    @staticmethod
    def coclusterer(matrix, M: int, N: int, debug: bool = False):
        """Legacy coclusterer class compatibility."""
        import warnings

        warnings.warn(
            "Legacy coclusterer class is deprecated. Use BiclusterAnalyzer instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        # Create modern analyzer with legacy-compatible interface
        analyzer = create_analyzer(k1=10, k2=10, tolerance=0.02)

        class LegacyWrapper:
            def __init__(self, analyzer_instance, matrix_data):
                self.analyzer = analyzer_instance
                self.matrix = matrix_data
                self.M = M
                self.N = N
                self.debug = debug
                self.biclusterList = []

            def cocluster(self, tor: float, k1: int, k2: int, atomOrNot: bool = False):
                # Update analyzer configuration
                self.analyzer.config.tolerance = tor
                self.analyzer.config.k1 = k1
                self.analyzer.config.k2 = k2

                # Perform analysis
                self.analyzer.fit(self.matrix)
                self.biclusterList = self.analyzer.get_biclusters()
                return self

            def printBiclusterList(
                self, save: bool = True, path: str = "result/biclusterList.txt"
            ):
                """Legacy print function compatibility."""
                if save:
                    from pathlib import Path
                    import json

                    Path(path).parent.mkdir(parents=True, exist_ok=True)
                    results = {
                        "biclusters": [bc.to_dict() for bc in self.biclusterList]
                    }

                    with open(path, "w") as f:
                        json.dump(results, f, indent=2)
                else:
                    for i, bc in enumerate(self.biclusterList):
                        print(f"bicluster {i}")
                        print(f"row members {bc.row_labels.tolist()}")
                        print(f"col members {bc.col_labels.tolist()}")
                        print(f"score {bc.score}")
                        print("------")

                return self.biclusterList

        return LegacyWrapper(analyzer, matrix)

    @staticmethod
    def score(X, subrowI, subcolJ) -> float:
        """Legacy score function compatibility."""
        import warnings

        warnings.warn(
            "Legacy score function is deprecated. Use CompatibilityScorer instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        # Create temporary bicluster for scoring
        scorer = CompatibilityScorer()
        temp_bicluster = Bicluster(row_indices=subrowI, col_indices=subcolJ, score=None)

        return scorer.score(X, temp_bicluster)


# Expose legacy functions for backward compatibility
coclusterer = LegacyCompatibilityLayer.coclusterer
score = LegacyCompatibilityLayer.score

# Define public API
__all__ = [
    # Core classes
    "Bicluster",
    "BiclusterConfig",
    "BiclusterAnalyzer",
    "ScoringMethod",
    "ClusteringMethod",
    # Detection and scoring
    "BiclusterDetector",
    "SVDBiclusterDetector",
    "ScoringStrategy",
    "CompatibilityScorer",
    # Visualization and data generation
    "BiclusterVisualizer",
    "SyntheticDataGenerator",
    "BiclusterSpec",
    "SyntheticDataConfig",
    # Pipeline
    "BiclusteringPipeline",
    "PipelineConfig",
    # Factory functions
    "create_analyzer",
    "create_pipeline",
    "create_synthetic_data",
    "create_synthetic_data_with_generator",
    "analyze_matrix",
    # Legacy compatibility
    "coclusterer",
    "score",
    # Package metadata
    "__version__",
    "__author__",
    "__email__",
]

# Configure package-level logging
import logging


def setup_logging(level: str = "INFO") -> None:
    """Configure package-wide logging settings."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# Initialize logging with INFO level by default
setup_logging()

# Package-level configuration
import warnings


def configure_warnings(action: str = "default") -> None:
    """Configure warning behavior for the package."""
    warnings.filterwarnings(action, category=DeprecationWarning, module=__name__)


# Set up deprecation warnings
configure_warnings("default")
