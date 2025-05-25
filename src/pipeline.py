"""
Complete Biclustering Analysis Pipeline

Provides end-to-end workflow automation for biclustering analysis, from data preparation
through algorithm execution to comprehensive result visualization and reporting.

The pipeline integrates all modern components into a cohesive, easy-to-use interface
that handles the complexity of coordinating multiple analysis steps.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union, Tuple
from pathlib import Path
import numpy as np
from numpy.typing import NDArray
import logging
import time
import json
from contextlib import contextmanager

# Import core components
from .bicluster import Bicluster
from .core import BiclusterAnalyzer, BiclusterConfig, ScoringMethod, ClusteringMethod
from .visualization import (
    BiclusterVisualizer,
    SyntheticDataGenerator,
    BiclusterSpec,
    SyntheticDataConfig,
    create_synthetic_data_with_generator,
)

# Type aliases
Matrix = NDArray[np.floating]


@dataclass
class PipelineConfig:
    """Comprehensive configuration for the biclustering analysis pipeline."""

    # Core analysis parameters
    k1: int = 10
    k2: int = 10
    tolerance: float = 0.05
    scoring_method: ScoringMethod = ScoringMethod.EXPONENTIAL
    clustering_method: ClusteringMethod = ClusteringMethod.KMEANS
    random_state: Optional[int] = 42

    # Analysis quality controls
    min_rows: int = 3
    min_cols: int = 3
    max_overlap: float = 0.3

    # Processing options
    parallel_processing: bool = True
    use_caching: bool = True

    # Output and reporting
    output_directory: str = "biclustering_results"
    save_intermediate_results: bool = True
    generate_visualizations: bool = True
    create_detailed_report: bool = True

    # Performance settings
    memory_limit_gb: Optional[float] = None
    max_execution_time_minutes: Optional[int] = None

    def to_bicluster_config(self) -> BiclusterConfig:
        """Convert to BiclusterConfig for analyzer initialization."""
        return BiclusterConfig(
            k1=self.k1,
            k2=self.k2,
            tolerance=self.tolerance,
            scoring_method=self.scoring_method,
            clustering_method=self.clustering_method,
            random_state=self.random_state,
            parallel=self.parallel_processing,
            min_rows=self.min_rows,
            min_cols=self.min_cols,
            max_overlap=self.max_overlap,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        config_dict = {}
        for field_info in self.__dataclass_fields__.values():
            value = getattr(self, field_info.name)
            if isinstance(value, (ScoringMethod, ClusteringMethod)):
                config_dict[field_info.name] = value.value
            else:
                config_dict[field_info.name] = value
        return config_dict


@dataclass
class PipelineResults:
    """Comprehensive results container for pipeline execution."""

    biclusters: List[Bicluster]
    execution_time: float
    matrix_shape: Tuple[int, int]
    configuration: PipelineConfig
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    quality_metrics: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Generate human-readable summary of results."""
        lines = [
            "BICLUSTERING ANALYSIS RESULTS",
            "=" * 40,
            f"Matrix dimensions: {self.matrix_shape[0]} × {self.matrix_shape[1]}",
            f"Execution time: {self.execution_time:.2f} seconds",
            f"Biclusters detected: {len(self.biclusters)}",
            f"Configuration: k1={self.configuration.k1}, k2={self.configuration.k2}",
            f"Tolerance: {self.configuration.tolerance}",
            f"Scoring method: {self.configuration.scoring_method.value}",
        ]

        if self.biclusters:
            lines.extend(["", "BICLUSTER SUMMARY:", "-" * 20])

            for i, bc in enumerate(self.biclusters[:5]):  # Show first 5
                lines.append(f"Bicluster {i+1}: {bc.shape} (score: {bc.score:.4f})")

            if len(self.biclusters) > 5:
                lines.append(f"... and {len(self.biclusters) - 5} more")

        return "\n".join(lines)


class BiclusteringPipeline:
    """
    Complete end-to-end biclustering analysis pipeline.

    Provides automated workflow management from data input through analysis
    execution to comprehensive result reporting and visualization.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.output_dir = Path(self.config.output_directory)

        # Initialize components
        self.analyzer = BiclusterAnalyzer(self.config.to_bicluster_config())
        self.visualizer = BiclusterVisualizer()
        self.logger = self._configure_logging()

        # State management
        self.input_matrix: Optional[Matrix] = None
        self.results: Optional[PipelineResults] = None
        self.ground_truth_biclusters: Optional[List[Bicluster]] = None
        self._execution_context: Dict[str, Any] = {}

        # Setup output directory
        if self.config.save_intermediate_results or self.config.generate_visualizations:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def _configure_logging(self) -> logging.Logger:
        """Configure pipeline-specific logging."""
        logger = logging.getLogger(f"{self.__class__.__name__}")

        if self.config.save_intermediate_results:
            log_path = self.output_dir / "pipeline.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)  # 确保目录存在
            handler = logging.FileHandler(log_path)
            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
            )
            logger.addHandler(handler)

        return logger

    def load_matrix(self, matrix: Matrix) -> BiclusteringPipeline:
        """Load input matrix for analysis."""
        if matrix.ndim != 2:
            raise ValueError("Input must be a 2D matrix")

        if (
            matrix.shape[0] < self.config.min_rows
            or matrix.shape[1] < self.config.min_cols
        ):
            raise ValueError(
                f"Matrix too small: {matrix.shape}. Minimum: ({self.config.min_rows}, {self.config.min_cols})"
            )

        self.input_matrix = matrix.astype(np.float64)
        self.logger.info(f"Loaded matrix with shape {matrix.shape}")

        return self

    def generate_synthetic_data(
        self,
        n_biclusters: int = 5,
        matrix_shape: Tuple[int, int] = (300, 250),
        bicluster_size_range: Tuple[int, int] = (20, 40),
        noise_level: float = 0.2,
    ) -> BiclusteringPipeline:
        """Generate synthetic test data with embedded biclusters."""

        self.logger.info(
            f"Generating synthetic data: {matrix_shape} with {n_biclusters} biclusters"
        )

        permuted_matrix, original_matrix, gt_biclusters, generator = (
            create_synthetic_data_with_generator(
                n_biclusters=n_biclusters,
                matrix_shape=matrix_shape,
                bicluster_size_range=bicluster_size_range,
                noise_level_spec=noise_level,
                random_state=self.config.random_state,
            )
        )

        self.input_matrix = permuted_matrix
        self.ground_truth_biclusters = gt_biclusters
        self._execution_context["synthetic_data"] = {
            "original_matrix": original_matrix,
            "generator": generator,
            "ground_truth_count": len(gt_biclusters),
        }

        # Save synthetic data if requested
        if self.config.save_intermediate_results:
            np.save(self.output_dir / "synthetic_input_matrix.npy", permuted_matrix)
            np.save(self.output_dir / "synthetic_original_matrix.npy", original_matrix)

            gt_data = [bc.to_dict() for bc in gt_biclusters]
            with open(self.output_dir / "ground_truth_biclusters.json", "w") as f:
                json.dump(gt_data, f, indent=2)

        return self

    @contextmanager
    def _execution_timer(self):
        """Context manager for timing pipeline execution."""
        start_time = time.time()
        start_memory = self._get_memory_usage()

        try:
            yield
        finally:
            end_time = time.time()
            end_memory = self._get_memory_usage()

            execution_time = end_time - start_time
            memory_delta = (
                end_memory - start_memory if end_memory and start_memory else None
            )

            self._execution_context.update(
                {
                    "execution_time": execution_time,
                    "memory_usage_delta_mb": memory_delta,
                    "start_time": start_time,
                    "end_time": end_time,
                }
            )

    def _get_memory_usage(self) -> Optional[float]:
        """Get current memory usage in MB."""
        try:
            import psutil

            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except ImportError:
            return None

    def fit(self, matrix: Optional[Matrix] = None) -> BiclusteringPipeline:
        """Execute the complete biclustering analysis."""

        if matrix is not None:
            self.load_matrix(matrix)

        if self.input_matrix is None:
            raise ValueError(
                "No input matrix available. Use load_matrix() or generate_synthetic_data() first."
            )

        self.logger.info("Starting biclustering analysis pipeline")

        with self._execution_timer():
            # Execute core analysis
            self.analyzer.fit(self.input_matrix)
            detected_biclusters = self.analyzer.get_biclusters()

            # Apply post-processing filters
            if self.config.max_overlap < 1.0:
                self.logger.info("Applying overlap filtering")
                detected_biclusters = self.analyzer.merge_overlapping(
                    self.config.max_overlap
                )

            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(detected_biclusters)

            # Create results object
            execution_time = self._execution_context.get("execution_time", 0.0)
            self.results = PipelineResults(
                biclusters=detected_biclusters,
                execution_time=execution_time,
                matrix_shape=self.input_matrix.shape,
                configuration=self.config,
                performance_metrics=self._execution_context.copy(),
                quality_metrics=quality_metrics,
            )

        self.logger.info(
            f"Analysis completed: {len(detected_biclusters)} biclusters detected in {self.results.execution_time:.2f}s"
        )

        # Generate outputs if requested
        if self.config.save_intermediate_results:
            self._save_results()

        if self.config.generate_visualizations:
            self._create_visualizations()

        if self.config.create_detailed_report:
            self._generate_report()

        return self

    def _calculate_quality_metrics(self, biclusters: List[Bicluster]) -> Dict[str, Any]:
        """Calculate various quality metrics for the detected biclusters."""
        if not biclusters:
            return {"total_biclusters": 0}

        scores = [bc.score for bc in biclusters if bc.score is not None]
        sizes = [bc.size for bc in biclusters]
        shapes = [bc.shape for bc in biclusters]

        metrics = {
            "total_biclusters": len(biclusters),
            "average_score": np.mean(scores) if scores else None,
            "score_std": np.std(scores) if scores else None,
            "average_size": np.mean(sizes),
            "size_std": np.std(sizes),
            "size_range": (min(sizes), max(sizes)),
            "average_rows": np.mean([s[0] for s in shapes]),
            "average_cols": np.mean([s[1] for s in shapes]),
            "total_coverage": sum(sizes)
            / (self.input_matrix.shape[0] * self.input_matrix.shape[1]),
        }

        # Ground truth comparison if available
        if self.ground_truth_biclusters:
            metrics.update(self._compare_with_ground_truth(biclusters))

        return metrics

    def _compare_with_ground_truth(self, detected: List[Bicluster]) -> Dict[str, Any]:
        """Compare detected biclusters with ground truth."""
        if not self.ground_truth_biclusters:
            return {}

        # Simple overlap-based matching
        matches = 0
        best_overlaps = []

        for gt_bc in self.ground_truth_biclusters:
            best_overlap = 0
            for det_bc in detected:
                try:
                    overlap = det_bc.jaccard_index(gt_bc)
                    best_overlap = max(best_overlap, overlap)
                except ValueError:
                    continue  # Skip incompatible biclusters

            best_overlaps.append(best_overlap)
            if best_overlap > 0.3:  # Threshold for considering a match
                matches += 1

        return {
            "ground_truth_count": len(self.ground_truth_biclusters),
            "detected_count": len(detected),
            "matches_found": matches,
            "precision": matches / len(detected) if detected else 0,
            "recall": matches / len(self.ground_truth_biclusters),
            "average_overlap": np.mean(best_overlaps) if best_overlaps else 0,
        }

    def _save_results(self) -> None:
        """Save analysis results to files."""
        if not self.results:
            return

        def _to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: _to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [_to_serializable(v) for v in obj]
            elif hasattr(obj, '__dict__'):
                return str(obj)
            else:
                return obj

        # Save biclusters as JSON
        results_data = {
            "configuration": self.config.to_dict(),
            "biclusters": [bc.to_dict() for bc in self.results.biclusters],
            "performance_metrics": self.results.performance_metrics,
            "quality_metrics": self.results.quality_metrics,
        }

        with open(self.output_dir / "analysis_results.json", "w") as f:
            json.dump(_to_serializable(results_data), f, indent=2)

        # Save input matrix
        np.save(self.output_dir / "input_matrix.npy", self.input_matrix)

        self.logger.info(f"Results saved to {self.output_dir}")

    def _create_visualizations(self) -> None:
        """Generate comprehensive visualizations."""
        if not self.results or not self.input_matrix.size:
            return

        viz_dir = self.output_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)

        try:
            # Matrix comparison plot
            self.visualizer.plot_matrix_comparison(
                self.input_matrix,
                self.results.biclusters,
                save_path=viz_dir / "matrix_with_biclusters.png",
            )

            # Statistics plots
            if self.results.biclusters:
                self.visualizer.plot_bicluster_statistics(
                    self.results.biclusters,
                    save_path=viz_dir / "bicluster_statistics.png",
                )

                self.visualizer.plot_individual_biclusters(
                    self.input_matrix,
                    self.results.biclusters,
                    max_to_plot=6,
                    save_dir=viz_dir,
                )

            self.logger.info(f"Visualizations saved to {viz_dir}")

        except Exception as e:
            self.logger.error(f"Error creating visualizations: {e}")

    def _generate_report(self) -> None:
        """Generate comprehensive analysis report."""
        if not self.results:
            return

        report_lines = [
            self.results.summary(),
            "",
            "DETAILED ANALYSIS:",
            "-" * 30,
            f"Input matrix shape: {self.input_matrix.shape}",
            f"Matrix density: {np.count_nonzero(self.input_matrix) / self.input_matrix.size:.3f}",
            f"Value range: [{np.min(self.input_matrix):.3f}, {np.max(self.input_matrix):.3f}]",
            "",
            "PERFORMANCE METRICS:",
            "-" * 20,
        ]

        for key, value in self.results.performance_metrics.items():
            if isinstance(value, float):
                report_lines.append(f"{key}: {value:.4f}")
            else:
                report_lines.append(f"{key}: {value}")

        report_lines.extend(["", "QUALITY METRICS:", "-" * 16])

        for key, value in self.results.quality_metrics.items():
            if isinstance(value, float):
                report_lines.append(f"{key}: {value:.4f}")
            else:
                report_lines.append(f"{key}: {value}")

        # Individual bicluster details
        if self.results.biclusters:
            report_lines.extend(["", "INDIVIDUAL BICLUSTER DETAILS:", "-" * 32])

            for i, bc in enumerate(self.results.biclusters):
                report_lines.extend(
                    [
                        f"Bicluster {i+1}:",
                        f"  Shape: {bc.shape}",
                        f"  Size: {bc.size} elements",
                        f"  Score: {bc.score:.6f}",
                        f"  Row coverage: {bc.shape[0]/self.input_matrix.shape[0]:.2%}",
                        f"  Col coverage: {bc.shape[1]/self.input_matrix.shape[1]:.2%}",
                        "",
                    ]
                )

        # Save report
        report_text = "\n".join(report_lines)
        with open(self.output_dir / "analysis_report.txt", "w") as f:
            f.write(report_text)

        self.logger.info(
            f"Detailed report saved to {self.output_dir / 'analysis_report.txt'}"
        )

    def get_biclusters(self) -> List[Bicluster]:
        """Get detected biclusters from the analysis."""
        if not self.results:
            raise ValueError("Pipeline has not been fitted yet. Call fit() first.")
        return self.results.biclusters

    def get_results(self) -> PipelineResults:
        """Get complete analysis results."""
        if not self.results:
            raise ValueError("Pipeline has not been fitted yet. Call fit() first.")
        return self.results

    def print_summary(self) -> None:
        """Print analysis summary to console."""
        if not self.results:
            print("No results available. Run fit() first.")
            return

        print(self.results.summary())


# Factory functions for easy pipeline creation
def create_pipeline(
    k1: int = 10,
    k2: int = 10,
    tolerance: float = 0.05,
    scoring_method: str = "exponential",
    clustering_method: str = "kmeans",
    output_directory: str = "biclustering_results",
    random_state: Optional[int] = 42,
    **kwargs,
) -> BiclusteringPipeline:
    """
    Create a configured biclustering pipeline with common parameters.

    Args:
        k1: Number of row clusters
        k2: Number of column clusters
        tolerance: Score threshold for bicluster validation
        scoring_method: Scoring algorithm ("exponential", "pearson", "compatibility")
        clustering_method: Clustering algorithm ("kmeans", "spectral")
        output_directory: Directory for saving results and visualizations
        random_state: Random seed for reproducibility
        **kwargs: Additional configuration parameters

    Returns:
        Configured BiclusteringPipeline instance
    """

    config = PipelineConfig(
        k1=k1,
        k2=k2,
        tolerance=tolerance,
        scoring_method=ScoringMethod(scoring_method),
        clustering_method=ClusteringMethod(clustering_method),
        output_directory=output_directory,
        random_state=random_state,
        **kwargs,
    )

    return BiclusteringPipeline(config)


def run_complete_analysis(
    matrix: Optional[Matrix] = None,
    synthetic_config: Optional[Dict[str, Any]] = None,
    analysis_config: Optional[Dict[str, Any]] = None,
    output_directory: str = "complete_analysis_results",
) -> BiclusteringPipeline:
    """
    Run complete end-to-end biclustering analysis with default settings.

    Args:
        matrix: Input matrix (if None, synthetic data will be generated)
        synthetic_config: Configuration for synthetic data generation
        analysis_config: Configuration for biclustering analysis
        output_directory: Directory for all outputs

    Returns:
        Completed pipeline with results
    """

    # Setup configuration
    pipeline_config = PipelineConfig(
        output_directory=output_directory,
        save_intermediate_results=True,
        generate_visualizations=True,
        create_detailed_report=True,
    )

    if analysis_config:
        for key, value in analysis_config.items():
            if hasattr(pipeline_config, key):
                setattr(pipeline_config, key, value)

    # Create and run pipeline
    pipeline = BiclusteringPipeline(pipeline_config)

    if matrix is not None:
        pipeline.load_matrix(matrix)
    else:
        # Generate synthetic data with provided or default configuration
        syn_config = synthetic_config or {}
        pipeline.generate_synthetic_data(**syn_config)

    pipeline.fit()

    return pipeline


if __name__ == "__main__":
    # Example usage demonstration
    import logging

    logging.basicConfig(level=logging.INFO)

    print("Running complete biclustering analysis pipeline demonstration...")

    # Run analysis with synthetic data
    pipeline = run_complete_analysis(
        synthetic_config={
            "n_biclusters": 4,
            "matrix_shape": (200, 150),
            "bicluster_size_range": (20, 40),
        },
        analysis_config={"k1": 8, "k2": 8, "tolerance": 0.03},
        output_directory="demo_pipeline_results",
    )

    # Print results
    pipeline.print_summary()

    print(f"\nAnalysis complete! Check '{pipeline.output_dir}' for detailed results.")
