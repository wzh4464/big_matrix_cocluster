"""
DiMergeCo Complete Example

Demonstrates the complete DiMergeCo framework for scalable biclustering.

Implements the algorithm from the paper:
"DiMergeCo: A Scalable Framework for Large-Scale Co-Clustering"

Key Components Demonstrated:
1. SVR (Singular Value Ratio) scoring - Definition from paper
2. Probabilistic matrix partitioning - Algorithm 2 with theoretical guarantees
3. Hierarchical merging - O(log n) binary tree aggregation
4. Complete end-to-end pipeline
"""

import numpy as np
import logging
from pathlib import Path

# Import DiMergeCo components
from src import (
    create_dimergeco_pipeline,
    create_synthetic_data_with_generator,
    PartitionConfig,
    HierarchicalMergeConfig,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def example_basic_dimergeco():
    """
    Basic DiMergeCo example with default parameters.

    Demonstrates the simplest way to use the complete DiMergeCo framework.
    """
    logger.info("=" * 60)
    logger.info("Example 1: Basic DiMergeCo with Default Parameters")
    logger.info("=" * 60)

    # Step 1: Generate synthetic data with embedded biclusters
    logger.info("\n[Step 1] Generating synthetic data...")
    matrix, _, ground_truth, _ = create_synthetic_data_with_generator(
        n_biclusters=5,
        matrix_shape=(1000, 800),
        bicluster_size_range=(50, 80),
        noise_level_spec=0.1,
        random_state=42,
    )
    logger.info(f"Matrix shape: {matrix.shape}")
    logger.info(f"Ground truth biclusters: {len(ground_truth)}")

    # Step 2: Create DiMergeCo pipeline
    logger.info("\n[Step 2] Creating DiMergeCo pipeline...")
    pipeline = create_dimergeco_pipeline(
        k1=8,
        k2=8,  # SVD clustering parameters
        tolerance=0.03,  # Quality threshold
        T_m=40,
        T_n=40,  # Partition thresholds (Algorithm 2)
        T_p=5,  # Partition iterations
        P_thresh=0.95,  # Detection probability guarantee (Theorem 2)
        overlap_threshold=0.3,  # Hierarchical merge threshold
        output_directory="examples/dimergeco_basic_results",
        random_state=42,
    )

    # Step 3: Run analysis
    logger.info("\n[Step 3] Running DiMergeCo analysis...")
    pipeline.load_matrix(matrix)
    pipeline.ground_truth_biclusters = ground_truth  # For comparison
    pipeline.fit()

    # Step 4: Get and display results
    logger.info("\n[Step 4] Results:")
    results = pipeline.get_results()
    print(results.summary())

    # Display quality metrics
    if "precision" in results.quality_metrics:
        print(f"\nComparison with ground truth:")
        print(f"  Precision: {results.quality_metrics['precision']:.3f}")
        print(f"  Recall: {results.quality_metrics['recall']:.3f}")
        print(f"  Average overlap: {results.quality_metrics['average_overlap']:.3f}")

    logger.info(f"\n✓ Results saved to: {pipeline.output_dir}")


def example_custom_configuration():
    """
    Advanced DiMergeCo example with custom configuration.

    Demonstrates fine-tuning of all three components:
    - SVR scoring parameters
    - Partition strategy parameters
    - Hierarchical merging parameters
    """
    logger.info("\n" + "=" * 60)
    logger.info("Example 2: Custom DiMergeCo Configuration")
    logger.info("=" * 60)

    # Generate data
    logger.info("\n[Step 1] Generating challenging synthetic data...")
    matrix, _, ground_truth, _ = create_synthetic_data_with_generator(
        n_biclusters=8,
        matrix_shape=(2000, 1500),
        bicluster_size_range=(80, 120),
        noise_level_spec=0.15,  # More noise = harder problem
        random_state=123,
    )
    logger.info(f"Matrix shape: {matrix.shape}")
    logger.info(f"Ground truth biclusters: {len(ground_truth)}")

    # Custom partition configuration
    logger.info("\n[Step 2] Configuring custom partition strategy...")
    partition_cfg = PartitionConfig(
        T_m=60,  # Larger minimum blocks for large matrix
        T_n=50,
        T_p=10,  # More iterations for better coverage
        T_p_max=20,
        P_thresh=0.98,  # Very high detection probability
        uniform_blocks=True,  # Use uniform block sizes
        random_state=123,
    )
    logger.info(f"  Partition config: T_m={partition_cfg.T_m}, T_n={partition_cfg.T_n}, T_p={partition_cfg.T_p}")

    # Custom hierarchical merge configuration
    logger.info("\n[Step 3] Configuring custom hierarchical merging...")
    merge_cfg = HierarchicalMergeConfig(
        overlap_threshold=0.25,  # More permissive overlap
        use_spatial_indexing=True,  # Enable O(1) spatial index
        spatial_grid_size=15,  # Higher resolution grid
        base_tolerance=0.02,  # Strict quality requirement
        level_penalty=0.05,  # Gradual strictness increase
        track_merge_history=True,  # Track provenance
    )
    logger.info(f"  Merge config: overlap_threshold={merge_cfg.overlap_threshold}")

    # Create pipeline with custom configuration
    from src import BiclusteringPipeline, PipelineConfig, ScoringMethod

    pipeline_cfg = PipelineConfig(
        k1=10,
        k2=10,
        tolerance=0.02,
        enable_dimergeco=True,
        use_svr_scoring=True,
        svr_normalized=True,
        partition_config=partition_cfg,
        hierarchical_merge_config=merge_cfg,
        output_directory="examples/dimergeco_custom_results",
        random_state=123,
    )

    pipeline = BiclusteringPipeline(pipeline_cfg)

    # Run analysis
    logger.info("\n[Step 4] Running analysis with custom configuration...")
    pipeline.load_matrix(matrix)
    pipeline.ground_truth_biclusters = ground_truth
    pipeline.fit()

    # Display results
    logger.info("\n[Step 5] Results:")
    results = pipeline.get_results()
    print(results.summary())

    # Show performance breakdown
    if results.performance_metrics:
        print(f"\nPerformance metrics:")
        print(f"  Execution time: {results.execution_time:.2f}s")
        if "memory_usage_delta_mb" in results.performance_metrics:
            mem_delta = results.performance_metrics["memory_usage_delta_mb"]
            if mem_delta:
                print(f"  Memory delta: {mem_delta:.1f} MB")

    logger.info(f"\n✓ Results saved to: {pipeline.output_dir}")


def example_comparative_analysis():
    """
    Compare DiMergeCo with standard biclustering.

    Demonstrates the advantages of DiMergeCo on the same data.
    """
    logger.info("\n" + "=" * 60)
    logger.info("Example 3: DiMergeCo vs Standard Biclustering")
    logger.info("=" * 60)

    # Generate data
    logger.info("\n[Step 1] Generating test data...")
    matrix, _, ground_truth, _ = create_synthetic_data_with_generator(
        n_biclusters=6,
        matrix_shape=(1500, 1200),
        bicluster_size_range=(60, 90),
        noise_level_spec=0.12,
        random_state=456,
    )
    logger.info(f"Matrix shape: {matrix.shape}")

    # Run standard biclustering
    logger.info("\n[Step 2] Running STANDARD biclustering...")
    from src import create_pipeline

    standard_pipeline = create_pipeline(
        k1=8,
        k2=8,
        tolerance=0.03,
        scoring_method="exponential",
        output_directory="examples/standard_results",
        random_state=456,
    )
    standard_pipeline.load_matrix(matrix)
    standard_pipeline.ground_truth_biclusters = ground_truth

    import time

    start = time.time()
    standard_pipeline.fit()
    standard_time = time.time() - start

    standard_results = standard_pipeline.get_results()
    logger.info(f"  Standard: {len(standard_results.biclusters)} biclusters in {standard_time:.2f}s")

    # Run DiMergeCo
    logger.info("\n[Step 3] Running DIMERGECO biclustering...")
    dimergeco_pipeline = create_dimergeco_pipeline(
        k1=8,
        k2=8,
        tolerance=0.03,
        T_m=50,
        T_n=50,
        T_p=5,
        P_thresh=0.95,
        output_directory="examples/dimergeco_comparison_results",
        random_state=456,
    )
    dimergeco_pipeline.load_matrix(matrix)
    dimergeco_pipeline.ground_truth_biclusters = ground_truth

    start = time.time()
    dimergeco_pipeline.fit()
    dimergeco_time = time.time() - start

    dimergeco_results = dimergeco_pipeline.get_results()
    logger.info(f"  DiMergeCo: {len(dimergeco_results.biclusters)} biclusters in {dimergeco_time:.2f}s")

    # Compare results
    logger.info("\n[Step 4] Comparison:")
    print("\n" + "=" * 50)
    print("COMPARISON: Standard vs DiMergeCo")
    print("=" * 50)

    print(f"\nBiclusters detected:")
    print(f"  Standard:  {len(standard_results.biclusters)}")
    print(f"  DiMergeCo: {len(dimergeco_results.biclusters)}")

    print(f"\nExecution time:")
    print(f"  Standard:  {standard_time:.2f}s")
    print(f"  DiMergeCo: {dimergeco_time:.2f}s")
    print(f"  Speedup:   {standard_time/dimergeco_time:.2f}x")

    if (
        "precision" in standard_results.quality_metrics
        and "precision" in dimergeco_results.quality_metrics
    ):
        print(f"\nPrecision:")
        print(f"  Standard:  {standard_results.quality_metrics['precision']:.3f}")
        print(f"  DiMergeCo: {dimergeco_results.quality_metrics['precision']:.3f}")

        print(f"\nRecall:")
        print(f"  Standard:  {standard_results.quality_metrics['recall']:.3f}")
        print(f"  DiMergeCo: {dimergeco_results.quality_metrics['recall']:.3f}")


if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  DiMergeCo: Scalable Co-Clustering Framework  ".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")
    print("\n")

    # Run examples
    try:
        example_basic_dimergeco()
    except Exception as e:
        logger.error(f"Example 1 failed: {e}", exc_info=True)

    try:
        example_custom_configuration()
    except Exception as e:
        logger.error(f"Example 2 failed: {e}", exc_info=True)

    try:
        example_comparative_analysis()
    except Exception as e:
        logger.error(f"Example 3 failed: {e}", exc_info=True)

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
    print("\nCheck the 'examples/' directory for detailed results and visualizations.")
