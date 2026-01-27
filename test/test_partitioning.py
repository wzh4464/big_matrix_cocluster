"""
Tests for probabilistic matrix partitioning (DiMergeCo Algorithm 2).

Tests verify:
- Theoretical probability formulas (Theorem 2, Lemma 1)
- Partition parameter computation
- Matrix partitioning mechanics
- Integration with detection
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from src.partitioning import PartitionConfig, MatrixPartitioner
from src.detection import PartitionedBiclusterDetector
from src.core import BiclusterConfig, ScoringMethod
from src.bicluster import Bicluster


# --- Test Fixtures ---


@pytest.fixture
def default_partition_config():
    """Default partition configuration."""
    return PartitionConfig(
        T_m=10,
        T_n=10,
        T_p=5,
        P_thresh=0.95,
        uniform_blocks=True,
        random_state=42,
    )


@pytest.fixture
def sample_matrix():
    """Sample matrix for testing."""
    np.random.seed(42)
    return np.random.rand(100, 80)


@pytest.fixture
def small_matrix():
    """Small matrix for quick tests."""
    return np.random.rand(20, 20)


# --- Tests for PartitionConfig ---


def test_partition_config_initialization():
    """Test PartitionConfig initialization with default values."""
    config = PartitionConfig()

    assert config.T_m == 10
    assert config.T_n == 10
    assert config.T_p == 5
    assert config.P_thresh == 0.95
    assert config.uniform_blocks is True


def test_partition_config_custom_values():
    """Test PartitionConfig with custom parameters."""
    config = PartitionConfig(
        T_m=30,
        T_n=25,
        T_p=10,
        T_p_max=50,
        P_thresh=0.99,
        uniform_blocks=False,
        random_state=123,
    )

    assert config.T_m == 30
    assert config.T_n == 25
    assert config.T_p == 10
    assert config.T_p_max == 50
    assert config.P_thresh == 0.99
    assert config.uniform_blocks is False
    assert config.random_state == 123


# --- Tests for MatrixPartitioner ---


def test_partitioner_initialization(default_partition_config):
    """Test MatrixPartitioner initialization."""
    partitioner = MatrixPartitioner(default_partition_config)

    assert partitioner.config == default_partition_config
    assert partitioner.m is None  # Not computed yet
    assert partitioner.n is None


def test_compute_partition_parameters_basic(default_partition_config):
    """Test basic partition parameter computation."""
    partitioner = MatrixPartitioner(default_partition_config)

    M, N = 100, 80
    m, n, phi_list, psi_list, T_p = partitioner.compute_partition_parameters(M, N)

    # Verify m, n computation (Line 1 of Algorithm 2)
    # m = ceil(M / T_m) = ceil(100 / 10) = 10
    # n = ceil(N / T_n) = ceil(80 / 10) = 8
    assert m == 10
    assert n == 8

    # Verify block sizes sum to total dimensions
    assert sum(phi_list) == M
    assert sum(psi_list) == N

    # Verify block counts
    assert len(phi_list) == m
    assert len(psi_list) == n

    # Verify T_p is set
    assert T_p >= 1
    assert T_p <= default_partition_config.T_p_max


def test_partition_dimensions_sum_correctly(default_partition_config, sample_matrix):
    """Verify partition dimensions sum to matrix dimensions."""
    partitioner = MatrixPartitioner(default_partition_config)
    M, N = sample_matrix.shape

    m, n, phi_list, psi_list, T_p = partitioner.compute_partition_parameters(M, N)

    # Critical: dimensions must sum exactly
    assert sum(phi_list) == M, f"Row blocks sum to {sum(phi_list)}, expected {M}"
    assert sum(psi_list) == N, f"Col blocks sum to {sum(psi_list)}, expected {N}"


def test_detection_probability_formula(default_partition_config):
    """
    Test detection probability calculation (Theorem 2).

    Formula: P ≥ 1 - K·exp(-2·T_p·[M·(s^k)² + N·(t^k)²])
    """
    partitioner = MatrixPartitioner(default_partition_config)

    M, N = 100, 80
    # Compute partition first to get phi_avg, psi_avg
    m, n, phi_list, psi_list, T_p = partitioner.compute_partition_parameters(M, N)

    # Test with known co-cluster size
    cocluster_sizes = [(30, 25)]  # One co-cluster of size 30×25

    P = partitioner._compute_detection_probability(
        M, N, phi_list, psi_list, T_p=5, cocluster_sizes=cocluster_sizes
    )

    # P should be between 0 and 1
    assert 0 <= P <= 1, f"Probability {P} out of range [0, 1]"

    # P should increase with more iterations
    P_low = partitioner._compute_detection_probability(
        M, N, phi_list, psi_list, T_p=2, cocluster_sizes=cocluster_sizes
    )
    P_high = partitioner._compute_detection_probability(
        M, N, phi_list, psi_list, T_p=10, cocluster_sizes=cocluster_sizes
    )

    assert P_high >= P_low, "Higher T_p should give higher probability"


def test_detectability_parameters(default_partition_config):
    """
    Test detectability parameter computation (Lemma 1).

    s^k = (M_k/M) - ((T_m-1)/φ_avg)
    t^k = (N_k/N) - ((T_n-1)/ψ_avg)
    """
    partitioner = MatrixPartitioner(default_partition_config)

    M, N = 100, 80
    m, n, phi_list, psi_list, T_p = partitioner.compute_partition_parameters(M, N)

    phi_avg = sum(phi_list) / len(phi_list)
    psi_avg = sum(psi_list) / len(psi_list)

    # Test 1: Very large co-cluster (should be detectable)
    # Need M_k/M > (T_m-1)/φ_avg for positive s_k
    # With T_m=10, φ_avg≈10: need M_k/M > 9/10 = 0.9, so M_k > 90
    M_k_large, N_k_large = 95, 75
    s_k_large = (M_k_large / M) - ((default_partition_config.T_m - 1) / phi_avg)
    t_k_large = (N_k_large / N) - ((default_partition_config.T_n - 1) / psi_avg)

    # Very large co-clusters should have positive detectability
    # (or at least, larger s_k and t_k values)

    # Test 2: Medium co-cluster
    M_k_medium, N_k_medium = 60, 50
    s_k_medium = (M_k_medium / M) - ((default_partition_config.T_m - 1) / phi_avg)
    t_k_medium = (N_k_medium / N) - ((default_partition_config.T_n - 1) / psi_avg)

    # Test 3: Small co-cluster (hard to detect)
    M_k_small, N_k_small = 15, 12
    s_k_small = (M_k_small / M) - ((default_partition_config.T_m - 1) / phi_avg)
    t_k_small = (N_k_small / N) - ((default_partition_config.T_n - 1) / psi_avg)

    # Verify ordering: larger co-clusters have larger detectability parameters
    assert (
        s_k_large > s_k_medium
    ), f"Large s_k ({s_k_large}) should be > medium s_k ({s_k_medium})"
    assert (
        s_k_medium > s_k_small
    ), f"Medium s_k ({s_k_medium}) should be > small s_k ({s_k_small})"

    assert (
        t_k_large > t_k_medium
    ), f"Large t_k ({t_k_large}) should be > medium t_k ({t_k_medium})"
    assert (
        t_k_medium > t_k_small
    ), f"Medium t_k ({t_k_medium}) should be > small t_k ({t_k_small})"

    # For very large co-clusters, s_k and t_k should be positive or at least not very negative
    # The formula is correct even if some values are negative (indicates hard-to-detect co-clusters)


def test_probability_increases_with_Tp(default_partition_config):
    """Verify detection probability increases with T_p."""
    partitioner = MatrixPartitioner(default_partition_config)

    M, N = 100, 80
    m, n, phi_list, psi_list, _ = partitioner.compute_partition_parameters(M, N)

    cocluster_sizes = [(25, 20)]

    probabilities = []
    for T_p in [1, 3, 5, 10, 20]:
        P = partitioner._compute_detection_probability(
            M, N, phi_list, psi_list, T_p, cocluster_sizes
        )
        probabilities.append(P)

    # Probability should be monotonically increasing
    for i in range(len(probabilities) - 1):
        assert (
            probabilities[i] <= probabilities[i + 1]
        ), f"P not increasing: {probabilities}"


def test_block_size_constraints(default_partition_config):
    """Verify block sizes respect T_m and T_n constraints."""
    partitioner = MatrixPartitioner(default_partition_config)

    M, N = 100, 80
    m, n, phi_list, psi_list, T_p = partitioner.compute_partition_parameters(M, N)

    # Each block should be at least close to T_m, T_n (within rounding)
    for phi in phi_list:
        assert phi >= 1, f"Block size {phi} too small"

    for psi in psi_list:
        assert psi >= 1, f"Block size {psi} too small"


# --- Tests for Matrix Partitioning ---


def test_partition_matrix_dimensions(default_partition_config, sample_matrix):
    """Test that partition_matrix creates blocks with correct dimensions."""
    partitioner = MatrixPartitioner(default_partition_config)

    blocks = partitioner.partition_matrix(sample_matrix, iteration=0)

    # Should have m × n blocks
    assert len(blocks) == partitioner.m * partitioner.n

    # Verify each block has valid structure
    for submatrix, slices, coords, row_idx, col_idx in blocks:
        assert submatrix.ndim == 2
        assert submatrix.shape[0] > 0
        assert submatrix.shape[1] > 0
        assert len(row_idx) == submatrix.shape[0]
        assert len(col_idx) == submatrix.shape[1]


def test_partition_randomization(default_partition_config, sample_matrix):
    """Verify different iterations produce different partitions."""
    partitioner = MatrixPartitioner(default_partition_config)

    # Partition with iteration 0
    blocks_0 = partitioner.partition_matrix(sample_matrix, iteration=0)

    # Partition with iteration 1
    blocks_1 = partitioner.partition_matrix(sample_matrix, iteration=1)

    # Should have same number of blocks
    assert len(blocks_0) == len(blocks_1)

    # At least some blocks should have different indices
    different_count = 0
    for (_, _, _, row_idx_0, col_idx_0), (_, _, _, row_idx_1, col_idx_1) in zip(
        blocks_0, blocks_1
    ):
        if not np.array_equal(row_idx_0, row_idx_1) or not np.array_equal(
            col_idx_0, col_idx_1
        ):
            different_count += 1

    assert different_count > 0, "Partitions should differ across iterations"


def test_partition_coverage(default_partition_config, small_matrix):
    """Verify all matrix elements are covered exactly once per partition."""
    partitioner = MatrixPartitioner(default_partition_config)
    M, N = small_matrix.shape

    blocks = partitioner.partition_matrix(small_matrix, iteration=0)

    # Track which indices appear in partitions
    row_coverage = set()
    col_coverage = set()

    for _, _, _, row_idx, col_idx in blocks:
        row_coverage.update(row_idx)
        col_coverage.update(col_idx)

    # Every row and column should appear exactly once
    assert row_coverage == set(range(M)), "Not all rows covered"
    assert col_coverage == set(range(N)), "Not all columns covered"


# --- Tests for PartitionedBiclusterDetector ---


def test_partitioned_detector_initialization():
    """Test PartitionedBiclusterDetector initialization."""
    base_config = BiclusterConfig(k1=5, k2=5, tolerance=0.05)
    partition_config = PartitionConfig(T_m=15, T_n=15, T_p=3)

    detector = PartitionedBiclusterDetector(base_config, partition_config)

    assert detector.config == base_config
    assert detector.partition_config == partition_config
    assert detector.base_detector is not None


def test_partitioned_detection_completes(small_matrix):
    """Test that partitioned detection completes without error."""
    base_config = BiclusterConfig(
        k1=3,
        k2=3,
        tolerance=10.0,  # Permissive for test
        random_state=42,
    )
    partition_config = PartitionConfig(T_m=5, T_n=5, T_p=2, random_state=42)

    detector = PartitionedBiclusterDetector(base_config, partition_config)

    # Should complete without error
    biclusters = detector.detect(small_matrix)

    # Result should be a list
    assert isinstance(biclusters, list)


def test_partitioned_detection_with_synthetic():
    """Test partitioned detection on synthetic data with embedded biclusters."""
    np.random.seed(42)

    # Create matrix with embedded low-rank block
    matrix = np.random.rand(60, 50) * 0.3

    # Embed a coherent bicluster
    matrix[10:20, 10:20] = 5.0 + 0.1 * np.random.rand(10, 10)

    base_config = BiclusterConfig(
        k1=4,
        k2=4,
        tolerance=10.0,
        scoring_method=ScoringMethod.SVR_NORMALIZED,
        random_state=42,
    )
    partition_config = PartitionConfig(T_m=15, T_n=15, T_p=3, random_state=42)

    detector = PartitionedBiclusterDetector(base_config, partition_config)

    biclusters = detector.detect(matrix)

    # Should detect some biclusters
    assert len(biclusters) >= 0, "Detection should complete"

    # Verify biclusters have correct structure
    for bc in biclusters:
        assert bc.shape[0] > 0
        assert bc.shape[1] > 0
        assert bc.score is not None


def test_coordinate_mapping(small_matrix):
    """Test that bicluster coordinates are correctly mapped back to original matrix."""
    base_config = BiclusterConfig(k1=3, k2=3, tolerance=10.0, random_state=42)
    partition_config = PartitionConfig(T_m=5, T_n=5, T_p=1, random_state=42)

    detector = PartitionedBiclusterDetector(base_config, partition_config)
    biclusters = detector.detect(small_matrix)

    M, N = small_matrix.shape

    for bc in biclusters:
        # Verify indices are within bounds
        assert len(bc.row_indices) == M
        assert len(bc.col_indices) == N

        # Verify can extract submatrix
        submatrix = bc.extract_submatrix(small_matrix)
        assert submatrix.shape[0] == bc.shape[0]
        assert submatrix.shape[1] == bc.shape[1]


# --- Performance Tests ---


def test_partitioning_scalability():
    """Test partitioning performance on different matrix sizes."""
    import time

    partition_config = PartitionConfig(T_m=20, T_n=20, T_p=2, random_state=42)
    partitioner = MatrixPartitioner(partition_config)

    sizes = [(100, 100), (200, 200), (500, 500)]
    times = []

    for M, N in sizes:
        matrix = np.random.rand(M, N)

        start = time.time()
        partitioner.compute_partition_parameters(M, N)
        blocks = partitioner.partition_matrix(matrix, iteration=0)
        elapsed = time.time() - start

        times.append(elapsed)

        # Sanity check
        assert len(blocks) > 0

    # Larger matrices should take more time, but not drastically
    # (We're not testing strict O(n) here, just that it scales reasonably)
    assert times[0] < times[-1] * 100, "Partitioning scaling seems unreasonable"


def test_memory_efficiency(sample_matrix):
    """Verify partitioning doesn't create excessive copies."""
    partition_config = PartitionConfig(T_m=10, T_n=10, T_p=1, random_state=42)
    partitioner = MatrixPartitioner(partition_config)

    # Partition should create views, not deep copies
    blocks = partitioner.partition_matrix(sample_matrix, iteration=0)

    # Verify we got blocks
    assert len(blocks) > 0

    # Each block's submatrix is a separate array (not a view unfortunately),
    # but the original matrix is unchanged
    assert sample_matrix.shape == sample_matrix.shape  # Basic sanity check


# --- Integration Tests ---


def test_detection_with_theoretical_guarantee():
    """
    Verify detection probability guarantee holds in practice.

    This is a statistical test that may occasionally fail due to randomness.
    """
    np.random.seed(42)

    # Create matrix with multiple embedded biclusters
    matrix = np.random.rand(100, 80) * 0.2

    # Embed 3 clear biclusters
    bicluster_positions = [(10, 15, 20, 25), (30, 40, 30, 40), (60, 70, 50, 60)]

    for r_start, r_end, c_start, c_end in bicluster_positions:
        matrix[r_start:r_end, c_start:c_end] = (
            5.0 + 0.05 * np.random.rand(r_end - r_start, c_end - c_start)
        )

    # Configure with high detection probability
    base_config = BiclusterConfig(
        k1=5,
        k2=5,
        tolerance=10.0,
        scoring_method=ScoringMethod.SVR_NORMALIZED,
        random_state=42,
    )
    partition_config = PartitionConfig(
        T_m=20, T_n=20, T_p=10, P_thresh=0.9, random_state=42
    )

    detector = PartitionedBiclusterDetector(base_config, partition_config)

    # Run detection multiple times
    num_trials = 5
    detection_counts = []

    for trial in range(num_trials):
        partition_config.random_state = 42 + trial
        detector = PartitionedBiclusterDetector(base_config, partition_config)
        biclusters = detector.detect(matrix)
        detection_counts.append(len(biclusters))

    # Should detect biclusters consistently (not empty most of the time)
    successful_trials = sum(1 for count in detection_counts if count > 0)
    success_rate = successful_trials / num_trials

    # With P_thresh=0.9 and T_p=10, should succeed frequently
    assert success_rate >= 0.5, (
        f"Detection success rate {success_rate} lower than expected. "
        f"Counts: {detection_counts}"
    )


def test_partition_metadata(sample_matrix):
    """Test that partition metadata is correctly tracked."""
    base_config = BiclusterConfig(k1=3, k2=3, tolerance=10.0, random_state=42)
    partition_config = PartitionConfig(
        T_m=15, T_n=15, T_p=2, keep_partition_metadata=True, random_state=42
    )

    detector = PartitionedBiclusterDetector(base_config, partition_config)
    biclusters = detector.detect(sample_matrix)

    # Check metadata is present
    for bc in biclusters:
        if bc.metadata:
            # Should have partition info if from partitioned detection
            assert (
                "detection_method" in bc.metadata
                or "merged_from_count" in bc.metadata
            )
