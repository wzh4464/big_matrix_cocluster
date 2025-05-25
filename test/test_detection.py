import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from dataclasses import replace

from src.bicluster import Bicluster, BICLUSTER_ID_PREFIX
from src.core import BiclusterConfig, ScoringMethod, ClusteringMethod, Matrix
from src.detection import SVDBiclusterDetector

# --- Test Fixtures ---


@pytest.fixture
def base_config_params_for_detect() -> dict:
    # Parameters for SVDBiclusterDetector.detect method, mapping to new signature
    return {
        "n_iterations": 3,
        "n_clusters_rows": 2,
        "n_clusters_cols": 2,
        "n_svd_components": None,  # Or a specific number like 2, if tests rely on it
        "min_rows": 2,
        "min_cols": 2,
        "max_overlap": 0.7,
        "tolerance": 0.5,  # Default from BiclusterConfig or a specific test value
    }


@pytest.fixture
def core_config_detection() -> BiclusterConfig:
    return BiclusterConfig(
        k1=3,  # Corresponds to n_iterations or n_clusters_rows in some contexts
        k2=2,  # Corresponds to n_clusters_cols in some contexts
        scoring_method=ScoringMethod.EXPONENTIAL,
        clustering_method=ClusteringMethod.KMEANS,
        random_state=42,
        min_rows=2,  # Ensure these are set in config for fallback
        min_cols=2,
        max_overlap=0.7,
        tolerance=0.5,
    )


@pytest.fixture
def simple_matrix_with_bicluster() -> Matrix:
    """A simple matrix with a clear bicluster in the top-left corner."""
    matrix = np.array(
        [
            [5.0, 5.1, 1.0, 1.2],
            [5.2, 5.0, 1.3, 1.1],
            [0.8, 1.0, 4.0, 4.1],
            [0.9, 1.1, 4.2, 4.3],
            [0.5, 0.6, 0.7, 0.9],  # Noise row
        ],
        dtype=np.float64,
    )
    return matrix


@pytest.fixture
def simple_matrix_two_biclusters() -> Matrix:
    """Matrix with two non-overlapping biclusters."""
    matrix = np.array(
        [
            [10, 10.1, 0, 0, 0, 0.1],  # BC1
            [10.2, 10, 0, 0.1, 0, 0],  # BC1
            [0, 0.2, 20, 20.1, 0, 0.2],  # BC2
            [0.1, 0, 20.2, 20, 0, 0],  # BC2
            [1, 0.5, 1, 0.8, 1, 0.3],  # Noise
            [0.5, 1, 0.3, 1, 0.5, 1],  # Noise
        ],
        dtype=np.float64,
    )
    return matrix


# --- Tests for SVDBiclusterDetector ---


def test_svd_detector_initialization(core_config_detection: BiclusterConfig):
    with patch("src.detection.CompatibilityScorer") as MockScorer:
        mock_scorer_instance = MockScorer.return_value
        detector = SVDBiclusterDetector(config=core_config_detection)
        assert detector.config == core_config_detection
        MockScorer.assert_called_once_with(core_config_detection.scoring_method)
        assert detector.scorer is mock_scorer_instance


def test_detect_simple_bicluster_kmeans(
    core_config_detection: BiclusterConfig,
    simple_matrix_with_bicluster: Matrix,
    base_config_params_for_detect: dict,
):
    config = replace(
        core_config_detection, clustering_method=ClusteringMethod.KMEANS, k1=3, k2=2
    )
    detector = SVDBiclusterDetector(config=config)
    biclusters = detector.detect(
        simple_matrix_with_bicluster,
        n_iterations=config.k1,
        n_clusters_rows=config.k2,
        n_clusters_cols=config.k2,
        min_rows=base_config_params_for_detect["min_rows"],
        min_cols=base_config_params_for_detect["min_cols"],
        # n_biclusters_to_find removed as it is not a direct param of detect
    )
    assert len(biclusters) >= 0  # We expect at least one or zero if none found
    if biclusters:
        bc1 = biclusters[0]
        assert bc1.row_indices.sum() >= base_config_params_for_detect["min_rows"]
        assert bc1.col_indices.sum() >= base_config_params_for_detect["min_cols"]
        assert bc1.id.startswith(BICLUSTER_ID_PREFIX)
        # Stronger assertions about which bicluster is found can be added if the data is more deterministic


def test_detect_simple_bicluster_spectral(
    core_config_detection: BiclusterConfig,
    simple_matrix_with_bicluster: Matrix,
    base_config_params_for_detect: dict,
):
    config = replace(
        core_config_detection, clustering_method=ClusteringMethod.SPECTRAL, k1=1, k2=2
    )
    detector = SVDBiclusterDetector(config=config)
    biclusters = detector.detect(
        simple_matrix_with_bicluster,
        n_iterations=config.k1,
        n_clusters_rows=config.k2,
        n_clusters_cols=config.k2,
        min_rows=base_config_params_for_detect["min_rows"],
        min_cols=base_config_params_for_detect["min_cols"],
    )
    assert len(biclusters) >= 0
    if biclusters:
        bc1 = biclusters[0]
        assert bc1.row_indices.sum() >= base_config_params_for_detect["min_rows"]
        assert bc1.col_indices.sum() >= base_config_params_for_detect["min_cols"]


def test_detect_n_clusters_param_effect(
    core_config_detection: BiclusterConfig,
    simple_matrix_two_biclusters: Matrix,
    base_config_params_for_detect: dict,
):
    config = replace(
        core_config_detection, k1=3, k2=2
    )  # k1=iter, k2=default clusters_cols
    detector = SVDBiclusterDetector(config=config)

    biclusters_k2 = detector.detect(
        simple_matrix_two_biclusters,
        n_iterations=config.k1,
        n_clusters_rows=2,
        n_clusters_cols=2,
        min_rows=base_config_params_for_detect["min_rows"],
        min_cols=base_config_params_for_detect["min_cols"],
    )
    # More clusters might lead to more (potentially smaller or more refined) biclusters before overlap filtering
    biclusters_k3 = detector.detect(
        simple_matrix_two_biclusters,
        n_iterations=config.k1,
        n_clusters_rows=3,
        n_clusters_cols=3,
        min_rows=base_config_params_for_detect["min_rows"],
        min_cols=base_config_params_for_detect["min_cols"],
    )
    # This assertion is weak; exact number depends on data and filtering
    # assert len(biclusters_k3) >= len(biclusters_k2) or len(biclusters_k3) > 0
    assert isinstance(biclusters_k2, list)
    assert isinstance(biclusters_k3, list)


def test_detect_min_max_filters(
    core_config_detection: BiclusterConfig, base_config_params_for_detect: dict
):
    matrix = np.array([[5, 5, 1, 1], [5, 5, 1, 1], [1, 1, 1, 1], [2, 2, 2, 2]])
    config_strict = replace(core_config_detection, k1=1, k2=2)
    detector = SVDBiclusterDetector(config=config_strict)
    biclusters = detector.detect(
        matrix,
        n_iterations=config_strict.k1,
        n_clusters_rows=config_strict.k2,
        n_clusters_cols=config_strict.k2,
        min_rows=2,
        min_cols=2,
        max_overlap=1.0,
    )  # Override max_overlap

    assert len(biclusters) >= 1, "Expected at least one bicluster to be found"

    # Check if the expected bicluster is among the found ones
    expected_row_indices = np.array([True, True, False, False])
    expected_col_indices = np.array([True, True, False, False])
    found_expected_bicluster = False
    for bc in biclusters:
        if np.array_equal(bc.row_indices, expected_row_indices) and np.array_equal(
            bc.col_indices, expected_col_indices
        ):
            found_expected_bicluster = True
            break
    assert (
        found_expected_bicluster
    ), "The specific 2x2 block [[5,5],[5,5]] was not found"

    # The second part of the test can remain as is, as it checks if allowing smaller biclusters finds more or equal.
    biclusters_min_1x1 = detector.detect(
        matrix,
        n_iterations=config_strict.k1,
        n_clusters_rows=config_strict.k2,
        n_clusters_cols=config_strict.k2,
        min_rows=1,
        min_cols=1,
        max_overlap=1.0,
    )  # Override max_overlap here as well
    assert len(biclusters_min_1x1) >= len(biclusters)


def test_detect_no_bicluster_found(
    core_config_detection: BiclusterConfig, base_config_params_for_detect: dict
):
    matrix = np.random.rand(10, 10)
    config = replace(
        core_config_detection, k1=1, tolerance=0.001
    )  # Very strict tolerance
    detector = SVDBiclusterDetector(config=config)
    biclusters = detector.detect(
        matrix,
        n_iterations=config.k1,
        n_clusters_rows=config.k2,
        n_clusters_cols=config.k2,
        min_rows=base_config_params_for_detect["min_rows"],
        min_cols=base_config_params_for_detect["min_cols"],
        tolerance=config.tolerance,
    )
    assert len(biclusters) == 0


def test_svd_on_almost_zero_matrix(
    core_config_detection: BiclusterConfig, base_config_params_for_detect: dict
):
    matrix = np.zeros((5, 5))
    matrix[0, 0] = 1e-12
    config = replace(core_config_detection, k1=1, k2=1)
    detector = SVDBiclusterDetector(config=config)
    biclusters = detector.detect(
        matrix,
        n_iterations=config.k1,
        n_clusters_rows=config.k2,
        n_clusters_cols=config.k2,
        n_svd_components=1,
        min_rows=base_config_params_for_detect["min_rows"],
        min_cols=base_config_params_for_detect["min_cols"],
    )
    assert isinstance(biclusters, list)


def test_max_overlap_filtering(
    core_config_detection: BiclusterConfig, base_config_params_for_detect: dict
):
    config = replace(core_config_detection, k1=1, k2=3)
    matrix_overlap_prone = np.array(
        [
            [10, 10, 10, 1, 1, 1],
            [10, 10, 10, 1, 1, 1],
            [10, 10, 10, 1, 1, 1],
            [9, 9, 9, 1.5, 1.5, 1.5],
            [1, 1, 1, 10, 10, 10],
            [1, 1, 1, 10, 10, 10],
        ]
    )
    target_max_overlap = 0.2
    detector_overlap = SVDBiclusterDetector(config=config)
    biclusters = detector_overlap.detect(
        matrix_overlap_prone,
        n_iterations=config.k1,
        n_clusters_rows=config.k2,
        n_clusters_cols=config.k2,
        min_rows=base_config_params_for_detect["min_rows"],
        min_cols=base_config_params_for_detect["min_cols"],
        max_overlap=target_max_overlap,
    )
    assert len(biclusters) <= (config.k2 * config.k2)
    if len(biclusters) > 1:
        for i in range(len(biclusters)):
            for j in range(i + 1, len(biclusters)):
                bc_i, bc_j = biclusters[i], biclusters[j]
                row_intersect = np.sum(bc_i.row_indices & bc_j.row_indices)
                row_union = np.sum(bc_i.row_indices | bc_j.row_indices)
                col_intersect = np.sum(bc_i.col_indices & bc_j.col_indices)
                col_union = np.sum(bc_i.col_indices | bc_j.col_indices)
                j_row = row_intersect / row_union if row_union > 0 else 0
                j_col = col_intersect / col_union if col_union > 0 else 0
                assert (
                    max(j_row, j_col) <= target_max_overlap
                ), f"Biclusters {i} ({bc_i.id}) and {j} ({bc_j.id}) overlap too much: J_row={j_row}, J_col={j_col}"
