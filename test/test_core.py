import pytest
import numpy as np
import json
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open
from numpy.testing import assert_array_equal

# Assuming your project structure allows this import path
# If big_matrix_cocluster is the root package:
from src.core import (
    BiclusterConfig,
    ScoringMethod,
    ClusteringMethod,
    BiclusterAnalyzer,
    create_analyzer,
    Matrix,
)
from src.bicluster import Bicluster  # For creating mock biclusters
from src.detection import SVDBiclusterDetector  # For isinstance check

# --- Tests for BiclusterConfig ---


def test_bicluster_config_defaults():
    config = BiclusterConfig()
    assert config.tolerance == 0.05
    assert config.k1 == 5
    assert config.k2 == 5
    assert config.scoring_method == ScoringMethod.EXPONENTIAL
    assert config.clustering_method == ClusteringMethod.KMEANS
    assert config.parallel is False
    assert config.random_state is None
    assert config.min_rows == 2
    assert config.min_cols == 2
    assert config.max_overlap == 1.0


def test_bicluster_config_custom_values():
    config = BiclusterConfig(
        tolerance=0.05,
        k1=5,
        k2=7,
        scoring_method=ScoringMethod.PEARSON,
        clustering_method=ClusteringMethod.SPECTRAL,
        parallel=False,
        random_state=100,
    )
    assert config.tolerance == 0.05
    assert config.k1 == 5
    assert config.k2 == 7
    assert config.scoring_method == ScoringMethod.PEARSON
    assert config.clustering_method == ClusteringMethod.SPECTRAL
    assert config.parallel is False
    assert config.random_state == 100


def test_bicluster_config_to_dict():
    config = BiclusterConfig(
        scoring_method=ScoringMethod.PEARSON,
        clustering_method=ClusteringMethod.SPECTRAL,
    )
    config_dict = config.to_dict()
    expected_dict = {
        "k1": 5,
        "k2": 5,
        "tolerance": 0.05,
        "scoring_method": "pearson",
        "clustering_method": "spectral",
        "random_state": None,
        "parallel": False,
        "min_rows": 2,
        "min_cols": 2,
        "max_overlap": 1.0,
    }
    assert config_dict == expected_dict


def test_bicluster_config_from_dict():
    config_data = {
        "tolerance": 0.01,
        "k1": 8,
        "k2": 8,
        "scoring_method": "compatibility",
        "clustering_method": "kmeans",
        "parallel": False,
        "random_state": None,
    }
    config = BiclusterConfig.from_dict(config_data)
    assert config.tolerance == 0.01
    assert config.k1 == 8
    assert config.k2 == 8
    assert config.scoring_method == ScoringMethod.COMPATIBILITY
    assert config.clustering_method == ClusteringMethod.KMEANS
    assert config.parallel is False
    assert config.random_state is None

    # Test with Enum objects already (should still work)
    config_data_enum = {"scoring_method": ScoringMethod.EXPONENTIAL}
    config_from_enum = BiclusterConfig.from_dict(config_data_enum)
    assert config_from_enum.scoring_method == ScoringMethod.EXPONENTIAL


# --- Tests for BiclusterAnalyzer ---


@pytest.fixture
def mock_bicluster_list() -> list[Bicluster]:
    # Create some mock Bicluster objects
    bc1 = Bicluster(
        np.array([True, False]), np.array([True, False, True]), 0.1, {"id": 1}
    )
    bc2 = Bicluster(
        np.array([False, True]), np.array([False, True, False]), 0.05, {"id": 2}
    )
    return [bc1, bc2]


@pytest.fixture
def mock_detector(mock_bicluster_list):
    detector = MagicMock()
    detector.detect.return_value = mock_bicluster_list
    return detector


@pytest.fixture
def mock_bicluster_config() -> BiclusterConfig:
    return BiclusterConfig(k1=2, k2=2, random_state=42)  # Simplified for some tests


@pytest.fixture
def sample_matrix_core() -> Matrix:
    return np.random.rand(10, 10)


@pytest.fixture
def analyzer_with_mock_detector_and_results(
    mock_bicluster_config: BiclusterConfig,
) -> tuple[BiclusterAnalyzer, list[Bicluster]]:
    # This fixture now returns the analyzer AND the list of biclusters it's supposed to find.
    mock_results = [
        Bicluster(
            np.array([True, True, False, False, False]),
            np.array([True, True, False, False, False]),
            0.1,
            {"id": "mock_bc_A"},
        ),
        Bicluster(
            np.array([False, False, True, True, True]),
            np.array([False, False, True, True, True]),
            0.2,
            {"id": "mock_bc_B"},
        ),
    ]
    with patch("src.core.SVDBiclusterDetector") as MockDetector:
        mock_detector_instance = MockDetector.return_value
        mock_detector_instance.detect.return_value = mock_results
        analyzer = BiclusterAnalyzer(config=mock_bicluster_config)
        # analyzer.detector is already set to an instance of SVDBiclusterDetector by BiclusterAnalyzer.__init__
        # We need to ensure *that specific instance* (analyzer.detector) has its detect method mocked.
        # The patch above mocks the class, so when BiclusterAnalyzer creates an SVDBiclusterDetector,
        # it gets a MagicMock. We then configure that mock.
        # To be absolutely sure, we can re-assign here if BiclusterAnalyzer creates its own detector instance.
        # However, the patch should ensure any SVDBiclusterDetector() call returns the mock.
        # Let's assume the patch works as intended for SVDBiclusterDetector instantiation inside BiclusterAnalyzer.
        # If analyzer.detector was *not* the MockDetector.return_value, we'd need to do:
        # analyzer.detector = mock_detector_instance OR patch.object(analyzer.detector, 'detect', return_value=mock_results)
        # To be absolutely sure, we can re-assign here if BiclusterAnalyzer creates its own detector instance.
        # However, the patch should ensure any SVDBiclusterDetector() call returns the mock.
        # Let's assume the patch works as intended for SVDBiclusterDetector instantiation inside BiclusterAnalyzer.
        # If analyzer.detector was *not* the MockDetector.return_value, we'd need to do:
        # analyzer.detector = mock_detector_instance OR patch.object(analyzer.detector, 'detect', return_value=mock_results)
        return analyzer, mock_results


def test_analyzer_initialization(
    analyzer_with_mock_detector_and_results: tuple[BiclusterAnalyzer, list[Bicluster]],
):
    analyzer, _ = analyzer_with_mock_detector_and_results
    assert isinstance(analyzer.config, BiclusterConfig)
    # The detector inside BiclusterAnalyzer will be a MagicMock due to the patch
    assert isinstance(analyzer.detector, MagicMock)


def test_analyzer_fit(
    analyzer_with_mock_detector_and_results: tuple[BiclusterAnalyzer, list[Bicluster]],
    sample_matrix_core: Matrix,
):
    analyzer, mock_results = analyzer_with_mock_detector_and_results
    returned_analyzer = analyzer.fit(sample_matrix_core)
    analyzer.detector.detect.assert_called_once_with(sample_matrix_core)  # type: ignore
    assert analyzer.results == mock_results
    assert returned_analyzer is analyzer


def test_analyzer_get_biclusters(
    analyzer_with_mock_detector_and_results: tuple[BiclusterAnalyzer, list[Bicluster]],
    sample_matrix_core: Matrix,
):
    analyzer, mock_results = analyzer_with_mock_detector_and_results
    with pytest.raises(ValueError, match="Analyzer has not been fitted yet"):
        analyzer.get_biclusters()
    analyzer.fit(sample_matrix_core)
    assert analyzer.get_biclusters() == mock_results


def test_analyzer_filter_biclusters(
    analyzer_with_mock_detector_and_results: tuple[BiclusterAnalyzer, list[Bicluster]],
    sample_matrix_core: Matrix,
):
    analyzer, mock_results = analyzer_with_mock_detector_and_results
    analyzer.fit(sample_matrix_core)
    # mock_results = [
    #     Bicluster(..., score=0.1, ...), # size 2*2=4
    #     Bicluster(..., score=0.2, ...), # size 3*3=9
    # ]

    # Filter by score (assuming mock_results[0].score=0.1, mock_results[1].score=0.2)
    filtered_score_gt_015 = analyzer.filter_biclusters(min_score=0.15)
    assert len(filtered_score_gt_015) == 1
    assert mock_results[1] in filtered_score_gt_015
    assert mock_results[0] not in filtered_score_gt_015

    # Filter by size
    # mock_results[0].size is 4 (2x2), mock_results[1].size is 9 (3x3)
    filtered_size_gt_5 = analyzer.filter_biclusters(min_size=5)
    assert len(filtered_size_gt_5) == 1
    assert mock_results[1] in filtered_size_gt_5
    assert mock_results[0] not in filtered_size_gt_5


def test_analyzer_merge_overlapping(
    analyzer_with_mock_detector_and_results: tuple[BiclusterAnalyzer, list[Bicluster]],
    sample_matrix_core: Matrix,
):
    analyzer, mock_results = analyzer_with_mock_detector_and_results

    # Create Biclusters that can be merged
    # Assume row/col indices are compatible masks of shape (10,10)
    R1 = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=bool)
    C1 = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=bool)
    BC1 = Bicluster(R1, C1, 0.1, {"id": "bc1"})

    R2 = np.array([0, 0, 1, 1, 1, 0, 0, 0, 0, 0], dtype=bool)  # Overlaps R1 at index 2
    C2 = np.array([0, 0, 1, 1, 1, 0, 0, 0, 0, 0], dtype=bool)  # Overlaps C1 at index 2
    BC2 = Bicluster(R2, C2, 0.2, {"id": "bc2"})

    R3 = np.array(
        [0, 0, 0, 0, 0, 1, 1, 1, 0, 0], dtype=bool
    )  # No overlap with BC1 or BC2
    C3 = np.array([0, 0, 0, 0, 0, 1, 1, 1, 0, 0], dtype=bool)
    BC3 = Bicluster(R3, C3, 0.3, {"id": "bc3"})

    analyzer.detector.detect.return_value = [BC1, BC2, BC3]
    analyzer.fit(
        sample_matrix_core
    )  # sample_matrix_core might need to be 10x10 for this test

    # Mock jaccard_index to control merging behavior
    # Here, we assume BC1 and BC2 will merge, BC3 remains separate.
    # We need to mock Bicluster.intersection_ratio for this test to be precise
    # without complex Bicluster setup.

    original_intersection_ratio = Bicluster.intersection_ratio

    def mock_ir(self, other):
        if (self.metadata.get("id") == "bc1" and other.metadata.get("id") == "bc2") or (
            self.metadata.get("id") == "bc2" and other.metadata.get("id") == "bc1"
        ):
            return 0.6  # Above threshold of 0.5
        return original_intersection_ratio(self, other)  # Default for others

    with patch.object(Bicluster, "intersection_ratio", mock_ir):
        merged_biclusters = analyzer.merge_overlapping(threshold=0.5)

    assert len(merged_biclusters) == 2  # BC1+BC2 merged, BC3 separate

    merged_one = None
    separate_one = None
    for bc in merged_biclusters:
        if bc.metadata.get("merged_from", 0) > 1:
            merged_one = bc
        else:
            separate_one = bc

    assert merged_one is not None
    assert separate_one is not None
    assert separate_one.metadata.get("id") == "bc3"
    assert merged_one.metadata.get("merged_from") == 2

    # Check merged indices (expected R1 | R2, C1 | C2)
    expected_merged_R = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=bool)
    expected_merged_C = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=bool)
    assert_array_equal(merged_one.row_indices, expected_merged_R)
    assert_array_equal(merged_one.col_indices, expected_merged_C)
    assert merged_one.score == pytest.approx((0.1 + 0.2) / 2)  # Average score


def test_analyzer_save_results(
    analyzer_with_mock_detector_and_results: tuple[BiclusterAnalyzer, list[Bicluster]],
    sample_matrix_core: Matrix,
    tmp_path: Path,
):
    analyzer, _ = analyzer_with_mock_detector_and_results
    analyzer.fit(sample_matrix_core)

    file_path = tmp_path / "results.json"
    analyzer.save_results(file_path)

    assert file_path.exists()
    with open(file_path, "r") as f:
        saved_data = json.load(f)

    assert "config" in saved_data
    assert "biclusters" in saved_data
    assert saved_data["config"] == analyzer.config.to_dict()
    assert len(saved_data["biclusters"]) == len(analyzer.results or [])
    if analyzer.results:
        assert saved_data["biclusters"][0]["id"] == analyzer.results[0].id


def test_analyzer_performance_tracking(
    analyzer_with_mock_detector_and_results: tuple[BiclusterAnalyzer, list[Bicluster]],
    sample_matrix_core: Matrix,
):
    analyzer, _ = analyzer_with_mock_detector_and_results
    with patch.object(analyzer, "logger") as mock_logger:
        with analyzer.performance_tracking():
            analyzer.fit(sample_matrix_core)
        mock_logger.info.assert_called()
        found_message = False
        for call_args in mock_logger.info.call_args_list:
            if (
                "Analysis completed in" in call_args[0][0]
                and "seconds" in call_args[0][0]
            ):
                found_message = True
                break
        assert (
            found_message
        ), "Logger was not called with performance message or message format changed."


def test_create_analyzer_factory():
    analyzer = create_analyzer(
        tolerance=0.1,
        k1=3,
        k2=4,
        scoring_method="pearson",
        clustering_method="spectral",
        random_state=123,
    )
    assert isinstance(analyzer, BiclusterAnalyzer)
    assert analyzer.config.tolerance == 0.1
    assert analyzer.config.k1 == 3
    assert analyzer.config.k2 == 4
    assert analyzer.config.scoring_method == ScoringMethod.PEARSON
    assert analyzer.config.clustering_method == ClusteringMethod.SPECTRAL
    assert analyzer.config.random_state == 123
    assert isinstance(analyzer.detector, SVDBiclusterDetector)


# To run: pytest test_core.py
