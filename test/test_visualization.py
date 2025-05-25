import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

from src.bicluster import Bicluster
from src.core import Matrix
from src.visualization import (
    BiclusterSpec,
    SyntheticDataConfig,
    SyntheticDataGenerator,
    BiclusterVisualizer,
    create_synthetic_data_with_generator,
)

# --- Constants for Testing ---
DEFAULT_MATRIX_SHAPE = (50, 40)
DEFAULT_RANDOM_STATE = 42

# --- Tests for BiclusterSpec ---


def test_bicluster_spec_creation():
    spec = BiclusterSpec(rows=10, cols=5, value=2.0, noise_level=0.2)
    assert spec.rows == 10
    assert spec.cols == 5
    assert spec.value == 2.0
    assert spec.noise_level == 0.2
    assert spec.size == 50


def test_bicluster_spec_defaults():
    spec = BiclusterSpec(rows=5, cols=5)
    assert spec.value == 1.0
    assert spec.noise_level == 0.1
    assert spec.size == 25


# --- Tests for SyntheticDataConfig ---


@pytest.fixture
def sample_bicluster_specs() -> list[BiclusterSpec]:
    return [
        BiclusterSpec(rows=10, cols=8, value=1.5, noise_level=0.1),
        BiclusterSpec(rows=12, cols=10, value=2.5, noise_level=0.15),
    ]


def test_synthetic_data_config_creation(sample_bicluster_specs):
    config = SyntheticDataConfig(
        matrix_shape=DEFAULT_MATRIX_SHAPE,
        bicluster_specs=sample_bicluster_specs,
        background_noise=0.05,
        random_state=DEFAULT_RANDOM_STATE,
    )
    assert config.matrix_shape == DEFAULT_MATRIX_SHAPE
    assert config.bicluster_specs == sample_bicluster_specs
    assert config.background_noise == 0.05
    assert config.random_state == DEFAULT_RANDOM_STATE


def test_synthetic_data_config_defaults(sample_bicluster_specs):
    config = SyntheticDataConfig(
        matrix_shape=DEFAULT_MATRIX_SHAPE, bicluster_specs=sample_bicluster_specs
    )
    assert config.background_noise == 0.0
    assert config.random_state == 42  # Default from dataclass field


def test_synthetic_data_config_post_init_validation(sample_bicluster_specs):
    # This test mainly ensures __post_init__ runs without error.
    # The current __post_init__ is a pass, so this is trivial.
    # If more complex validation is added, this test should be expanded.
    try:
        SyntheticDataConfig(
            matrix_shape=DEFAULT_MATRIX_SHAPE, bicluster_specs=sample_bicluster_specs
        )
    except Exception as e:
        pytest.fail(f"SyntheticDataConfig __post_init__ raised an exception: {e}")


# --- Tests for SyntheticDataGenerator ---


@pytest.fixture
def simple_generator_config() -> SyntheticDataConfig:
    specs = [BiclusterSpec(rows=5, cols=4, value=10.0, noise_level=0.1)]
    return SyntheticDataConfig(
        matrix_shape=(10, 8),
        bicluster_specs=specs,
        background_noise=0.01,
        random_state=DEFAULT_RANDOM_STATE,
    )


@pytest.fixture
def generator(simple_generator_config: SyntheticDataConfig) -> SyntheticDataGenerator:
    return SyntheticDataGenerator(config=simple_generator_config)


def test_synthetic_data_generator_initialization(
    generator: SyntheticDataGenerator, simple_generator_config: SyntheticDataConfig
):
    assert generator.config == simple_generator_config
    assert generator.logger is not None
    assert generator._rng is not None
    # Check if rng is seeded correctly (indirectly, by checking one random number if needed, but init is enough for now)
    # Example: first_rand = generator._rng.rand(); np.random.seed(DEFAULT_RANDOM_STATE); assert first_rand == np.random.rand()


def test_generate_bicluster_data(generator: SyntheticDataGenerator):
    spec = BiclusterSpec(rows=3, cols=2, value=5.0, noise_level=0.1)
    # Reset generator's internal RNG for this specific test part if needed, or use a new one.
    # For _generate_bicluster_data, it uses generator._rng, so its state matters.
    # We can fix the generator._rng state before this call for reproducibility here.
    generator._rng = np.random.RandomState(
        DEFAULT_RANDOM_STATE
    )  # Reset for predictable noise

    submatrix = generator._generate_bicluster_data(spec)
    assert submatrix.shape == (spec.rows, spec.cols)

    # Check if the mean is close to spec.value (considering noise)
    # The noise is normal(0, noise_level), so mean should be around spec.value
    expected_mean = spec.value
    # Tolerance for mean check can be related to noise_level / sqrt(size)
    assert_allclose(
        np.mean(submatrix), expected_mean, atol=spec.noise_level * 3
    )  # Wider tolerance, mean is sensitive

    # Check if std dev is related to noise_level
    # This is harder to assert precisely without knowing the exact distribution of means of noisy signals.
    # A rough check: std dev of noise should be spec.noise_level.
    # Signal part has std dev 0. So std dev of (signal+noise) should be close to std dev of noise.
    std_dev = np.std(submatrix)
    assert std_dev > 0  # It should have some variance due to noise
    # A more specific check would require a fixed random seed and pre-calculated expected std dev.
    # For now, checking it's not zero and roughly in a plausible range if value is not 0.
    if spec.value != 0:
        assert std_dev < spec.value  # Assuming noise is smaller than signal typically


def test_create_structured_matrix_and_gt(simple_generator_config: SyntheticDataConfig):
    # Use a config with multiple small biclusters that fit sequentially
    specs = [
        BiclusterSpec(rows=3, cols=2, value=1.0, noise_level=0.01),
        BiclusterSpec(rows=2, cols=3, value=2.0, noise_level=0.01),
    ]
    config = SyntheticDataConfig(
        matrix_shape=(10, 10),  # Enough space
        bicluster_specs=specs,
        background_noise=0.01,
        random_state=123,
    )
    generator_multi = SyntheticDataGenerator(config)
    matrix, gt_biclusters = generator_multi._create_structured_matrix_and_gt()

    assert matrix.shape == config.matrix_shape
    assert len(gt_biclusters) == len(specs)

    # Check first bicluster
    bc1_spec = specs[0]
    gt_bc1 = gt_biclusters[0]
    sub_bc1 = matrix[0 : bc1_spec.rows, 0 : bc1_spec.cols]
    assert_allclose(
        np.mean(sub_bc1),
        bc1_spec.value,
        atol=bc1_spec.noise_level * 5 + config.background_noise * 5,
    )
    assert gt_bc1.shape == (bc1_spec.rows, bc1_spec.cols)
    assert gt_bc1.metadata["value"] == bc1_spec.value
    assert_array_equal(
        gt_bc1.row_indices,
        np.array(
            [True] * bc1_spec.rows + [False] * (config.matrix_shape[0] - bc1_spec.rows)
        ),
    )
    assert_array_equal(
        gt_bc1.col_indices,
        np.array(
            [True] * bc1_spec.cols + [False] * (config.matrix_shape[1] - bc1_spec.cols)
        ),
    )

    # Check second bicluster (placed after first one, row-wise)
    bc2_spec = specs[1]
    gt_bc2 = gt_biclusters[1]
    # Placement: starts at row=bc1_spec.rows, col=0
    start_row_bc2 = bc1_spec.rows
    sub_bc2 = matrix[start_row_bc2 : start_row_bc2 + bc2_spec.rows, 0 : bc2_spec.cols]
    assert_allclose(
        np.mean(sub_bc2),
        bc2_spec.value,
        atol=bc2_spec.noise_level * 5 + config.background_noise * 5,
    )
    assert gt_bc2.shape == (bc2_spec.rows, bc2_spec.cols)

    expected_bc2_row_indices = np.zeros(config.matrix_shape[0], dtype=bool)
    expected_bc2_row_indices[start_row_bc2 : start_row_bc2 + bc2_spec.rows] = True
    assert_array_equal(gt_bc2.row_indices, expected_bc2_row_indices)
    assert_array_equal(
        gt_bc2.col_indices,
        np.array(
            [True] * bc2_spec.cols + [False] * (config.matrix_shape[1] - bc2_spec.cols)
        ),
    )
    assert len(generator_multi.ground_truth_biclusters_info) == len(specs)


def test_create_structured_matrix_gt_bicluster_too_large(
    simple_generator_config: SyntheticDataConfig,
):
    # Test scenario where a bicluster spec is too large to fit
    large_spec = [
        BiclusterSpec(rows=simple_generator_config.matrix_shape[0] + 1, cols=5)
    ]
    config_too_large = SyntheticDataConfig(
        matrix_shape=simple_generator_config.matrix_shape,
        bicluster_specs=large_spec,
        random_state=DEFAULT_RANDOM_STATE,
    )
    generator_tl = SyntheticDataGenerator(config_too_large)
    # This should log a warning and skip the bicluster
    matrix, gt_biclusters = generator_tl._create_structured_matrix_and_gt()
    assert len(gt_biclusters) == 0
    assert matrix.shape == config_too_large.matrix_shape
    # Ensure background noise is still present (mean should be around 0 if background is 0 and loc is 0)
    assert_allclose(
        np.mean(matrix), 0, atol=config_too_large.background_noise * 5 + 0.1
    )  # Wider tolerance for overall matrix mean


def test_apply_permutation(generator: SyntheticDataGenerator):
    original_matrix = np.arange(20).reshape((5, 4))
    permuted_matrix = generator._apply_permutation(original_matrix)
    assert permuted_matrix.shape == original_matrix.shape
    assert generator.row_permutation is not None
    assert generator.col_permutation is not None
    assert len(generator.row_permutation) == original_matrix.shape[0]
    assert len(generator.col_permutation) == original_matrix.shape[1]
    # Check if all elements are preserved (sum should be same)
    assert np.sum(permuted_matrix) == np.sum(original_matrix)
    # Check if it's actually permuted (not same as original, unless permutation is identity)
    if not (
        np.all(generator.row_permutation == np.arange(original_matrix.shape[0]))
        and np.all(generator.col_permutation == np.arange(original_matrix.shape[1]))
    ):
        assert not np.array_equal(permuted_matrix, original_matrix)


def test_permute_ground_truth(generator: SyntheticDataGenerator):
    # Setup: Create a dummy structured matrix and GT biclusters, then apply permutation
    rng_gt = np.random.RandomState(1)
    structured_matrix = rng_gt.rand(6, 5)
    true_bc_rows = np.array([True, True, False, False, False, False])
    true_bc_cols = np.array([True, True, True, False, False])
    gt_bc = Bicluster(true_bc_rows, true_bc_cols, metadata={"id": "gt_0"})

    # Apply a known permutation for testing
    generator.row_permutation = np.array(
        [2, 0, 1, 3, 5, 4]
    )  # Original row 0 goes to 2, 1 to 0, 2 to 1 etc.
    generator.col_permutation = np.array(
        [1, 0, 3, 2, 4]
    )  # Original col 0 goes to 1, 1 to 0, 2 to 3 etc.

    permuted_gt_list = generator._permute_ground_truth([gt_bc])
    assert len(permuted_gt_list) == 1
    permuted_gt_bc = permuted_gt_list[0]

    # Expected permuted indices based on the known permutation
    # Original selected rows: 0, 1. Permutation maps 0->2, 1->0. So permuted selection is at indices 0, 2 of the permuted_row_indices array.
    # However, permuted_row_indices = bc.row_indices[self.row_permutation] means the *values* from original row_indices are reordered according to row_permutation.
    # So, if row_indices = [T,T,F,F,F,F] and row_permutation = [2,0,1,3,5,4]
    # permuted_row_indices will be [row_indices[2], row_indices[0], row_indices[1], row_indices[3], row_indices[5], row_indices[4]]
    # = [F, T, T, F, F, F]
    expected_perm_rows = true_bc_rows[generator.row_permutation]
    expected_perm_cols = true_bc_cols[generator.col_permutation]

    assert_array_equal(permuted_gt_bc.row_indices, expected_perm_rows)
    assert_array_equal(permuted_gt_bc.col_indices, expected_perm_cols)
    assert permuted_gt_bc.metadata["is_permuted_gt"] is True


def test_permute_ground_truth_no_permutation_error(generator: SyntheticDataGenerator):
    generator.row_permutation = None  # Ensure no permutation set
    with pytest.raises(RuntimeError, match="Permutation has not been applied yet"):
        generator._permute_ground_truth([MagicMock(spec=Bicluster)])


def test_generate_full(simple_generator_config: SyntheticDataConfig):
    # Test the main generate() method
    generator_full = SyntheticDataGenerator(config=simple_generator_config)
    p_matrix, s_matrix, p_gt_bcs = generator_full.generate()

    assert p_matrix.shape == simple_generator_config.matrix_shape
    assert s_matrix.shape == simple_generator_config.matrix_shape
    assert len(p_gt_bcs) == len(simple_generator_config.bicluster_specs)

    if p_gt_bcs:
        assert p_gt_bcs[0].metadata.get("is_permuted_gt") is True
        # Check that permuted GT biclusters are not empty if original spec was not empty
        original_spec = simple_generator_config.bicluster_specs[0]
        if original_spec.rows > 0 and original_spec.cols > 0:
            assert p_gt_bcs[0].size > 0  # or shape check

    # Check content preservation (sum) after permutation for the actual generated matrix
    assert_allclose(
        np.sum(p_matrix), np.sum(s_matrix), rtol=1e-5
    )  # Sums might differ slightly due to float precision if any in-place ops happened, but shouldn't here.

    assert generator_full.row_permutation is not None
    assert generator_full.col_permutation is not None
    assert len(generator_full.ground_truth_biclusters_info) == len(
        simple_generator_config.bicluster_specs
    )


def test_generate_full_no_specs():
    config_no_specs = SyntheticDataConfig(matrix_shape=(10, 10), bicluster_specs=[])
    generator_no_specs = SyntheticDataGenerator(config_no_specs)
    p_matrix, s_matrix, p_gt_bcs = generator_no_specs.generate()
    assert len(p_gt_bcs) == 0
    assert p_matrix.shape == (10, 10)
    # Matrix should be just background noise
    assert_allclose(
        np.mean(p_matrix), 0, atol=config_no_specs.background_noise * 5 + 0.1
    )


def test_get_ground_truth_bicluster_info_dicts(generator: SyntheticDataGenerator):
    assert generator.get_ground_truth_bicluster_info_dicts() == []  # Before generate
    p_matrix, s_matrix, p_gt_bcs = generator.generate()
    info_dicts = generator.get_ground_truth_bicluster_info_dicts()
    assert len(info_dicts) == len(generator.config.bicluster_specs)
    if info_dicts:
        assert "id" in info_dicts[0]
        assert info_dicts[0]["id"].startswith("gt_")


def test_get_permutation_indices(generator: SyntheticDataGenerator):
    assert generator.get_permutation_indices() is None  # Before generate
    generator.generate()
    row_perm, col_perm = generator.get_permutation_indices()
    assert row_perm is not None
    assert col_perm is not None
    assert len(row_perm) == generator.config.matrix_shape[0]
    assert len(col_perm) == generator.config.matrix_shape[1]


# --- Tests for BiclusterVisualizer ---


@pytest.fixture
def visualizer() -> BiclusterVisualizer:
    return BiclusterVisualizer(figsize=(6, 3), dpi=75)  # Smaller for tests


@pytest.fixture
def sample_matrix_vis() -> Matrix:
    return np.random.rand(20, 15)


@pytest.fixture
def sample_biclusters_vis(sample_matrix_vis: Matrix) -> list[Bicluster]:
    # Create a couple of sample Bicluster objects for visualization tests
    rows, cols = sample_matrix_vis.shape
    bc1_rows = np.zeros(rows, dtype=bool)
    bc1_rows[0:5] = True
    bc1_cols = np.zeros(cols, dtype=bool)
    bc1_cols[0:5] = True
    bc1 = Bicluster(bc1_rows, bc1_cols, score=0.1, metadata={"id": "vis_bc1"})

    bc2_rows = np.zeros(rows, dtype=bool)
    bc2_rows[10:15] = True
    bc2_cols = np.zeros(cols, dtype=bool)
    bc2_cols[8:12] = True
    bc2 = Bicluster(bc2_rows, bc2_cols, score=0.2, metadata={"id": "vis_bc2"})
    return [bc1, bc2]


@patch("matplotlib.pyplot.show")  # Mock plt.show for all visualizer tests
def test_visualizer_plot_matrix_heatmap_no_error(
    mock_show, visualizer: BiclusterVisualizer, sample_matrix_vis: Matrix
):
    try:
        visualizer.plot_matrix_heatmap(sample_matrix_vis, title="Test Heatmap")
    except Exception as e:
        pytest.fail(f"plot_matrix_heatmap raised an exception: {e}")
    mock_show.assert_called_once()  # Ensure show was called (or would have been)


@patch("matplotlib.pyplot.show")
@patch("matplotlib.pyplot.savefig")
def test_visualizer_plot_matrix_heatmap_save(
    mock_savefig,
    mock_show,
    visualizer: BiclusterVisualizer,
    sample_matrix_vis: Matrix,
    tmp_path: Path,
):
    save_path = tmp_path / "heatmap.png"
    visualizer.plot_matrix_heatmap(sample_matrix_vis, save_path=save_path)
    mock_savefig.assert_called_once_with(save_path)
    mock_show.assert_called_once()


@patch("matplotlib.pyplot.show")
def test_visualizer_plot_matrix_comparison_no_error(
    mock_show,
    visualizer: BiclusterVisualizer,
    sample_matrix_vis: Matrix,
    sample_biclusters_vis: list[Bicluster],
):
    try:
        visualizer.plot_matrix_comparison(sample_matrix_vis, sample_biclusters_vis)
        visualizer.plot_matrix_comparison(
            sample_matrix_vis, []
        )  # Test with no biclusters
    except Exception as e:
        pytest.fail(f"plot_matrix_comparison raised an exception: {e}")
    assert mock_show.call_count == 2


@patch("matplotlib.pyplot.show")
def test_visualizer_plot_bicluster_statistics_no_error(
    mock_show, visualizer: BiclusterVisualizer, sample_biclusters_vis: list[Bicluster]
):
    try:
        visualizer.plot_bicluster_statistics(sample_biclusters_vis)
        visualizer.plot_bicluster_statistics(
            []
        )  # Test with no biclusters (should warn and return)
    except Exception as e:
        pytest.fail(f"plot_bicluster_statistics raised an exception: {e}")
    # mock_show might be called once or not at all if no biclusters and it returns early.
    # If it always creates a figure, then it would be called.
    # Based on current code, if not biclusters, it logs and returns. So show is only called if biclusters.
    mock_show.assert_called_once()


@patch("matplotlib.pyplot.show")
def test_visualizer_plot_individual_biclusters_no_error(
    mock_show,
    visualizer: BiclusterVisualizer,
    sample_matrix_vis: Matrix,
    sample_biclusters_vis: list[Bicluster],
):
    try:
        visualizer.plot_individual_biclusters(
            sample_matrix_vis, sample_biclusters_vis, max_to_plot=1
        )
        visualizer.plot_individual_biclusters(
            sample_matrix_vis, []
        )  # Test with no biclusters
    except Exception as e:
        pytest.fail(f"plot_individual_biclusters raised an exception: {e}")
    if sample_biclusters_vis:  # show is called if there are biclusters to plot
        mock_show.assert_called_once()
    else:  # if no biclusters, show might not be called
        mock_show.assert_not_called()  # Or check call_count based on actual behavior for empty list


def test_create_bicluster_overlay_matrix(
    visualizer: BiclusterVisualizer,
    sample_matrix_vis: Matrix,
    sample_biclusters_vis: list[Bicluster],
):
    overlay = visualizer._create_bicluster_overlay_matrix(
        sample_matrix_vis.shape, sample_biclusters_vis
    )
    assert overlay.shape == sample_matrix_vis.shape
    assert overlay.dtype == int

    unique_values = np.unique(overlay)
    # Expected unique values: 0 (background) and 1, 2 (for the two biclusters)
    expected_unique_values = [0, 1, 2]
    assert_array_equal(np.sort(unique_values), np.sort(expected_unique_values))

    # Check if bicluster 1 is marked with 1
    bc1 = sample_biclusters_vis[0]
    for r_idx in np.where(bc1.row_indices)[0]:
        for c_idx in np.where(bc1.col_indices)[0]:
            assert overlay[r_idx, c_idx] == 1

    # Check if bicluster 2 is marked with 2
    bc2 = sample_biclusters_vis[1]
    for r_idx in np.where(bc2.row_indices)[0]:
        for c_idx in np.where(bc2.col_indices)[0]:
            # This assertion needs to be careful if biclusters can overlap.
            # The current _create_bicluster_overlay_matrix has last-one-wins for overlaps.
            # Our sample_biclusters_vis are non-overlapping, so this is fine.
            assert overlay[r_idx, c_idx] == 2


@patch("matplotlib.pyplot.show")
@patch.object(BiclusterVisualizer, "plot_matrix_comparison")
@patch.object(BiclusterVisualizer, "plot_bicluster_statistics")
@patch.object(BiclusterVisualizer, "plot_individual_biclusters")
@patch("pathlib.Path.mkdir")
def test_visualizer_create_report_visualizations(
    mock_mkdir,
    mock_plot_individual,
    mock_plot_stats,
    mock_plot_compare,
    mock_show,  # To catch any stray plt.show() calls within the tested method itself
    visualizer: BiclusterVisualizer,
    sample_matrix_vis: Matrix,
    sample_biclusters_vis: list[Bicluster],
    tmp_path: Path,
):
    report_dir = tmp_path / "report"
    visualizer.create_report_visualizations(
        sample_matrix_vis, sample_biclusters_vis, report_dir
    )

    mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
    mock_plot_compare.assert_called_once_with(
        sample_matrix_vis,
        sample_biclusters_vis,
        save_path=report_dir / "matrix_with_biclusters_overlay.png",
    )
    mock_plot_stats.assert_called_once_with(
        sample_biclusters_vis, save_path=report_dir / "bicluster_statistics.png"
    )
    mock_plot_individual.assert_called_once_with(
        sample_matrix_vis,
        sample_biclusters_vis,
        max_to_plot=6,  # default from create_report_visualizations
        save_dir=report_dir,
    )


# --- Tests for factory function ---


def test_create_synthetic_data_with_generator():
    p_mat, s_mat, gt_bcs, gen = create_synthetic_data_with_generator(
        n_biclusters=2,
        matrix_shape=(30, 25),
        bicluster_size_range=(5, 10),
        random_state=123,
    )
    assert p_mat.shape == (30, 25)
    assert s_mat.shape == (30, 25)
    assert len(gt_bcs) == 2
    assert isinstance(gen, SyntheticDataGenerator)
    assert gen.config.random_state == 123
    assert len(gen.config.bicluster_specs) == 2
    if gt_bcs:
        assert gt_bcs[0].size > 0


# </rewritten_file>
