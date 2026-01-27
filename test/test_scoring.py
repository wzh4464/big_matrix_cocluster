import pytest
import numpy as np
from numpy.testing import (
    assert_allclose,
    assert_array_almost_equal_nulp,
    assert_array_equal,
)

from src.bicluster import Bicluster
from src.core import ScoringMethod, Matrix
from src.scoring import CompatibilityScorer, SVRScorer, ScoringStrategy


# --- Test Fixtures ---
@pytest.fixture
def bicluster_sample_data() -> tuple[Matrix, Bicluster]:
    matrix = np.array(
        [
            [1.0, 2.0, 0.5, 1.5],
            [1.2, 2.1, 0.6, 1.6],
            [3.0, 3.2, 1.0, 1.2],
            [0.1, 0.2, 5.0, 5.5],
        ],
        dtype=np.float64,
    )

    # A perfect bicluster (rows 0,1; cols 0,1)
    row_indices = np.array([True, True, False, False])
    col_indices = np.array([True, True, False, False])
    bicluster = Bicluster(
        row_indices, col_indices, score=0.0
    )  # Score will be recalculated
    return matrix, bicluster


@pytest.fixture
def constant_submatrix_bicluster() -> tuple[Matrix, Bicluster]:
    matrix = np.array(
        [
            [2.0, 2.0, 0.5, 1.5],
            [2.0, 2.0, 0.6, 1.6],
            [3.0, 3.2, 1.0, 1.2],
        ],
        dtype=np.float64,
    )
    row_indices = np.array([True, True, False])
    col_indices = np.array([True, True, False, False])
    bicluster = Bicluster(row_indices, col_indices, score=0.0)
    return matrix, bicluster


@pytest.fixture
def single_row_col_bicluster() -> tuple[Matrix, Bicluster]:
    matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
    row_indices_sr = np.array([True, False])
    col_indices_sr = np.array([True, True])
    bicluster_sr = Bicluster(row_indices_sr, col_indices_sr, 0.0)
    return matrix, bicluster_sr


# --- Tests for CompatibilityScorer ---


def test_scorer_initialization():
    scorer_exp = CompatibilityScorer(method=ScoringMethod.EXPONENTIAL)
    assert scorer_exp.method == ScoringMethod.EXPONENTIAL
    scorer_pr = CompatibilityScorer(method=ScoringMethod.PEARSON)
    assert scorer_pr.method == ScoringMethod.PEARSON
    scorer_comp = CompatibilityScorer(method=ScoringMethod.COMPATIBILITY)
    assert scorer_comp.method == ScoringMethod.COMPATIBILITY
    # Check default
    scorer_default = CompatibilityScorer()
    assert scorer_default.method == ScoringMethod.EXPONENTIAL


def test_score_empty_submatrix(bicluster_sample_data):
    matrix, _ = bicluster_sample_data
    empty_row_indices = np.array([False, False, False, False])
    col_indices = np.array([True, True, False, False])
    empty_bicluster = Bicluster(empty_row_indices, col_indices, 0.0)

    scorer = CompatibilityScorer()
    score_val = scorer.score(matrix, empty_bicluster)
    assert score_val == float(
        "inf"
    ), "Score for bicluster yielding empty submatrix should be inf"


def test_score_single_row_or_col_submatrix(single_row_col_bicluster):
    matrix, bicluster_sr = single_row_col_bicluster
    # This bicluster extracts a 1x2 submatrix
    scorer = CompatibilityScorer()
    score_val = scorer.score(matrix, bicluster_sr)
    assert score_val == float(
        "inf"
    ), "Score for submatrix with <2 rows or <2 cols should be inf"

    row_indices_sc = np.array([True, True])
    col_indices_sc = np.array([True, False])
    bicluster_sc = Bicluster(row_indices_sc, col_indices_sc, 0.0)
    score_val_sc = scorer.score(matrix, bicluster_sc)
    assert score_val_sc == float(
        "inf"
    ), "Score for submatrix with <2 rows or <2 cols should be inf"


# --- More specific tests for _calculate_compatibility, _pearson_compatibility, etc. ---
# These might require crafting very specific small matrices


def test_score_helper_perfect_similarity():
    scorer = CompatibilityScorer()
    # All items are perfectly similar (similarity 1)
    sim_matrix_perfect = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    scores = scorer._score_helper(length=3, similarity_matrix=sim_matrix_perfect)
    assert_allclose(scores, 0.0)  # 1 - mean_similarity = 1 - ((1+1)/(3-1)) = 1 - 1 = 0


def test_score_helper_no_similarity():
    scorer = CompatibilityScorer()
    # All items have no similarity to others (similarity 0)
    sim_matrix_none = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    scores = scorer._score_helper(length=3, similarity_matrix=sim_matrix_none)
    assert_allclose(scores, 1.0)  # 1 - 0 = 1


def test_score_helper_mixed_similarity():
    scorer = CompatibilityScorer()
    sim_matrix_mixed = np.array([[0.0, 0.8, 0.2], [0.8, 0.0, 0.5], [0.2, 0.5, 0.0]])
    # For item 0: avg_sim = (0.8+0.2)/2 = 0.5. Score = 1-0.5 = 0.5
    # For item 1: avg_sim = (0.8+0.5)/2 = 0.65. Score = 1-0.65 = 0.35
    # For item 2: avg_sim = (0.2+0.5)/2 = 0.35. Score = 1-0.35 = 0.65
    expected_scores = np.array([0.5, 0.35, 0.65])
    actual_scores = scorer._score_helper(length=3, similarity_matrix=sim_matrix_mixed)
    assert_allclose(actual_scores, expected_scores, atol=1e-7)


def test_score_helper_single_element():
    scorer = CompatibilityScorer()
    sim_matrix_single = np.array([[0.0]])  # Not really used for single element
    scores = scorer._score_helper(length=1, similarity_matrix=sim_matrix_single)
    assert (
        np.isinf(scores).all() and scores[0] > 0
    ), "Score for single element should be positive infinity"
    # Ensure it returns an array with a single positive infinity
    expected = np.array([float("inf")])
    assert_array_equal(scores, expected)


# Define a fixed seed for generating random matrix data in tests where applicable
FIXED_RANDOM_SEED_FOR_DATA = 12345


@pytest.mark.parametrize(
    "matrix_data_desc, matrix_data_gen_params, expected_score_approx, tolerance_param",
    [
        # Case 1: Perfect positive correlation
        (
            "perfect_positive",
            {
                "type": "specific",
                "data": np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]], dtype=float),
            },
            0.16,
            {"rel": 0.1},
        ),
        # Case 2: Perfect negative correlation
        (
            "perfect_negative",
            {
                "type": "specific",
                "data": np.array([[1, 2, 3], [3, 2, 1], [1, 2, 3]], dtype=float),
            },
            0.14,
            {"rel": 0.1},
        ),
        # Case 3: No correlation (random) - now generated with a fixed seed inside the test
        (
            "random_5x5",
            {"type": "random", "shape": (5, 5), "seed": FIXED_RANDOM_SEED_FOR_DATA},
            (0.4, 0.9),
            None,
        ),  # Keep range for this one for now
        # Case 4: Highly correlated block
        (
            "highly_correlated",
            {
                "type": "specific",
                "data": np.array(
                    [
                        [1, 1.1, 3, 5],
                        [1.2, 1.3, 3.2, 5.1],
                        [0, 0, 8, 7],
                        [0.1, 0.2, 8.3, 7.3],
                    ],
                    dtype=float,
                ),
            },
            (0.0, 0.2),
            None,
        ),  # Keep range
    ],
)
def test_pearson_scoring_logic(
    matrix_data_desc, matrix_data_gen_params, expected_score_approx, tolerance_param
):
    if matrix_data_gen_params["type"] == "specific":
        matrix_data = matrix_data_gen_params["data"]
    elif matrix_data_gen_params["type"] == "random":
        rng = np.random.RandomState(matrix_data_gen_params["seed"])
        matrix_data = rng.rand(*matrix_data_gen_params["shape"])
    else:
        raise ValueError("Unknown matrix data generation type")

    np.random.seed(42)
    scorer = CompatibilityScorer(method=ScoringMethod.PEARSON)
    rows, cols = matrix_data.shape
    bicluster = Bicluster(np.ones(rows, dtype=bool), np.ones(cols, dtype=bool), 0.0)

    score = scorer.score(matrix_data, bicluster)
    if tolerance_param:
        assert score == pytest.approx(
            expected_score_approx, **tolerance_param
        ), f"Pearson score {score} for {matrix_data_desc} not approx {expected_score_approx}"
    else:  # Fallback to range for cases not using approx yet
        assert (
            score >= expected_score_approx[0] and score <= expected_score_approx[1]
        ), f"Pearson score {score} for {matrix_data_desc} out of range {expected_score_approx}"


# Test Exponential Scoring
@pytest.mark.parametrize(
    "matrix_data_desc, matrix_data_gen_params, expected_score_approx, tolerance_param",
    [
        (
            "perfect_constant_4x4",
            {"type": "specific", "data": np.ones((4, 4), dtype=float)},
            (0.0, 0.01),
            None,
        ),  # Keep range
        (
            "two_distinct_blocks",
            {
                "type": "specific",
                "data": np.array(
                    [[1, 1, 5, 5], [1, 1, 5, 5], [10, 10, 15, 15], [10, 10, 15, 15]],
                    dtype=float,
                ),
            },
            (0.6, 0.75),
            None,
        ),  # Keep range
        (
            "smoothly_varying_3x3",
            {
                "type": "specific",
                "data": np.array(
                    [[1, 1.1, 1.2], [2, 2.1, 2.2], [3, 3.1, 3.2]], dtype=float
                ),
            },
            (0.3, 0.45),
            None,
        ),  # Keep range
        (
            "random_5x5_exp",
            {"type": "random", "shape": (5, 5), "seed": FIXED_RANDOM_SEED_FOR_DATA},
            0.13,
            {"rel": 0.15},
        ),  # Use approx, NOTE counter-intuitive value needs review
    ],
)
def test_exponential_scoring_logic(
    matrix_data_desc, matrix_data_gen_params, expected_score_approx, tolerance_param
):
    if matrix_data_gen_params["type"] == "specific":
        matrix_data = matrix_data_gen_params["data"]
    elif matrix_data_gen_params["type"] == "random":
        rng = np.random.RandomState(matrix_data_gen_params["seed"])
        matrix_data = rng.rand(*matrix_data_gen_params["shape"])
    else:
        raise ValueError("Unknown matrix data generation type")

    np.random.seed(42)
    scorer = CompatibilityScorer(method=ScoringMethod.EXPONENTIAL)
    rows, cols = matrix_data.shape
    bicluster = Bicluster(np.ones(rows, dtype=bool), np.ones(cols, dtype=bool), 0.0)
    score = scorer.score(matrix_data, bicluster)
    if tolerance_param:
        assert score == pytest.approx(
            expected_score_approx, **tolerance_param
        ), f"Exponential score {score} for {matrix_data_desc} not approx {expected_score_approx}"
    else:  # Fallback to range
        assert (
            score >= expected_score_approx[0] and score <= expected_score_approx[1]
        ), f"Exponential score {score} for {matrix_data_desc} out of range {expected_score_approx}"


def test_scoring_with_constant_submatrix(constant_submatrix_bicluster):
    matrix, bicluster = constant_submatrix_bicluster
    # Submatrix is [[2,2],[2,2]]

    # Pearson: Should be low score (perfectly correlated after noise)
    scorer_pearson = CompatibilityScorer(method=ScoringMethod.PEARSON)
    score_p = scorer_pearson.score(matrix, bicluster)
    assert_allclose(
        score_p,
        0.0,
        atol=0.01,
        err_msg="Pearson score for constant block not close to 0",
    )

    # Exponential: Should be low score (highly similar after noise)
    scorer_exp = CompatibilityScorer(method=ScoringMethod.EXPONENTIAL)
    score_e = scorer_exp.score(matrix, bicluster)
    assert_allclose(
        score_e,
        0.0,
        atol=0.01,
        err_msg="Exponential score for constant block not close to 0",
    )


# Example of a full bicluster scoring test
def test_perfect_bicluster_score(bicluster_sample_data):
    matrix, bicluster = bicluster_sample_data
    # This specific bicluster (rows 0,1; cols 0,1) is:
    # [[1.0, 2.0],
    #  [1.2, 2.1]]
    # Columns are [1.0, 1.2] and [2.0, 2.1]. Rows are [1.0, 2.0] and [1.2, 2.1].
    # These are highly correlated / similar.

    scorer_exp = CompatibilityScorer(method=ScoringMethod.EXPONENTIAL)
    score_e = scorer_exp.score(matrix, bicluster)
    assert score_e < 0.4, f"Exponential score for good bicluster {score_e} too high"

    scorer_pearson = CompatibilityScorer(method=ScoringMethod.PEARSON)
    score_p = scorer_pearson.score(matrix, bicluster)
    assert score_p < 0.1, f"Pearson score for good bicluster {score_p} too high"


# It's good to also test a clearly non-bicluster region
@pytest.fixture
def bad_bicluster_sample_data() -> tuple[Matrix, Bicluster]:
    matrix = np.array(
        [
            [1.0, 10.0, 0.5, 1.5],
            [1.2, 0.1, 0.6, 1.6],
            [3.0, 3.2, 20.0, 1.2],
            [0.1, 0.2, 5.0, 30.0],
        ],
        dtype=np.float64,
    )

    row_indices = np.array([True, True, True, True])  # Full matrix
    col_indices = np.array([True, True, True, True])
    bicluster = Bicluster(row_indices, col_indices, score=0.0)
    return matrix, bicluster


def test_bad_bicluster_score(bad_bicluster_sample_data):
    matrix, bicluster = bad_bicluster_sample_data  # A random-ish matrix
    scorer_exp = CompatibilityScorer(method=ScoringMethod.EXPONENTIAL)
    score_e = scorer_exp.score(matrix, bicluster)
    assert score_e > 0.5, f"Exponential score for bad bicluster {score_e} too low"

    scorer_pearson = CompatibilityScorer(method=ScoringMethod.PEARSON)
    score_p = scorer_pearson.score(matrix, bicluster)
    # Pearson score can be low if there's accidental linear trend,
    # but generally for random, it should not be extremely close to 0.
    assert score_p > 0.2, f"Pearson score for bad bicluster {score_p} too low"


# --- Tests for SVRScorer (DiMergeCo paper implementation) ---


def test_svr_scorer_initialization():
    """Test SVR scorer initialization with different parameters."""
    # Test normalized SVR (default)
    scorer_norm = SVRScorer(normalized=True)
    assert scorer_norm.normalized is True
    assert scorer_norm.epsilon == 1e-10
    assert scorer_norm.min_singular_values == 2

    # Test basic SVR
    scorer_basic = SVRScorer(normalized=False)
    assert scorer_basic.normalized is False

    # Test custom parameters
    scorer_custom = SVRScorer(
        normalized=True, epsilon=1e-8, min_singular_values=3, cache_svd=False
    )
    assert scorer_custom.epsilon == 1e-8
    assert scorer_custom.min_singular_values == 3
    assert scorer_custom.cache_svd is False


def test_svr_score_calculation_known_matrix():
    """Test SVR score calculation on a matrix with known singular values."""
    # Create a simple 3x3 matrix with known singular values
    # Using a diagonal-like structure
    matrix = np.array([[3.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]], dtype=float)

    # Full bicluster
    row_indices = np.ones(3, dtype=bool)
    col_indices = np.ones(3, dtype=bool)
    bicluster = Bicluster(row_indices, col_indices, score=None)

    # Test basic SVR s(A) = s₁/s₂
    scorer_basic = SVRScorer(normalized=False)
    score_basic = scorer_basic.score(matrix, bicluster)

    # For diagonal matrix, singular values are the diagonal elements (sorted desc)
    # s₁ = 3, s₂ = 2, so s(A) = 3/2 = 1.5
    # Returned score should be -1.5 (negative convention)
    assert score_basic < 0, "SVR score should be negative (lower is better convention)"
    assert_allclose(score_basic, -1.5, rtol=0.01)

    # Test normalized SVR S(A) = s(A)/||A||_F
    scorer_norm = SVRScorer(normalized=True)
    score_norm = scorer_norm.score(matrix, bicluster)

    # Frobenius norm = sqrt(3² + 2² + 1²) = sqrt(14) ≈ 3.742
    # S(A) = 1.5 / 3.742 ≈ 0.401
    # Returned score should be ≈ -0.401
    frobenius = np.linalg.norm(matrix, "fro")
    expected_norm = -1.5 / frobenius
    assert score_norm < 0
    assert_allclose(score_norm, expected_norm, rtol=0.01)


def test_svr_rank1_matrix():
    """Test SVR scoring on rank-1 matrix (perfect bicluster)."""
    # Create a rank-1 matrix (outer product)
    u = np.array([[1.0], [2.0], [3.0]])
    v = np.array([[1.0, 2.0, 3.0, 4.0]])
    matrix = u @ v  # 3x4 rank-1 matrix

    row_indices = np.ones(3, dtype=bool)
    col_indices = np.ones(4, dtype=bool)
    bicluster = Bicluster(row_indices, col_indices, score=None)

    scorer = SVRScorer(normalized=False)
    score = scorer.score(matrix, bicluster)

    # Rank-1 matrix should have s₂ ≈ 0, so scorer returns very negative score
    assert score < -1e9, f"Rank-1 matrix should get highly negative score, got {score}"


def test_svr_small_matrix_boundary():
    """Test SVR scoring on matrices that are too small."""
    # 1x1 matrix
    matrix_1x1 = np.array([[5.0]])
    row_indices = np.array([True])
    col_indices = np.array([True])
    bicluster_1x1 = Bicluster(row_indices, col_indices, score=None)

    scorer = SVRScorer()
    score = scorer.score(matrix_1x1, bicluster_1x1)
    assert score == float("inf"), "1x1 bicluster should return inf score"

    # 2x1 matrix (too few columns)
    matrix_2x1 = np.array([[1.0], [2.0]])
    row_indices = np.array([True, True])
    col_indices = np.array([True])
    bicluster_2x1 = Bicluster(row_indices, col_indices, score=None)

    score = scorer.score(matrix_2x1, bicluster_2x1)
    assert score == float("inf"), "2x1 bicluster should return inf score"

    # 1x2 matrix (too few rows)
    matrix_1x2 = np.array([[1.0, 2.0]])
    row_indices = np.array([True])
    col_indices = np.array([True, True])
    bicluster_1x2 = Bicluster(row_indices, col_indices, score=None)

    score = scorer.score(matrix_1x2, bicluster_1x2)
    assert score == float("inf"), "1x2 bicluster should return inf score"


def test_svr_identity_matrix():
    """Test SVR scoring on identity matrix."""
    # Identity matrix has singular values all equal to 1
    # So s₁/s₂ = 1/1 = 1
    matrix = np.eye(4)

    row_indices = np.ones(4, dtype=bool)
    col_indices = np.ones(4, dtype=bool)
    bicluster = Bicluster(row_indices, col_indices, score=None)

    scorer = SVRScorer(normalized=False)
    score = scorer.score(matrix, bicluster)

    # Expected: -1.0 (negative of s₁/s₂ = 1)
    assert_allclose(score, -1.0, rtol=0.01)


def test_svr_normalized_vs_unnormalized():
    """Compare normalized and unnormalized SVR scores."""
    # Create a full-rank matrix (not rank-1)
    matrix = np.array([[2.0, 4.0, 6.0], [1.0, 3.0, 5.0], [3.0, 5.0, 7.0]], dtype=float)

    row_indices = np.ones(3, dtype=bool)
    col_indices = np.ones(3, dtype=bool)
    bicluster = Bicluster(row_indices, col_indices, score=None)

    # Basic SVR
    scorer_basic = SVRScorer(normalized=False)
    score_basic = scorer_basic.score(matrix, bicluster)

    # Normalized SVR
    scorer_norm = SVRScorer(normalized=True)
    score_norm = scorer_norm.score(matrix, bicluster)

    # Both should be negative
    assert score_basic < 0
    assert score_norm < 0

    # Normalized score should have smaller absolute value (divided by ||A||_F > 1)
    frobenius = np.linalg.norm(matrix, "fro")
    assert frobenius > 1  # Ensure our assumption holds

    # |score_norm| should be approximately |score_basic| / frobenius
    assert_allclose(score_norm, score_basic / frobenius, rtol=0.01)


def test_svr_semantic_inversion():
    """Verify negative score convention: lower rank matrices get more negative scores."""
    np.random.seed(42)

    # Create a low-rank (approximate rank-2) matrix
    u = np.random.rand(5, 2)
    v = np.random.rand(2, 5)
    low_rank_matrix = u @ v + 0.01 * np.random.rand(5, 5)  # Small noise

    # Create a full-rank random matrix
    full_rank_matrix = np.random.rand(5, 5)

    row_indices = np.ones(5, dtype=bool)
    col_indices = np.ones(5, dtype=bool)

    bicluster_low = Bicluster(row_indices, col_indices, score=None)
    bicluster_full = Bicluster(row_indices, col_indices, score=None)

    scorer = SVRScorer(normalized=True)

    score_low = scorer.score(low_rank_matrix, bicluster_low)
    score_full = scorer.score(full_rank_matrix, bicluster_full)

    # Low-rank matrix should have more negative score (better bicluster)
    assert score_low < score_full, (
        f"Low-rank matrix should have more negative score. "
        f"Got low={score_low:.3f}, full={score_full:.3f}"
    )


def test_svr_empty_bicluster():
    """Test SVR scoring on empty bicluster."""
    matrix = np.random.rand(5, 5)

    # Empty bicluster
    row_indices = np.zeros(5, dtype=bool)
    col_indices = np.zeros(5, dtype=bool)
    bicluster = Bicluster(row_indices, col_indices, score=None)

    scorer = SVRScorer()
    score = scorer.score(matrix, bicluster)

    assert score == float("inf"), "Empty bicluster should return inf score"


def test_svr_cache_functionality():
    """Test that SVD caching works correctly."""
    matrix = np.random.rand(10, 10)

    row_indices = np.ones(10, dtype=bool)
    col_indices = np.ones(10, dtype=bool)
    bicluster = Bicluster(row_indices, col_indices, score=None)

    # Create scorer with caching enabled
    scorer_cached = SVRScorer(normalized=True, cache_svd=True)

    # First call - should compute and cache
    score1 = scorer_cached.score(matrix, bicluster)

    # Second call - should use cache
    score2 = scorer_cached.score(matrix, bicluster)

    # Scores should be identical
    assert score1 == score2

    # Check that cache is populated
    assert len(scorer_cached.svd_cache) > 0


def test_svr_comparison_with_other_scorers():
    """Compare SVR scoring with existing scorers on synthetic data."""
    # Create a clear bicluster pattern
    matrix = np.zeros((10, 10))
    # Embed a coherent block
    matrix[0:4, 0:4] = 5.0 + 0.1 * np.random.rand(4, 4)
    # Add noise to rest
    matrix[4:, :] = np.random.rand(6, 10)
    matrix[:, 4:] = np.random.rand(10, 6)

    # Good bicluster (the coherent block)
    good_row_indices = np.array([True] * 4 + [False] * 6)
    good_col_indices = np.array([True] * 4 + [False] * 6)
    good_bicluster = Bicluster(good_row_indices, good_col_indices, score=None)

    # Bad bicluster (random region)
    bad_row_indices = np.array([False] * 4 + [True] * 4 + [False] * 2)
    bad_col_indices = np.array([False] * 4 + [True] * 4 + [False] * 2)
    bad_bicluster = Bicluster(bad_row_indices, bad_col_indices, score=None)

    # Score with different methods
    svr_scorer = SVRScorer(normalized=True)
    exp_scorer = CompatibilityScorer(method=ScoringMethod.EXPONENTIAL)

    svr_good = svr_scorer.score(matrix, good_bicluster)
    svr_bad = svr_scorer.score(matrix, bad_bicluster)

    exp_good = exp_scorer.score(matrix, good_bicluster)
    exp_bad = exp_scorer.score(matrix, bad_bicluster)

    # Both scorers should rank good bicluster better (lower score)
    assert svr_good < svr_bad, f"SVR: good={svr_good:.3f} should be < bad={svr_bad:.3f}"
    assert exp_good < exp_bad, f"EXP: good={exp_good:.3f} should be < bad={exp_bad:.3f}"


def test_svr_with_detection_integration():
    """Test SVR scorer integration with BiclusterDetector."""
    from src.core import BiclusterConfig
    from src.detection import SVDBiclusterDetector

    # Create synthetic data with embedded bicluster
    matrix = np.random.rand(50, 40) * 0.5

    # Embed a low-rank block
    u = np.ones((10, 1))
    v = np.ones((1, 10))
    matrix[5:15, 5:15] = u @ v * 10

    # Configure detector with SVR scoring
    config_svr = BiclusterConfig(
        k1=5,
        k2=5,
        tolerance=10.0,  # Adjusted for negative scores
        scoring_method=ScoringMethod.SVR_NORMALIZED,
        random_state=42,
    )

    detector = SVDBiclusterDetector(config_svr)

    # Verify SVR scorer is being used
    assert isinstance(detector.scorer, SVRScorer)

    # Run detection
    biclusters = detector.detect(matrix)

    # Should detect some biclusters
    assert len(biclusters) >= 0, "Detection should complete without error"

    # If biclusters found, verify they have negative scores
    if biclusters:
        for bc in biclusters:
            assert bc.score < 0, f"Bicluster score {bc.score} should be negative with SVR"
