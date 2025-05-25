import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
from big_matrix_cocluster.bicluster import Bicluster, BICLUSTER_ID_PREFIX
from big_matrix_cocluster.core import Matrix # Assuming Matrix is an alias for np.ndarray

# Constants for testing
R_IDX_A = np.array([True, True, False, False, True])
C_IDX_A = np.array([False, True, True, False])
SCORE_A = 0.8
META_A = {'name': 'A'}

R_IDX_B = np.array([False, True, True, True, False])
C_IDX_B = np.array([True, False, True, True])
SCORE_B = 0.6
META_B = {'name': 'B'}

# Overlapping with A
R_IDX_C = np.array([True, True, False, True, False]) # 3 rows
C_IDX_C = np.array([False, True, False, False])      # 1 col
SCORE_C = 0.9

# No overlap with A
R_IDX_D = np.array([False, False, False, True, True])
C_IDX_D = np.array([True, False, False, False])


@pytest.fixture
def bicluster_a() -> Bicluster:
    return Bicluster(R_IDX_A.copy(), C_IDX_A.copy(), SCORE_A, META_A.copy())

@pytest.fixture
def bicluster_b() -> Bicluster:
    return Bicluster(R_IDX_B.copy(), C_IDX_B.copy(), SCORE_B, META_B.copy()) #Intentionally use META_B to test equality more deeply

@pytest.fixture
def bicluster_c() -> Bicluster: # Overlaps with A only on rows, different col intersection behavior
    return Bicluster(row_indices=R_IDX_C.copy(), col_indices=C_IDX_C.copy(), score=SCORE_C)

@pytest.fixture
def bicluster_d() -> Bicluster: # No overlap with A
    return Bicluster(row_indices=R_IDX_D.copy(), col_indices=C_IDX_D.copy(), score=0.5)
    
@pytest.fixture
def sample_matrix() -> Matrix:
    # Matrix compatible with R_IDX_A/B/C (5 rows), C_IDX_A/B/C (4 cols)
    return np.arange(20).reshape((5, 4)).astype(np.float64)

def test_bicluster_creation_and_id(bicluster_a: Bicluster):
    assert bicluster_a.score == SCORE_A
    assert bicluster_a.metadata == META_A
    assert bicluster_a.id.startswith(BICLUSTER_ID_PREFIX)
    assert len(bicluster_a.id) == len(BICLUSTER_ID_PREFIX) + 8 # Example: bc_1234abcd

    bc_no_meta = Bicluster(R_IDX_A, C_IDX_A, score=0.5)
    assert bc_no_meta.metadata is None

    bc_default_score_meta = Bicluster(R_IDX_A, C_IDX_A)
    assert bc_default_score_meta.score is None
    assert bc_default_score_meta.metadata is None

def test_bicluster_properties(bicluster_a: Bicluster):
    assert bicluster_a.shape == (3, 2) # 3 rows, 2 cols selected
    assert bicluster_a.size == 6
    assert_array_equal(bicluster_a.row_labels, np.array([0, 1, 4]))
    assert_array_equal(bicluster_a.col_labels, np.array([1, 2]))

def test_bicluster_empty_properties():
    empty_rows = Bicluster(np.zeros(5, dtype=bool), C_IDX_A.copy())
    assert empty_rows.shape == (0,2)
    assert empty_rows.size == 0
    assert_array_equal(empty_rows.row_labels, [])

    empty_cols = Bicluster(R_IDX_A.copy(), np.zeros(4, dtype=bool))
    assert empty_cols.shape == (3,0)
    assert empty_cols.size == 0
    assert_array_equal(empty_cols.col_labels, [])

def test_bicluster_validation():
    with pytest.raises(TypeError, match="row_indices must be a boolean NumPy array"):
        Bicluster(np.array([1,2,3]), C_IDX_A, 0.5)
    with pytest.raises(TypeError, match="col_indices must be a boolean NumPy array"):
        Bicluster(R_IDX_A, np.array(["a","b"]), 0.5)
    with pytest.raises(TypeError, match="score must be a float or None"):
        Bicluster(R_IDX_A, C_IDX_A, score="high") # type: ignore
    with pytest.raises(TypeError, match="metadata must be a dictionary or None"):
        Bicluster(R_IDX_A, C_IDX_A, metadata=[1,2,3]) # type: ignore


def test_extract_submatrix(bicluster_a: Bicluster, sample_matrix: Matrix):
    submatrix = bicluster_a.extract_submatrix(sample_matrix)
    expected_submatrix = np.array([
        [1, 2],   # row 0, cols 1,2 from sample_matrix
        [5, 6],   # row 1, cols 1,2
        [17, 18]  # row 4, cols 1,2
    ])
    assert_array_equal(submatrix, expected_submatrix)

def test_extract_submatrix_empty(sample_matrix: Matrix):
    bc_empty_rows = Bicluster(np.zeros(5, dtype=bool), C_IDX_A.copy())
    sub_empty_r = bc_empty_rows.extract_submatrix(sample_matrix)
    assert sub_empty_r.shape == (0,2)

    bc_empty_cols = Bicluster(R_IDX_A.copy(), np.zeros(4, dtype=bool))
    sub_empty_c = bc_empty_cols.extract_submatrix(sample_matrix)
    assert sub_empty_c.shape == (3,0)

def test_extract_submatrix_errors(bicluster_a: Bicluster, sample_matrix: Matrix):
    # Matrix with wrong dimensions for masks
    wrong_dim_matrix_rows = np.random.rand(3, 4) # bicluster_a has 5 row indices
    with pytest.raises(ValueError, match="Indices length .* exceeds matrix dimensions"):
        bicluster_a.extract_submatrix(wrong_dim_matrix_rows)

    wrong_dim_matrix_cols = np.random.rand(5, 3) # bicluster_a has 4 col indices
    with pytest.raises(ValueError, match="Indices length .* exceeds matrix dimensions"):
        bicluster_a.extract_submatrix(wrong_dim_matrix_cols)

    # Non 2D matrix
    non_2d_matrix = np.random.rand(5)
    with pytest.raises(ValueError, match="Original matrix must be 2-dimensional"):
        bicluster_a.extract_submatrix(non_2d_matrix)


def test_to_dict_and_from_dict(bicluster_a: Bicluster):
    data_dict = bicluster_a.to_dict()
    assert data_dict['row_labels'] == [0, 1, 4]
    assert data_dict['col_labels'] == [1, 2]
    assert data_dict['score'] == SCORE_A
    assert data_dict['shape'] == (3, 2)
    assert data_dict['metadata'] == META_A
    assert data_dict['original_row_dim'] == len(R_IDX_A)
    assert data_dict['original_col_dim'] == len(C_IDX_A)
    assert data_dict['id'] == bicluster_a.id

    reconstructed_bicluster = Bicluster.from_dict(data_dict)
    assert_array_equal(reconstructed_bicluster.row_indices, bicluster_a.row_indices)
    assert_array_equal(reconstructed_bicluster.col_indices, bicluster_a.col_indices)
    assert reconstructed_bicluster.score == bicluster_a.score
    assert reconstructed_bicluster.metadata == bicluster_a.metadata
    assert reconstructed_bicluster.shape == bicluster_a.shape
    assert reconstructed_bicluster.id == bicluster_a.id # Check ID reconstruction

    # Test from_dict with explicit original dimensions and no id
    smaller_dict = {'row_labels': [0,1], 'col_labels': [0], 'score': 0.5, 'original_row_dim': 3, 'original_col_dim':2}
    bc_from_small = Bicluster.from_dict(smaller_dict) # No id passed
    assert bc_from_small.id is not None
    assert bc_from_small.id.startswith(BICLUSTER_ID_PREFIX)
    assert_array_equal(bc_from_small.row_labels, [0,1])
    assert_array_equal(bc_from_small.col_labels, [0])
    assert len(bc_from_small.row_indices) == 3
    assert len(bc_from_small.col_indices) == 2

    # Test from_dict with minimal data (no original dimensions, rely on max label)
    minimal_dict = {'row_labels': [0,2], 'col_labels': [1], 'score': 0.1, 'id': 'test_id_min'}
    bc_from_min = Bicluster.from_dict(minimal_dict)
    assert len(bc_from_min.row_indices) == 3 # max_label 2 + 1
    assert len(bc_from_min.col_indices) == 2 # max_label 1 + 1
    assert bc_from_min.id == 'test_id_min'


def test_bicluster_equality(bicluster_a: Bicluster):
    bc_a_copy = Bicluster(R_IDX_A.copy(), C_IDX_A.copy(), SCORE_A, META_B.copy(), id=bicluster_a.id)
    assert bc_a_copy == bicluster_a
    assert hash(bc_a_copy) == hash(bicluster_a)

    # Different ID, same content -> should be False by current __eq__ logic (ID first)
    # To test content equality for different ID objects, __eq__ would need to handle it.
    # For now, if IDs are different, they are not equal.
    bc_a_diff_id = Bicluster(R_IDX_A.copy(), C_IDX_A.copy(), SCORE_A, META_B.copy(), id="some_other_id")
    assert bc_a_diff_id != bicluster_a

    bc_b = Bicluster(R_IDX_B.copy(), C_IDX_B.copy(), SCORE_B, META_B.copy())
    assert bicluster_a != bc_b
    assert bc_b != "not a bicluster"

    # Test with different content but same ID (should not happen with default_factory, but testable)
    # To make this test meaningful, we would need to bypass the default_factory for ID or have a way to set it post-init for testing.
    # However, if IDs are the same, __eq__ returns True. If IDs differ, then content is checked.
    # Let's check content difference when IDs are different.
    bc_a_content_diff = Bicluster(R_IDX_A.copy(), C_IDX_A.copy(), score=0.1, metadata=META_B.copy(), id="another_unique_id")
    bc_a_ref_for_content = Bicluster(R_IDX_A.copy(), C_IDX_A.copy(), score=SCORE_A, metadata=META_B.copy(), id="yet_another_id")
    assert bc_a_content_diff != bc_a_ref_for_content # Score differs

def test_bicluster_overlaps_and_jaccard(bicluster_a: Bicluster, bicluster_b: Bicluster):
    # R_A = {0,1,4} (3), C_A = {1,2} (2). Size_A = 6.
    # R_B = {1,2,3} (3), C_B = {0,2,3} (3). Size_B = 9.
    # Row Intersection: {1} (count=1)
    # Col Intersection: {2} (count=1)
    # Intersection Area: 1 * 1 = 1
    assert bicluster_a.intersection_area(bicluster_b) == 1
    
    # Union Area: Size_A + Size_B - Intersection = 6 + 9 - 1 = 14
    assert bicluster_a.union_area(bicluster_b) == 14
    
    # Jaccard Index: Intersection / Union = 1 / 14
    expected_jaccard = 1 / 14
    assert_allclose(bicluster_a.jaccard_index(bicluster_b), expected_jaccard)
    
    # Test overlaps_with based on this Jaccard index
    assert not bicluster_a.overlaps_with(bicluster_b) # Default threshold 0.25; 1/14 is ~0.0714
    assert bicluster_a.overlaps_with(bicluster_b, threshold=0.05)
    assert not bicluster_a.overlaps_with(bicluster_b, threshold=0.1)

    # Test with non-overlapping bicluster
    no_overlap_rows = np.array([False, False, False, False, False]) # Ensure different length if needed or handle in Bicluster
    # For this test, ensure R_IDX_A and no_overlap_rows have same length for valid comparison
    # Assuming Bicluster expects full dimension boolean arrays
    if len(no_overlap_rows) != len(R_IDX_A):
        no_overlap_rows = np.zeros(len(R_IDX_A), dtype=bool)
        
    bc_no_overlap = Bicluster(no_overlap_rows, C_IDX_A.copy(), 0.1)
    assert bicluster_a.intersection_area(bc_no_overlap) == 0
    assert bicluster_a.jaccard_index(bc_no_overlap) == 0.0
    assert not bicluster_a.overlaps_with(bc_no_overlap)

    # Test with itself
    assert bicluster_a.intersection_area(bicluster_a) == bicluster_a.size
    assert_allclose(bicluster_a.jaccard_index(bicluster_a), 1.0)
    assert bicluster_a.overlaps_with(bicluster_a, threshold=0.5)

def test_bicluster_with_no_selected_rows_or_cols(bicluster_a: Bicluster):
    bc_no_rows = Bicluster(np.zeros(5, dtype=bool), C_IDX_A.copy(), 0.1)
    assert bc_no_rows.shape == (0, 2)
    assert bc_no_rows.size == 0
    assert_array_equal(bc_no_rows.row_labels, [])
    assert_array_equal(bc_no_rows.col_labels, [1, 2])

    bc_no_cols = Bicluster(R_IDX_A.copy(), np.zeros(4, dtype=bool), 0.1)
    assert bc_no_cols.shape == (3, 0)
    assert bc_no_cols.size == 0
    assert_array_equal(bc_no_cols.row_labels, [0, 1, 4])
    assert_array_equal(bc_no_cols.col_labels, [])

    bc_none_selected = Bicluster(np.zeros(5, dtype=bool), np.zeros(4, dtype=bool), 0.1)
    assert bc_none_selected.shape == (0, 0)
    assert bc_none_selected.size == 0
    assert_array_equal(bc_none_selected.row_labels, [])
    assert_array_equal(bc_none_selected.col_labels, [])

    # Test Jaccard with empty bicluster
    # bicluster_a is now correctly passed as a Bicluster instance
    assert bc_none_selected.jaccard_index(bicluster_a) == 0.0 
    assert bicluster_a.jaccard_index(bc_none_selected) == 0.0

def test_intersection_ratio(bicluster_a: Bicluster, bicluster_b: Bicluster):
    # A.size = 6, B.size = 9, Intersection = 1
    # Smaller area = min(6,9) = 6
    # Ratio = 1/6
    assert_allclose(bicluster_a.intersection_ratio(bicluster_b), 1/6)
    assert_allclose(bicluster_b.intersection_ratio(bicluster_a), 1/6) # Symmetrical if using min area

    # Test with itself
    assert_allclose(bicluster_a.intersection_ratio(bicluster_a), 1.0)

    # Test with no overlap
    no_overlap_rows = np.zeros(len(R_IDX_A), dtype=bool)
    bc_no_overlap = Bicluster(no_overlap_rows, C_IDX_A.copy(), 0.1)
    assert bicluster_a.intersection_ratio(bc_no_overlap) == 0.0

    # Test with one bicluster being empty
    bc_empty = Bicluster(np.zeros(len(R_IDX_A), dtype=bool), np.zeros(len(C_IDX_A), dtype=bool), 0.1)
    assert bc_empty.size == 0
    assert bicluster_a.intersection_ratio(bc_empty) == 0.0
    assert bc_empty.intersection_ratio(bicluster_a) == 0.0

# Example of how to run this with pytest:
# Ensure pytest and numpy are installed: pip install pytest numpy
# Navigate to the directory containing this file and your bicluster.py
# Run: pytest test_bicluster.py 