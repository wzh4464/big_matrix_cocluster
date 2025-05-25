"""
File: /bicluster.py
Created Date: Wednesday August 30th 2023
Author: Zihan Wu
-----
Last Modified: Monday, 30th October 2023 4:55:30 pm
Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
-----
HISTORY:
Date      		By   	Comments
----------		------	---------------------------------------------------------

30-10-2023		Zihan	Add idxtoLabel
"""

from __future__ import annotations
from dataclasses import dataclass, field, fields
from typing import Tuple, Dict, Any, Optional, Union
import numpy as np
from numpy.typing import NDArray
import uuid

# Assuming Matrix, BoolArray, IntArray are defined in core.py or a common types file
# For direct use here, let's redefine or import them if they are not globally available.
# If they are in core.py, this would be: from .core import Matrix, BoolArray, IntArray
# To keep bicluster.py self-contained for its definition or importable by core.py without circularity:
Matrix = NDArray[np.floating]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.integer]

# Define the prefix for Bicluster IDs
BICLUSTER_ID_PREFIX = "bc_"


@dataclass
class Bicluster:
    """Modern bicluster representation with enhanced functionality."""

    row_indices: BoolArray
    col_indices: BoolArray
    score: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    id: str = field(
        default_factory=lambda: f"{BICLUSTER_ID_PREFIX}{uuid.uuid4().hex[:8]}"
    )

    def __post_init__(self):
        """Validate bicluster after initialization."""
        if (
            not isinstance(self.row_indices, np.ndarray)
            or self.row_indices.dtype != np.bool_
        ):
            raise TypeError("row_indices must be a boolean NumPy array.")
        if (
            not isinstance(self.col_indices, np.ndarray)
            or self.col_indices.dtype != np.bool_
        ):
            raise TypeError("col_indices must be a boolean NumPy array.")
        if self.score is not None and not isinstance(self.score, (int, float)):
            raise TypeError("score must be a float or None.")
        if self.metadata is not None and not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dictionary or None.")
        if len(self.row_indices) == 0 or len(self.col_indices) == 0:
            # This check might be too restrictive if an empty matrix can produce empty biclusters
            # However, a bicluster usually implies selection from a matrix, so non-empty indices are typical.
            # Consider if a bicluster can meaningfully exist with zero rows/cols selected from a larger matrix.
            pass  # Allow empty indices if they represent selection from a larger matrix
        if (
            np.sum(self.row_indices) == 0
            and np.sum(self.col_indices) == 0
            and len(self.row_indices) > 0
            and len(self.col_indices) > 0
        ):
            # This is a bicluster that selects no rows and no columns from a matrix of certain dimensions
            pass  # This is a valid "empty" bicluster in some contexts
        elif np.sum(self.row_indices) == 0 or np.sum(self.col_indices) == 0:
            # A bicluster that selects some rows but no columns, or some columns but no rows.
            # This typically means its area is 0. Such biclusters might be filtered out later.
            pass

    @property
    def row_labels(self) -> IntArray:
        """Get row indices as integer labels."""
        return np.where(self.row_indices)[0].astype(np.int64)

    @property
    def col_labels(self) -> IntArray:
        """Get column indices as integer labels."""
        return np.where(self.col_indices)[0].astype(np.int64)

    @property
    def shape(self) -> tuple[int, int]:
        """Get shape of the bicluster (number of selected rows, number of selected columns)."""
        return (int(np.sum(self.row_indices)), int(np.sum(self.col_indices)))

    @property
    def size(self) -> int:
        """Get total number of elements in the bicluster (selected rows * selected columns)."""
        return self.shape[0] * self.shape[1]

    def overlaps_with(self, other: Bicluster, threshold: float = 0.25) -> bool:
        """Check if this bicluster overlaps with another."""
        if (
            self.row_indices.shape != other.row_indices.shape
            or self.col_indices.shape != other.col_indices.shape
        ):
            # Biclusters are from matrices of different dimensions or incompatible index arrays
            # This check assumes row_indices and col_indices are masks for the *same original matrix dimensions*
            raise ValueError(
                "Cannot compare biclusters from different underlying matrix shapes or with incompatible index dimensions."
            )

        jaccard = self.jaccard_index(other)
        return jaccard > threshold

    def intersection_area(self, other: Bicluster) -> int:
        """Calculate the area of intersection with another bicluster."""
        if (
            self.row_indices.shape != other.row_indices.shape
            or self.col_indices.shape != other.col_indices.shape
        ):
            raise ValueError(
                "Cannot compute intersection for biclusters with incompatible index dimensions."
            )

        row_intersection = np.sum(self.row_indices & other.row_indices)
        col_intersection = np.sum(self.col_indices & other.col_indices)
        return int(row_intersection * col_intersection)

    def union_area(self, other: Bicluster) -> int:
        """Calculate the area of the union with another bicluster."""
        # Union Area = Area(A) + Area(B) - IntersectionArea(A, B)
        return int(self.size + other.size - self.intersection_area(other))

    def jaccard_index(self, other: Bicluster) -> float:
        """Calculate Jaccard index (intersection over union area) with another bicluster."""
        intersection = self.intersection_area(other)
        union = self.union_area(other)
        return float(intersection / union) if union > 0 else 0.0

    # The provided code had intersection_ratio. Jaccard Index is a more standard term for IoU.
    # If `intersection_ratio` means something different (e.g., related to only one of the biclusters sizes),
    # it should be clarified. Assuming it was intended to be Jaccard Index / IoU.
    def intersection_ratio(self, other: Bicluster) -> float:
        """Calculate the ratio of the intersection area to the smaller of the two biclusters' areas."""
        intersect_area = self.intersection_area(other)
        if intersect_area == 0:
            return 0.0
        smaller_area = min(self.size, other.size)
        if smaller_area == 0:
            return 0.0  # Avoid division by zero if one bicluster is empty
        return float(intersect_area / smaller_area)

    def extract_submatrix(self, matrix: Matrix) -> Matrix:
        """Extract the submatrix corresponding to this bicluster.
        Assumes row_indices and col_indices are masks for the given matrix.
        """
        if matrix.ndim != 2:
            raise ValueError("Original matrix must be 2-dimensional.")
        if (
            len(self.row_indices) > matrix.shape[0]
            or len(self.col_indices) > matrix.shape[1]
        ):
            # This indicates the boolean masks are larger than the matrix they are applied to.
            # It could be an error in how masks were created or if a different matrix is passed.
            raise ValueError(
                f"Indices length ({len(self.row_indices)}, {len(self.col_indices)}) "
                f"exceeds matrix dimensions ({matrix.shape[0]}, {matrix.shape[1]})."
            )
        # Ensure indices are not out of bounds for the current matrix if they are shorter
        # This is generally handled if row_indices and col_indices are always full-size masks.
        # If they can be shorter, padding or alignment is needed.
        # Assuming they are full-size masks corresponding to the original matrix dimensions.

        # Ensure masks are not longer than the matrix dimensions they are applied to
        # This check is important if the Bicluster object can be applied to different matrices.
        current_row_indices = self.row_indices[: matrix.shape[0]]
        current_col_indices = self.col_indices[: matrix.shape[1]]

        # Extract rows and then columns
        sub_matrix_rows = matrix[current_row_indices, :]
        sub_matrix = sub_matrix_rows[:, current_col_indices]
        return sub_matrix

    def to_dict(self) -> Dict[str, Any]:
        """Convert bicluster to dictionary for serialization."""
        return {
            "id": self.id,
            "row_labels": [int(x) for x in self.row_labels],  # Store integer labels
            "col_labels": [int(x) for x in self.col_labels],  # Store integer labels
            "score": self.score,
            "shape": self.shape,  # Shape of the selected submatrix
            "metadata": self.metadata,
            # Store original dimensions for robust reconstruction from labels
            "original_row_dim": len(self.row_indices),
            "original_col_dim": len(self.col_indices),
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        original_row_dim: Optional[int] = None,
        original_col_dim: Optional[int] = None,
    ) -> Bicluster:
        """Create a Bicluster instance from a dictionary.
        Requires original matrix dimensions to reconstruct boolean masks if not provided in dict.
        """
        row_labels = np.array(data["row_labels"])
        col_labels = np.array(data["col_labels"])

        # Determine original dimensions
        # Priority: explicit args > dict values > max_label + 1
        final_row_dim = (
            original_row_dim
            if original_row_dim is not None
            else data.get("original_row_dim")
        )
        final_col_dim = (
            original_col_dim
            if original_col_dim is not None
            else data.get("original_col_dim")
        )

        if final_row_dim is None:
            final_row_dim = np.max(row_labels) + 1 if len(row_labels) > 0 else 0
        if final_col_dim is None:
            final_col_dim = np.max(col_labels) + 1 if len(col_labels) > 0 else 0

        row_indices = np.zeros(final_row_dim, dtype=bool)
        if len(row_labels) > 0:
            row_indices[row_labels] = True

        col_indices = np.zeros(final_col_dim, dtype=bool)
        if len(col_labels) > 0:
            col_indices[col_labels] = True

        # Get ID from data or generate a new one if not present
        bicluster_id = data.get("id")
        score = data.get("score")
        metadata = data.get("metadata", {})

        if bicluster_id is not None:
            return cls(
                row_indices=row_indices,
                col_indices=col_indices,
                score=score,
                metadata=metadata,
                id=bicluster_id,
            )
        else:
            # Let the default_factory generate the id
            return cls(
                row_indices=row_indices,
                col_indices=col_indices,
                score=score,
                metadata=metadata,
            )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Bicluster):
            return NotImplemented
        # If IDs are explicitly set and different, they are different objects for sure.
        # If one id is None (e.g. not yet set) or they are the same, then compare content.
        if self.id is not None and other.id is not None and self.id != other.id:
            return False  # This was the original primary check that caused issues with mock comparison

        # If IDs are the same, or one/both are None, then fall back to content comparison.
        # This makes equality more about content if IDs aren't decisive.
        if (
            self.id == other.id and self.id is not None
        ):  # if IDs match and are not None, they are equal
            return True

        # Fallback to content comparison if IDs don't provide a quick answer
        if (
            self.row_indices.shape != other.row_indices.shape
            or self.col_indices.shape != other.col_indices.shape
        ):
            return False

        return (
            np.array_equal(self.row_indices, other.row_indices)
            and np.array_equal(self.col_indices, other.col_indices)
            and self.score == other.score
            # Metadata comparison can be tricky, decide if it's part of equality
            # For now, let's assume metadata equality is also required if IDs don't match.
            and self.metadata == other.metadata
        )

    def __hash__(self) -> int:
        # Make hashable based on content if ID is not always the primary key for equality.
        # Warning: If Bicluster objects are mutable and used in sets/dicts, this can be problematic.
        # For now, assuming ID is mostly unique or content defines it.
        return hash(
            (
                self.id,  # Include id in hash
                self.row_indices.tobytes(),
                self.col_indices.tobytes(),
                self.score,
                # Making metadata hashable is complex; common practice is to exclude mutable dicts
                # or convert to a frozenset of items if metadata structure is stable.
                # tuple(sorted(self.metadata.items())) if self.metadata else None
                # For simplicity, if metadata is part of equality, it makes hashing harder unless IDs are primary.
            )
        )


if __name__ == "__main__":
    # test bicluster
    bi = Bicluster(np.array([True, False, True]), np.array([True, False, False]), 1.0)
    print(bi.row_labels)
    print(bi.col_labels)
