# scoring.py
# This file will contain scoring algorithms.
from __future__ import annotations
from typing import Protocol
import numpy as np
from numpy.typing import NDArray
import logging

# Assuming ScoringMethod and Bicluster are in core.py and bicluster.py respectively:
from .core import ScoringMethod, Matrix # Assuming Matrix is defined in core.py
from .bicluster import Bicluster # Assuming Bicluster is defined in bicluster.py

class ScoringStrategy(Protocol):
    """Protocol for bicluster scoring strategies."""
    
    def score(self, matrix: Matrix, bicluster: Bicluster) -> float:
        """Calculate score for a bicluster."""
        ...


class CompatibilityScorer(ScoringStrategy):
    """Compatibility-based scoring implementation."""
    
    def __init__(self, method: ScoringMethod = ScoringMethod.EXPONENTIAL, noise_level: float = 1e-6):
        self.method = method
        self.noise_level = noise_level
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def score(self, matrix: Matrix, bicluster: Bicluster) -> float:
        """Calculate compatibility score for bicluster."""
        bicluster_id_for_log = bicluster.id
        if bicluster.metadata and 'id' in bicluster.metadata: # Prefer metadata ID if explicitly set for some reason
            bicluster_id_for_log = bicluster.metadata['id']
        
        try:
            submatrix = bicluster.extract_submatrix(matrix)
            if submatrix.size == 0:
                self.logger.debug(f"Bicluster {bicluster_id_for_log} results in empty submatrix. Score is inf.")
                return float('inf')
            if submatrix.shape[0] < 2 or submatrix.shape[1] < 2:
                self.logger.debug(f"Bicluster {bicluster_id_for_log} submatrix too small ({submatrix.shape}). Score is inf.")
                return float('inf')
            return self._calculate_compatibility(submatrix)
        except ValueError as ve:
            self.logger.warning(f"ValueError for bicluster {bicluster_id_for_log}: {ve}. Returning inf score.")
            return float('inf')
        except Exception as e:
            self.logger.error(f"Unexpected error for bicluster {bicluster_id_for_log}: {e}", exc_info=True)
            return float('inf')
    
    def _calculate_compatibility(self, submatrix: Matrix) -> float:
        """Calculate compatibility score for submatrix."""
        # Add small noise to prevent issues with constant rows/cols in Pearson or perfect similarity
        noisy_submatrix = submatrix + self.noise_level * np.random.randn(*submatrix.shape)

        if self.method == ScoringMethod.PEARSON:
            return self._pearson_compatibility(noisy_submatrix)
        elif self.method == ScoringMethod.EXPONENTIAL:
            return self._exponential_compatibility(noisy_submatrix)
        elif self.method == ScoringMethod.COMPATIBILITY:
            # This was an option, but not fully defined; defaulting to exponential or needs implementation
            self.logger.warning("COMPATIBILITY scoring not fully implemented, using EXPONENTIAL.")
            return self._exponential_compatibility(noisy_submatrix)
        else:
            raise ValueError(f"Unknown scoring method: {self.method}")
    
    def _pearson_compatibility(self, submatrix: Matrix) -> float:
        num_rows, num_cols = submatrix.shape
        if num_rows < 2 or num_cols < 2:
            return float('inf') # Pearson requires at least 2x2

        # Row compatibility (average Pearson correlation between row pairs)
        row_corr_sum = 0
        for i in range(num_rows):
            for j in range(i + 1, num_rows):
                # Correlation coef returns NaN if one vector is constant. 
                # The noise addition should mitigate this, but handle just in case.
                corr = np.corrcoef(submatrix[i, :], submatrix[j, :])[0, 1]
                row_corr_sum += np.abs(corr) if not np.isnan(corr) else 0
        
        num_row_pairs = num_rows * (num_rows - 1) / 2
        avg_row_corr = row_corr_sum / num_row_pairs if num_row_pairs > 0 else 1.0

        # Column compatibility (average Pearson correlation between column pairs)
        col_corr_sum = 0
        for i in range(num_cols):
            for j in range(i + 1, num_cols):
                corr = np.corrcoef(submatrix[:, i], submatrix[:, j])[0, 1]
                col_corr_sum += np.abs(corr) if not np.isnan(corr) else 0

        num_col_pairs = num_cols * (num_cols - 1) / 2
        avg_col_corr = col_corr_sum / num_col_pairs if num_col_pairs > 0 else 1.0

        # Score is 1 minus the average of row and column correlations (closer to 0 is better)
        return float(1.0 - (avg_row_corr + avg_col_corr) / 2.0)

    def _exponential_compatibility(self, submatrix: Matrix) -> float:
        num_rows, num_cols = submatrix.shape
        if num_rows < 1 or num_cols < 1: # Can operate on 1xN or Nx1, though score_helper handles < 2
            return float('inf')

        # Row compatibility
        row_sim_matrix = np.zeros((num_rows, num_rows))
        if num_rows >= 2:
            for i in range(num_rows):
                for j in range(i + 1, num_rows):
                    # Using squared Euclidean distance, then exp transform
                    dist_sq = np.sum((submatrix[i, :] - submatrix[j, :])**2)
                    row_sim_matrix[i, j] = row_sim_matrix[j, i] = np.exp(-dist_sq / num_cols)
            row_scores = self._score_helper(num_rows, row_sim_matrix)
            avg_row_score = np.mean(row_scores) if len(row_scores) > 0 else 0.0
        else:
            avg_row_score = 0.0 # Perfect score if only one row (nothing to compare to)

        # Column compatibility
        col_sim_matrix = np.zeros((num_cols, num_cols))
        if num_cols >= 2:
            for i in range(num_cols):
                for j in range(i + 1, num_cols):
                    dist_sq = np.sum((submatrix[:, i] - submatrix[:, j])**2)
                    col_sim_matrix[i, j] = col_sim_matrix[j, i] = np.exp(-dist_sq / num_rows)
            col_scores = self._score_helper(num_cols, col_sim_matrix)
            avg_col_score = np.mean(col_scores) if len(col_scores) > 0 else 0.0
        else:
            avg_col_score = 0.0 # Perfect score if only one col

        # Final score is the average of row and column scores
        # If one dimension is 1, its score is 0, so it relies on the other dimension's score.
        if num_rows < 2 and num_cols < 2: # e.g. 1x1 matrix, should be perfect score by this logic
            return 0.0
        if num_rows < 2: # e.g. 1xN matrix, score is avg_col_score
            return float(avg_col_score)
        if num_cols < 2: # e.g. Nx1 matrix, score is avg_row_score
            return float(avg_row_score)
            
        return float((avg_row_score + avg_col_score) / 2.0)

    def _score_helper(self, length: int, similarity_matrix: Matrix) -> np.ndarray:
        if length < 2:
            return np.array([float('inf')], dtype=np.float64) 
        
        sum_similarity = np.sum(similarity_matrix, axis=1)
        mean_similarity_per_item = sum_similarity / (length - 1)
        scores = 1.0 - mean_similarity_per_item
        return scores.astype(np.float64) 