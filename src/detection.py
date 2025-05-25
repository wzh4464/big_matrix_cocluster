# detection.py
# This file will contain detection algorithms.
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import logging

# Assuming BiclusterConfig, Matrix, Bicluster, ScoringMethod, ClusteringMethod are defined in .core and .bicluster
from .core import (
    BiclusterConfig,
    Matrix,
    ClusteringMethod,
    ScoringMethod,
)  # Add ScoringMethod
from .bicluster import (
    Bicluster,
    IntArray,
)  # IntArray might be needed from .core or defined in .bicluster
from .scoring import CompatibilityScorer  # Import the specific scorer


class BiclusterDetector(ABC):
    """Abstract base class for bicluster detection algorithms."""

    def __init__(self, config: BiclusterConfig):
        self.config = config
        # Pass the scoring_method from config to the Scorer
        self.scorer = CompatibilityScorer(config.scoring_method)
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def detect(self, matrix: Matrix, **kwargs) -> List[Bicluster]:
        """Detect biclusters in the given matrix."""
        pass

    def validate_bicluster(
        self,
        matrix: Matrix,
        bicluster: Bicluster,
        min_rows: int,
        min_cols: int,
        tolerance: float,
    ) -> bool:
        """Validate if a bicluster meets quality criteria."""
        # Score is calculated. Lower score is better.
        score = self.scorer.score(matrix, bicluster)
        bicluster.score = score  # Store the calculated score in the bicluster object

        # Validate score against tolerance
        # Validate size (e.g., must have at least 2x2 = 4 elements)
        # The problem description implies bicluster.size is num_rows * num_cols
        # A common minimum is 2 rows and 2 columns.
        meets_size_criteria = (
            bicluster.shape[0] >= min_rows and bicluster.shape[1] >= min_cols
        )

        return score < tolerance and meets_size_criteria


class SVDBiclusterDetector(BiclusterDetector):
    """SVD-based bicluster detection implementation."""

    def __init__(self, config: BiclusterConfig):
        super().__init__(config)
        # Cache for SVD results: key is hash_of_matrix_data + shape, value is (U, S, Vh)
        self.svd_cache: Dict[str, Tuple[Matrix, NDArray, Matrix]] = {}

    def detect(
        self,
        matrix: Matrix,
        n_iterations: Optional[int] = None,
        n_clusters_rows: Optional[int] = None,  # Renamed from n_clusters to be specific
        n_clusters_cols: Optional[int] = None,
        n_svd_components: Optional[int] = None,
        min_rows: Optional[int] = None,
        min_cols: Optional[int] = None,
        max_overlap: Optional[float] = None,
        tolerance: Optional[float] = None,
        # n_biclusters_to_find is harder to directly integrate here as detection is generative
    ) -> List[Bicluster]:

        # Use overridden parameters if provided, else use from config
        cfg_k1 = (
            n_iterations if n_iterations is not None else self.config.k1
        )  # k1 is iterations for SVD refinement (not directly used like this now)
        # For now, map n_iterations to k1 if provided for row clustering config.
        cfg_k2_rows = (
            n_clusters_rows if n_clusters_rows is not None else self.config.k1
        )  # Default k1 for row clusters
        cfg_k2_cols = (
            n_clusters_cols if n_clusters_cols is not None else self.config.k2
        )  # Default k2 for col clusters
        cfg_n_svd = n_svd_components  # This will be passed to _compute_svd
        cfg_min_rows = min_rows if min_rows is not None else self.config.min_rows
        cfg_min_cols = min_cols if min_cols is not None else self.config.min_cols
        cfg_max_overlap = (
            max_overlap if max_overlap is not None else self.config.max_overlap
        )
        cfg_tolerance = tolerance if tolerance is not None else self.config.tolerance

        self.logger.info(
            f"Starting bicluster detection on {matrix.shape} matrix using {self.config.clustering_method.value} and {self.config.scoring_method.value}"
        )

        if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 2:
            self.logger.warning(
                "Matrix is too small for SVD-based biclustering. Returning empty list."
            )
            return []

        # Perform SVD
        try:
            U, S, Vh = self._compute_svd(matrix, n_components_override=cfg_n_svd)
        except Exception as e:
            self.logger.error(f"SVD computation failed: {e}. Returning empty list.")
            return []

        # Determine number of components for clustering based on k1, k2 from config
        # k1 affects row clustering (U), k2 affects column clustering (Vh.T or V)
        # The features for clustering are derived from singular vectors scaled by singular values.
        # For row features: U_k = U[:, :k] @ diag(S[:k]) or similar
        # For col features: V_k = Vh.T[:, :k] @ diag(S[:k]) or (diag(S[:k]) @ Vh[:k, :]).T

        # The original code used U @ np.diag(S) for rows and (np.diag(S) @ Vh).T for columns.
        # This uses all available singular values/vectors from TruncatedSVD.
        # It might be better to select a number of components for U and Vh based on k1/k2 or variance explained.
        # For now, follow the provided structure but ensure k for clustering is appropriate.

        num_svd_components = S.shape[0]
        if num_svd_components == 0:
            self.logger.warning("SVD returned 0 singular values. Cannot proceed.")
            return []

        # Ensure k1 and k2 are not larger than available features/samples
        # For U (num_samples, num_svd_components), clustering on rows means samples are rows.
        # For Vh.T (num_features, num_svd_components), clustering on cols means samples are columns.

        actual_k_rows = min(
            cfg_k2_rows,
            matrix.shape[0],
            U.shape[1] if U.shape[1] > 0 else matrix.shape[0],
        )
        actual_k_cols = min(
            cfg_k2_cols,
            matrix.shape[1],
            Vh.shape[0] if Vh.shape[0] > 0 else matrix.shape[1],
        )

        if actual_k_rows <= 1 or actual_k_cols <= 1:
            self.logger.warning(
                f"Number of clusters (k1={actual_k_rows}, k2={actual_k_cols}) too small. Min 2 required. Returning empty list."
            )
            return []

        # Row features: U scaled by S. Number of features for clustering is num_svd_components.
        row_features = U @ np.diag(S)
        # Col features: V (which is Vh.T) scaled by S. Vh is (num_svd_components, matrix.shape[1])
        # So (np.diag(S) @ Vh).T is (matrix.shape[1], num_svd_components)
        col_features = (np.diag(S) @ Vh).T

        if (
            row_features.shape[0] < actual_k_rows
            or col_features.shape[0] < actual_k_cols
        ):
            self.logger.warning(
                f"Not enough samples for clustering: rows ({row_features.shape[0]} vs k1={actual_k_rows}), cols ({col_features.shape[0]} vs k2={actual_k_cols}). Adjusting k."
            )
            actual_k_rows = min(actual_k_rows, row_features.shape[0])
            actual_k_cols = min(actual_k_cols, col_features.shape[0])
            if actual_k_rows <= 1 or actual_k_cols <= 1:  # Check again after adjustment
                self.logger.error("Adjusted k still too small. Aborting detection.")
                return []

        try:
            row_labels = self._cluster_features(row_features, actual_k_rows)
            col_labels = self._cluster_features(col_features, actual_k_cols)
        except Exception as e:
            self.logger.error(f"Clustering failed: {e}. Returning empty list.")
            return []

        # Pass effective min_rows, min_cols, tolerance to generate_biclusters (or its validator call)
        generated_biclusters = self._generate_biclusters(
            matrix,
            row_labels,
            col_labels,
            actual_k_rows,
            actual_k_cols,
            cfg_min_rows,
            cfg_min_cols,
            cfg_tolerance,
        )

        # Filter by overlap
        if cfg_max_overlap is not None and cfg_max_overlap < 1.0:
            final_biclusters = self._filter_by_overlap(
                generated_biclusters, cfg_max_overlap
            )
        else:
            final_biclusters = generated_biclusters

        self.logger.info(
            f"Detected {len(final_biclusters)} valid biclusters after overlap filtering."
        )
        return final_biclusters

    def _compute_svd(
        self, matrix: Matrix, n_components_override: Optional[int] = None
    ) -> Tuple[Matrix, NDArray, Matrix]:
        """Compute SVD with caching. Result (U, S, Vh)."""
        # Hash the matrix data for caching. Using tobytes() for hash.
        # Ensure matrix is C-contiguous for tobytes() to be consistent if that matters.
        matrix_for_hash = np.ascontiguousarray(matrix)
        matrix_hash = hash(matrix_for_hash.tobytes())
        cache_key = f"{matrix_hash}_{matrix.shape}_{self.config.random_state}"

        if cache_key not in self.svd_cache:
            from sklearn.decomposition import TruncatedSVD

            n_components = (
                n_components_override
                if n_components_override is not None
                else min(matrix.shape) - 1
            )
            if n_components >= min(matrix.shape):
                n_components = min(matrix.shape) - 1
            if n_components <= 0:
                raise ValueError(f"n_components must be > 0. Got {n_components}")

            svd = TruncatedSVD(
                n_components=n_components, random_state=self.config.random_state
            )

            # fit_transform returns U*S (transformed data)
            U_transformed = svd.fit_transform(matrix)
            S_singular_values = svd.singular_values_
            Vh_components = svd.components_  # This is Vh

            # Recover U from U_transformed = U @ diag(S)
            # U = U_transformed @ diag(1/S)
            # Need to handle S_singular_values being zero or very small to avoid division by zero.
            S_inv = np.zeros_like(S_singular_values)
            S_inv[S_singular_values > 1e-9] = (
                1.0 / S_singular_values[S_singular_values > 1e-9]
            )
            U_recovered = U_transformed @ np.diag(S_inv)

            self.svd_cache[cache_key] = (U_recovered, S_singular_values, Vh_components)

        return self.svd_cache[cache_key]

    def _cluster_features(self, features: Matrix, n_clusters: int) -> IntArray:
        """Cluster features using specified method. Returns integer labels array."""
        if n_clusters <= 1:  # KMeans requires n_clusters >= 2 usually.
            self.logger.warning(
                f"n_clusters is {n_clusters}, which is too small for clustering. Returning all zeros."
            )
            return np.zeros(features.shape[0], dtype=int)

        if features.shape[0] < n_clusters:
            self.logger.warning(
                f"Number of samples ({features.shape[0]}) is less than n_clusters ({n_clusters}). Adjusting n_clusters."
            )
            n_clusters = features.shape[0]
            if n_clusters <= 1:
                return np.zeros(features.shape[0], dtype=int)

        if self.config.clustering_method == ClusteringMethod.KMEANS:
            from sklearn.cluster import KMeans

            kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=self.config.random_state,
                n_init="auto",  # Suppress warning for future changes in scikit-learn
            )
            return kmeans.fit_predict(features).astype(np.int64)
        elif self.config.clustering_method == ClusteringMethod.SPECTRAL:
            from sklearn.cluster import SpectralClustering

            spectral = SpectralClustering(
                n_clusters=n_clusters,
                random_state=self.config.random_state,
                affinity="nearest_neighbors",  # Common default, can be configured
            )
            return spectral.fit_predict(features).astype(np.int64)
        else:
            # Fallback or error for unimplemented methods
            raise NotImplementedError(
                f"Clustering method {self.config.clustering_method.value} not implemented in SVDBiclusterDetector"
            )

    def _generate_biclusters(
        self,
        matrix: Matrix,
        row_labels: IntArray,
        col_labels: IntArray,
        num_row_clusters: int,
        num_col_clusters: int,
        min_rows: int,
        min_cols: int,
        tolerance: float,
    ) -> List[Bicluster]:
        """Generate and validate biclusters from cluster labels."""
        biclusters = []

        # Original matrix dimensions are needed for Bicluster boolean masks
        original_rows, original_cols = matrix.shape

        for i in range(num_row_clusters):  # Iterate up to actual_k1 used
            for j in range(num_col_clusters):  # Iterate up to actual_k2 used
                # Create boolean masks for rows and columns belonging to current cluster pair (i,j)
                row_mask = np.zeros(original_rows, dtype=bool)
                current_row_indices_for_cluster_i = np.where(row_labels == i)[0]
                row_mask[current_row_indices_for_cluster_i] = True

                col_mask = np.zeros(original_cols, dtype=bool)
                current_col_indices_for_cluster_j = np.where(col_labels == j)[0]
                col_mask[current_col_indices_for_cluster_j] = True

                # Basic check: skip if cluster results in no rows or no columns selected
                if not np.any(row_mask) or not np.any(col_mask):
                    continue

                # Create candidate bicluster
                # The score will be computed and set during validation.
                candidate = Bicluster(
                    row_indices=row_mask,
                    col_indices=col_mask,
                    score=float(
                        "nan"
                    ),  # Placeholder, will be set by validate_bicluster
                    metadata={"cluster_id": (i, j), "detection_method": "SVD"},
                )

                # Validate bicluster (this will also calculate and set its score)
                if self.validate_bicluster(
                    matrix, candidate, min_rows, min_cols, tolerance
                ):
                    biclusters.append(candidate)

        return biclusters

    def _filter_by_overlap(
        self, biclusters: List[Bicluster], max_overlap: float
    ) -> List[Bicluster]:
        if not biclusters or max_overlap >= 1.0:
            return biclusters
        # Sort by score (lower is better)
        biclusters.sort(
            key=lambda bc: bc.score if bc.score is not None else float("inf")
        )
        selected_biclusters: List[Bicluster] = []
        for candidate_bc in biclusters:
            is_too_overlapping = False
            for selected_bc in selected_biclusters:
                # Using Jaccard Index for overlap here, can be customized
                # Check both row and column Jaccard index independently
                row_intersect = np.sum(
                    candidate_bc.row_indices & selected_bc.row_indices
                )
                row_union = np.sum(candidate_bc.row_indices | selected_bc.row_indices)
                col_intersect = np.sum(
                    candidate_bc.col_indices & selected_bc.col_indices
                )
                col_union = np.sum(candidate_bc.col_indices | selected_bc.col_indices)

                j_row = row_intersect / row_union if row_union > 0 else 0.0
                j_col = col_intersect / col_union if col_union > 0 else 0.0

                if (
                    max(j_row, j_col) > max_overlap
                ):  # If either row or col Jaccard exceeds, it's too overlapping
                    is_too_overlapping = True
                    break
            if not is_too_overlapping:
                selected_biclusters.append(candidate_bc)
        return selected_biclusters
