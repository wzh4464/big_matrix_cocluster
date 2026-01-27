"""
Hierarchical Merging Strategy for DiMergeCo

Implements O(log n) complexity hierarchical bicluster merging using binary tree
structure and spatial indexing for efficient overlap detection.

Key Components:
    - HierarchicalMergeConfig: Configuration for merge strategy
    - HierarchicalMerger: Implements binary tree merging with progressive filtering
    - BiclusterSpatialIndex: Spatial grid index for O(1) overlap queries

References:
    DiMergeCo paper: Hierarchical merging strategy (Section on Result Aggregation)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Set, Optional
from collections import defaultdict
import numpy as np
from numpy.typing import NDArray
import math
import logging

from .bicluster import Bicluster

# Type aliases
Matrix = NDArray[np.floating]


@dataclass
class MergeEvent:
    """Record of a merge operation for provenance tracking."""

    level: int
    biclusters_merged: int
    input_count: int
    output_count: int
    metadata: Dict


@dataclass
class HierarchicalMergeConfig:
    """
    Configuration for hierarchical merging strategy.

    Attributes:
        overlap_threshold: Jaccard index threshold for considering overlap
        overlap_metric: Metric for overlap ("jaccard" or "intersection_ratio")
        merge_method: Method for combining overlapping biclusters
        score_aggregation: How to combine scores ("weighted_mean", "min", "mean")
        base_tolerance: Base quality threshold for filtering
        level_penalty: Quality penalty increase per level (higher=more strict)
        use_spatial_indexing: Whether to use spatial grid index
        spatial_grid_size: Number of grid cells per dimension
        min_merge_size: Minimum cluster size for merging
        max_merge_candidates: Maximum candidates to consider per bicluster
        track_merge_history: Whether to record merge events
        track_provenance: Whether to track bicluster origins
    """

    # Overlap detection
    overlap_threshold: float = 0.3  # Jaccard threshold
    overlap_metric: str = "jaccard"  # "jaccard" or "intersection_ratio"

    # Merging strategy
    merge_method: str = "quality_weighted"  # Quality-weighted merging
    score_aggregation: str = "weighted_mean"  # Score combination method

    # Progressive filtering
    base_tolerance: float = 0.05  # Base quality threshold
    level_penalty: float = 0.1  # Quality requirement increase per level

    # Performance optimization
    use_spatial_indexing: bool = True
    spatial_grid_size: int = 10  # Grid resolution

    # Merge controls
    min_merge_size: int = 2
    max_merge_candidates: int = 10

    # Metadata tracking
    track_merge_history: bool = True
    track_provenance: bool = True


class BiclusterSpatialIndex:
    """
    Grid-based spatial index for efficient bicluster overlap queries.

    Reduces O(n²) overlap checking to O(n log n) average case by
    partitioning space into grid cells and only checking biclusters
    in nearby cells.
    """

    def __init__(self, matrix_shape: Tuple[int, int], grid_size: int = 10):
        """
        Initialize spatial index.

        Args:
            matrix_shape: (rows, cols) dimensions of the matrix
            grid_size: Number of grid cells per dimension
        """
        self.rows, self.cols = matrix_shape
        self.grid_size = grid_size

        # Cell dimensions
        self.row_cell_size = max(1, self.rows // grid_size)
        self.col_cell_size = max(1, self.cols // grid_size)

        # Grid storage: (cell_row, cell_col) -> List[Bicluster]
        self.grid: Dict[Tuple[int, int], List[Bicluster]] = defaultdict(list)

        self.logger = logging.getLogger(self.__class__.__name__)

    def _get_cells(self, bicluster: Bicluster) -> Set[Tuple[int, int]]:
        """
        Get all grid cells covered by a bicluster.

        Args:
            bicluster: Bicluster to map to cells

        Returns:
            Set of (cell_row, cell_col) tuples
        """
        row_labels = bicluster.row_labels
        col_labels = bicluster.col_labels

        # Map indices to cell coordinates
        row_cells = set(min(r // self.row_cell_size, self.grid_size - 1) for r in row_labels)
        col_cells = set(min(c // self.col_cell_size, self.grid_size - 1) for c in col_labels)

        # Cartesian product of row and column cells
        return {(r, c) for r in row_cells for c in col_cells}

    def insert(self, bicluster: Bicluster) -> None:
        """
        Insert bicluster into spatial index.

        Args:
            bicluster: Bicluster to insert
        """
        cells = self._get_cells(bicluster)
        for cell in cells:
            self.grid[cell].append(bicluster)

    def query_overlapping(
        self, bicluster: Bicluster, threshold: float
    ) -> List[Bicluster]:
        """
        Query biclusters with actual overlap exceeding threshold.

        Args:
            bicluster: Query bicluster
            threshold: Jaccard index threshold

        Returns:
            List of overlapping biclusters
        """
        # Get candidate biclusters from nearby cells
        cells = self._get_cells(bicluster)
        candidates = set()

        for cell in cells:
            candidates.update(self.grid.get(cell, []))

        # Verify actual overlap
        overlapping = []
        for candidate in candidates:
            if candidate.id == bicluster.id:
                continue

            try:
                jaccard = bicluster.jaccard_index(candidate)
                if jaccard > threshold:
                    overlapping.append(candidate)
            except ValueError:
                # Incompatible dimensions, skip
                continue

        return overlapping

    def size(self) -> int:
        """Get total number of biclusters in index."""
        return sum(len(bclist) for bclist in self.grid.values())


class HierarchicalMerger:
    """
    Hierarchical bicluster merger with O(log n) complexity.

    Implements binary tree merging strategy where biclusters are
    progressively merged level-by-level with increasing quality
    requirements at higher levels.
    """

    def __init__(self, config: HierarchicalMergeConfig):
        """
        Initialize hierarchical merger.

        Args:
            config: Merge configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.merge_history: List[MergeEvent] = []

    def merge_partitions(
        self,
        partition_biclusters: List[List[Bicluster]],
        matrix_shape: Tuple[int, int],
    ) -> List[Bicluster]:
        """
        Hierarchically merge biclusters from multiple partitions.

        Implements binary tree merging:
        1. Initialize: Each partition's biclusters as leaf nodes
        2. Iterate: ⌈log₂(P)⌉ levels, pairwise merging at each level
        3. Each merge: Use spatial index for overlap detection → quality-weighted merge
        4. Progressive filtering: Higher levels have stricter quality requirements

        Args:
            partition_biclusters: List of bicluster lists from each partition
            matrix_shape: (rows, cols) dimensions of original matrix

        Returns:
            Final merged bicluster list from root node
        """
        if not partition_biclusters:
            return []

        # Flatten to get total count
        total_biclusters = sum(len(p) for p in partition_biclusters)
        if total_biclusters == 0:
            return []

        # Compute tree depth
        tree_depth = math.ceil(math.log2(len(partition_biclusters)))

        self.logger.info(
            f"Hierarchical merge: {len(partition_biclusters)} partitions, "
            f"depth={tree_depth}, total biclusters={total_biclusters}"
        )

        # Initialize current level with partitions
        current_level = partition_biclusters.copy()
        level = 0

        # Bottom-up merging
        while len(current_level) > 1:
            self.logger.info(f"Merging level {level}: {len(current_level)} groups")

            next_level = []

            # Pairwise merging
            for i in range(0, len(current_level), 2):
                if i + 1 < len(current_level):
                    # Merge pair
                    left = current_level[i]
                    right = current_level[i + 1]
                    merged = self._merge_bicluster_groups(
                        left, right, matrix_shape, level
                    )
                else:
                    # Odd node promoted to next level
                    merged = current_level[i]

                next_level.append(merged)

            # Record merge event
            if self.config.track_merge_history:
                self.merge_history.append(
                    MergeEvent(
                        level=level,
                        biclusters_merged=sum(len(g) for g in current_level),
                        input_count=len(current_level),
                        output_count=len(next_level),
                        metadata={"matrix_shape": matrix_shape},
                    )
                )

            current_level = next_level
            level += 1

        final_biclusters = current_level[0]
        self.logger.info(f"Final merged: {len(final_biclusters)} biclusters")

        return final_biclusters

    def _merge_bicluster_groups(
        self,
        left: List[Bicluster],
        right: List[Bicluster],
        matrix_shape: Tuple[int, int],
        level: int,
    ) -> List[Bicluster]:
        """
        Merge two groups of biclusters using spatial indexing.

        Key innovations:
        1. Spatial index avoids O(n²) comparisons
        2. Quality-weighted merging preserves best features
        3. Progressive filtering: higher levels = stricter quality

        Args:
            left: First bicluster group
            right: Second bicluster group
            matrix_shape: Matrix dimensions
            level: Current merge level

        Returns:
            Merged bicluster list
        """
        # Build spatial index if enabled
        if self.config.use_spatial_indexing:
            spatial_index = BiclusterSpatialIndex(matrix_shape, self.config.spatial_grid_size)
            for bc in left + right:
                spatial_index.insert(bc)

        # Merge results
        merged_result = []
        processed = set()

        # Sort by score (better biclusters first)
        all_biclusters = sorted(
            left + right,
            key=lambda bc: bc.score if bc.score is not None else float("inf"),
        )

        for bc in all_biclusters:
            if bc.id in processed:
                continue

            # Find overlapping candidates
            if self.config.use_spatial_indexing:
                candidates = spatial_index.query_overlapping(
                    bc, self.config.overlap_threshold
                )
            else:
                candidates = self._find_overlapping_naive(bc, all_biclusters, processed)

            # Limit candidates
            candidates = candidates[: self.config.max_merge_candidates]

            if len(candidates) == 0:
                # No overlap, keep as-is
                merged_result.append(bc)
                processed.add(bc.id)
            else:
                # Quality-weighted merge
                merge_cluster = [bc] + [c for c in candidates if c.id not in processed]
                merged_bc = self._quality_weighted_merge(merge_cluster)

                merged_result.append(merged_bc)
                processed.update(m.id for m in merge_cluster)

        # Progressive filtering: stricter at higher levels
        quality_threshold = self.config.base_tolerance * (
            1 + level * self.config.level_penalty
        )

        filtered = [
            bc
            for bc in merged_result
            if bc.score is not None and bc.score < quality_threshold
        ]

        self.logger.debug(
            f"Level {level}: {len(merged_result)} → {len(filtered)} after filtering "
            f"(threshold={quality_threshold:.4f})"
        )

        return filtered

    def _find_overlapping_naive(
        self,
        bicluster: Bicluster,
        candidates: List[Bicluster],
        processed: Set[str],
    ) -> List[Bicluster]:
        """
        Find overlapping biclusters using naive O(n) search.

        Fallback when spatial indexing is disabled.

        Args:
            bicluster: Query bicluster
            candidates: All biclusters to check
            processed: Already processed IDs to skip

        Returns:
            List of overlapping biclusters
        """
        overlapping = []

        for candidate in candidates:
            if candidate.id == bicluster.id or candidate.id in processed:
                continue

            try:
                jaccard = bicluster.jaccard_index(candidate)
                if jaccard > self.config.overlap_threshold:
                    overlapping.append(candidate)
            except ValueError:
                continue

        return overlapping

    def _quality_weighted_merge(self, biclusters: List[Bicluster]) -> Bicluster:
        """
        Merge multiple biclusters using quality-weighted strategy.

        Better quality biclusters contribute more to the merged result.

        Args:
            biclusters: Biclusters to merge

        Returns:
            Merged bicluster
        """
        # Compute weights (lower score = higher quality = higher weight)
        scores = np.array(
            [bc.score if bc.score is not None else float("inf") for bc in biclusters]
        )

        # Handle infinite scores
        valid_scores = scores[np.isfinite(scores)]
        if len(valid_scores) == 0:
            # All scores invalid, use uniform weights
            weights = np.ones(len(biclusters)) / len(biclusters)
        else:
            max_score = np.max(valid_scores)

            # Invert scores: lower score → higher weight
            weights = max_score - scores + 1e-6
            weights = np.where(np.isfinite(weights), weights, 1e-6)
            weights = weights / np.sum(weights)

        # Weighted union of indices
        row_union = np.zeros_like(biclusters[0].row_indices, dtype=bool)
        col_union = np.zeros_like(biclusters[0].col_indices, dtype=bool)

        threshold = self.config.min_merge_size / len(biclusters)

        for bc, weight in zip(biclusters, weights):
            if weight > threshold:
                row_union |= bc.row_indices
                col_union |= bc.col_indices

        # Aggregate score based on configuration
        if self.config.score_aggregation == "weighted_mean":
            merged_score = float(np.sum(weights * scores))
        elif self.config.score_aggregation == "min":
            merged_score = float(np.min(scores))
        elif self.config.score_aggregation == "mean":
            merged_score = float(np.mean(scores))
        else:
            merged_score = float(np.sum(weights * scores))

        # Build metadata
        metadata = {
            "merge_method": self.config.merge_method,
            "merged_from_count": len(biclusters),
            "original_scores": scores.tolist(),
            "merge_weights": weights.tolist(),
        }

        if self.config.track_provenance:
            metadata["provenance"] = [
                bc.metadata.get("partition_iteration", -1)
                for bc in biclusters
                if bc.metadata
            ]

        return Bicluster(
            row_indices=row_union,
            col_indices=col_union,
            score=merged_score,
            metadata=metadata,
        )

    def get_merge_statistics(self) -> Dict:
        """
        Get statistics about the merge process.

        Returns:
            Dictionary with merge statistics
        """
        if not self.merge_history:
            return {"levels": 0, "total_events": 0}

        return {
            "levels": len(self.merge_history),
            "total_events": len(self.merge_history),
            "total_biclusters_processed": sum(
                event.biclusters_merged for event in self.merge_history
            ),
            "events": [
                {
                    "level": event.level,
                    "input_count": event.input_count,
                    "output_count": event.output_count,
                    "reduction_ratio": event.input_count / max(event.output_count, 1),
                }
                for event in self.merge_history
            ],
        }
