"""
Probabilistic Matrix Partitioning for DiMergeCo Algorithm

Implements the probabilistic matrix partitioning strategy from the DiMergeCo paper.
Provides theoretical guarantees for co-cluster detection based on Theorem 2 and Lemma 1.

Key Components:
    - PartitionConfig: Configuration for partition parameters
    - MatrixPartitioner: Implements probabilistic partitioning with theoretical guarantees

References:
    DiMergeCo paper, Algorithm 2: Probabilistic Matrix Partitioning
    Theorem 2: Detection probability guarantee
    Lemma 1: Detectability parameters
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import numpy as np
from numpy.typing import NDArray
import math
import logging

# Type aliases
Matrix = NDArray[np.floating]


@dataclass
class PartitionConfig:
    """
    Configuration for probabilistic matrix partitioning.

    Strictly corresponds to paper notation and parameters from Algorithm 2.

    Attributes:
        T_m: Minimum row threshold (paper: T_m)
        T_n: Minimum column threshold (paper: T_n)
        T_p: Number of partitioning iterations (paper: T_p)
        T_p_max: Maximum iterations allowed
        P_thresh: Minimum detection probability threshold (paper: P_thresh)
        alpha: Failure probability (1 - P_thresh)
        m: Number of row blocks (computed by algorithm)
        n: Number of column blocks (computed by algorithm)
        phi: Row block sizes φ_i (computed by algorithm)
        psi: Column block sizes ψ_j (computed by algorithm)
        uniform_blocks: Whether to use uniform or adaptive block sizing
        epsilon: Minimum block size as proportion of matrix dimension
        merge_threshold: Jaccard threshold for result aggregation
        keep_partition_metadata: Whether to track partition provenance
        random_state: Random seed for reproducibility
    """

    # Basic parameters (paper notation)
    T_m: int = 10  # Minimum row threshold
    T_n: int = 10  # Minimum column threshold
    T_p: int = 5   # Partitioning iterations
    T_p_max: int = 20  # Maximum iterations

    # Probability guarantee parameters
    P_thresh: float = 0.95  # Minimum detection probability
    alpha: float = 0.05     # Failure probability (1 - P_thresh)

    # Block size parameters (computed by algorithm)
    m: Optional[int] = None  # Number of row blocks
    n: Optional[int] = None  # Number of column blocks
    phi: Optional[List[int]] = None  # Row block sizes
    psi: Optional[List[int]] = None  # Column block sizes

    # Partitioning strategy
    uniform_blocks: bool = True  # Uniform vs adaptive block sizing
    epsilon: float = 0.01  # Minimum block size ratio

    # Aggregation strategy
    merge_threshold: float = 0.3  # Jaccard threshold for merging
    keep_partition_metadata: bool = True
    random_state: Optional[int] = None


class MatrixPartitioner:
    """
    Probabilistic matrix partitioning with theoretical guarantees.

    Implements Algorithm 2 from the DiMergeCo paper, providing detection
    probability guarantees based on Theorem 2.

    The partitioner divides a matrix into blocks and performs randomized
    sampling to ensure robust co-cluster detection with high probability.
    """

    def __init__(self, config: PartitionConfig):
        """
        Initialize matrix partitioner.

        Args:
            config: Partition configuration parameters
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Computed parameters (set by compute_partition_parameters)
        self.m: Optional[int] = None
        self.n: Optional[int] = None
        self.phi_list: Optional[List[int]] = None
        self.psi_list: Optional[List[int]] = None
        self.T_p_computed: Optional[int] = None

    def compute_partition_parameters(
        self,
        M: int,
        N: int,
        estimated_cocluster_sizes: Optional[List[Tuple[int, int]]] = None,
    ) -> Tuple[int, int, List[int], List[int], int]:
        """
        Compute optimal partition parameters.

        Strictly implements Algorithm 2, Lines 1-15 from the paper:
        - Line 1: Initialize m, n from T_m, T_n
        - Lines 2-4: Initialize φ and ψ uniformly
        - Lines 6-13: Adaptive optimization (if enabled)
        - Lines 12-13: Optimize T_p until P ≥ P_thresh

        Args:
            M: Number of rows in matrix
            N: Number of columns in matrix
            estimated_cocluster_sizes: Optional list of (rows, cols) for known co-clusters

        Returns:
            Tuple of (m, n, phi_list, psi_list, T_p)
        """
        # Line 1: Initialize minimum block counts
        m = max(2, math.ceil(M / self.config.T_m))
        n = max(2, math.ceil(N / self.config.T_n))

        # Line 2: Uniform initialization of block sizes
        phi_list = [math.ceil(M / m)] * m
        psi_list = [math.ceil(N / n)] * n

        # Line 3: Adjust last block to match total dimensions
        phi_list[-1] = M - sum(phi_list[:-1])
        psi_list[-1] = N - sum(psi_list[:-1])

        # Ensure non-negative sizes
        phi_list[-1] = max(1, phi_list[-1])
        psi_list[-1] = max(1, psi_list[-1])

        T_p = self.config.T_p

        if not self.config.uniform_blocks and estimated_cocluster_sizes:
            # Lines 6-13: Adaptive optimization
            m, n, phi_list, psi_list, T_p = self._adaptive_refinement(
                M, N, m, n, phi_list, psi_list, estimated_cocluster_sizes
            )

        T_p = min(T_p, self.config.T_p_max)

        # Compute and log detection probability
        P_estimated = self._compute_detection_probability(
            M, N, phi_list, psi_list, T_p, estimated_cocluster_sizes or []
        )

        self.logger.info(
            f"Computed partition: {m}×{n} blocks, T_p={T_p}, P={P_estimated:.4f}"
        )

        # Store computed parameters
        self.m = m
        self.n = n
        self.phi_list = phi_list
        self.psi_list = psi_list
        self.T_p_computed = T_p

        return m, n, phi_list, psi_list, T_p

    def _compute_detection_probability(
        self,
        M: int,
        N: int,
        phi_list: List[int],
        psi_list: List[int],
        T_p: int,
        cocluster_sizes: List[Tuple[int, int]],
    ) -> float:
        """
        Compute detection probability per Theorem 2.

        Formula: P ≥ 1 - K·exp(-2·T_p·[M·(s^k)² + N·(t^k)²])

        where s^k, t^k are detectability parameters from Lemma 1:
        s^k = (M_k/M) - ((T_m-1)/φ_avg)
        t^k = (N_k/N) - ((T_n-1)/ψ_avg)

        Args:
            M: Number of matrix rows
            N: Number of matrix columns
            phi_list: Row block sizes
            psi_list: Column block sizes
            T_p: Number of partition iterations
            cocluster_sizes: List of (M_k, N_k) co-cluster dimensions

        Returns:
            Estimated detection probability P
        """
        if not cocluster_sizes:
            # No co-clusters to detect, trivially satisfied
            return 1.0

        K = len(cocluster_sizes)
        phi_avg = sum(phi_list) / len(phi_list)
        psi_avg = sum(psi_list) / len(psi_list)

        # Compute worst-case detectability parameters (Lemma 1)
        worst_case_s = float("inf")
        worst_case_t = float("inf")

        for M_k, N_k in cocluster_sizes:
            # Lemma 1 formulas
            s_k = (M_k / M) - ((self.config.T_m - 1) / phi_avg)
            t_k = (N_k / N) - ((self.config.T_n - 1) / psi_avg)

            worst_case_s = min(worst_case_s, s_k)
            worst_case_t = min(worst_case_t, t_k)

        # Boundary case: co-cluster too small for reliable detection
        if worst_case_s <= 0 or worst_case_t <= 0:
            self.logger.warning(
                f"Co-cluster too small for reliable detection: s={worst_case_s:.4f}, t={worst_case_t:.4f}"
            )
            return 0.0

        # Compute probability (Theorem 2)
        exponent = -2 * T_p * (M * worst_case_s**2 + N * worst_case_t**2)

        # Prevent numerical overflow
        if exponent < -700:  # exp(-700) ≈ 0
            P = 1.0
        else:
            P = 1.0 - K * np.exp(exponent)

        return max(0.0, min(1.0, P))  # Clamp to [0,1]

    def _adaptive_refinement(
        self,
        M: int,
        N: int,
        m: int,
        n: int,
        phi_list: List[int],
        psi_list: List[int],
        estimated_cocluster_sizes: List[Tuple[int, int]],
    ) -> Tuple[int, int, List[int], List[int], int]:
        """
        Adaptive block size optimization (Algorithm 2, Lines 6-13).

        Iteratively refines block sizes and T_p until P ≥ P_thresh
        or T_p reaches T_p_max.

        Args:
            M, N: Matrix dimensions
            m, n: Initial block counts
            phi_list, psi_list: Initial block sizes
            estimated_cocluster_sizes: Known co-cluster dimensions

        Returns:
            Refined (m, n, phi_list, psi_list, T_p)
        """
        T_p = self.config.T_p
        iteration = 0
        max_iterations = 100  # Safety limit

        while iteration < max_iterations:
            # Compute current probability
            P = self._compute_detection_probability(
                M, N, phi_list, psi_list, T_p, estimated_cocluster_sizes
            )

            self.logger.debug(f"Refinement iteration {iteration}: P={P:.4f}, T_p={T_p}")

            if P >= self.config.P_thresh or T_p >= self.config.T_p_max:
                break

            # Lines 8-9: Expand blocks overlapping with co-clusters
            for M_k, N_k in estimated_cocluster_sizes:
                if M_k >= self.config.T_m and N_k >= self.config.T_n:
                    phi_list, psi_list = self._expand_overlapping_blocks(
                        M, N, phi_list, psi_list, M_k, N_k
                    )

            # Line 10: Redistribute to maintain total dimensions
            phi_list, psi_list = self._redistribute_blocks(M, N, phi_list, psi_list)

            # Lines 12-13: Increase sampling iterations
            if P < self.config.P_thresh:
                delta_T = self._compute_required_iterations(
                    M, N, phi_list, psi_list, estimated_cocluster_sizes, P
                )
                T_p = min(T_p + delta_T, self.config.T_p_max)

            iteration += 1

        if iteration >= max_iterations:
            self.logger.warning("Adaptive refinement reached max iterations")

        return m, n, phi_list, psi_list, T_p

    def _expand_overlapping_blocks(
        self,
        M: int,
        N: int,
        phi_list: List[int],
        psi_list: List[int],
        M_k: int,
        N_k: int,
    ) -> Tuple[List[int], List[int]]:
        """
        Expand blocks that overlap with a co-cluster (Algorithm 2, Lines 8-9).

        Args:
            M, N: Matrix dimensions
            phi_list, psi_list: Current block sizes
            M_k, N_k: Co-cluster dimensions

        Returns:
            Updated (phi_list, psi_list)
        """
        # Simple heuristic: increase blocks proportionally to co-cluster size
        row_expansion = max(1, M_k // len(phi_list))
        col_expansion = max(1, N_k // len(psi_list))

        new_phi = [min(phi + row_expansion, M) for phi in phi_list]
        new_psi = [psi + col_expansion for psi in psi_list]

        return new_phi, new_psi

    def _redistribute_blocks(
        self, M: int, N: int, phi_list: List[int], psi_list: List[int]
    ) -> Tuple[List[int], List[int]]:
        """
        Redistribute block sizes to maintain total dimensions (Algorithm 2, Line 10).

        Args:
            M, N: Matrix dimensions
            phi_list, psi_list: Current block sizes

        Returns:
            Redistributed (phi_list, psi_list)
        """
        # Normalize to match total dimensions
        total_phi = sum(phi_list)
        total_psi = sum(psi_list)

        if total_phi != M:
            # Proportional scaling
            scale = M / total_phi
            phi_list = [max(1, int(phi * scale)) for phi in phi_list]
            # Adjust last block for exact match
            phi_list[-1] = M - sum(phi_list[:-1])

        if total_psi != N:
            scale = N / total_psi
            psi_list = [max(1, int(psi * scale)) for psi in psi_list]
            psi_list[-1] = N - sum(psi_list[:-1])

        return phi_list, psi_list

    def _compute_required_iterations(
        self,
        M: int,
        N: int,
        phi_list: List[int],
        psi_list: List[int],
        cocluster_sizes: List[Tuple[int, int]],
        current_P: float,
    ) -> int:
        """
        Compute additional iterations needed to reach P_thresh.

        Based on inverting Theorem 2 formula.

        Args:
            M, N: Matrix dimensions
            phi_list, psi_list: Current block sizes
            cocluster_sizes: Co-cluster dimensions
            current_P: Current detection probability

        Returns:
            Number of additional iterations needed
        """
        if not cocluster_sizes or current_P >= self.config.P_thresh:
            return 0

        K = len(cocluster_sizes)
        phi_avg = sum(phi_list) / len(phi_list)
        psi_avg = sum(psi_list) / len(psi_list)

        # Compute detectability parameters
        worst_s = float("inf")
        worst_t = float("inf")
        for M_k, N_k in cocluster_sizes:
            s_k = (M_k / M) - ((self.config.T_m - 1) / phi_avg)
            t_k = (N_k / N) - ((self.config.T_n - 1) / psi_avg)
            worst_s = min(worst_s, s_k)
            worst_t = min(worst_t, t_k)

        if worst_s <= 0 or worst_t <= 0:
            return self.config.T_p_max  # Need maximum iterations

        # Solve for T_p: 1 - K*exp(-2*T_p*(M*s² + N*t²)) ≥ P_thresh
        # => exp(-2*T_p*(M*s² + N*t²)) ≤ (1 - P_thresh)/K
        # => -2*T_p*(M*s² + N*t²) ≤ ln((1 - P_thresh)/K)
        # => T_p ≥ -ln((1 - P_thresh)/K) / (2*(M*s² + N*t²))

        target_prob = self.config.P_thresh
        if (1 - target_prob) / K <= 0:
            return 1  # Already satisfied

        numerator = -np.log((1 - target_prob) / K)
        denominator = 2 * (M * worst_s**2 + N * worst_t**2)

        if denominator <= 0:
            return self.config.T_p_max

        required_T_p = math.ceil(numerator / denominator)
        current_T_p = self.T_p_computed or self.config.T_p

        delta = max(1, required_T_p - current_T_p)
        return min(delta, self.config.T_p_max - current_T_p)

    def partition_matrix(
        self, matrix: Matrix, iteration: int = 0
    ) -> List[Tuple[Matrix, Tuple[slice, slice], Tuple[int, int], np.ndarray, np.ndarray]]:
        """
        Partition matrix with random shuffling.

        Each iteration uses different random permutation to ensure
        diverse sampling across partitions.

        Args:
            matrix: Input matrix to partition
            iteration: Current partition iteration (for random seed)

        Returns:
            List of tuples containing:
            - submatrix: The partitioned block
            - (row_slice, col_slice): Slice objects for the block
            - (block_i, block_j): Block coordinates
            - block_row_indices: Original row indices in this block
            - block_col_indices: Original column indices in this block
        """
        M, N = matrix.shape

        # Ensure parameters are computed
        if self.m is None or self.n is None:
            self.m, self.n, self.phi_list, self.psi_list, _ = (
                self.compute_partition_parameters(M, N)
            )

        # Random shuffling for this iteration
        rng = np.random.RandomState(
            self.config.random_state + iteration
            if self.config.random_state is not None
            else None
        )

        row_indices = np.arange(M)
        col_indices = np.arange(N)

        rng.shuffle(row_indices)
        rng.shuffle(col_indices)

        # Create blocks
        blocks = []
        row_start = 0

        for i, phi_i in enumerate(self.phi_list):
            col_start = 0

            for j, psi_j in enumerate(self.psi_list):
                # Get shuffled indices for this block
                block_row_indices = sorted(row_indices[row_start : row_start + phi_i])
                block_col_indices = sorted(col_indices[col_start : col_start + psi_j])

                # Extract submatrix
                submatrix = matrix[np.ix_(block_row_indices, block_col_indices)]

                row_slice = slice(row_start, row_start + phi_i)
                col_slice = slice(col_start, col_start + psi_j)

                blocks.append(
                    (
                        submatrix,
                        (row_slice, col_slice),
                        (i, j),
                        np.array(block_row_indices),
                        np.array(block_col_indices),
                    )
                )

                col_start += psi_j

            row_start += phi_i

        return blocks

    def get_partition_metadata(self) -> Dict:
        """
        Get metadata about the computed partition.

        Returns:
            Dictionary with partition parameters and statistics
        """
        if self.m is None:
            return {"computed": False}

        return {
            "computed": True,
            "m": self.m,
            "n": self.n,
            "total_blocks": self.m * self.n,
            "phi_list": self.phi_list,
            "psi_list": self.psi_list,
            "T_p": self.T_p_computed,
            "avg_block_size_rows": sum(self.phi_list) / len(self.phi_list),
            "avg_block_size_cols": sum(self.psi_list) / len(self.psi_list),
        }
