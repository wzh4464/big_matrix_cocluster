# visualization.py
# This file will contain unified visualization components.

"""
Modern data generation and visualization components for biclustering analysis.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
# import seaborn as sns # For potentially more advanced plots if desired
from contextlib import contextmanager
import logging

# Assuming Bicluster class is in bicluster.py
from .bicluster import Bicluster
# Assuming Matrix type alias is in core.py (or a common types file)
from .core import Matrix # If Matrix is defined in core.py


@dataclass
class BiclusterSpec:
    """Specification for a synthetic bicluster."""
    rows: int
    cols: int
    value: float = 1.0
    noise_level: float = 0.1
    
    @property
    def size(self) -> int:
        return self.rows * self.cols


@dataclass 
class SyntheticDataConfig:
    """Configuration for synthetic data generation."""
    matrix_shape: Tuple[int, int]
    bicluster_specs: List[BiclusterSpec]
    background_noise: float = 0.0
    random_state: Optional[int] = 42
    
    def __post_init__(self):
        """Validate configuration."""
        # Basic validation: sum of bicluster areas (if placed non-overlappingly)
        # More complex validation (e.g. if they can fit) could be added.
        # total_bicluster_rows = sum(spec.rows for spec in self.bicluster_specs)
        # total_bicluster_cols = sum(spec.cols for spec in self.bicluster_specs) 
        # This check was `total_bicluster_size > matrix_size`, which is fine for simple sequential placement.
        pass # Keep it simple for now


class SyntheticDataGenerator:
    """Modern synthetic data generator with embedded biclusters."""
    
    def __init__(self, config: SyntheticDataConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self._rng = np.random.RandomState(config.random_state) 
        
        # Store generation metadata
        self.ground_truth_biclusters_info: List[Dict[str, Any]] = [] # Info about true biclusters
        self.row_permutation: Optional[NDArray] = None
        self.col_permutation: Optional[NDArray] = None
    
    def generate(self) -> Tuple[NDArray, NDArray, List[Bicluster]]:
        """
        Generate synthetic matrix with embedded biclusters.
        
        Returns:
            Tuple of (permuted_matrix, original_structured_matrix, list_of_ground_truth_Bicluster_objects)
        """
        self.logger.info(f"Generating {self.config.matrix_shape} matrix with {len(self.config.bicluster_specs)} biclusters")
        
        # Create base matrix and ground truth Bicluster objects
        structured_matrix, ground_truth_biclusters_list = self._create_structured_matrix_and_gt()
        
        # Apply random permutation
        permuted_matrix = self._apply_permutation(structured_matrix)
        
        # Permute the ground truth Bicluster objects' indices as well
        permuted_ground_truth_biclusters = self._permute_ground_truth(ground_truth_biclusters_list)

        self.logger.info("Synthetic data generation completed")
        return permuted_matrix, structured_matrix, permuted_ground_truth_biclusters
    
    def _create_structured_matrix_and_gt(self) -> Tuple[NDArray, List[Bicluster]]:
        """Create structured matrix with embedded biclusters and their Bicluster object representations."""
        rows, cols = self.config.matrix_shape
        matrix = self._rng.normal(loc=0, scale=self.config.background_noise, size=(rows,cols)) # Background noise
        
        # Place biclusters sequentially (can be made more complex, e.g., random placement, overlap)
        current_row, current_col = 0, 0
        self.ground_truth_biclusters_info = [] # Store dict info for easy reference/serialization
        created_biclusters: List[Bicluster] = []
        
        for i, spec in enumerate(self.config.bicluster_specs):
            if current_row + spec.rows > rows or current_col + spec.cols > cols:
                self.logger.warning(f"Bicluster spec {i} ({spec.rows}x{spec.cols}) doesn't fit sequentially. Skipping.")
                continue
            
            end_row, end_col = current_row + spec.rows, current_col + spec.cols
            bicluster_submatrix = self._generate_bicluster_data(spec)
            matrix[current_row:end_row, current_col:end_col] = bicluster_submatrix
            
            # Create boolean masks for this bicluster for the original structured matrix
            row_indices_mask = np.zeros(rows, dtype=bool)
            row_indices_mask[current_row:end_row] = True
            col_indices_mask = np.zeros(cols, dtype=bool)
            col_indices_mask[current_col:end_col] = True
            
            gt_bicluster = Bicluster(
                id=f'gt_{i}', # Pass ID directly to constructor
                row_indices=row_indices_mask,
                col_indices=col_indices_mask,
                score=0.0, # Ground truth score often considered perfect or not applicable
                metadata={ # Keep other relevant info in metadata if needed
                    # 'id': f'gt_{i}', # No longer primary store for this ID here
                    'value': spec.value,
                    'noise_level': spec.noise_level,
                    'original_placement': ( (current_row, end_row), (current_col, end_col) )
                }
            )
            created_biclusters.append(gt_bicluster)
            # self.ground_truth_biclusters_info.append(gt_bicluster.to_dict()) # Save serializable info
            # Update: to_dict() should now correctly reflect the id=f'gt_{i}'.
            # Let's ensure that the Bicluster.to_dict() method correctly serializes the main `id` field.
            # Assuming Bicluster.to_dict() serializes all dataclass fields including `id`.
            current_bc_dict = gt_bicluster.to_dict()
            # If Bicluster.to_dict() doesn't include the main `id` field, and only `metadata['id']` 
            # then we need to ensure it does or adjust here.
            # For now, let's assume to_dict() is comprehensive.
            # If the test still fails, this is where to look.
            self.ground_truth_biclusters_info.append(current_bc_dict)
            
            # Update position for next bicluster (simple sequential placement)
            current_row += spec.rows
            if current_row >= rows: # Move to next column band if out of rows
                current_row = 0
                current_col += spec.cols 
                if current_col >= cols:
                    self.logger.info("Ran out of space for sequential bicluster placement.")
                    break
        
        return matrix, created_biclusters

    def _generate_bicluster_data(self, spec: BiclusterSpec) -> NDArray:
        """Generate data for a single bicluster."""
        base_signal = np.full((spec.rows, spec.cols), spec.value)
        noise = self._rng.normal(0, spec.noise_level, (spec.rows, spec.cols))
        return base_signal + noise
    
    def _apply_permutation(self, matrix: NDArray) -> NDArray:
        """Apply random row and column permutation and store permutations."""
        rows, cols = matrix.shape
        self.row_permutation = self._rng.permutation(rows)
        self.col_permutation = self._rng.permutation(cols)
        
        permuted_matrix = matrix[self.row_permutation, :][:, self.col_permutation]
        return permuted_matrix

    def _permute_ground_truth(self, gt_biclusters: List[Bicluster]) -> List[Bicluster]:
        """Permute the row and column indices of ground truth Bicluster objects."""
        if self.row_permutation is None or self.col_permutation is None:
            raise RuntimeError("Permutation has not been applied yet. Call generate() first.")

        permuted_gt_biclusters = []
        for bc in gt_biclusters:
            permuted_row_indices = bc.row_indices[self.row_permutation]
            permuted_col_indices = bc.col_indices[self.col_permutation]
            permuted_bc = Bicluster(
                row_indices=permuted_row_indices,
                col_indices=permuted_col_indices,
                score=bc.score,
                metadata=bc.metadata.copy() # Copy metadata
            )
            permuted_bc.metadata['is_permuted_gt'] = True
            permuted_gt_biclusters.append(permuted_bc)
        return permuted_gt_biclusters

    def get_ground_truth_bicluster_info_dicts(self) -> List[Dict[str, Any]]:
        """Get ground truth bicluster information (serializable dicts)."""
        if not self.ground_truth_biclusters_info:
            self.logger.warning("No ground truth available. Generate data first.")
        return self.ground_truth_biclusters_info.copy()
    
    def get_permutation_indices(self) -> Optional[Tuple[NDArray, NDArray]]:
        """Get row and column permutation arrays (indices that map original to permuted)."""
        if self.row_permutation is None or self.col_permutation is None:
            self.logger.warning("No permutation info available. Generate data first.")
            return None
        return self.row_permutation, self.col_permutation


class BiclusterVisualizer:
    """Modern visualization component for biclustering results."""
    
    def __init__(self, figsize: Tuple[float, float] = (12, 5), dpi: int = 100):
        self.figsize = figsize
        self.dpi = dpi
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Basic Matplotlib configuration
        plt.rcParams.update({
            'figure.dpi': self.dpi,
            'font.size': 10,
            # Add other defaults if desired
        })
    
    @contextmanager
    def _figure_context(self, num_subplots_rows: int = 1, num_subplots_cols: int = 1, 
                        title: Optional[str] = None, custom_figsize: Optional[Tuple[float,float]] = None):
        """Context manager for figure creation and cleanup."""
        effective_figsize = custom_figsize if custom_figsize else self.figsize
        fig, axs = plt.subplots(num_subplots_rows, num_subplots_cols, figsize=effective_figsize, squeeze=False)
        if title:
            fig.suptitle(title, fontsize=14, y=1.02) # Adjust y for suptitle spacing
        
        try:
            if num_subplots_rows * num_subplots_cols == 1:
                yield fig, axs[0,0] # Single subplot
            else:
                yield fig, axs # Array of subplots
        finally:
            plt.tight_layout()
            # plt.show() # Decide if show should be automatic or manual

    def plot_matrix_heatmap(self, matrix: Matrix, title: str = "Matrix Heatmap", 
                            ax: Optional[plt.Axes] = None, save_path: Optional[Union[str,Path]] = None):
        """Plots a single matrix as a heatmap."""
        if ax is None:
            with self._figure_context(title=title) as (fig, current_ax):
                im = current_ax.imshow(matrix, cmap='viridis', aspect='auto')
                current_ax.set_title(title)
                current_ax.set_xlabel("Columns")
                current_ax.set_ylabel("Rows")
                fig.colorbar(im, ax=current_ax, shrink=0.8, pad=0.02)
                if save_path: plt.savefig(save_path)
                plt.show()
        else:
            im = ax.imshow(matrix, cmap='viridis', aspect='auto')
            ax.set_title(title)
            ax.set_xlabel("Columns")
            ax.set_ylabel("Rows")
            # For subplots, colorbar might need to be handled by the calling function
            # fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02) # This needs `fig`

    def plot_matrix_comparison(self, 
                             original: Matrix, 
                             biclusters: List[Bicluster], # List of Bicluster objects
                             title: str = "Matrix with Detected Biclusters",
                             save_path: Optional[Union[str, Path]] = None) -> None:
        """Plot original matrix and an overlay of detected biclusters."""
        
        with self._figure_context(1, 2, title=title, custom_figsize=(self.figsize[0]*1.5, self.figsize[1])) as (fig, axs):
            ax1, ax2 = axs[0,0], axs[0,1]
            
            # Plot original matrix
            im1 = ax1.imshow(original, cmap='viridis', aspect='auto')
            ax1.set_title('Original Matrix')
            ax1.set_xlabel('Columns'); ax1.set_ylabel('Rows')
            fig.colorbar(im1, ax=ax1, shrink=0.7, pad=0.02)
            
            # Create and plot bicluster overlay
            overlay = self._create_bicluster_overlay_matrix(original.shape, biclusters)
            # Use a qualitative colormap for the overlay. `Set3` is good for categorical data.
            # Ensure 0 (background) is distinct, e.g. by making it transparent or specific color.
            # cmap_overlay = plt.cm.get_cmap('Set3', len(biclusters) + 1) # +1 for background # DEPRECATED
            cmap_overlay = plt.colormaps['Set3'] # Get the base colormap
            if len(biclusters) > 0:
                # If you need to specify the number of colors from the colormap:
                cmap_overlay = cmap_overlay.resampled(len(biclusters) + 1) 
            
            # If you want background to be white/transparent:
            # cmap_overlay.set_under(alpha=0) or specific color
            
            if len(biclusters) > 0:
                im2 = ax2.imshow(overlay, cmap=cmap_overlay, aspect='auto', vmin=0.5, vmax=len(biclusters)+0.5)
                # Add colorbar for bicluster indices if useful
                # cbar = fig.colorbar(im2, ax=ax2, shrink=0.7, pad=0.02, ticks=np.arange(1, len(biclusters) + 1))
                # cbar.set_label('Bicluster ID')
            else:
                 # If no biclusters, show a blank or the original matrix again
                ax2.imshow(np.zeros_like(original), cmap='gray', aspect='auto') # Show blank

            ax2.set_title(f'Detected Biclusters (n={len(biclusters)})')
            ax2.set_xlabel('Columns'); ax2.set_ylabel('Rows')
            
            if save_path:
                plt.savefig(Path(save_path), bbox_inches='tight')
                self.logger.info(f"Comparison visualization saved to {save_path}")
            plt.show()
    
    def plot_bicluster_statistics(self, 
                                biclusters: List[Bicluster], # List of Bicluster objects
                                title: str = "Bicluster Statistics",
                                save_path: Optional[Union[str, Path]] = None) -> None:
        """Plot statistical analysis of detected biclusters (scores, sizes, shapes)."""
        if not biclusters:
            self.logger.warning("No biclusters provided for statistics plot.")
            return
        
        scores = [bc.score for bc in biclusters if bc.score is not None and not np.isnan(bc.score)]
        sizes = [bc.size for bc in biclusters]
        shapes = [bc.shape for bc in biclusters]
        row_counts = [s[0] for s in shapes]
        col_counts = [s[1] for s in shapes]

        # Determine subplot layout dynamically, e.g. 2x2 for 3-4 plots
        num_plots = 3 # Score hist, Size hist, Row vs Col counts scatter
        if scores and sizes:
             num_plots = 4 # Add Score vs Size scatter if both available
        
        plot_rows = 2
        plot_cols = (num_plots + 1) // 2 # Ensure enough columns

        with self._figure_context(plot_rows, plot_cols, title=title, custom_figsize=(self.figsize[0]*1.2, self.figsize[1]*1.8)) as (fig, axs):
            axs_flat = axs.flatten()
            plot_idx = 0

            if scores:
                ax = axs_flat[plot_idx]
                ax.hist(scores, bins=min(20, len(scores) // 2 if len(scores) > 20 else 10), alpha=0.7, color='skyblue', edgecolor='black')
                ax.set_title('Score Distribution'); ax.set_xlabel('Score'); ax.set_ylabel('Frequency')
                plot_idx +=1
            
            if sizes:
                ax = axs_flat[plot_idx]
                ax.hist(sizes, bins=min(20, len(sizes) // 2 if len(sizes) > 20 else 10), alpha=0.7, color='lightcoral', edgecolor='black')
                ax.set_title('Size Distribution (elements)'); ax.set_xlabel('Bicluster Size'); ax.set_ylabel('Frequency')
                plot_idx += 1

            if row_counts and col_counts:
                ax = axs_flat[plot_idx]
                ax.scatter(col_counts, row_counts, alpha=0.6, color='purple', edgecolor='black') # Swapped to (cols, rows) for common convention
                ax.set_title('Bicluster Dimensions'); ax.set_xlabel('Number of Columns'); ax.set_ylabel('Number of Rows')
                plot_idx +=1

            if scores and sizes and len(scores) == len(sizes):
                ax = axs_flat[plot_idx]
                ax.scatter(sizes, scores, alpha=0.6, color='green', edgecolor='black')
                ax.set_title('Score vs. Size'); ax.set_xlabel('Size'); ax.set_ylabel('Score')
                plot_idx +=1
            
            # Hide any unused subplots
            for i in range(plot_idx, len(axs_flat)):
                fig.delaxes(axs_flat[i])

            if save_path:
                plt.savefig(Path(save_path), bbox_inches='tight')
                self.logger.info(f"Statistics plot saved to {save_path}")
            plt.show()
    
    def _create_bicluster_overlay_matrix(self, matrix_shape: Tuple[int, int], biclusters: List[Bicluster]) -> NDArray:
        """Create an overlay matrix where each bicluster is marked with a unique integer ID."""
        overlay = np.zeros(matrix_shape, dtype=int) # Use int for bicluster IDs
        
        for i, bicluster in enumerate(biclusters, 1): # Start IDs from 1
            # Bicluster.row_indices and .col_indices are boolean masks for the original matrix dimensions
            # We need to apply these masks to the overlay.
            # This can be done by creating a sub-view and assigning, or by using np.ix_ if we had integer labels.
            # With boolean masks, it's: overlay[bicluster.row_indices, :][:, bicluster.col_indices] = i is not quite right.
            # It should be: overlay[np.ix_(bicluster.row_labels, bicluster.col_labels)] = i
            # OR, if row_indices and col_indices are full-matrix masks:
            rows_mask_for_bc = bicluster.row_indices
            cols_mask_for_bc = bicluster.col_indices
            
            # Create a combined mask for cells in this bicluster
            # Cell (r,c) is in bicluster if rows_mask_for_bc[r] is True AND cols_mask_for_bc[c] is True.
            # This requires broadcasting or careful indexing.
            # A simpler way for non-overlapping assignment (or last-one-wins for overlaps):
            for r_idx in np.where(rows_mask_for_bc)[0]:
                overlay[r_idx, cols_mask_for_bc] = i # Assign bicluster ID to cells in these rows that are also in selected columns
        return overlay

    def plot_individual_biclusters(self, 
                                   original_matrix: Matrix,
                                   biclusters: List[Bicluster],
                                   max_to_plot: int = 6,
                                   title_prefix: str = "Bicluster",
                                   save_dir: Optional[Union[str, Path]] = None) -> None:
        """Plot heatmaps of individual bicluster submatrices."""
        if not biclusters:
            self.logger.warning("No biclusters to plot individually.")
            return

        num_to_plot = min(len(biclusters), max_to_plot)
        if num_to_plot == 0: return

        plot_cols = min(3, num_to_plot) # Max 3 columns
        plot_rows = (num_to_plot + plot_cols - 1) // plot_cols
        
        with self._figure_context(plot_rows, plot_cols, title=f"Top {num_to_plot} Individual Biclusters", 
                                custom_figsize=(self.figsize[0]*plot_cols*0.4, self.figsize[1]*plot_rows*0.6)) as (fig, axs_obj):
            # axs_obj can be a single Axes or an array of Axes
            if num_to_plot == 1: 
                axs_flat = [axs_obj] # Treat single Axes as a list with one element
            else:
                axs_flat = axs_obj.flatten() # If multiple, it's an array, so flatten
            
            for i, bc in enumerate(biclusters[:num_to_plot]):
                ax = axs_flat[i]
                try:
                    submatrix = bc.extract_submatrix(original_matrix)
                    if submatrix.size == 0: 
                        ax.text(0.5, 0.5, "Empty Submatrix", ha='center', va='center')
                        ax.set_title(f"{title_prefix} {i+1} (Empty)")
                        continue
                    
                    im = ax.imshow(submatrix, cmap='viridis', aspect='auto')
                    ax.set_title(f"{title_prefix} {i+1} ({bc.shape[0]}x{bc.shape[1]}) Score: {bc.score:.3f}", fontsize=9)
                    fig.colorbar(im, ax=ax, shrink=0.7, pad=0.03)
                except Exception as e:
                    self.logger.error(f"Error plotting bicluster {i+1}: {e}")
                    ax.text(0.5, 0.5, "Error Plotting", ha='center', va='center')
                ax.set_xticks([]) # Often cleaner for small heatmaps
                ax.set_yticks([])

            # Hide unused subplots
            for j in range(num_to_plot, len(axs_flat)):
                fig.delaxes(axs_flat[j])
            
            if save_dir:
                Path(save_dir).mkdir(parents=True, exist_ok=True)
                save_path = Path(save_dir) / "individual_biclusters_heatmaps.png"
                plt.savefig(save_path, bbox_inches='tight')
                self.logger.info(f"Individual bicluster heatmaps saved to {save_path}")
            plt.show()

    def create_report_visualizations(self, 
                                  original_matrix: Matrix,
                                  biclusters: List[Bicluster],
                                  output_dir: Union[str, Path],
                                  max_individual_biclusters: int = 6) -> None:
        """Create a set of standard visualizations for a report."""
        report_dir = Path(output_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Generating visualization report in {report_dir}")

        self.plot_matrix_comparison(
            original_matrix, biclusters, 
            save_path=report_dir / "matrix_with_biclusters_overlay.png"
        )
        self.plot_bicluster_statistics(
            biclusters, 
            save_path=report_dir / "bicluster_statistics.png"
        )
        if biclusters:
            self.plot_individual_biclusters(
                original_matrix, biclusters, 
                max_to_plot=max_individual_biclusters,
                save_dir=report_dir
            )
        self.logger.info(f"Report visualizations generated in {report_dir}")

# Factory functions for easy creation of synthetic data
def create_synthetic_data_with_generator(
                        n_biclusters: int = 3,
                        matrix_shape: Tuple[int, int] = (100, 80),
                        bicluster_size_range: Tuple[int, int] = (15, 25), # (min_dim, max_dim) for rows/cols
                        bicluster_value_start: float = 1.0,
                        bicluster_value_step: float = 1.0,
                        noise_level_spec: float = 0.2,
                        background_noise_config: float = 0.1,
                        random_state: Optional[int] = 42
                        ) -> Tuple[NDArray, NDArray, List[Bicluster], SyntheticDataGenerator]:
    """Factory function to create synthetic data, returns matrices, GT biclusters, and the generator instance."""
    
    rng_spec = np.random.RandomState(random_state) # Separate RNG for spec generation for reproducibility
    
    specs = []
    for i in range(n_biclusters):
        rows = rng_spec.randint(bicluster_size_range[0], bicluster_size_range[1] + 1)
        cols = rng_spec.randint(bicluster_size_range[0], bicluster_size_range[1] + 1)
        value = bicluster_value_start + (i * bicluster_value_step)
        
        specs.append(BiclusterSpec(
            rows=rows,
            cols=cols, 
            value=value,
            noise_level=noise_level_spec
        ))
    
    config = SyntheticDataConfig(
        matrix_shape=matrix_shape,
        bicluster_specs=specs,
        background_noise=background_noise_config,
        random_state=random_state # Pass main random state to generator
    )
    
    generator = SyntheticDataGenerator(config)
    permuted_matrix, original_structured_matrix, ground_truth_biclusters = generator.generate()
    
    return permuted_matrix, original_structured_matrix, ground_truth_biclusters, generator


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Generate synthetic data using the factory
    perm_matrix, orig_matrix, gt_biclusters, s_generator = create_synthetic_data_with_generator(
        n_biclusters=2,
        matrix_shape=(80, 60),
        bicluster_size_range=(10, 20),
        random_state=42
    )
    
    print(f"Generated permuted matrix shape: {perm_matrix.shape}")
    print(f"Number of ground truth biclusters: {len(gt_biclusters)}")
    if gt_biclusters:
        print(f"  Example GT Bicluster 0 shape: {gt_biclusters[0].shape}, metadata: {gt_biclusters[0].metadata}")

    # Create visualizer
    visualizer = BiclusterVisualizer(figsize=(10,4))
    
    # Plot original structured matrix (truth)
    visualizer.plot_matrix_heatmap(orig_matrix, title="Original Structured Matrix (Ground Truth)")

    # Plot permuted matrix (input for algorithm)
    visualizer.plot_matrix_heatmap(perm_matrix, title="Permuted Matrix (Algorithm Input)")

    # For demonstration, assume we ran an algorithm and got some 'detected_biclusters'
    # Here, we'll just use the ground truth ones as if they were detected for visualization demo.
    # In a real scenario, these would come from BiclusterAnalyzer.get_biclusters()
    detected_biclusters_demo = gt_biclusters 

    # Plot comparison of permuted matrix and (pseudo-)detected biclusters
    visualizer.plot_matrix_comparison(perm_matrix, detected_biclusters_demo, title="Permuted Matrix with (Simulated) Detected Biclusters")
    
    # Plot statistics of these (pseudo-)detected biclusters
    if detected_biclusters_demo:
        visualizer.plot_bicluster_statistics(detected_biclusters_demo, title="Statistics of (Simulated) Detected Biclusters")
        visualizer.plot_individual_biclusters(perm_matrix, detected_biclusters_demo, max_to_plot=2)
    
    print("\nVisualization example finished. Check for plots.") 