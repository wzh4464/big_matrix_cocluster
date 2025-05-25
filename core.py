# core.py
# This file will contain BiclusterAnalyzer, BiclusterDetector, and related configuration classes.

"""
Modern Python implementation of biclustering algorithms.
Redesigned with proper OOP principles and modern Python features.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict, fields
from typing import Protocol, Optional, List, Tuple, Dict, Any, Union
from pathlib import Path
import numpy as np
from numpy.typing import NDArray
import logging
from contextlib import contextmanager
from enum import Enum
import json

# Type aliases for better readability
Matrix = NDArray[np.floating]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.integer]


class ScoringMethod(Enum):
    """Available scoring methods for bicluster evaluation."""
    PEARSON = "pearson"
    EXPONENTIAL = "exponential"
    COMPATIBILITY = "compatibility"


class ClusteringMethod(Enum):
    """Available clustering methods."""
    KMEANS = "kmeans"
    SPECTRAL = "spectral"


@dataclass
class BiclusterConfig:
    """Configuration for biclustering algorithm."""
    k1: int = 5
    k2: int = 5
    tolerance: float = 0.05
    scoring_method: ScoringMethod = ScoringMethod.EXPONENTIAL
    clustering_method: ClusteringMethod = ClusteringMethod.KMEANS
    random_state: Optional[int] = None
    parallel: bool = False
    min_rows: int = 2
    min_cols: int = 2
    max_overlap: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        data = asdict(self)
        data['scoring_method'] = self.scoring_method.value
        data['clustering_method'] = self.clustering_method.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BiclusterConfig:
        """Create config from dictionary."""
        # Create a new dict to avoid modifying the input `data` if it's used elsewhere.
        processed_data = data.copy()
        if 'scoring_method' in processed_data and isinstance(processed_data['scoring_method'], str):
            processed_data['scoring_method'] = ScoringMethod(processed_data['scoring_method'])
        if 'clustering_method' in processed_data and isinstance(processed_data['clustering_method'], str):
            processed_data['clustering_method'] = ClusteringMethod(processed_data['clustering_method'])
        
        # Get all field names from the dataclass
        config_fields = {f.name for f in fields(cls)}
        # Filter data to only include keys that are actual fields of BiclusterConfig
        filtered_data = {k: v for k, v in processed_data.items() if k in config_fields}
        return cls(**filtered_data)

# Forward declaration for Bicluster if needed by BiclusterAnalyzer early
# However, BiclusterAnalyzer mostly uses List[Bicluster] which works with string annotation
# from .bicluster import Bicluster # This will be a circular import if Bicluster imports from core
# Instead, we'll use string type hints like 'Bicluster' if necessary or ensure Bicluster doesn't import from core.
# For now, BiclusterAnalyzer uses List['Bicluster'], which is fine with Bicluster in a separate file.

from .detection import SVDBiclusterDetector # Import specific detector
from .bicluster import Bicluster # BiclusterAnalyzer methods return List[Bicluster]

class BiclusterAnalyzer:
    """High-level interface for biclustering analysis."""
    
    def __init__(self, config: Optional[BiclusterConfig] = None):
        self.config = config or BiclusterConfig()
        self.detector = SVDBiclusterDetector(self.config) # Use the imported detector
        self.results: Optional[List[Bicluster]] = None
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def fit(self, matrix: Matrix) -> BiclusterAnalyzer:
        """Fit the analyzer to the given matrix."""
        self.logger.info("Starting biclustering analysis")
        self.results = self.detector.detect(matrix)
        return self
    
    def get_biclusters(self) -> List[Bicluster]:
        """Get detected biclusters."""
        if self.results is None:
            raise ValueError("Analyzer has not been fitted yet")
        return self.results
    
    def filter_biclusters(self, min_score: Optional[float] = None, 
                         min_size: Optional[int] = None) -> List[Bicluster]:
        """Filter biclusters based on criteria."""
        biclusters = self.get_biclusters()
        
        if min_score is not None:
            biclusters = [bc for bc in biclusters if bc.score >= min_score]
        
        if min_size is not None:
            biclusters = [bc for bc in biclusters if bc.size >= min_size]
        
        return biclusters
    
    def merge_overlapping(self, threshold: float = 0.5) -> List[Bicluster]:
        """Merge overlapping biclusters."""
        biclusters = self.get_biclusters().copy()
        merged = []
        
        while biclusters:
            current = biclusters.pop(0)
            to_merge = []
            
            # Find overlapping biclusters
            i = 0
            while i < len(biclusters):
                if current.intersection_ratio(biclusters[i]) > threshold:
                    to_merge.append(biclusters.pop(i))
                else:
                    i += 1
            
            # Merge if necessary
            if to_merge:
                current = self._merge_biclusters([current] + to_merge)
            
            merged.append(current)
        
        # Update self.results to reflect merged biclusters if this is the desired behavior
        # self.results = merged 
        return merged
    
    def _merge_biclusters(self, biclusters_to_merge: List[Bicluster]) -> Bicluster:
        """Merge multiple biclusters into one."""
        # Combine indices using logical OR
        row_indices = biclusters_to_merge[0].row_indices.copy()
        col_indices = biclusters_to_merge[0].col_indices.copy()
        
        for bc in biclusters_to_merge[1:]:
            row_indices |= bc.row_indices
            col_indices |= bc.col_indices
        
        # Calculate new score (could be improved, e.g., re-score the merged bicluster)
        # For now, using the score of the first bicluster or average
        avg_score = np.mean([bc.score for bc in biclusters_to_merge])
        
        return Bicluster(
            row_indices=row_indices,
            col_indices=col_indices,
            score=avg_score, # Or re-calculate score on the merged bicluster
            metadata={'merged_from': len(biclusters_to_merge), 'original_scores': [bc.score for bc in biclusters_to_merge]}
        )
    
    def save_results(self, filepath: Union[str, Path]) -> None:
        """Save analysis results to file."""
        filepath = Path(filepath)
        
        results_data = {
            'config': self.config.to_dict(),
            'biclusters': [bc.to_dict() for bc in self.get_biclusters()]
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2)
    
    @contextmanager
    def performance_tracking(self):
        """Context manager for performance tracking."""
        import time
        start_time = time.time()
        
        try:
            yield
        finally:
            end_time = time.time()
            elapsed_time = end_time - start_time
            self.logger.info(f"Analysis completed in {elapsed_time:.2f} seconds")
            # Optionally store this performance metric
            if hasattr(self, 'performance_metrics'):
                 self.performance_metrics['analysis_time'] = elapsed_time


# Example usage and factory functions
def create_analyzer(tolerance: float = 0.02, 
                   k1: int = 10, 
                   k2: int = 10,
                   scoring_method: str = "exponential",
                   clustering_method: str = "kmeans",
                   random_state: Optional[int] = 42) -> BiclusterAnalyzer:
    """Factory function to create configured analyzer."""
    config = BiclusterConfig(
        tolerance=tolerance,
        k1=k1,
        k2=k2,
        scoring_method=ScoringMethod(scoring_method),
        clustering_method=ClusteringMethod(clustering_method),
        random_state=random_state
    )
    return BiclusterAnalyzer(config)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create synthetic data
    np.random.seed(42)
    matrix_data = np.random.rand(100, 50)
    
    # Create and run analyzer
    # Note: SVDBiclusterDetector and Bicluster classes need to be defined or imported for this to run
    # This __main__ block might be better in pipeline.py or a dedicated examples script
    # For now, assuming necessary classes are available via imports for this example.
    
    analyzer_instance = create_analyzer(tolerance=0.05, k1=5, k2=5)
    
    with analyzer_instance.performance_tracking():
        analyzer_instance.fit(matrix_data)
    
    # Get results
    biclusters_found = analyzer_instance.get_biclusters()
    print(f"Found {len(biclusters_found)} biclusters")
    
    for i, bc_item in enumerate(biclusters_found):
        print(f"Bicluster {i}: shape={bc_item.shape}, score={bc_item.score:.4f}") 