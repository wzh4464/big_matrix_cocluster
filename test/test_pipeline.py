import pytest
import numpy as np
from pathlib import Path
import shutil
from src.pipeline import BiclusteringPipeline, PipelineConfig, run_complete_analysis

# ====== Fixtures ======

@pytest.fixture
def temp_output_dir(tmp_path):
    yield tmp_path
    shutil.rmtree(tmp_path, ignore_errors=True)

@pytest.fixture
def basic_pipeline(temp_output_dir):
    config = PipelineConfig(
        k1=2, k2=2, output_directory=str(temp_output_dir),
        save_intermediate_results=True, generate_visualizations=False, create_detailed_report=False
    )
    return BiclusteringPipeline(config)

# ====== Tests ======

def test_generate_synthetic_and_fit(basic_pipeline):
    basic_pipeline.generate_synthetic_data(n_biclusters=1, matrix_shape=(10, 10), bicluster_size_range=(3, 5), noise_level=0.1)
    basic_pipeline.fit()
    results = basic_pipeline.get_results()
    assert results is not None
    assert results.matrix_shape == (10, 10)
    assert len(results.biclusters) > 0

def test_pipeline_print_summary(capsys, basic_pipeline):
    basic_pipeline.generate_synthetic_data(n_biclusters=1, matrix_shape=(10, 10), bicluster_size_range=(3, 5), noise_level=0.1)
    basic_pipeline.fit()
    basic_pipeline.print_summary()
    captured = capsys.readouterr()
    assert "BICLUSTERING ANALYSIS RESULTS" in captured.out

def test_run_complete_analysis(temp_output_dir):
    pipeline = run_complete_analysis(
        synthetic_config={"n_biclusters": 2, "matrix_shape": (12, 8), "bicluster_size_range": (2, 4)},
        analysis_config={"k1": 2, "k2": 2, "tolerance": 0.05},
        output_directory=str(temp_output_dir),
    )
    results = pipeline.get_results()
    assert results is not None
    assert results.matrix_shape == (12, 8)
    assert len(results.biclusters) > 0
    assert (Path(temp_output_dir) / "analysis_report.txt").exists()

@pytest.mark.parametrize("shape", [(8, 8), (20, 10)])
def test_pipeline_various_shapes(basic_pipeline, shape):
    basic_pipeline.generate_synthetic_data(n_biclusters=1, matrix_shape=shape, bicluster_size_range=(2, 4), noise_level=0.1)
    basic_pipeline.fit()
    results = basic_pipeline.get_results()
    assert results.matrix_shape == shape

def test_pipeline_no_matrix_error(basic_pipeline):
    with pytest.raises(ValueError):
        basic_pipeline.fit() 