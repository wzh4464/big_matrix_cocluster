import pytest
import numpy as np

import src


def test_imports_and_version():
    assert hasattr(src, "create_analyzer")
    assert hasattr(src, "create_synthetic_data")
    assert hasattr(src, "analyze_matrix")
    assert hasattr(src, "coclusterer")
    assert hasattr(src, "score")
    assert isinstance(src.__version__, str)


def test_create_analyzer_and_bicluster():
    analyzer = src.create_analyzer(k1=2, k2=2, tolerance=0.1)
    assert hasattr(analyzer, "fit")
    matrix = np.random.rand(8, 8)
    analyzer.fit(matrix)
    biclusters = analyzer.get_biclusters()
    assert isinstance(biclusters, list)


def test_create_synthetic_data():
    matrix, orig, gt_bics, gen = src.create_synthetic_data(n_biclusters=2, matrix_shape=(10, 10))
    assert matrix.shape == (10, 10)
    assert orig.shape == (10, 10)
    assert isinstance(gt_bics, list)
    assert hasattr(gen, "generate")


def test_analyze_matrix():
    matrix = np.random.rand(10, 10)
    biclusters = src.analyze_matrix(matrix, k1=2, k2=2, tolerance=0.1)
    assert isinstance(biclusters, list)
    # 测试 return_pipeline
    biclusters2, pipeline = src.analyze_matrix(matrix, k1=2, k2=2, tolerance=0.1, return_pipeline=True)
    assert isinstance(biclusters2, list)
    assert hasattr(pipeline, "get_biclusters")


def test_modern_analyzer_and_scorer():
    matrix = np.random.rand(6, 6)
    analyzer = src.create_analyzer(k1=2, k2=2, tolerance=0.1)
    analyzer.fit(matrix)
    biclusters = analyzer.get_biclusters()
    assert isinstance(biclusters, list)
    # 推荐的评分方式
    from src.scoring import CompatibilityScorer
    scorer = CompatibilityScorer()
    subrowI = np.array([True, False, True, False, False, True])
    subcolJ = np.array([True, True, False, False, False, False])
    from src.bicluster import Bicluster
    bicluster = Bicluster(subrowI, subcolJ)
    score_val = scorer.score(matrix, bicluster)
    assert isinstance(score_val, float) 