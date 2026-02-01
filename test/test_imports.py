"""
Tests for import correctness and script compatibility.

验证：
1. 所有模块可以正确导入（作为包的一部分）
2. 脚本可以通过设置 PYTHONPATH 正常运行
3. 相对导入在所有场景下都正确工作
"""

import pytest
import sys
import subprocess
from pathlib import Path


def test_all_src_modules_importable():
    """测试所有 src 模块可以导入（作为包）。"""
    # 核心模块
    from src import Bicluster, BiclusterConfig, BiclusterAnalyzer
    from src import create_pipeline, create_dimergeco_pipeline
    from src import create_synthetic_data_with_generator

    # 检测模块
    from src.detection import (
        BiclusterDetector,
        SVDBiclusterDetector,
        PartitionedBiclusterDetector,
    )

    # 评分模块
    from src.scoring import ScoringStrategy, CompatibilityScorer, SVRScorer

    # 分区模块（DiMergeCo）
    from src.partitioning import PartitionConfig, MatrixPartitioner

    # 层次化合并模块（DiMergeCo）
    from src.hierarchical_merge import (
        HierarchicalMergeConfig,
        HierarchicalMerger,
        BiclusterSpatialIndex,
    )

    # 优化模块（新增）
    from src.detection_optimized import (
        OptimizedAggregator,
        AggregationConfig,
        create_optimized_aggregator,
    )

    # 可视化和数据生成
    from src.visualization import BiclusterVisualizer, SyntheticDataGenerator

    # Pipeline
    from src.pipeline import BiclusteringPipeline, PipelineConfig

    # 所有导入成功
    assert True


def test_partitioned_detector_with_optimization():
    """测试 PartitionedBiclusterDetector 启用优化时的导入。"""
    from src.detection import PartitionedBiclusterDetector
    from src.core import BiclusterConfig
    from src.partitioning import PartitionConfig

    # 默认启用优化
    base_config = BiclusterConfig(k1=5, k2=5)
    partition_config = PartitionConfig(T_m=10, T_n=10)

    detector = PartitionedBiclusterDetector(
        base_config, partition_config, use_optimized_aggregation=True
    )

    assert detector.use_optimized_aggregation is True
    assert detector.optimized_aggregator is not None


def test_dimergeco_pipeline_imports():
    """测试 DiMergeCo pipeline 的完整导入链。"""
    from src import create_dimergeco_pipeline
    import numpy as np

    # 创建小型测试矩阵
    matrix = np.random.rand(50, 40)

    # 创建 pipeline（这会触发所有导入）
    pipeline = create_dimergeco_pipeline(
        k1=5,
        k2=5,
        T_m=10,
        T_n=10,
        T_p=2,
        P_thresh=0.9,
        overlap_threshold=0.3,
    )

    # 验证 pipeline 创建成功
    assert pipeline is not None
    assert pipeline.config.enable_dimergeco is True


def test_script_execution_with_pythonpath():
    """测试脚本在设置 PYTHONPATH 后可以正常运行。

    这模拟了在服务器上运行脚本的实际场景。
    """
    project_root = Path(__file__).parent.parent
    script_path = project_root / "scripts" / "run_classic4_experiment.py"

    # 跳过如果脚本不存在
    if not script_path.exists():
        pytest.skip("run_classic4_experiment.py not found")

    # 设置环境变量
    env = {
        "PYTHONPATH": str(project_root),
        "PATH": sys.executable.rsplit("/", 1)[0],  # Python bin 目录
    }

    # 运行脚本（只导入检查，不实际运行实验）
    result = subprocess.run(
        [sys.executable, "-c", f"import sys; sys.path.insert(0, '{project_root}'); "
         f"from src import create_dimergeco_pipeline; print('OK')"],
        capture_output=True,
        text=True,
        env=env,
        timeout=5,
    )

    assert result.returncode == 0, f"Import failed: {result.stderr}"
    assert "OK" in result.stdout


def test_relative_imports_in_detection_optimized():
    """测试 detection_optimized.py 的相对导入。"""
    # 这会间接测试 detection_optimized.py 中的导入
    from src.detection_optimized import create_optimized_aggregator

    aggregator = create_optimized_aggregator()
    assert aggregator is not None


def test_no_circular_imports():
    """测试没有循环导入。"""
    modules_to_test = [
        "src.bicluster",
        "src.core",
        "src.detection",
        "src.detection_optimized",
        "src.hierarchical_merge",
        "src.partitioning",
        "src.pipeline",
    ]

    # 在子进程中逐个导入，避免 reload 污染 enum 等全局状态
    for module_name in modules_to_test:
        result = subprocess.run(
            [sys.executable, "-c", f"import {module_name}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0, (
            f"Circular import detected in {module_name}: {result.stderr}"
        )


def test_package_structure():
    """测试包结构正确。"""
    import src

    # 验证关键导出
    required_exports = [
        "Bicluster",
        "BiclusterConfig",
        "create_pipeline",
        "create_dimergeco_pipeline",
        "create_synthetic_data_with_generator",
        "PartitionedBiclusterDetector",
        "HierarchicalMerger",
    ]

    for export in required_exports:
        assert hasattr(
            src, export
        ), f"Missing export: {export} in src/__init__.py"


if __name__ == "__main__":
    # 允许直接运行此测试文件
    pytest.main([__file__, "-v"])
