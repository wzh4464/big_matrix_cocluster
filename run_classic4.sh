#!/bin/bash
# 便捷脚本：运行 CLASSIC4 实验
# 自动设置 PYTHONPATH 并运行实验

# 获取脚本所在目录（项目根目录）
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# 设置 PYTHONPATH 包含项目根目录
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# 运行实验
echo "============================================"
echo "CLASSIC4 DiMergeCo 实验"
echo "============================================"
echo "项目路径: ${SCRIPT_DIR}"
echo "PYTHONPATH: ${PYTHONPATH}"
echo "============================================"
echo ""

python3 "${SCRIPT_DIR}/scripts/run_classic4_experiment.py" "$@"

exit $?
