#!/bin/bash
# DiMergeCo大规模实验 - 服务器运行脚本
#
# 用法：
#   ./experiment/run_server.sh classic4              # 运行CLASSIC4实验
#   ./experiment/run_server.sh large                 # 运行大规模实验
#   ./experiment/run_server.sh batch                 # 批量运行
#   ./experiment/run_server.sh custom 15000 12000 8  # 自定义规模

set -e  # 遇到错误立即退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查Python环境
check_python() {
    echo -e "${YELLOW}检查Python环境...${NC}"

    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}错误: 未找到Python3${NC}"
        exit 1
    fi

    python_version=$(python3 --version | cut -d' ' -f2)
    echo -e "${GREEN}✓ Python版本: $python_version${NC}"

    # 检查必要的包
    if ! python3 -c "import numpy" 2>/dev/null; then
        echo -e "${RED}错误: 未安装numpy${NC}"
        echo "请运行: pip install -e ."
        exit 1
    fi

    echo -e "${GREEN}✓ 依赖包检查通过${NC}"
}

# 显示系统信息
show_system_info() {
    echo -e "${YELLOW}系统信息:${NC}"

    # CPU信息
    if command -v nproc &> /dev/null; then
        cpu_cores=$(nproc)
        echo "  CPU核心数: $cpu_cores"
    fi

    # 内存信息
    if command -v free &> /dev/null; then
        total_mem=$(free -h | awk '/^Mem:/ {print $2}')
        avail_mem=$(free -h | awk '/^Mem:/ {print $7}')
        echo "  总内存: $total_mem"
        echo "  可用内存: $avail_mem"
    fi

    # 磁盘空间
    if command -v df &> /dev/null; then
        disk_avail=$(df -h . | awk 'NR==2 {print $4}')
        echo "  可用磁盘空间: $disk_avail"
    fi

    echo ""
}

# 设置环境变量
setup_env() {
    export PYTHONPATH=$(pwd)
    export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
    export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}

    echo -e "${YELLOW}环境变量:${NC}"
    echo "  PYTHONPATH=$PYTHONPATH"
    echo "  OMP_NUM_THREADS=$OMP_NUM_THREADS"
    echo ""
}

# 运行实验
run_experiment() {
    local exp_type=$1
    shift  # 移除第一个参数

    # 创建输出目录
    mkdir -p results logs

    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local log_file="logs/${exp_type}_${timestamp}.log"

    echo -e "${GREEN}开始运行 $exp_type 实验...${NC}"
    echo "日志文件: $log_file"
    echo ""

    case $exp_type in
        classic4|large|xlarge|amazon)
            python3 experiment/run_large_scale_experiments.py \
                --dataset $exp_type \
                --output results/$exp_type \
                2>&1 | tee $log_file
            ;;

        batch)
            local datasets=${1:-"classic4 large"}
            echo "批量运行数据集: $datasets"
            python3 experiment/run_large_scale_experiments.py \
                --batch \
                --batch-datasets $datasets \
                --output results/batch_${timestamp} \
                2>&1 | tee $log_file
            ;;

        custom)
            local rows=${1:-10000}
            local cols=${2:-8000}
            local biclusters=${3:-6}
            echo "自定义规模: ${rows}×${cols}, ${biclusters} biclusters"
            python3 experiment/run_large_scale_experiments.py \
                --custom \
                --rows $rows \
                --cols $cols \
                --biclusters $biclusters \
                --output results/custom_${timestamp} \
                2>&1 | tee $log_file
            ;;

        test)
            echo -e "${YELLOW}运行快速测试...${NC}"
            python3 experiment/simple_test.py 2>&1 | tee $log_file
            ;;

        *)
            echo -e "${RED}未知的实验类型: $exp_type${NC}"
            show_usage
            exit 1
            ;;
    esac

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ 实验完成！${NC}"
        echo "查看结果: results/$exp_type/"
        echo "查看日志: $log_file"
    else
        echo -e "${RED}✗ 实验失败！${NC}"
        echo "检查日志: $log_file"
        exit 1
    fi
}

# 显示用法
show_usage() {
    cat << EOF
${GREEN}DiMergeCo 大规模实验 - 服务器运行脚本${NC}

用法:
    $0 <实验类型> [参数...]

实验类型:
    test                           快速测试（验证环境）
    classic4                       CLASSIC4规模 (6,461×4,667)
    large                          大规模 (10,000×8,000)
    xlarge                         超大规模 (20,000×15,000)
    amazon                         Amazon规模 (123,321×23,379, 需要32GB+内存)
    batch [数据集列表]             批量运行（默认: classic4 large）
    custom <rows> <cols> <biclusters>  自定义规模

示例:
    $0 test                        # 快速测试
    $0 classic4                    # 运行CLASSIC4
    $0 batch "classic4 large xlarge"  # 批量运行
    $0 custom 15000 12000 8        # 自定义15K×12K

后台运行:
    nohup $0 classic4 > run.log 2>&1 &

使用screen:
    screen -S dimergeco
    $0 classic4
    # Ctrl+A, D 分离会话

环境变量:
    OMP_NUM_THREADS=8 $0 classic4  # 设置线程数

更多帮助:
    查看 experiment/SERVER_GUIDE.md

EOF
}

# 主函数
main() {
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}DiMergeCo 大规模实验运行脚本${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""

    # 检查参数
    if [ $# -eq 0 ]; then
        show_usage
        exit 0
    fi

    # 检查环境
    check_python
    show_system_info
    setup_env

    # 运行实验
    run_experiment "$@"
}

# 执行主函数
main "$@"
