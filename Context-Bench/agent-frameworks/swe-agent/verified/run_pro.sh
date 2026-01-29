#!/bin/bash
# SWE-bench Pro 批量运行脚本
# 使用方法: ./run_pro.sh [num_workers] [instance_filter]
# 例如: ./run_pro.sh 4 "instance_ansible__ansible-.*"

set -e

# 配置参数
NUM_WORKERS=${1:-4}  # 默认 4 个并发
FILTER=${2:-".*"}    # 默认运行所有 Pro 实例
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/config/azure_gpt5.yaml"
OUTPUT_DIR="${SCRIPT_DIR}/output_pro"
LOG_FILE="${SCRIPT_DIR}/run_pro_$(date +%Y%m%d_%H%M%S).log"

# 切换到 SWE-agent 目录
cd "$SCRIPT_DIR"

# 设置 PYTHONPATH 以确保使用当前目录下的代码
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "SWE-bench Pro 批量运行"
echo "=========================================="
echo "并发数: $NUM_WORKERS"
echo "过滤器: $FILTER"
echo "配置文件: $CONFIG_FILE"
echo "输出目录: $OUTPUT_DIR"
echo "日志文件: $LOG_FILE"
echo "=========================================="

# 运行 SWE-agent
# 我们使用 python -m sweagent.run.run_batch 来运行，确保使用最新的代码
PYTHONPATH="$SCRIPT_DIR" python3 -m sweagent.run.run_batch \
    --instances.type swe_bench \
    --instances.subset pro \
    --instances.split test \
    --instances.filter "$FILTER" \
    --config "$CONFIG_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --num_workers "$NUM_WORKERS" \
    2>&1 | tee "$LOG_FILE"

echo "运行完成！输出保存在: $OUTPUT_DIR"
