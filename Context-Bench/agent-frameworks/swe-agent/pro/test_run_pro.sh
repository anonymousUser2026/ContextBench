#!/bin/bash

# ====================================
# SWE-agent 测试脚本 - 运行 2 个 ansible 实例
# ====================================
# 用途：在运行完整批次前先测试几个实例（使用 ansible 以降低难度）
# ====================================

set -e

# 配置变量
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MISSING_INSTANCES_FILE="${SCRIPT_DIR}/missing_pro.txt"
TEST_INSTANCES_FILE="${SCRIPT_DIR}/test_instances.txt"
OUTPUT_DIR="${SCRIPT_DIR}/trajectories/gpt-5__test_pro"
CONFIG_FILE="${SCRIPT_DIR}/config/azure_gpt5_multilingual.yaml"

# SWE-bench Pro 相关路径（支持环境变量或相对路径）
SWE_BENCH_PRO_DIR="${SWE_BENCH_PRO_DIR:-../SWE-bench_Pro-os}"
FULL_INSTANCES_YAML="${SWE_BENCH_PRO_DIR}/SWE-agent/data/instances.yaml"
FILTERED_INSTANCES_YAML="${SCRIPT_DIR}/data/test_instances.yaml"

DOCKERHUB_USERNAME="jefzda"
TEST_COUNT=2  # 测试 2 个 ansible 实例

# 颜色输出
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}[TEST MODE]${NC} Running ${TEST_COUNT} ansible instances from missing_pro.txt"
echo ""

# 检查 missing_pro.txt 是否存在
if [ ! -f "$MISSING_INSTANCES_FILE" ]; then
    echo -e "${RED}错误: 未找到 $MISSING_INSTANCES_FILE${NC}"
    exit 1
fi

# 创建测试实例文件（从 missing_pro.txt 中筛选 ansible 实例，取前 2 个）
grep 'instance_ansible__ansible' "$MISSING_INSTANCES_FILE" | head -n ${TEST_COUNT} > "$TEST_INSTANCES_FILE"

echo "Test instances:"
cat "$TEST_INSTANCES_FILE"
echo ""

# 检查是否存在完整的 instances.yaml
if [ ! -f "$FULL_INSTANCES_YAML" ]; then
    echo -e "${YELLOW}警告: 未找到完整的 instances.yaml${NC}"
    echo "正在生成..."
    cd "$SWE_BENCH_PRO_DIR"
    python helper_code/generate_sweagent_instances.py \
        --dockerhub_username "$DOCKERHUB_USERNAME" \
        --output_path "$FULL_INSTANCES_YAML"
    cd "$SCRIPT_DIR"
fi

# 使用 Python 创建过滤后的 YAML
mkdir -p "$(dirname "$FILTERED_INSTANCES_YAML")"

python3 << PYTHON_SCRIPT
import yaml

full_yaml_path = "$FULL_INSTANCES_YAML"
filtered_yaml_path = "$FILTERED_INSTANCES_YAML"
test_instances_file = "$TEST_INSTANCES_FILE"

# 读取测试实例ID
with open(test_instances_file, 'r') as f:
    wanted_ids = set(line.strip() for line in f if line.strip())

print(f"Loading {len(wanted_ids)} test instances")

# 读取完整的 instances.yaml
with open(full_yaml_path, 'r') as f:
    all_instances = yaml.safe_load(f)

# 过滤出需要的实例
filtered_instances = [
    inst for inst in all_instances 
    if inst.get('instance_id') in wanted_ids
]

print(f"Found {len(filtered_instances)} matching instances")

# 写入过滤后的 YAML
with open(filtered_yaml_path, 'w') as f:
    yaml.dump(filtered_instances, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

print(f"Created test instances file: {filtered_yaml_path}")
PYTHON_SCRIPT

echo ""
echo -e "${GREEN}Starting test run with ${TEST_COUNT} instances...${NC}"
echo ""

cd "$SCRIPT_DIR"

python -m sweagent.run.run_batch \
    --config "$CONFIG_FILE" \
    --agent.model.name gpt-5 \
    --agent.model.per_instance_cost_limit 0 \
    --output_dir "$OUTPUT_DIR" \
    --num_workers 1 \
    --random_delay_multiplier 0.5 \
    --instances.type file \
    --instances.path "$FILTERED_INSTANCES_YAML" \
    --instances.shuffle=False \
    --instances.deployment.type=docker \
    --instances.deployment.startup_timeout 600
    # --instances.deployment.python_standalone_dir ""

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}[SUCCESS]${NC} Test run completed!"
    echo "Results: $OUTPUT_DIR"
    echo ""
    echo "If everything looks good, run the full batch with:"
    echo "  bash run_missing_pro.sh"
    
    # 清理测试文件
    rm -f "$TEST_INSTANCES_FILE"
else
    echo ""
    echo -e "${RED}[ERROR]${NC} Test run failed!"
    exit 1
fi
