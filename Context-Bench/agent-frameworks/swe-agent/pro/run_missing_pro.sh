#!/bin/bash

# ====================================
# SWE-agent 批量运行脚本 - SWE-bench Pro
# ====================================
# 功能：
# 1. 从missing_pro.txt读取需要运行的实例ID
# 2. 生成完整的instances.yaml（如果不存在）
# 3. 创建过滤后的instances YAML文件
# 4. 运行SWE-agent
# ====================================

set -e  # 遇到错误立即退出

# 配置变量
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MISSING_INSTANCES_FILE="${SCRIPT_DIR}/missing_pro.txt"
OUTPUT_DIR="${SCRIPT_DIR}/trajectories/gpt-5__missing_pro"
CONFIG_FILE="${SCRIPT_DIR}/config/azure_gpt5_multilingual.yaml"

# SWE-bench Pro 相关路径（支持环境变量或相对路径）
SWE_BENCH_PRO_DIR="${SWE_BENCH_PRO_DIR:-../SWE-bench_Pro-os}"
HELPER_CODE_DIR="${SWE_BENCH_PRO_DIR}/helper_code"
FULL_INSTANCES_YAML="${SWE_BENCH_PRO_DIR}/SWE-agent/data/instances.yaml"
FILTERED_INSTANCES_YAML="${SCRIPT_DIR}/data/missing_pro_instances.yaml"

# Docker Hub 用户名（用于生成镜像URI）
DOCKERHUB_USERNAME="jefzda"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查文件是否存在
if [ ! -f "$MISSING_INSTANCES_FILE" ]; then
    log_error "Missing instances file not found: $MISSING_INSTANCES_FILE"
    exit 1
fi

# 统计要运行的实例数量
INSTANCE_COUNT=$(wc -l < "$MISSING_INSTANCES_FILE")
log_info "Found ${INSTANCE_COUNT} instances in ${MISSING_INSTANCES_FILE}"

echo ""
echo "==========================================="
echo "Running SWE-agent on ${INSTANCE_COUNT} instances from missing_pro.txt"
echo "==========================================="
echo ""

# 步骤1: 生成完整的 instances.yaml（如果不存在）
if [ ! -f "$FULL_INSTANCES_YAML" ]; then
    log_warning "Full instances.yaml not found, generating it now..."
    log_info "This may take a few minutes..."
    
    cd "$SWE_BENCH_PRO_DIR"
    python helper_code/generate_sweagent_instances.py \
        --dockerhub_username "$DOCKERHUB_USERNAME" \
        --output_path "$FULL_INSTANCES_YAML"
    
    if [ $? -eq 0 ]; then
        log_success "Generated full instances.yaml"
    else
        log_error "Failed to generate instances.yaml"
        exit 1
    fi
else
    log_success "Full instances.yaml already exists"
fi

# 步骤2: 创建过滤后的 instances YAML 文件
log_info "Creating filtered instances YAML from missing_pro.txt..."

# 创建目标目录
mkdir -p "$(dirname "$FILTERED_INSTANCES_YAML")"

# 使用 Python 脚本来过滤 instances
python3 << PYTHON_SCRIPT
import yaml
import sys

# 读取完整的 instances.yaml
full_yaml_path = "$FULL_INSTANCES_YAML"
filtered_yaml_path = "$FILTERED_INSTANCES_YAML"
missing_instances_file = "$MISSING_INSTANCES_FILE"

try:
    # 读取需要的实例ID列表
    with open(missing_instances_file, 'r') as f:
        wanted_ids = set(line.strip() for line in f if line.strip())
    
    print(f"Loading {len(wanted_ids)} instance IDs from {missing_instances_file}")
    
    # 读取完整的 instances.yaml
    with open(full_yaml_path, 'r') as f:
        all_instances = yaml.safe_load(f)
    
    print(f"Loaded {len(all_instances)} instances from {full_yaml_path}")
    
    # 过滤出需要的实例
    filtered_instances = [
        inst for inst in all_instances 
        if inst.get('instance_id') in wanted_ids
    ]
    
    print(f"Filtered to {len(filtered_instances)} matching instances")
    
    # 写入过滤后的 YAML
    with open(filtered_yaml_path, 'w') as f:
        yaml.dump(filtered_instances, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    
    print(f"Successfully wrote filtered instances to {filtered_yaml_path}")
    
    # 显示未找到的实例（如果有）
    found_ids = set(inst['instance_id'] for inst in filtered_instances)
    missing_ids = wanted_ids - found_ids
    if missing_ids:
        print(f"\nWarning: {len(missing_ids)} instance IDs not found in full instances.yaml:")
        for mid in sorted(list(missing_ids))[:10]:
            print(f"  - {mid}")
        if len(missing_ids) > 10:
            print(f"  ... and {len(missing_ids) - 10} more")
    
    sys.exit(0)
    
except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)
PYTHON_SCRIPT

if [ $? -eq 0 ]; then
    log_success "Created filtered instances YAML: $FILTERED_INSTANCES_YAML"
else
    log_error "Failed to create filtered instances YAML"
    exit 1
fi

echo ""
log_info "Configuration:"
log_info "  - Config file: $CONFIG_FILE"
log_info "  - Instances file: $FILTERED_INSTANCES_YAML"
log_info "  - Output directory: $OUTPUT_DIR"
log_info "  - Number of workers: 10"
echo ""

# 步骤3: 运行 SWE-agent
log_info "Starting SWE-agent batch run..."
echo ""

cd "$SCRIPT_DIR"

python -m sweagent.run.run_batch \
    --config "$CONFIG_FILE" \
    --agent.model.name gpt-5 \
    --agent.model.per_instance_cost_limit 0 \
    --output_dir "$OUTPUT_DIR" \
    --num_workers 4 \
    --random_delay_multiplier 1 \
    --instances.type file \
    --instances.path "$FILTERED_INSTANCES_YAML" \
    --instances.shuffle=False \
    --instances.deployment.type=docker \
    --instances.deployment.startup_timeout 600 \
    --instances.deployment.python_standalone_dir ""

# 检查运行结果
if [ $? -eq 0 ]; then
    echo ""
    log_success "SWE-agent batch run completed successfully!"
    log_info "Results saved to: $OUTPUT_DIR"
    
    # 显示预测结果文件
    if [ -f "${OUTPUT_DIR}/all_preds.jsonl" ]; then
        log_success "Predictions file: ${OUTPUT_DIR}/all_preds.jsonl"
    fi
else
    echo ""
    log_error "SWE-agent batch run failed!"
    exit 1
fi
