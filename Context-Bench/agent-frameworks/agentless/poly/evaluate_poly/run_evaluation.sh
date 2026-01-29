#!/bin/bash
# Multi.csv 评测运行脚本
# 此脚本确保在正确的虚拟环境中运行评测

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# 激活虚拟环境
if [ -d ".venv" ]; then
    echo "激活虚拟环境: .venv"
    source .venv/bin/activate
else
    echo "警告: 未找到 .venv 目录，使用系统 Python"
fi

# 设置环境变量
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# 运行评测脚本
python evaluate_multi/evaluate_multi.py "$@"

