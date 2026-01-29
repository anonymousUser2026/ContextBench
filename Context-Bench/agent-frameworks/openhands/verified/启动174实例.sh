#!/bin/bash
##############################################################################
# 后台运行 174 个 Verified 实例测试（使用 conda 环境）
##############################################################################

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# OpenHands repo root (defaults to this directory). Override if needed:
#   export OPENHANDS_RUN_DIR=/path/to/openhands-verified
export OPENHANDS_RUN_DIR="${OPENHANDS_RUN_DIR:-$SCRIPT_DIR}"
# Conda init script location (override if your conda is elsewhere)
CONDA_SH="${CONDA_SH:-/root/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-openhands}"

echo "========================================="
echo "启动 174 个 Verified 实例测试"
echo "并发数：4"
echo "========================================="
echo ""

# 严格模式：只跑 all_verified_174.txt 里的 174 个 instance_id
LIST_FILE="${OPENHANDS_RUN_DIR}/all_verified_174.txt"
SWE_CONFIG_FILE="${OPENHANDS_RUN_DIR}/evaluation/benchmarks/swe_bench/config.toml"

# 把清单写入 evaluation/benchmarks/swe_bench/config.toml（run_infer.py 会从这里读取 selected_ids）
if [ ! -f "$LIST_FILE" ]; then
    echo "❌ 未找到清单文件：$LIST_FILE"
    exit 1
fi

echo "🧩 同步 selected_ids（严格只跑 174）..."
python3 - <<'PY'
import os
from pathlib import Path

run_dir = Path(os.environ["OPENHANDS_RUN_DIR"])
list_file = run_dir / "all_verified_174.txt"
cfg_file = run_dir / "evaluation/benchmarks/swe_bench/config.toml"

ids = [line.strip() for line in list_file.read_text(encoding="utf-8").splitlines() if line.strip()]
ids_unique = list(dict.fromkeys(ids))  # 保序去重

if len(ids) != 174:
    raise SystemExit(f"清单行数不是 174：{len(ids)}（文件：{list_file}）")
if len(ids_unique) != 174:
    raise SystemExit(f"清单存在重复 instance_id：unique={len(ids_unique)}（文件：{list_file}）")

lines = []
lines.append("# Auto-generated. STRICT: only run these 174 Verified instance_ids.")
lines.append("selected_ids = [")
for x in ids_unique:
    # TOML 单引号字符串，避免转义复杂
    lines.append(f"    '{x}',")
lines.append("]")
cfg_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"Wrote {len(ids_unique)} selected_ids -> {cfg_file}")
PY

# 再次校验（防止意外写错路径）
python3 - <<'PY'
import toml
import os
from pathlib import Path

cfg = str(Path(os.environ["OPENHANDS_RUN_DIR"]) / "evaluation/benchmarks/swe_bench/config.toml")
data=toml.load(cfg)
ids=data.get("selected_ids", [])
assert isinstance(ids, list), "selected_ids 不是 list"
assert len(ids)==174, f"selected_ids 数量不是 174：{len(ids)}"
print("✅ selected_ids 校验通过：174")
PY

# 设置日志文件
LOG_FILE="verified_174_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="verified_test.pid"

# 检查并停止旧进程
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "发现运行中的测试（PID: $OLD_PID），正在停止..."
        kill "$OLD_PID" 2>/dev/null || true
        sleep 2
        if ps -p "$OLD_PID" > /dev/null 2>&1; then
            kill -9 "$OLD_PID" 2>/dev/null || true
        fi
        echo "✅ 旧测试已停止"
    fi
fi

# 启动测试
echo "🚀 启动后台测试..."

# 重要：固定复用你指定的输出目录（便于续跑自动跳过已完成）
# 该目录名对应的 eval-note 为：v1.2.1-no-hint-summarizer_for_eval-run_1
# 注意：该目录可能混入历史运行的结果；你已确认“测完后筛选即可”。
FIXED_OUTPUT_DIR="evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Verified-test/CodeActAgent/gpt-5_maxiter_200_N_v1.2.1-no-hint-summarizer_for_eval-run_1"

nohup bash -c "
source \"$CONDA_SH\"
conda activate \"$CONDA_ENV\"
export ITERATIVE_EVAL_MODE=true
export EVAL_CONDENSER=summarizer_for_eval
cd \"$OPENHANDS_RUN_DIR\"
./evaluation/benchmarks/swe_bench/scripts/run_infer.sh \
    llm.forge_gpt5 \
    HEAD \
    CodeActAgent \
    174 \
    200 \
    4 \
    princeton-nlp/SWE-bench_Verified \
    test
" > "$LOG_FILE" 2>&1 &

# 保存 PID
echo $! > "$PID_FILE"

# 启动自动清理（每完成一个实例就清理无用镜像）
PRUNE_PID_FILE="docker_prune.pid"
OUTPUT_FILE="$FIXED_OUTPUT_DIR/output.critic_attempt_1.jsonl"
nohup ./auto_prune_on_complete.sh "$OUTPUT_FILE" "$PID_FILE" 300 "docker_prune.log" >/dev/null 2>&1 &
echo $! > "$PRUNE_PID_FILE"

echo ""
echo "✅ 测试已启动"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📋 测试信息"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "进程 ID：$(cat $PID_FILE)"
echo "日志文件：$LOG_FILE"
echo "实例数量：174（全部 Verified）"
echo "并发数：4"
echo "预计时间：5-9 小时"
echo "严格模式：只跑 all_verified_174.txt（174 个）"
echo "输出目录：$FIXED_OUTPUT_DIR（固定复用，用于自动跳过已完成）"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 监控命令"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "实时查看日志："
echo "  tail -f $LOG_FILE"
echo ""
echo "检查进程："
echo "  ps -fp \$(cat $PID_FILE)"
echo ""
echo "查看并发任务："
echo "  watch -n 5 'docker ps | grep openhands'"
echo ""
echo "查看进度："
echo "  grep -c completed $LOG_FILE"
echo ""
echo "停止测试："
echo "  kill \$(cat $PID_FILE)"
echo ""
echo "停止自动清理："
echo "  kill \$(cat $PRUNE_PID_FILE)"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 等待几秒确认启动
sleep 5

if ps -p $(cat "$PID_FILE") > /dev/null 2>&1; then
    echo ""
    echo "✅ 进程运行正常"
    echo ""
    echo "提示：可以按 Ctrl+C 关闭此窗口，测试会继续运行"
else
    echo ""
    echo "⚠️  进程可能已退出，请检查日志："
    echo "  tail -50 $LOG_FILE"
fi

echo ""

