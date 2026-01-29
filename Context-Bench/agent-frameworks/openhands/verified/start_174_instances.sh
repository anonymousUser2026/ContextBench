#!/bin/bash
##############################################################################
# 后台运行 174 个 Verified 实例测试
# 自动配置，无需交互
##############################################################################

set -e

# 创建日志目录
LOG_DIR="./logs_verified_test"
mkdir -p "$LOG_DIR"

# 生成时间戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/verified_174_$TIMESTAMP.log"
PID_FILE="$LOG_DIR/verified_test.pid"

echo "========================================="
echo "后台运行 Verified 测试"
echo "实例数：174（全部 Verified 实例）"
echo "并发数：4"
echo "========================================="
echo ""

# 检查配置文件
if [ ! -f "config.toml" ]; then
    echo "❌ 错误：config.toml 文件不存在！"
    exit 1
fi

# 检查 API 密钥
if grep -q 'api_key = ""' config.toml; then
    echo "❌ 错误：请先在 config.toml 中配置您的 API 密钥！"
    exit 1
fi

echo "✅ 配置已就绪"
echo ""

# 检查是否已有测试在运行
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "⚠️  发现运行中的测试（PID: $OLD_PID）"
        echo "正在停止旧的测试..."
        kill "$OLD_PID" 2>/dev/null || true
        sleep 2
        if ps -p "$OLD_PID" > /dev/null 2>&1; then
            echo "强制终止旧进程..."
            kill -9 "$OLD_PID" 2>/dev/null || true
        fi
        rm -f "$PID_FILE"
        echo "✅ 旧测试已停止"
        echo ""
    fi
fi

# 生成 verified_instances.txt（如果不存在）
if [ ! -f "verified_instances.txt" ]; then
    echo "生成 Verified 实例列表..."
    grep "^Verified" selected_500_instances.csv | cut -d',' -f2 > verified_instances.txt
    echo "✅ 已生成 verified_instances.txt（174 个实例）"
    echo ""
fi

# 显示测试配置
echo "📊 测试配置："
echo "  - 实例数量：174（全部 Verified）"
echo "  - 数据集：princeton-nlp/SWE-bench_Verified"
echo "  - 模型：GPT-5"
echo "  - 并发数：4"
echo "  - 最大迭代：100"
echo "  - 日志文件：$LOG_FILE"
echo ""

# 设置环境变量
export ITERATIVE_EVAL_MODE=true
export EVAL_CONDENSER=summarizer_for_eval

# 启动后台测试
echo "🚀 启动测试..."
echo ""

# 创建临时脚本
cat > /tmp/run_verified_174.sh << 'EOF'
#!/bin/bash
# Run from the OpenHands repo root.
# Override with OPENHANDS_RUN_DIR if you keep the repo elsewhere.
cd "${OPENHANDS_RUN_DIR:-$(pwd)}"

# 设置环境变量
export ITERATIVE_EVAL_MODE=true
export EVAL_CONDENSER=summarizer_for_eval

# 运行测试
./evaluation/benchmarks/swe_bench/scripts/run_infer.sh \
    llm.eval_gpt5 \
    HEAD \
    CodeActAgent \
    174 \
    100 \
    4 \
    princeton-nlp/SWE-bench_Verified \
    test
EOF

chmod +x /tmp/run_verified_174.sh

# 后台运行
nohup /tmp/run_verified_174.sh > "$LOG_FILE" 2>&1 &
TEST_PID=$!

# 保存 PID
echo "$TEST_PID" > "$PID_FILE"

echo "✅ 测试已在后台启动"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📋 测试信息"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "进程 ID：$TEST_PID"
echo "日志文件：$LOG_FILE"
echo "PID 文件：$PID_FILE"
echo ""
echo "预计完成时间：5-9 小时（4 并发）"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 监控命令"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "实时查看日志："
echo "  tail -f $LOG_FILE"
echo ""
echo "查看测试状态："
echo "  ./manage_background_test.sh"
echo ""
echo "检查进程："
echo "  ps -fp $TEST_PID"
echo ""
echo "查看并发任务："
echo "  watch -n 5 'docker ps | grep openhands'"
echo ""
echo "查看进度："
echo "  grep -c completed $LOG_FILE"
echo ""
echo "停止测试："
echo "  kill $TEST_PID"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 等待几秒确认进程启动
sleep 3

if ps -p "$TEST_PID" > /dev/null 2>&1; then
    echo ""
    echo "✅ 进程运行正常"
    echo ""
    echo "提示：此窗口可以关闭，测试会继续在后台运行"
    echo ""
else
    echo ""
    echo "❌ 进程启动失败，请检查日志："
    echo "  cat $LOG_FILE"
    rm -f "$PID_FILE"
    exit 1
fi


