#!/bin/bash
##############################################################################
# 后台运行 Verified 测试脚本
# 自动处理日志和进程管理
##############################################################################

# 创建日志目录
LOG_DIR="./logs_verified_test"
mkdir -p "$LOG_DIR"

# 生成时间戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/verified_test_$TIMESTAMP.log"
PID_FILE="$LOG_DIR/verified_test.pid"

echo "========================================="
echo "后台运行 Verified 测试"
echo "========================================="
echo ""
echo "日志文件：$LOG_FILE"
echo "PID 文件：$PID_FILE"
echo ""

# 检查是否已有测试在运行
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "⚠️  警告：已有测试在运行（PID: $OLD_PID）"
        echo ""
        read -p "是否终止旧的测试并启动新测试？(y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            kill "$OLD_PID"
            echo "已终止旧进程"
        else
            echo "已取消"
            exit 0
        fi
    fi
fi

# 启动后台测试
echo "🚀 启动测试..."
echo ""

nohup ./test_verified_only.sh > "$LOG_FILE" 2>&1 &
TEST_PID=$!

# 保存 PID
echo "$TEST_PID" > "$PID_FILE"

echo "✅ 测试已在后台启动"
echo ""
echo "进程 ID：$TEST_PID"
echo "日志文件：$LOG_FILE"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 监控命令："
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "实时查看日志："
echo "  tail -f $LOG_FILE"
echo ""
echo "检查进程状态："
echo "  ps -fp $TEST_PID"
echo ""
echo "终止测试："
echo "  kill $TEST_PID"
echo ""
echo "查看 Docker 容器（并发任务）："
echo "  watch -n 5 'docker ps | grep openhands'"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 等待几秒确认进程启动
sleep 3

if ps -p "$TEST_PID" > /dev/null 2>&1; then
    echo "✅ 进程运行正常"
    echo ""
    echo "开始实时显示日志（Ctrl+C 退出监控，不会终止测试）："
    echo ""
    sleep 2
    tail -f "$LOG_FILE"
else
    echo "❌ 进程启动失败，请检查日志："
    echo "  cat $LOG_FILE"
    rm -f "$PID_FILE"
    exit 1
fi

