#!/bin/bash
##############################################################################
# 管理后台测试脚本
# 用于查看状态、监控日志、停止测试
##############################################################################

LOG_DIR="./logs_verified_test"
PID_FILE="$LOG_DIR/verified_test.pid"

# 显示菜单
show_menu() {
    echo "========================================="
    echo "后台测试管理"
    echo "========================================="
    echo ""
    echo "1) 查看测试状态"
    echo "2) 实时查看日志"
    echo "3) 查看所有日志文件"
    echo "4) 查看 Docker 容器（并发任务）"
    echo "5) 查看系统资源使用"
    echo "6) 停止测试"
    echo "7) 清理旧日志"
    echo "0) 退出"
    echo ""
}

# 检查测试状态
check_status() {
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📊 测试状态"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    if [ ! -f "$PID_FILE" ]; then
        echo "❌ 没有运行中的测试"
        return
    fi
    
    PID=$(cat "$PID_FILE")
    
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "✅ 测试正在运行"
        echo ""
        echo "进程 ID：$PID"
        echo ""
        echo "进程详情："
        ps -fp "$PID"
        echo ""
        echo "运行时长："
        ps -o etime= -p "$PID"
        echo ""
    else
        echo "❌ 测试进程已停止（PID: $PID）"
        echo ""
        echo "最新日志文件："
        ls -lt "$LOG_DIR"/*.log 2>/dev/null | head -1
    fi
}

# 实时查看日志
view_logs() {
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📝 实时日志（Ctrl+C 退出）"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    LATEST_LOG=$(ls -t "$LOG_DIR"/*.log 2>/dev/null | head -1)
    
    if [ -z "$LATEST_LOG" ]; then
        echo "❌ 没有找到日志文件"
        return
    fi
    
    echo "日志文件：$LATEST_LOG"
    echo ""
    sleep 1
    tail -f "$LATEST_LOG"
}

# 列出所有日志
list_logs() {
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📂 所有日志文件"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    if [ ! -d "$LOG_DIR" ]; then
        echo "❌ 日志目录不存在"
        return
    fi
    
    ls -lht "$LOG_DIR"/*.log 2>/dev/null || echo "没有日志文件"
    echo ""
    
    read -p "要查看某个日志文件吗？输入文件名（或按 Enter 跳过）: " logfile
    if [ -n "$logfile" ]; then
        if [ -f "$LOG_DIR/$logfile" ]; then
            less "$LOG_DIR/$logfile"
        else
            echo "文件不存在"
        fi
    fi
}

# 查看 Docker 容器
view_containers() {
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🐳 Docker 容器（并发任务）"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    echo "OpenHands 相关容器："
    docker ps | head -1
    docker ps | grep openhands || echo "没有 OpenHands 容器在运行"
    echo ""
    
    read -p "持续监控？(y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo ""
        echo "每 5 秒刷新一次（Ctrl+C 退出）："
        echo ""
        watch -n 5 'docker ps | grep -E "CONTAINER|openhands"'
    fi
}

# 查看系统资源
view_resources() {
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "💻 系统资源使用"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    echo "CPU 和内存："
    top -bn1 | head -5
    echo ""
    
    echo "内存使用详情："
    free -h
    echo ""
    
    echo "磁盘使用："
    df -h | grep -E "Filesystem|/$"
    echo ""
    
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            echo "测试进程资源使用："
            ps -o pid,ppid,%cpu,%mem,vsz,rss,cmd -p "$PID"
            echo ""
        fi
    fi
}

# 停止测试
stop_test() {
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🛑 停止测试"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    if [ ! -f "$PID_FILE" ]; then
        echo "❌ 没有运行中的测试"
        return
    fi
    
    PID=$(cat "$PID_FILE")
    
    if ! ps -p "$PID" > /dev/null 2>&1; then
        echo "❌ 测试进程已停止（PID: $PID）"
        rm -f "$PID_FILE"
        return
    fi
    
    echo "⚠️  将停止测试进程（PID: $PID）"
    echo ""
    read -p "确认停止？(y/n) " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo ""
        echo "正在停止测试..."
        kill "$PID"
        sleep 2
        
        if ps -p "$PID" > /dev/null 2>&1; then
            echo "进程未响应，强制终止..."
            kill -9 "$PID"
        fi
        
        rm -f "$PID_FILE"
        echo "✅ 测试已停止"
    else
        echo "已取消"
    fi
}

# 清理旧日志
cleanup_logs() {
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🧹 清理旧日志"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    if [ ! -d "$LOG_DIR" ]; then
        echo "❌ 日志目录不存在"
        return
    fi
    
    LOG_COUNT=$(ls "$LOG_DIR"/*.log 2>/dev/null | wc -l)
    
    if [ "$LOG_COUNT" -eq 0 ]; then
        echo "没有日志文件"
        return
    fi
    
    echo "找到 $LOG_COUNT 个日志文件"
    echo ""
    ls -lht "$LOG_DIR"/*.log
    echo ""
    
    read -p "保留最新几个日志文件？(输入数字，0=全部删除): " keep_count
    
    if ! [[ "$keep_count" =~ ^[0-9]+$ ]]; then
        echo "无效输入"
        return
    fi
    
    if [ "$keep_count" -eq 0 ]; then
        read -p "确认删除所有日志？(y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -f "$LOG_DIR"/*.log
            echo "✅ 已删除所有日志"
        fi
    else
        ls -t "$LOG_DIR"/*.log | tail -n +$((keep_count + 1)) | xargs rm -f 2>/dev/null
        REMAINING=$(ls "$LOG_DIR"/*.log 2>/dev/null | wc -l)
        echo "✅ 已清理，保留 $REMAINING 个最新日志"
    fi
}

# 主循环
while true; do
    clear
    show_menu
    read -p "请选择 (0-7): " choice
    echo ""
    
    case $choice in
        1) check_status ;;
        2) view_logs ;;
        3) list_logs ;;
        4) view_containers ;;
        5) view_resources ;;
        6) stop_test ;;
        7) cleanup_logs ;;
        0) echo "再见！"; exit 0 ;;
        *) echo "无效选择" ;;
    esac
    
    echo ""
    read -p "按 Enter 继续..." 
done

