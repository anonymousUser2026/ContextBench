#!/bin/bash
# 清理项目缓存和残留进程

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "开始清理项目缓存和残留进程"
echo "=========================================="
echo ""

# ============================================
# 1. 清理 MagentLess 残留进程（排除 proxy）
# ============================================
echo "1. 查找并终止 MagentLess 残留进程（保留 proxy）..."

# 查找相关进程（排除 proxy）
PROCESSES=(
    "evaluate_multi.py"
    "generate_traj_json.py"
    "convert_patches.py"
    "extract_patch_from_traj.py"
)

KILLED_COUNT=0
for proc in "${PROCESSES[@]}"; do
    # 查找进程 PID（排除 proxy）
    PIDS=$(ps aux | grep -E "[p]ython.*$proc|$proc" | grep -v "openai_proxy" | grep -v "proxy" | awk '{print $2}' || true)
    
    if [ -n "$PIDS" ]; then
        echo "  找到 $proc 进程: $PIDS"
        for pid in $PIDS; do
            if kill -0 "$pid" 2>/dev/null; then
                CMD=$(ps -p "$pid" -o cmd= 2>/dev/null | head -c 80 || echo "$proc")
                echo "    终止进程 $pid: $CMD"
                kill -TERM "$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
                KILLED_COUNT=$((KILLED_COUNT + 1))
            fi
        done
    fi
done

# 查找 MagentLess 相关进程（排除 proxy）
MAGENTLESS_PIDS=$(ps aux | grep -E "[p]ython.*MagentLess|MagentLess.*[p]ython" | grep -v "openai_proxy" | grep -v "proxy" | grep -v grep | awk '{print $2}' || true)
if [ -n "$MAGENTLESS_PIDS" ]; then
    echo "  找到 MagentLess 相关进程: $MAGENTLESS_PIDS"
    for pid in $MAGENTLESS_PIDS; do
        if kill -0 "$pid" 2>/dev/null; then
            CMD=$(ps -p "$pid" -o cmd= 2>/dev/null | head -c 80 || echo "MagentLess")
            echo "    终止进程 $pid: $CMD"
            kill -TERM "$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
            KILLED_COUNT=$((KILLED_COUNT + 1))
        fi
    done
fi

# 专门查找 MagentLess 脚本进程（更彻底）
MAGENTLESS_SCRIPTS=(
    "agentless/fl/localize.py"
    "agentless/fl/retrieve.py"
    "agentless/fl/combine.py"
    "agentless/repair/repair.py"
    "agentless/repair/rerank.py"
    "localize.py"
    "retrieve.py"
    "combine.py"
    "repair.py"
    "rerank.py"
)

for script in "${MAGENTLESS_SCRIPTS[@]}"; do
    SCRIPT_PIDS=$(ps aux | grep -E "[p]ython.*$script|$script" | grep -v "openai_proxy" | grep -v "proxy" | grep -v grep | awk '{print $2}' || true)
    if [ -n "$SCRIPT_PIDS" ]; then
        echo "  找到 MagentLess 脚本进程 ($script): $SCRIPT_PIDS"
        for pid in $SCRIPT_PIDS; do
            if kill -0 "$pid" 2>/dev/null; then
                CMD=$(ps -p "$pid" -o cmd= 2>/dev/null | head -c 80 || echo "$script")
                echo "    终止进程 $pid: $CMD"
                kill -TERM "$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
                KILLED_COUNT=$((KILLED_COUNT + 1))
            fi
        done
    fi
done

if [ $KILLED_COUNT -eq 0 ]; then
    echo "  ✓ 无 MagentLess 残留进程（proxy 已保留）"
else
    echo "  ✓ 已终止 $KILLED_COUNT 个 MagentLess 进程（proxy 已保留）"
fi

# 等待进程完全退出
sleep 2

echo ""

# ============================================
# 2. 清理 MagentLess 结果目录
# ============================================
echo "2. 清理 MagentLess 结果目录..."

if [ -d "$PROJECT_ROOT/MagentLess/results" ]; then
    # 只清理 Multi_* 开头的目录（不清理其他评测的结果）
    MULTI_DIRS=$(find "$PROJECT_ROOT/MagentLess/results" -maxdepth 1 -type d -name "Multi_*" 2>/dev/null || true)
    
    if [ -n "$MULTI_DIRS" ]; then
        TOTAL_SIZE=0
        for dir in $MULTI_DIRS; do
            if [ -d "$dir" ]; then
                SIZE=$(du -sk "$dir" 2>/dev/null | cut -f1 || echo "0")
                TOTAL_SIZE=$((TOTAL_SIZE + SIZE))
                echo "  删除: $dir ($(numfmt --to=iec-i --suffix=B $((SIZE * 1024)) 2>/dev/null || echo "${SIZE}KB"))"
                rm -rf "$dir"
            fi
        done
        if [ $TOTAL_SIZE -gt 0 ]; then
            echo "  ✓ 已清理 $(numfmt --to=iec-i --suffix=B $((TOTAL_SIZE * 1024)) 2>/dev/null || echo "${TOTAL_SIZE}KB")"
        fi
    else
        echo "  ✓ 无 Multi_* 结果目录"
    fi
else
    echo "  ✓ results 目录不存在"
fi

echo ""

# ============================================
# 3. 清理 playground 临时仓库目录
# ============================================
echo "3. 清理 playground 临时仓库目录..."

PLAYGROUND_DIR="$PROJECT_ROOT/MagentLess/playground"
if [ -d "$PLAYGROUND_DIR" ]; then
    # 统计目录数量
    DIR_COUNT=$(find "$PLAYGROUND_DIR" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l || echo "0")
    
    if [ "$DIR_COUNT" -gt 0 ]; then
        SIZE=$(du -sk "$PLAYGROUND_DIR" 2>/dev/null | cut -f1 || echo "0")
        echo "  删除 $DIR_COUNT 个临时仓库目录 ($(numfmt --to=iec-i --suffix=B $((SIZE * 1024)) 2>/dev/null || echo "${SIZE}KB"))"
        rm -rf "$PLAYGROUND_DIR"/*
        echo "  ✓ 已清理 playground 目录"
    else
        echo "  ✓ playground 目录为空"
    fi
else
    echo "  ✓ playground 目录不存在"
fi

echo ""

# ============================================
# 4. 清理 MagentLess 运行时日志文件
# ============================================
echo "4. 清理 MagentLess 运行时日志文件..."

# 只清理 MagentLess/results 目录下的日志文件
MAGENTLESS_LOGS=$(find "$PROJECT_ROOT/MagentLess/results" -type f -name "*.log" 2>/dev/null || true)

if [ -n "$MAGENTLESS_LOGS" ]; then
    COUNT=$(echo "$MAGENTLESS_LOGS" | wc -l)
    echo "  删除 $COUNT 个 MagentLess 日志文件"
    echo "$MAGENTLESS_LOGS" | xargs rm -f 2>/dev/null || true
    echo "  ✓ 已清理 MagentLess 日志文件"
else
    echo "  ✓ 无 MagentLess 日志文件"
fi

echo ""

# ============================================
# 5. 清理不完整的 traj.json 文件（可选）
# ============================================
echo "5. 检查不完整的 traj.json 文件..."

TRAJ_DIR="$PROJECT_ROOT/results/Multi/trajs"
if [ -d "$TRAJ_DIR" ]; then
    # 查找可能不完整的 traj.json（文件很小或格式错误）
    INCOMPLETE_COUNT=0
    while IFS= read -r traj_file; do
        if [ -f "$traj_file" ]; then
            # 检查文件大小（小于 100 字节可能不完整）
            SIZE=$(stat -f%z "$traj_file" 2>/dev/null || stat -c%s "$traj_file" 2>/dev/null || echo "0")
            if [ "$SIZE" -lt 100 ]; then
                echo "  删除不完整的文件: $traj_file ($SIZE 字节)"
                rm -f "$traj_file"
                INCOMPLETE_COUNT=$((INCOMPLETE_COUNT + 1))
            elif ! python3 -m json.tool "$traj_file" >/dev/null 2>&1; then
                echo "  删除格式错误的文件: $traj_file"
                rm -f "$traj_file"
                INCOMPLETE_COUNT=$((INCOMPLETE_COUNT + 1))
            fi
        fi
    done < <(find "$TRAJ_DIR" -name "*_traj.json" -type f 2>/dev/null || true)
    
    if [ $INCOMPLETE_COUNT -eq 0 ]; then
        echo "  ✓ 无 incomplete traj.json 文件"
    else
        echo "  ✓ 已清理 $INCOMPLETE_COUNT 个不完整的 traj.json 文件"
    fi
else
    echo "  ✓ trajs 目录不存在"
fi

echo ""

# ============================================
# 总结
# ============================================
echo "=========================================="
echo "清理完成！"
echo "=========================================="
echo ""
echo "已清理内容："
echo "  - MagentLess 残留进程: $KILLED_COUNT 个（proxy 已保留）"
echo "  - MagentLess 不完整结果目录"
echo "  - playground 临时仓库"
echo "  - MagentLess 运行时日志"
echo "  - 不完整的 traj.json 文件"
echo ""
echo "注意："
echo "  - proxy 进程未清理"
echo "  - results/Multi/ 目录中的完整 traj.json 文件已保留"
echo "  - 数据文件（data/）未清理"
echo "  - 符号链接（MagentLess/data/）未清理"
echo "  - Python 缓存和其他工具缓存未清理（非有害）"
echo ""

