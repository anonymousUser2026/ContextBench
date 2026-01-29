#!/bin/bash
##############################################################################
# 自动清理：每当 output.jsonl 增加一行（一个实例完成）就清理无用镜像
##############################################################################

set -euo pipefail

OUTPUT_FILE="${1:-}"
PID_FILE="${2:-verified_test.pid}"
SLEEP_SEC="${3:-300}"
LOG_FILE="${4:-docker_prune.log}"

if [ -z "$OUTPUT_FILE" ]; then
  echo "用法: $0 <output.jsonl> [pid_file] [sleep_sec] [log_file]" | tee -a "$LOG_FILE"
  exit 1
fi

echo "$(date '+%F %T') 启动自动清理，监控: $OUTPUT_FILE" | tee -a "$LOG_FILE"
echo "$(date '+%F %T') PID 文件: $PID_FILE, 间隔: ${SLEEP_SEC}s" | tee -a "$LOG_FILE"

last_count=0

while true; do
  # 如果主测试进程已结束，则退出
  if [ -f "$PID_FILE" ]; then
    test_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    if [ -n "$test_pid" ] && ! ps -p "$test_pid" >/dev/null 2>&1; then
      echo "$(date '+%F %T') 测试进程已结束，停止自动清理" | tee -a "$LOG_FILE"
      exit 0
    fi
  fi

  if [ -f "$OUTPUT_FILE" ]; then
    count="$(wc -l < "$OUTPUT_FILE" | tr -d ' ')"
    if [ "$count" -gt "$last_count" ]; then
      echo "$(date '+%F %T') 发现新增完成实例: $last_count -> $count，开始清理" | tee -a "$LOG_FILE"
      # 仅清理已停止的 OpenHands 容器（不动镜像）
      docker ps -a --filter "status=exited" --filter "name=openhands" -q | xargs -r docker rm | tee -a "$LOG_FILE" || true
      echo "$(date '+%F %T') 清理完成" | tee -a "$LOG_FILE"
      last_count="$count"
    fi
  fi

  sleep "$SLEEP_SEC"
done

