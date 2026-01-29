#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  PYTHON="python"
fi

# Prometheus traj roots (user-provided defaults)
PROMETHEUS_VERIFIED_ROOT="${PROMETHEUS_VERIFIED_ROOT:-traj/prometheus/verified}"
PROMETHEUS_PRO_ROOT="${PROMETHEUS_PRO_ROOT:-traj/prometheus/pro}"
PROMETHEUS_POLY_ROOT="${PROMETHEUS_POLY_ROOT:-traj/prometheus/poly}"
PROMETHEUS_MULTI_ROOT="${PROMETHEUS_MULTI_ROOT:-traj/prometheus/multi}"

# Output / cache / gold
OUT_ROOT="${CONTEXTBENCH_RESULTS_ROOT:-results}"
CACHE_DIR="${CONTEXTBENCH_CACHE:-./repos}"
GOLD_PATH="${CONTEXTBENCH_GOLD:-results/gold/contextbench_verified.gold.jsonl}"

mkdir -p "$(dirname "$GOLD_PATH")"
if [[ "${SKIP_GOLD_EXPORT:-0}" != "1" && "$GOLD_PATH" == "results/gold/contextbench_verified.gold.jsonl" && ! -s "$GOLD_PATH" ]]; then
  "$PYTHON" scripts/export_gold_contextbench_verified.py
fi

# Run controls
PROGRESS_EVERY="${PROGRESS_EVERY:-25}"  # 0 disables
RESUME="${RESUME:-1}"                  # 1: skip completed instances based on stderr logs
LIMIT="${LIMIT:-0}"                    # 0 means no limit

# Model label used in results path
MODEL="${PROMETHEUS_MODEL:-gpt-5}"

processed=0
started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

instance_from_file() {
  local b
  b="$(basename "$1")"
  b="${b%.log}"
  echo "$b"
}

is_success_log() {
  local log_path="$1"
  [[ -s "$log_path" ]] || return 1
  grep -q "EVALUATION:" "$log_path" || return 1
  grep -q "Results written to" "$log_path" || return 1
  ! grep -q "Traceback (most recent call last)" "$log_path" || return 1
  return 0
}

maybe_progress() {
  local bench="$1"
  local instance="$2"
  if [[ "${PROGRESS_EVERY}" -eq 0 ]]; then
    return 0
  fi
  if (( processed % PROGRESS_EVERY == 0 )); then
    echo "progress: processed=${processed} agent=prometheus model=${MODEL} bench=${bench} last=${instance} started_at=${started_at}" >&2
  fi
}

should_take() {
  if [[ "$LIMIT" -eq 0 ]]; then
    return 0
  fi
  [[ "$processed" -lt "$LIMIT" ]]
}

run_one() {
  local bench="$1"
  local traj_file="$2"
  local instance="$3"

  local out_dir="${OUT_ROOT}/prometheus/${MODEL}/${bench}"
  mkdir -p "$out_dir"
  local stderr_log="${out_dir}/${instance}.stderr.log"

  if [[ "${RESUME}" -eq 1 ]] && is_success_log "$stderr_log"; then
    return 0
  fi

  local tmp_out
  tmp_out="$(mktemp)"

  set +e
  "$PYTHON" -m contextbench.evaluate \
    --gold "$GOLD_PATH" \
    --pred "$traj_file" \
    --cache "$CACHE_DIR" \
    --out "$tmp_out" \
    >/dev/null 2> >(tee "$stderr_log" >&2)
  rc=$?
  set -e

  if [[ $rc -ne 0 ]]; then
    echo "FAILED: agent=prometheus model=${MODEL} bench=${bench} traj=${traj_file}" >&2
    rm -f "$tmp_out"
    return 1
  fi

  if [[ -s "$tmp_out" ]]; then
    cat "$tmp_out" >> "${out_dir}/all.jsonl"
  fi
  rm -f "$tmp_out"
}

loop_dir() {
  local bench="$1"
  local root="$2"
  local f instance
  local total idx
  total="$(find "$root" -type f -name '*.log' | wc -l | tr -d ' ')"
  idx=0

  if [[ ! -d "$root" ]]; then
    echo "SKIP: missing traj dir: $root" >&2
    return 0
  fi

  while IFS= read -r f; do
    instance="$(instance_from_file "$f")"
    idx=$((idx + 1))
    echo "[${idx}/${total}] agent=prometheus model=${MODEL} bench=${bench} instance=${instance}" >&2
    should_take || break
    run_one "$bench" "$f" "$instance" || true
    processed=$((processed + 1))
    maybe_progress "$bench" "$instance"
  done < <(find "$root" -type f -name '*.log' | sort)
}

loop_dir "Verified" "$PROMETHEUS_VERIFIED_ROOT"
loop_dir "Pro" "$PROMETHEUS_PRO_ROOT"
loop_dir "Poly" "$PROMETHEUS_POLY_ROOT"
loop_dir "Multi" "$PROMETHEUS_MULTI_ROOT"

echo "done: processed=$processed"


