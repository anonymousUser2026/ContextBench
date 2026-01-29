#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  PYTHON="python"
fi

BENCH_FILTER="${BENCH_FILTER:-all}"   # Pro|Poly|Verified|all
LIMIT="${LIMIT:-0}"                   # 0 means no limit
PROGRESS_EVERY="${PROGRESS_EVERY:-10}" # print progress every N processed
RESUME="${RESUME:-1}"                 # 1: skip completed instances

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bench) BENCH_FILTER="${2:?}"; shift 2 ;;
    --limit) LIMIT="${2:?}"; shift 2 ;;
    --no-resume) RESUME=0; shift ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

GOLD_PATH="${CONTEXTBENCH_GOLD:-results/gold/contextbench_verified.gold.jsonl}"
mkdir -p "$(dirname "$GOLD_PATH")"

# Export gold if needed
if [[ "${SKIP_GOLD_EXPORT:-0}" != "1" && "$GOLD_PATH" == "results/gold/contextbench_verified.gold.jsonl" && ! -s "$GOLD_PATH" ]]; then
  echo "Exporting gold annotations..." >&2
  "$PYTHON" scripts/export_gold_contextbench_verified.py
fi

processed=0
started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

bench_title() {
  case "$1" in
    multi) echo "Multi" ;;
    pro) echo "Pro" ;;
    poly) echo "Poly" ;;
    verified) echo "Verified" ;;
    Multi|Pro|Poly|Verified) echo "$1" ;;
    *) echo "$1" ;;
  esac
}

is_success_log() {
  local log_path="$1"
  [[ -s "$log_path" ]] || return 1
  grep -q "EVALUATION:" "$log_path" || return 1
  grep -q "Results written to" "$log_path" || return 1
  ! grep -q "Traceback (most recent call last)" "$log_path" || return 1
  return 0
}

run_one() {
  local agent="$1"
  local model="$2"
  local bench="$3"
  local traj_file="$4"

  local out_root="${CONTEXTBENCH_RESULTS_ROOT:-results}"
  local cache_dir="${CONTEXTBENCH_CACHE:-./repos}"
  local out_dir="${out_root}/${agent}/${model}/${bench}"
  
  mkdir -p "$out_dir"
  
  # For output.jsonl files, we process all instances in one go
  local base_name="$(basename "$(dirname "$traj_file")")"
  local stderr_log="${out_dir}/${base_name}.stderr.log"
  local tmp_out="${out_dir}/${base_name}.tmp.jsonl"

  if [[ "${RESUME}" -eq 1 ]] && is_success_log "$stderr_log"; then
    echo "  Skipping (already completed): $traj_file" >&2
    return 0
  fi

  echo "  Processing: $traj_file" >&2
  
  set +e
  "$PYTHON" -m contextbench.evaluate \
    --gold "$GOLD_PATH" \
    --pred "$traj_file" \
    --cache "$cache_dir" \
    --out "$tmp_out" \
    2>"$stderr_log"
  rc=$?
  set -e

  if [[ $rc -ne 0 ]]; then
    echo "  FAILED: $traj_file (see $stderr_log)" >&2
    rm -f "$tmp_out"
    return 1
  fi

  if [[ -s "$tmp_out" ]]; then
    cat "$tmp_out" >> "${out_dir}/all.jsonl"
    rm -f "$tmp_out"
  fi
  
  echo "  SUCCESS: $traj_file" >&2
}

maybe_progress() {
  local agent="$1"
  local model="$2"
  local bench="$3"
  if [[ "${PROGRESS_EVERY}" -eq 0 ]]; then
    return 0
  fi
  if (( processed % PROGRESS_EVERY == 0 )); then
    echo "Progress: processed=${processed} files, agent=${agent}, model=${model}, bench=${bench}, started_at=${started_at}" >&2
  fi
}

should_take() {
  if [[ "$LIMIT" -eq 0 ]]; then
    return 0
  fi
  [[ "$processed" -lt "$LIMIT" ]]
}

openhands_loop() {
  local f bench agent model
  agent="openhands"
  
  echo "=== Starting OpenHands trajectory evaluation ===" >&2
  echo "Agent: $agent" >&2
  echo "Bench filter: $BENCH_FILTER" >&2
  echo "Gold path: $GOLD_PATH" >&2
  echo "" >&2
  
  while IFS= read -r f; do
    # Determine bench from path
    if [[ "$f" == *"/pro/"* ]]; then
      bench="Pro"
    elif [[ "$f" == *"/poly/"* ]]; then
      bench="Poly"
    elif [[ "$f" == *"/multi/"* ]]; then
      bench="Multi"
    elif [[ "$f" == *"/verified/"* ]]; then
      bench="Verified"
    else
      # Try to extract from path
      bench="$(echo "$f" | sed -E 's#^traj/openhands/([^/]+)/.*#\1#')"
      bench="$(bench_title "$bench")"
    fi
    
    if [[ "$BENCH_FILTER" != "all" && "$bench" != "$BENCH_FILTER" ]]; then
      continue
    fi
    
    # Determine model from path
    if [[ "$f" == *"gpt-5"* ]]; then
      model="gpt-5"
    elif [[ "$f" == *"gpt-4"* ]]; then
      model="gpt-4"
    elif [[ "$f" == *"claude"* ]]; then
      model="claude"
    else
      model="default"
    fi
    
    should_take || break
    
    echo "" >&2
    echo "[$((processed + 1))] Bench: $bench, Model: $model" >&2
    run_one "$agent" "$model" "$bench" "$f" || true
    processed=$((processed + 1))
    maybe_progress "$agent" "$model" "$bench"
  done < <(find traj/openhands -type f -name 'output.jsonl' | sort)
}

openhands_loop

echo "" >&2
echo "=== Evaluation complete ===" >&2
echo "Total trajectory files processed: $processed" >&2
echo "Results saved to: results/openhands/" >&2

