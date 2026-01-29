#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  PYTHON="python"
fi

BENCH_FILTER="${BENCH_FILTER:-all}"   # Multi|Pro|Poly|Verified|all
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

is_success_log() {
  local log_path="$1"
  [[ -s "$log_path" ]] || return 1
  grep -q "EVALUATION:" "$log_path" || return 1
  grep -q "Results written to" "$log_path" || return 1
  ! grep -q "Traceback (most recent call last)" "$log_path" || return 1
  return 0
}

run_one_instance() {
  local agent="$1"
  local model="$2"
  local bench="$3"
  local context_file="$4"
  local instance="$5"

  local out_root="${CONTEXTBENCH_RESULTS_ROOT:-results}"
  local cache_dir="${CONTEXTBENCH_CACHE:-./repos}"
  local out_dir="${out_root}/${agent}/${model}/${bench}"
  
  mkdir -p "$out_dir"
  
  local stderr_log="${out_dir}/${instance}.stderr.log"
  local tmp_out="${out_dir}/${instance}.tmp.jsonl"

  if [[ "${RESUME}" -eq 1 ]] && is_success_log "$stderr_log"; then
    echo "  ✓ Already completed: $instance" >&2
    return 0
  fi

  echo "  → Processing: $instance" >&2
  
  set +e
  "$PYTHON" -m contextbench.evaluate \
    --gold "$GOLD_PATH" \
    --pred "$context_file" \
    --cache "$cache_dir" \
    --out "$tmp_out" \
    2>"$stderr_log"
  rc=$?
  set -e

  if [[ $rc -ne 0 ]]; then
    echo "  ✗ FAILED: $instance (see $stderr_log)" >&2
    rm -f "$tmp_out"
    return 1
  fi

  if [[ -s "$tmp_out" ]]; then
    cat "$tmp_out" >> "${out_dir}/all.jsonl"
    rm -f "$tmp_out"
  fi
  
  echo "  ✓ SUCCESS: $instance" >&2
}

maybe_progress() {
  local agent="$1"
  local model="$2"
  local bench="$3"
  if [[ "${PROGRESS_EVERY}" -eq 0 ]]; then
    return 0
  fi
  if (( processed % PROGRESS_EVERY == 0 )); then
    echo "Progress: ${processed} files processed (agent=${agent}, model=${model}, bench=${bench})" >&2
  fi
}

should_take() {
  if [[ "$LIMIT" -eq 0 ]]; then
    return 0
  fi
  [[ "$processed" -lt "$LIMIT" ]]
}

instance_from_file() {
  local b
  b="$(basename "$1")"
  b="${b%.context.json}"
  b="${b%.patch_context.txt}"
  b="${b%.checkpoints.jsonl}"
  b="${b%.traj}"
  echo "$b"
}

process_multi() {
  local agent="sweagent"
  local model="default"
  local bench="Multi"
  
  if [[ "$BENCH_FILTER" != "all" && "$BENCH_FILTER" != "Multi" ]]; then
    return 0
  fi
  
  echo "" >&2
  echo "=== Processing Multi Benchmark ===" >&2
  
  local multi_dir="traj/sweagent/multi"
  if [[ ! -d "$multi_dir" ]]; then
    echo "Warning: Multi directory not found: $multi_dir" >&2
    return 0
  fi
  
  # Clear all.jsonl for Multi if not resuming
  local out_dir="results/${agent}/${model}/${bench}"
  mkdir -p "$out_dir"
  if [[ "${RESUME}" -ne 1 ]]; then
    : > "${out_dir}/all.jsonl"
  fi
  
  # Process multi_forcel directory (context.json files with extended format)
  local forcel_dir="${multi_dir}/multi_forcel"
  if [[ -d "$forcel_dir" ]]; then
    echo "  Processing multi_forcel (context.json files)..." >&2
    while IFS= read -r context_file; do
      local instance="$(instance_from_file "$context_file")"
      should_take || break
      run_one_instance "$agent" "$model" "$bench" "$context_file" "$instance" || true
      processed=$((processed + 1))
      maybe_progress "$agent" "$model" "$bench"
    done < <(find "$forcel_dir" -maxdepth 1 -type f -name '*.context.json' | sort)
  fi
  
  # Process multi_wjm directory (patch_context.txt files)
  local wjm_dir="${multi_dir}/multi_wjm"
  if [[ -d "$wjm_dir" ]]; then
    echo "  Processing multi_wjm (patch_context.txt files)..." >&2
    while IFS= read -r context_file; do
      local instance="$(basename "$(dirname "$(dirname "$context_file")")")"
      should_take || break
      run_one_instance "$agent" "$model" "$bench" "$context_file" "$instance" || true
      processed=$((processed + 1))
      maybe_progress "$agent" "$model" "$bench"
    done < <(find "$wjm_dir" -type f -name '*.patch_context.txt' | sort)
  fi
  
  echo "  Multi benchmark complete" >&2
}

process_poly() {
  local agent="sweagent"
  local model="default"
  local bench="Poly"
  
  if [[ "$BENCH_FILTER" != "all" && "$BENCH_FILTER" != "Poly" ]]; then
    return 0
  fi
  
  echo "" >&2
  echo "=== Processing Poly Benchmark ===" >&2
  
  local poly_dir="traj/sweagent/Poly"
  if [[ ! -d "$poly_dir" ]]; then
    echo "Warning: Poly directory not found: $poly_dir" >&2
    return 0
  fi
  
  # Clear all.jsonl for Poly if not resuming
  local out_dir="results/${agent}/${model}/${bench}"
  mkdir -p "$out_dir"
  if [[ "${RESUME}" -ne 1 ]]; then
    : > "${out_dir}/all.jsonl"
  fi
  
  # Process patch_context.txt files
  while IFS= read -r context_file; do
    local instance="$(basename "$(dirname "$context_file")")"
    should_take || break
    run_one_instance "$agent" "$model" "$bench" "$context_file" "$instance" || true
    processed=$((processed + 1))
    maybe_progress "$agent" "$model" "$bench"
  done < <(find "$poly_dir" -type f -name '*.patch_context.txt' | sort)
  
  echo "  Poly benchmark complete" >&2
}

process_pro() {
  local agent="sweagent"
  local model="default"
  local bench="Pro"
  
  if [[ "$BENCH_FILTER" != "all" && "$BENCH_FILTER" != "Pro" ]]; then
    return 0
  fi
  
  echo "" >&2
  echo "=== Processing Pro Benchmark ===" >&2
  
  local pro_dir="traj/sweagent/pro"
  if [[ ! -d "$pro_dir" ]]; then
    echo "Warning: Pro directory not found: $pro_dir" >&2
    return 0
  fi
  
  # Clear all.jsonl for Pro if not resuming
  local out_dir="results/${agent}/${model}/${bench}"
  mkdir -p "$out_dir"
  if [[ "${RESUME}" -ne 1 ]]; then
    : > "${out_dir}/all.jsonl"
  fi
  
  # Process pro_forcel directory (.traj files with patch_context in info.patch_context - extended format)
  local forcel_dir="${pro_dir}/pro_forcel"
  if [[ -d "$forcel_dir" ]]; then
    echo "  Processing pro_forcel (.traj files)..." >&2
    while IFS= read -r traj_file; do
      # Extract instance_id from directory name: keep instance_ prefix
      local dir_name="$(basename "$(dirname "$traj_file")")"
      local instance="$dir_name"  # Keep 'instance_' prefix
      should_take || break
      run_one_instance "$agent" "$model" "$bench" "$traj_file" "$instance" || true
      processed=$((processed + 1))
      maybe_progress "$agent" "$model" "$bench"
    done < <(find "$forcel_dir" -type f -name '*.traj' | sort)
  fi
  
  # Process checkpoints.jsonl files (other pro directories)
  while IFS= read -r checkpoint_file; do
    local instance="$(instance_from_file "$checkpoint_file")"
    should_take || break
    run_one_instance "$agent" "$model" "$bench" "$checkpoint_file" "$instance" || true
    processed=$((processed + 1))
    maybe_progress "$agent" "$model" "$bench"
  done < <(find "$pro_dir" -type f -name '*.checkpoints.jsonl' | sort)
  
  echo "  Pro benchmark complete" >&2
}

process_verified() {
  local agent="sweagent"
  local model="default"
  local bench="Verified"
  
  if [[ "$BENCH_FILTER" != "all" && "$BENCH_FILTER" != "Verified" ]]; then
    return 0
  fi
  
  echo "" >&2
  echo "=== Processing Verified Benchmark ===" >&2
  
  local verified_dir="traj/sweagent/Verified"
  if [[ ! -d "$verified_dir" ]]; then
    echo "Warning: Verified directory not found: $verified_dir" >&2
    return 0
  fi
  
  # Clear all.jsonl for Verified if not resuming
  local out_dir="results/${agent}/${model}/${bench}"
  mkdir -p "$out_dir"
  if [[ "${RESUME}" -ne 1 ]]; then
    : > "${out_dir}/all.jsonl"
  fi
  
  # Process patch_context.txt files
  while IFS= read -r context_file; do
    local instance="$(basename "$(dirname "$context_file")")"
    should_take || break
    run_one_instance "$agent" "$model" "$bench" "$context_file" "$instance" || true
    processed=$((processed + 1))
    maybe_progress "$agent" "$model" "$bench"
  done < <(find "$verified_dir" -type f -name '*.patch_context.txt' | sort)
  
  echo "  Verified benchmark complete" >&2
}

# Main execution
echo "========================================" >&2
echo "SWE-agent Trajectory Evaluation" >&2
echo "========================================" >&2
echo "Bench filter: $BENCH_FILTER" >&2
echo "Gold path: $GOLD_PATH" >&2
echo "Resume mode: $RESUME" >&2
echo "" >&2

# Process each benchmark
process_multi
process_poly
process_pro
process_verified

echo "" >&2
echo "========================================" >&2
echo "Evaluation Complete" >&2
echo "========================================" >&2
echo "Total instances processed: $processed" >&2
echo "Results saved to: results/sweagent/" >&2
