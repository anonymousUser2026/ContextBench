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

run_one_batch() {
  local agent="$1"
  local model="$2"
  local bench="$3"
  local traj_file="$4"
  local batch_name="$5"

  local out_root="${CONTEXTBENCH_RESULTS_ROOT:-results}"
  local cache_dir="${CONTEXTBENCH_CACHE:-./repos}"
  local out_dir="${out_root}/${agent}/${model}/${bench}"
  
  mkdir -p "$out_dir"
  
  local stderr_log="${out_dir}/${batch_name}.stderr.log"
  local tmp_out="${out_dir}/${batch_name}.tmp.jsonl"

  if [[ "${RESUME}" -eq 1 ]] && is_success_log "$stderr_log"; then
    echo "  ✓ Already completed: $batch_name" >&2
    return 0
  fi

  echo "  → Processing: $batch_name" >&2
  
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
    echo "  ✗ FAILED: $batch_name (see $stderr_log)" >&2
    rm -f "$tmp_out"
    return 1
  fi

  if [[ -s "$tmp_out" ]]; then
    cat "$tmp_out" >> "${out_dir}/all.jsonl"
    rm -f "$tmp_out"
  fi
  
  echo "  ✓ SUCCESS: $batch_name" >&2
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

process_multi() {
  local agent="openhands"
  local model="gpt-5"
  local bench="Multi"
  
  if [[ "$BENCH_FILTER" != "all" && "$BENCH_FILTER" != "Multi" ]]; then
    return 0
  fi
  
  echo "" >&2
  echo "=== Processing Multi Benchmark ===" >&2
  
  local multi_dir="traj/openhands/Multi"
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
  
  # Process each language file
  local lang_files=(c cpp go java javascript rust typescript)
  for lang in "${lang_files[@]}"; do
    local lang_file="${multi_dir}/${lang}.jsonl"
    if [[ ! -f "$lang_file" ]]; then
      echo "  Warning: Language file not found: $lang_file" >&2
      continue
    fi
    
    should_take || break
    
    run_one_batch "$agent" "$model" "$bench" "$lang_file" "$lang" || true
    processed=$((processed + 1))
    maybe_progress "$agent" "$model" "$bench"
  done
  
  echo "  Multi benchmark complete" >&2
}

process_poly() {
  local agent="openhands"
  local model="gpt-5"
  local bench="Poly"
  
  if [[ "$BENCH_FILTER" != "all" && "$BENCH_FILTER" != "Poly" ]]; then
    return 0
  fi
  
  echo "" >&2
  echo "=== Processing Poly Benchmark ===" >&2
  
  local poly_dir="traj/openhands/poly/AmazonScience__SWE-PolyBench-test/CodeActAgent"
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
  
  # Process each batch directory
  while IFS= read -r batch_dir; do
    local batch_name="$(basename "$batch_dir")"
    local output_file="${batch_dir}/output.jsonl"
    
    if [[ ! -f "$output_file" ]]; then
      echo "  Warning: output.jsonl not found in $batch_name" >&2
      continue
    fi
    
    should_take || break
    
    run_one_batch "$agent" "$model" "$bench" "$output_file" "$batch_name" || true
    processed=$((processed + 1))
    maybe_progress "$agent" "$model" "$bench"
  done < <(find "$poly_dir" -mindepth 1 -maxdepth 1 -type d -name "gpt-5_maxiter_200_N_batch-*-of-12" | sort)
  
  echo "  Poly benchmark complete" >&2
}

process_pro() {
  local agent="openhands"
  local model="gpt-5"
  local bench="Pro"
  
  if [[ "$BENCH_FILTER" != "all" && "$BENCH_FILTER" != "Pro" ]]; then
    return 0
  fi
  
  echo "" >&2
  echo "=== Processing Pro Benchmark ===" >&2
  
  local pro_dir="traj/openhands/pro/ScaleAI__SWE-bench_Pro-test/CodeActAgent"
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
  
  # Process each batch directory
  while IFS= read -r batch_dir; do
    local batch_name="$(basename "$batch_dir")"
    local output_file="${batch_dir}/output.jsonl"
    
    if [[ ! -f "$output_file" ]]; then
      echo "  Warning: output.jsonl not found in $batch_name" >&2
      continue
    fi
    
    should_take || break
    
    run_one_batch "$agent" "$model" "$bench" "$output_file" "$batch_name" || true
    processed=$((processed + 1))
    maybe_progress "$agent" "$model" "$bench"
  done < <(find "$pro_dir" -mindepth 1 -maxdepth 1 -type d -name "gpt-5_maxiter_200_N_pro-batch-*-of-6" | sort)
  
  echo "  Pro benchmark complete" >&2
}

process_verified() {
  local agent="openhands"
  local bench="Verified"
  
  if [[ "$BENCH_FILTER" != "all" && "$BENCH_FILTER" != "Verified" ]]; then
    return 0
  fi
  
  echo "" >&2
  echo "=== Processing Verified Benchmark ===" >&2
  
  local verified_dir="traj/openhands/verified"
  if [[ ! -d "$verified_dir" ]]; then
    echo "Warning: Verified directory not found: $verified_dir" >&2
    return 0
  fi
  
  # Find all output*.jsonl files (including critic attempts)
  while IFS= read -r output_file; do
    local filename="$(basename "$output_file")"
    local model_name
    
    # Extract model name from parent directory or default to gpt-5
    local parent_dir="$(basename "$(dirname "$(dirname "$output_file")")")"
    if [[ "$parent_dir" == "verified" ]]; then
      # Direct verified dir - check filename or directory name for model
      if [[ "$filename" == *"gpt-4"* ]]; then
        model_name="gpt-4"
      elif [[ "$filename" == *"claude"* ]]; then
        model_name="claude"
      else
        model_name="default"
      fi
    else
      # Check if parent dir contains model info
      if [[ "$parent_dir" == *"gpt-5"* ]]; then
        model_name="gpt-5"
      elif [[ "$parent_dir" == *"gpt-4"* ]]; then
        model_name="gpt-4"
      else
        model_name="default"
      fi
    fi
    
    # Create batch name from filename
    local batch_name="${filename%.jsonl}"
    
    should_take || break
    
    run_one_batch "$agent" "$model_name" "$bench" "$output_file" "$batch_name" || true
    processed=$((processed + 1))
    maybe_progress "$agent" "$model_name" "$bench"
  done < <(find "$verified_dir" -maxdepth 1 -type f -name "output*.jsonl" | sort)
  
  echo "  Verified benchmark complete" >&2
}

# Main execution
echo "========================================" >&2
echo "OpenHands Trajectory Evaluation (Fixed)" >&2
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
echo "Total batches processed: $processed" >&2
echo "Results saved to: results/openhands/" >&2
