#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  PYTHON="python"
fi

AGENT_FILTER="${AGENT_FILTER:-all}"   # openhands|sweagent|all
BENCH_FILTER="${BENCH_FILTER:-all}"   # Multi|Pro|Poly|Verified|all
LIMIT="${LIMIT:-0}"                   # 0 means no limit
PROGRESS_EVERY="${PROGRESS_EVERY:-25}" # print progress every N processed
RESUME="${RESUME:-1}"                 # 1: skip completed instances

while [[ $# -gt 0 ]]; do
  case "$1" in
    --agent) AGENT_FILTER="${2:?}"; shift 2 ;;
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
  local out_dir_override="${6:-}"

  local out_root="${CONTEXTBENCH_RESULTS_ROOT:-results}"
  local cache_dir="${CONTEXTBENCH_CACHE:-.cache/repos}"
  local out_dir="${out_dir_override:-${out_root}/${agent}/${model}/${bench}}"
  
  mkdir -p "$out_dir"
  
  local stderr_log="${out_dir}/${batch_name}.stderr.log"
  local tmp_out="${out_dir}/${batch_name}.tmp.jsonl"

  if [[ "${RESUME}" -eq 1 ]] && is_success_log "$stderr_log"; then
    echo "  ✓ Skipped (completed): $batch_name" >&2
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

  # If available, split the combined stderr into per-instance logs.
  if [[ -f "scripts/split_eval_stderr.py" ]]; then
    "$PYTHON" scripts/split_eval_stderr.py --log "$stderr_log" --out-dir "${out_dir}/instances" >/dev/null 2>&1 || true
  fi
}

run_one_instance() {
  local agent="$1"
  local model="$2"
  local bench="$3"
  local traj_file="$4"
  local instance_id="$5"

  local out_root="${CONTEXTBENCH_RESULTS_ROOT:-results}"
  local cache_dir="${CONTEXTBENCH_CACHE:-.cache/repos}"
  local out_dir="${out_root}/${agent}/${model}/${bench}"
  
  mkdir -p "$out_dir"
  
  local stderr_log="${out_dir}/${instance_id}.stderr.log"
  local tmp_out="${out_dir}/${instance_id}.tmp.jsonl"

  if [[ "${RESUME}" -eq 1 ]] && is_success_log "$stderr_log"; then
    return 0
  fi

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
    rm -f "$tmp_out"
    return 1
  fi

  if [[ -s "$tmp_out" ]]; then
    cat "$tmp_out" >> "${out_dir}/all.jsonl"
    rm -f "$tmp_out"
  fi
}

maybe_progress() {
  local agent="$1"
  local model="$2"
  local bench="$3"
  if [[ "${PROGRESS_EVERY}" -eq 0 ]]; then
    return 0
  fi
  if (( processed % PROGRESS_EVERY == 0 )); then
    echo "Progress: ${processed} processed (agent=${agent}, model=${model}, bench=${bench}, elapsed=$(($(date +%s) - $(date -d "$started_at" +%s 2>/dev/null || echo 0)))s)" >&2
  fi
}

should_take() {
  if [[ "$LIMIT" -eq 0 ]]; then
    return 0
  fi
  [[ "$processed" -lt "$LIMIT" ]]
}

# ============================================================================
# OpenHands Processing Functions
# ============================================================================

openhands_multi() {
  local agent="openhands"
  local model="gpt-5"
  local bench="Multi"
  
  if [[ "$BENCH_FILTER" != "all" && "$BENCH_FILTER" != "Multi" ]]; then
    return 0
  fi
  
  echo "" >&2
  echo "=== OpenHands: Multi Benchmark ===" >&2
  
  local multi_dir="traj/openhands/Multi"
  if [[ ! -d "$multi_dir" ]]; then
    echo "  Warning: Directory not found: $multi_dir" >&2
    return 0
  fi
  
  # Results stored per language:
  # results/openhands/gpt-5/Multi/<lang>/{all.jsonl, <lang>.stderr.log, instances/*.stderr.log}
  
  # Process each language file
  local lang_files=(c cpp go java javascript rust typescript)
  for lang in "${lang_files[@]}"; do
    local lang_file="${multi_dir}/${lang}.jsonl"
    if [[ ! -f "$lang_file" ]]; then
      continue
    fi
    
    should_take || break

    local lang_out_dir="results/${agent}/${model}/${bench}/${lang}"
    mkdir -p "$lang_out_dir"
    if [[ "${RESUME}" -ne 1 ]]; then
      : > "${lang_out_dir}/all.jsonl"
    fi

    run_one_batch "$agent" "$model" "$bench" "$lang_file" "$lang" "$lang_out_dir" || true
    processed=$((processed + 1))
    maybe_progress "$agent" "$model" "$bench"
  done
}

openhands_poly() {
  local agent="openhands"
  local model="gpt-5"
  local bench="Poly"
  
  if [[ "$BENCH_FILTER" != "all" && "$BENCH_FILTER" != "Poly" ]]; then
    return 0
  fi
  
  echo "" >&2
  echo "=== OpenHands: Poly Benchmark ===" >&2
  
  local poly_dir="traj/openhands/poly/AmazonScience__SWE-PolyBench-test/CodeActAgent"
  if [[ ! -d "$poly_dir" ]]; then
    echo "  Warning: Directory not found: $poly_dir" >&2
    return 0
  fi
  
  local out_dir="results/${agent}/${model}/${bench}"
  mkdir -p "$out_dir"
  if [[ "${RESUME}" -ne 1 ]]; then
    : > "${out_dir}/all.jsonl"
  fi
  
  while IFS= read -r batch_dir; do
    local batch_name="$(basename "$batch_dir")"
    local output_file="${batch_dir}/output.jsonl"
    
    if [[ ! -f "$output_file" ]]; then
      continue
    fi
    
    should_take || break
    
    run_one_batch "$agent" "$model" "$bench" "$output_file" "$batch_name" || true
    processed=$((processed + 1))
    maybe_progress "$agent" "$model" "$bench"
  done < <(find "$poly_dir" -mindepth 1 -maxdepth 1 -type d -name "gpt-5_maxiter_200_N_batch-*-of-12" | sort)
}

openhands_pro() {
  local agent="openhands"
  local model="gpt-5"
  local bench="Pro"
  
  if [[ "$BENCH_FILTER" != "all" && "$BENCH_FILTER" != "Pro" ]]; then
    return 0
  fi
  
  echo "" >&2
  echo "=== OpenHands: Pro Benchmark ===" >&2
  
  local pro_dir="traj/openhands/pro/ScaleAI__SWE-bench_Pro-test/CodeActAgent"
  if [[ ! -d "$pro_dir" ]]; then
    echo "  Warning: Directory not found: $pro_dir" >&2
    return 0
  fi
  
  local out_dir="results/${agent}/${model}/${bench}"
  mkdir -p "$out_dir"
  if [[ "${RESUME}" -ne 1 ]]; then
    : > "${out_dir}/all.jsonl"
  fi
  
  while IFS= read -r batch_dir; do
    local batch_name="$(basename "$batch_dir")"
    local output_file="${batch_dir}/output.jsonl"
    
    if [[ ! -f "$output_file" ]]; then
      continue
    fi
    
    should_take || break
    
    run_one_batch "$agent" "$model" "$bench" "$output_file" "$batch_name" || true
    processed=$((processed + 1))
    maybe_progress "$agent" "$model" "$bench"
  done < <(find "$pro_dir" -mindepth 1 -maxdepth 1 -type d -name "gpt-5_maxiter_200_N_pro-batch-*-of-6" | sort)
}

openhands_verified() {
  local agent="openhands"
  local bench="Verified"
  
  if [[ "$BENCH_FILTER" != "all" && "$BENCH_FILTER" != "Verified" ]]; then
    return 0
  fi
  
  echo "" >&2
  echo "=== OpenHands: Verified Benchmark ===" >&2
  
  local verified_dir="traj/openhands/verified"
  if [[ ! -d "$verified_dir" ]]; then
    echo "  Warning: Directory not found: $verified_dir" >&2
    return 0
  fi
  
  # Check for llm_completions directory (new format)
  local llm_completions_dir="${verified_dir}/llm_completions"
  if [[ -d "$llm_completions_dir" ]]; then
    echo "  Using llm_completions format (per-instance directories)" >&2
    
    local model_name="default"
    local out_dir="results/${agent}/${model_name}/${bench}"
    mkdir -p "$out_dir"
    if [[ "${RESUME}" -ne 1 ]]; then
      : > "${out_dir}/all.jsonl"
    fi
    
    local total_dirs=$(find "$llm_completions_dir" -mindepth 1 -maxdepth 1 -type d | wc -l)
    echo "  Found $total_dirs instance directories" >&2
    
    local count=0
    while IFS= read -r instance_dir; do
      local instance_id="$(basename "$instance_dir")"
      
      should_take || break
      
      run_one_instance "$agent" "$model_name" "$bench" "$instance_dir" "$instance_id" || true
      processed=$((processed + 1))
      count=$((count + 1))
      
      if (( count % 25 == 0 )); then
        echo "  Progress: $count/$total_dirs instances" >&2
      fi
      
      maybe_progress "$agent" "$model_name" "$bench"
    done < <(find "$llm_completions_dir" -mindepth 1 -maxdepth 1 -type d | sort)
    
    echo "  Processed $count instances" >&2
    
  else
    # Fallback to output.jsonl format (if exists)
    echo "  Using output.jsonl format" >&2
    
    while IFS= read -r output_file; do
      local filename="$(basename "$output_file")"
      
      # Determine model from filename
      local model_name="default"
      if [[ "$filename" == *"gpt-5"* ]]; then
        model_name="gpt-5"
      elif [[ "$filename" == *"gpt-4"* ]]; then
        model_name="gpt-4"
      elif [[ "$filename" == *"claude"* ]]; then
        model_name="claude"
      fi
      
      local out_dir="results/${agent}/${model_name}/${bench}"
      mkdir -p "$out_dir"
      if [[ "${RESUME}" -ne 1 ]]; then
        : > "${out_dir}/all.jsonl"
      fi
      
      local batch_name="${filename%.jsonl}"
      
      should_take || break
      
      run_one_batch "$agent" "$model_name" "$bench" "$output_file" "$batch_name" || true
      processed=$((processed + 1))
      maybe_progress "$agent" "$model_name" "$bench"
    done < <(find "$verified_dir" -maxdepth 1 -type f -name "output*.jsonl" | sort)
  fi
}

# ============================================================================
# SWE-agent Processing Functions
# ============================================================================

sweagent_verified() {
  local agent="sweagent"
  local model="default"
  local bench="Verified"
  
  if [[ "$BENCH_FILTER" != "all" && "$BENCH_FILTER" != "Verified" ]]; then
    return 0
  fi
  
  echo "" >&2
  echo "=== SWE-agent: Verified Benchmark ===" >&2
  
  local traj_dir="traj/sweagent/Verified"
  if [[ ! -d "$traj_dir" ]]; then
    echo "  Warning: Directory not found: $traj_dir" >&2
    return 0
  fi
  
  local out_dir="results/${agent}/${model}/${bench}"
  mkdir -p "$out_dir"
  if [[ "${RESUME}" -ne 1 ]]; then
    : > "${out_dir}/all.jsonl"
  fi
  
  local total_dirs=$(find "$traj_dir" -mindepth 1 -maxdepth 1 -type d ! -name ".*" | wc -l)
  echo "  Found $total_dirs instance directories" >&2
  
  local count=0
  while IFS= read -r instance_dir; do
    local instance_id="$(basename "$instance_dir")"
    local checkpoint_file="${instance_dir}/${instance_id}.checkpoints.jsonl"
    
    if [[ ! -f "$checkpoint_file" ]]; then
      continue
    fi
    
    should_take || break
    
    run_one_instance "$agent" "$model" "$bench" "$checkpoint_file" "$instance_id" || true
    processed=$((processed + 1))
    count=$((count + 1))
    
    if (( count % 25 == 0 )); then
      echo "  Progress: $count/$total_dirs instances" >&2
    fi
    
    maybe_progress "$agent" "$model" "$bench"
  done < <(find "$traj_dir" -mindepth 1 -maxdepth 1 -type d ! -name ".*" | sort)
  
  echo "  Processed $count instances" >&2
}

sweagent_poly() {
  local agent="sweagent"
  local model="default"
  local bench="Poly"
  
  if [[ "$BENCH_FILTER" != "all" && "$BENCH_FILTER" != "Poly" ]]; then
    return 0
  fi
  
  echo "" >&2
  echo "=== SWE-agent: Poly Benchmark ===" >&2
  
  local poly_dir="traj/sweagent/Poly"
  if [[ ! -d "$poly_dir" ]]; then
    echo "  Warning: Directory not found: $poly_dir" >&2
    return 0
  fi
  
  local out_dir="results/${agent}/${model}/${bench}"
  mkdir -p "$out_dir"
  if [[ "${RESUME}" -ne 1 ]]; then
    : > "${out_dir}/all.jsonl"
  fi
  
  local total_dirs=$(find "$poly_dir" -mindepth 1 -maxdepth 1 -type d | wc -l)
  echo "  Found $total_dirs instance directories" >&2
  
  local count=0
  while IFS= read -r instance_dir; do
    local instance_id="$(basename "$instance_dir")"
    
    # Find checkpoint file
    local checkpoint_file
    checkpoint_file=$(find "$instance_dir" -maxdepth 1 -type f -name "*.checkpoints.jsonl" | head -1)
    
    if [[ ! -f "$checkpoint_file" ]]; then
      continue
    fi
    
    should_take || break
    
    run_one_instance "$agent" "$model" "$bench" "$checkpoint_file" "$instance_id" || true
    processed=$((processed + 1))
    count=$((count + 1))
    
    if (( count % 25 == 0 )); then
      echo "  Progress: $count/$total_dirs instances" >&2
    fi
    
    maybe_progress "$agent" "$model" "$bench"
  done < <(find "$poly_dir" -mindepth 1 -maxdepth 1 -type d | sort)
  
  echo "  Processed $count instances" >&2
}

sweagent_pro() {
  local agent="sweagent"
  local model="default"
  local bench="Pro"
  
  if [[ "$BENCH_FILTER" != "all" && "$BENCH_FILTER" != "Pro" ]]; then
    return 0
  fi
  
  echo "" >&2
  echo "=== SWE-agent: Pro Benchmark ===" >&2
  
  local pro_dir="traj/sweagent/pro/pro_bench"
  if [[ ! -d "$pro_dir" ]]; then
    echo "  Warning: Directory not found: $pro_dir" >&2
    return 0
  fi
  
  local out_dir="results/${agent}/${model}/${bench}"
  mkdir -p "$out_dir"
  if [[ "${RESUME}" -ne 1 ]]; then
    : > "${out_dir}/all.jsonl"
  fi
  
  local total_dirs=$(find "$pro_dir" -mindepth 1 -maxdepth 1 -type d | wc -l)
  echo "  Found $total_dirs instance directories" >&2
  
  local count=0
  while IFS= read -r instance_dir; do
    local instance_id="$(basename "$instance_dir")"
    
    # Find checkpoint file (pattern may vary)
    local checkpoint_file
    checkpoint_file=$(find "$instance_dir" -maxdepth 1 -type f -name "*.checkpoints.jsonl" | head -1)
    
    if [[ ! -f "$checkpoint_file" ]]; then
      continue
    fi
    
    should_take || break
    
    run_one_instance "$agent" "$model" "$bench" "$checkpoint_file" "$instance_id" || true
    processed=$((processed + 1))
    count=$((count + 1))
    
    if (( count % 10 == 0 )); then
      echo "  Progress: $count/$total_dirs instances" >&2
    fi
    
    maybe_progress "$agent" "$model" "$bench"
  done < <(find "$pro_dir" -mindepth 1 -maxdepth 1 -type d | sort)
  
  echo "  Processed $count instances" >&2
}

sweagent_multi() {
  local agent="sweagent"
  local model="default"
  local bench="Multi"
  
  if [[ "$BENCH_FILTER" != "all" && "$BENCH_FILTER" != "Multi" ]]; then
    return 0
  fi
  
  echo "" >&2
  echo "=== SWE-agent: Multi Benchmark ===" >&2
  
  local multi_dir="traj/sweagent/multi"
  if [[ ! -d "$multi_dir" ]]; then
    echo "  Warning: Directory not found: $multi_dir" >&2
    return 0
  fi
  
  # Check if directory is empty
  if [[ -z "$(ls -A "$multi_dir" 2>/dev/null)" ]]; then
    echo "  Directory is empty, skipping" >&2
    return 0
  fi
  
  echo "  Multi directory exists but is empty or not implemented" >&2
}

# ============================================================================
# Main Execution
# ============================================================================

echo "========================================" >&2
echo "Context-Bench Trajectory Evaluation" >&2
echo "========================================" >&2
echo "Agent filter: $AGENT_FILTER" >&2
echo "Bench filter: $BENCH_FILTER" >&2
echo "Gold path: $GOLD_PATH" >&2
echo "Resume mode: $RESUME" >&2
echo "Started at: $started_at" >&2
echo "" >&2

case "$AGENT_FILTER" in
  all)
    # Process OpenHands
    openhands_multi
    openhands_poly
    openhands_pro
    openhands_verified
    
    # Process SWE-agent
    sweagent_verified
    sweagent_poly
    sweagent_pro
    sweagent_multi
    ;;
    
  openhands)
    openhands_multi
    openhands_poly
    openhands_pro
    openhands_verified
    ;;
    
  sweagent)
    sweagent_verified
    sweagent_poly
    sweagent_pro
    sweagent_multi
    ;;
    
  *)
    echo "ERROR: Unknown agent: $AGENT_FILTER" >&2
    echo "Valid options: openhands, sweagent, all" >&2
    exit 2
    ;;
esac

echo "" >&2
echo "========================================" >&2
echo "Evaluation Complete" >&2
echo "========================================" >&2
echo "Total files/instances processed: $processed" >&2
echo "Results saved to: results/" >&2
echo "Finished at: $(date -u +%Y-%m-%dT%H:%M:%SZ)" >&2
