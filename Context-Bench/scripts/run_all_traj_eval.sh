#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  PYTHON="python"
fi

AGENT_FILTER="${AGENT_FILTER:-all}"   # agentless|miniswe|sweagent|all
BENCH_FILTER="${BENCH_FILTER:-all}"   # Multi|Pro|Poly|Verified|all
LIMIT="${LIMIT:-0}"                  # 0 means no limit
PROGRESS_EVERY="${PROGRESS_EVERY:-25}"  # print progress every N processed (0 disables)
RESUME="${RESUME:-1}"                # 1: skip completed instances based on stderr logs

while [[ $# -gt 0 ]]; do
  case "$1" in
    --agent) AGENT_FILTER="${2:?}"; shift 2 ;;
    --bench) BENCH_FILTER="${2:?}"; shift 2 ;;
    --limit) LIMIT="${2:?}"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

GOLD_PATH="${CONTEXTBENCH_GOLD:-results/gold/contextbench_verified.gold.jsonl}"
mkdir -p "$(dirname "$GOLD_PATH")"
if [[ "${SKIP_GOLD_EXPORT:-0}" != "1" && "$GOLD_PATH" == "results/gold/contextbench_verified.gold.jsonl" && ! -s "$GOLD_PATH" ]]; then
  "$PYTHON" scripts/export_gold_contextbench_verified.py
fi

processed=0
declare -A cleared
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

instance_from_file() {
  local b
  b="$(basename "$1")"
  b="${b%.checkpoints.jsonl}"
  b="${b%.traj.json}"
  b="${b%_traj.json}"
  echo "$b"
}

maybe_clear_all() {
  local out_dir="$1"
  if [[ -z "${cleared[$out_dir]+x}" ]]; then
    mkdir -p "$out_dir"
    if [[ "${RESUME}" -eq 1 ]]; then
      : >> "${out_dir}/all.jsonl"
    else
      : > "${out_dir}/all.jsonl"
    fi
    cleared["$out_dir"]=1
  fi
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
  local instance="$5"

  local out_root="${CONTEXTBENCH_RESULTS_ROOT:-results}"
  local cache_dir="${CONTEXTBENCH_CACHE:-./repos}"
  local out_dir="${out_root}/${agent}/${model}/${bench}"
  maybe_clear_all "$out_dir"

  local tmp_out
  tmp_out="$(mktemp)"
  local stderr_log="${out_dir}/${instance}.stderr.log"

  if [[ "${RESUME}" -eq 1 ]] && is_success_log "$stderr_log"; then
    rm -f "$tmp_out"
    return 0
  fi

  set +e
  "$PYTHON" -m contextbench.evaluate \
    --gold "$GOLD_PATH" \
    --pred "$traj_file" \
    --cache "$cache_dir" \
    --out "$tmp_out" \
    >/dev/null 2>"$stderr_log"
  rc=$?
  set -e

  if [[ $rc -ne 0 ]]; then
    echo "FAILED: agent=${agent} model=${model} bench=${bench} traj=${traj_file}" >&2
    rm -f "$tmp_out"
    return 1
  fi

  if [[ -s "$tmp_out" ]]; then
    cat "$tmp_out" >> "${out_dir}/all.jsonl"
  fi
  rm -f "$tmp_out"
}

maybe_progress() {
  local agent="$1"
  local model="$2"
  local bench="$3"
  local instance="$4"
  if [[ "${PROGRESS_EVERY}" -eq 0 ]]; then
    return 0
  fi
  if (( processed % PROGRESS_EVERY == 0 )); then
    echo "progress: processed=${processed} agent=${agent} model=${model} bench=${bench} last=${instance} started_at=${started_at}" >&2
  fi
}

should_take() {
  if [[ "$LIMIT" -eq 0 ]]; then
    return 0
  fi
  [[ "$processed" -lt "$LIMIT" ]]
}

agentless_loop() {
  local f bench instance agent model
  agent="agentless"
  model="default"
  while IFS= read -r f; do
    bench="$(echo "$f" | sed -E 's#^traj/agentless/([^/]+)/.*#\1#')"
    if [[ "$BENCH_FILTER" != "all" && "$bench" != "$BENCH_FILTER" ]]; then
      continue
    fi
    instance="$(instance_from_file "$f")"
    should_take || break
    run_one "$agent" "$model" "$bench" "$f" "$instance" || true
    processed=$((processed + 1))
    maybe_progress "$agent" "$model" "$bench" "$instance"
  done < <(find traj/agentless -type f -name '*_traj.json' | sort)
}

miniswe_loop() {
  local f trajset bench_token bench model instance agent
  agent="miniswe"
  while IFS= read -r f; do
    trajset="$(basename "$(dirname "$(dirname "$f")")")"
    bench_token="$(echo "$trajset" | awk -F_ '{print $2}')"
    model="$(echo "$trajset" | awk -F_ '{print $3}')"
    bench="$(bench_title "$bench_token")"
    if [[ "$BENCH_FILTER" != "all" && "$bench" != "$BENCH_FILTER" ]]; then
      continue
    fi
    instance="$(instance_from_file "$f")"
    should_take || break
    run_one "$agent" "$model" "$bench" "$f" "$instance" || true
    processed=$((processed + 1))
    maybe_progress "$agent" "$model" "$bench" "$instance"
  done < <(find traj/miniswe -type f -name '*.traj.json' | sort)
}

sweagent_loop() {
  local f bench model instance agent dir
  agent="sweagent"
  model="default"
  while IFS= read -r f; do
    if [[ "$f" == traj/sweagent/pro/* ]]; then
      bench="Pro"
    elif [[ "$f" == traj/sweagent/multi/* ]]; then
      bench="Multi"
    elif [[ "$f" == traj/sweagent/traj_verified-swe/* ]]; then
      bench="Verified"
    else
      # Fallback: try to parse patterns like traj_<bench>_* when present.
      dir="$(basename "$(dirname "$(dirname "$f")")")"
      if [[ "$dir" == traj_* ]]; then
        bench="$(echo "$dir" | sed -E 's/^traj_([a-z]+).*/\1/')"
        bench="$(bench_title "$bench")"
      else
        bench="Verified"
      fi
    fi
    if [[ "$BENCH_FILTER" != "all" && "$bench" != "$BENCH_FILTER" ]]; then
      continue
    fi
    instance="$(instance_from_file "$f")"
    should_take || break
    run_one "$agent" "$model" "$bench" "$f" "$instance" || true
    processed=$((processed + 1))
    maybe_progress "$agent" "$model" "$bench" "$instance"
  done < <(find traj/sweagent -type f -name '*.checkpoints.jsonl' 2>/dev/null | sort)
}

case "$AGENT_FILTER" in
  all)
    agentless_loop
    miniswe_loop
    sweagent_loop
    ;;
  agentless) agentless_loop ;;
  miniswe) miniswe_loop ;;
  sweagent) sweagent_loop ;;
  *) echo "Unknown --agent: $AGENT_FILTER" >&2; exit 2 ;;
esac

echo "done: processed=$processed"


