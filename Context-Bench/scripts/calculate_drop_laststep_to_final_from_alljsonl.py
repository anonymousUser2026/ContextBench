#!/usr/bin/env python3
"""
Compute evidence drop from results/*/all.jsonl (no traj files needed).

Goal: "How much gold context is discarded from the last exploration step to final?"

We compute per instance, per granularity (file / symbol / line):

Let:
  gold_size = |C_G| (from all.jsonl)
  seen_last = coverage_last_step * gold_size
  final_gold = intersection_final (gold items present in final)

We want "fraction of seen gold that gets discarded", so we clamp:
  kept_from_seen = min(final_gold, seen_last)

Then:
  keep = kept_from_seen / seen_last     if seen_last > 0 else 0
  drop = 1 - keep                       if seen_last > 0 else 1

Note:
- We use the last step's reported coverage as a proxy for "seen gold so far".
- If a run can include gold in final that was not seen in exploration, the clamp
  prevents negative drop.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence


def load_jsonl(path: Path) -> List[dict]:
    items: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def quantile(sorted_vals: Sequence[float], q: float) -> float:
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = (len(sorted_vals) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def summarize(vals: List[float]) -> dict:
    if not vals:
        return {"mean": None, "p50": None, "p90": None, "p99": None}
    sv = sorted(vals)
    return {
        "mean": sum(sv) / len(sv),
        "p50": quantile(sv, 0.50),
        "p90": quantile(sv, 0.90),
        "p99": quantile(sv, 0.99),
    }


def compute_drop_from_instance(inst: dict, dim: str) -> dict:
    # Some rows are error records like {"instance_id": "...", "error": "..."}.
    if "final" not in inst or "trajectory" not in inst:
        return {
            "error": inst.get("error", "missing_final_or_trajectory"),
            "gold_size": 0,
            "seen_last": 0.0,
            "final_intersection": 0,
            "kept_from_seen": 0.0,
            "keep": 0.0,
            "drop": 1.0,
            "lost_from_seen": 0.0,
        }

    final_by_dim = inst.get("final") or {}
    traj = inst.get("trajectory") or {}
    steps = traj.get("steps") or []

    if dim not in final_by_dim:
        return {
            "error": f"missing_final_dim:{dim}",
            "gold_size": 0,
            "seen_last": 0.0,
            "final_intersection": 0,
            "kept_from_seen": 0.0,
            "keep": 0.0,
            "drop": 1.0,
            "lost_from_seen": 0.0,
        }

    final = final_by_dim[dim]
    last_cov = float(steps[-1].get("coverage", {}).get(dim, 0.0)) if steps else 0.0
    final_cov = float(final.get("coverage", 0.0))

    gold_size = int(final["gold_size"])
    final_intersection = int(final["intersection"])

    seen_last = last_cov * gold_size  # may be fractional
    if seen_last <= 0.0:
        return {
            "gold_size": gold_size,
            "seen_last": 0.0,
            "final_intersection": final_intersection,
            "kept_from_seen": 0.0,
            "keep": 0.0,
            "drop": 1.0,
            "lost_from_seen": 0.0,
            "last_cov": last_cov,
            "final_cov": final_cov,
            "recall_drop": max(0.0, last_cov - final_cov),
        }

    kept_from_seen = min(float(final_intersection), seen_last)
    keep = kept_from_seen / seen_last
    drop = 1.0 - keep
    lost = seen_last - kept_from_seen
    return {
        "gold_size": gold_size,
        "seen_last": seen_last,
        "final_intersection": final_intersection,
        "kept_from_seen": kept_from_seen,
        "keep": keep,
        "drop": drop,
        "lost_from_seen": lost,
        "last_cov": last_cov,
        "final_cov": final_cov,
        "recall_drop": max(0.0, last_cov - final_cov),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=Path("/root/lh/Context-Bench/results/miniswe"),
        help="Directory containing model/benchmark/all.jsonl",
    )
    parser.add_argument(
        "--only",
        type=str,
        default="",
        help="Optional filter like 'claude45/Multi' (matches results_dir relative paths)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (defaults to <results_dir>/drop_laststep_to_final_from_alljsonl.json)",
    )
    args = parser.parse_args()

    results_dir: Path = args.results_dir
    output_path: Path = args.output or (results_dir / "drop_laststep_to_final_from_alljsonl.json")

    all_jsonls = sorted(results_dir.glob("**/all.jsonl"))
    if args.only:
        needle = args.only.strip("/").lower()
        all_jsonls = [p for p in all_jsonls if needle in str(p.relative_to(results_dir)).lower()]

    dims = ("file", "symbol", "line")

    per_file: Dict[str, dict] = {}
    per_instance: Dict[str, dict] = {}

    for jsonl_path in all_jsonls:
        rel = str(jsonl_path.relative_to(results_dir))
        parts = jsonl_path.relative_to(results_dir).parts
        if len(parts) < 3:
            continue
        model = parts[0]
        benchmark = parts[1]

        instances = load_jsonl(jsonl_path)

        drops_by_dim: Dict[str, List[float]] = {d: [] for d in dims}
        lost_by_dim: Dict[str, List[float]] = {d: [] for d in dims}
        recall_drop_by_dim: Dict[str, List[float]] = {d: [] for d in dims}
        error_rows = 0

        for inst in instances:
            instance_id = inst["instance_id"]
            key = f"{model}/{benchmark}/{instance_id}"

            per_dim = {d: compute_drop_from_instance(inst, d) for d in dims}
            if "final" not in inst or "trajectory" not in inst:
                error_rows += 1
            per_instance[key] = {
                "model": model,
                "benchmark": benchmark,
                "instance_id": instance_id,
                "num_steps": int(inst.get("num_steps", len((inst.get("trajectory") or {}).get("steps") or []))),
                "error": inst.get("error"),
                "per_dim": per_dim,
            }

            for d in dims:
                drops_by_dim[d].append(float(per_dim[d]["drop"]))
                lost_by_dim[d].append(float(per_dim[d]["lost_from_seen"]))
                if "recall_drop" in per_dim[d]:
                    recall_drop_by_dim[d].append(float(per_dim[d]["recall_drop"]))

        per_file[rel] = {
            "model": model,
            "benchmark": benchmark,
            "num_instances": len(instances),
            "error_rows": error_rows,
            "drop_last_to_final": {d: summarize(drops_by_dim[d]) for d in dims},
            "recall_drop_last_to_final": {d: summarize(recall_drop_by_dim[d]) for d in dims},
            "lost_last_to_final": {
                d: {
                    "mean": (sum(lost_by_dim[d]) / len(lost_by_dim[d])) if lost_by_dim[d] else None,
                    "p50": quantile(sorted(lost_by_dim[d]), 0.50) if lost_by_dim[d] else None,
                    "p90": quantile(sorted(lost_by_dim[d]), 0.90) if lost_by_dim[d] else None,
                }
                for d in dims
            },
        }

    output = {
        "meta": {
            "results_dir": str(results_dir),
            "only": args.only,
            "num_all_jsonl_files": len(all_jsonls),
            "dims": list(dims),
        },
        "per_file": per_file,
        "per_instance": per_instance,
    }
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    # Markdown summary with numbers only
    md_path = output_path.parent / "laststep_to_final_summary.md"

    # Compute weighted overall means (across instances) for drop and recall_drop
    overall = {
        d: {"drop_sum": 0.0, "drop_n": 0, "recall_drop_sum": 0.0, "recall_drop_n": 0} for d in dims
    }
    overall_by_group: Dict[str, dict] = {}
    for inst in per_instance.values():
        group = f"{inst['model']}/{inst['benchmark']}"
        if group not in overall_by_group:
            overall_by_group[group] = {
                d: {"drop_sum": 0.0, "drop_n": 0, "recall_drop_sum": 0.0, "recall_drop_n": 0} for d in dims
            }
        for d in dims:
            drop_v = float(inst["per_dim"][d]["drop"])
            overall[d]["drop_sum"] += drop_v
            overall[d]["drop_n"] += 1
            overall_by_group[group][d]["drop_sum"] += drop_v
            overall_by_group[group][d]["drop_n"] += 1

            rd_v = inst["per_dim"][d].get("recall_drop")
            if rd_v is not None:
                overall[d]["recall_drop_sum"] += float(rd_v)
                overall[d]["recall_drop_n"] += 1
                overall_by_group[group][d]["recall_drop_sum"] += float(rd_v)
                overall_by_group[group][d]["recall_drop_n"] += 1

    def mean_or_none(s: float, n: int) -> float | None:
        return (s / n) if n else None

    md_lines: List[str] = []
    md_lines.append("## Last-step → Final summary (from `all.jsonl`)")
    md_lines.append("")
    md_lines.append("- Metrics")
    md_lines.append("  - **drop**: fraction of *seen-last-step* gold that is discarded before final (per your convention).")
    md_lines.append("  - **recall_drop**: `max(0, last_step_coverage - final_coverage)` (coverage treated as recall).")
    md_lines.append("")
    md_lines.append("### Overall (weighted by instances)")
    md_lines.append("")
    md_lines.append("| dim | drop_mean | recall_drop_mean |")
    md_lines.append("|---|---:|---:|")
    for d in dims:
        md_lines.append(
            f"| {d} | {mean_or_none(overall[d]['drop_sum'], overall[d]['drop_n']):.6f} | "
            f"{mean_or_none(overall[d]['recall_drop_sum'], overall[d]['recall_drop_n']):.6f} |"
        )
    md_lines.append("")

    # Aggregate by model (across all benchmarks), weighted by instances
    overall_by_model: Dict[str, dict] = {}
    for inst in per_instance.values():
        model = str(inst["model"])
        if model not in overall_by_model:
            overall_by_model[model] = {
                d: {"drop_sum": 0.0, "drop_n": 0, "recall_drop_sum": 0.0, "recall_drop_n": 0} for d in dims
            }
        for d in dims:
            drop_v = float(inst["per_dim"][d]["drop"])
            overall_by_model[model][d]["drop_sum"] += drop_v
            overall_by_model[model][d]["drop_n"] += 1

            rd_v = inst["per_dim"][d].get("recall_drop")
            if rd_v is not None:
                overall_by_model[model][d]["recall_drop_sum"] += float(rd_v)
                overall_by_model[model][d]["recall_drop_n"] += 1

    md_lines.append("### By model (weighted by instances)")
    md_lines.append("")
    md_lines.append("| model | file_drop | symbol_drop | line_drop | file_recall_drop | symbol_recall_drop | line_recall_drop |")
    md_lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for model in sorted(overall_by_model.keys()):
        row = overall_by_model[model]
        md_lines.append(
            f"| {model} | "
            f"{mean_or_none(row['file']['drop_sum'], row['file']['drop_n']):.6f} | "
            f"{mean_or_none(row['symbol']['drop_sum'], row['symbol']['drop_n']):.6f} | "
            f"{mean_or_none(row['line']['drop_sum'], row['line']['drop_n']):.6f} | "
            f"{mean_or_none(row['file']['recall_drop_sum'], row['file']['recall_drop_n']):.6f} | "
            f"{mean_or_none(row['symbol']['recall_drop_sum'], row['symbol']['recall_drop_n']):.6f} | "
            f"{mean_or_none(row['line']['recall_drop_sum'], row['line']['recall_drop_n']):.6f} |"
        )
    md_lines.append("")

    md_lines.append("### By model/benchmark (weighted by instances)")
    md_lines.append("")
    md_lines.append("| group | file_drop | symbol_drop | line_drop | file_recall_drop | symbol_recall_drop | line_recall_drop |")
    md_lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for group in sorted(overall_by_group.keys()):
        row = overall_by_group[group]
        md_lines.append(
            f"| {group} | "
            f"{mean_or_none(row['file']['drop_sum'], row['file']['drop_n']):.6f} | "
            f"{mean_or_none(row['symbol']['drop_sum'], row['symbol']['drop_n']):.6f} | "
            f"{mean_or_none(row['line']['drop_sum'], row['line']['drop_n']):.6f} | "
            f"{mean_or_none(row['file']['recall_drop_sum'], row['file']['recall_drop_n']):.6f} | "
            f"{mean_or_none(row['symbol']['recall_drop_sum'], row['symbol']['recall_drop_n']):.6f} | "
            f"{mean_or_none(row['line']['recall_drop_sum'], row['line']['recall_drop_n']):.6f} |"
        )
    md_lines.append("")
    md_lines.append("### Per `all.jsonl` file")
    md_lines.append("")
    md_lines.append("| file | N | err | file_drop_mean | symbol_drop_mean | line_drop_mean | file_recall_drop_mean | symbol_recall_drop_mean | line_recall_drop_mean |")
    md_lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for rel, s in sorted(per_file.items()):
        d_drop = s["drop_last_to_final"]
        d_rd = s["recall_drop_last_to_final"]
        md_lines.append(
            f"| {rel} | {s['num_instances']} | {s['error_rows']} | "
            f"{d_drop['file']['mean']:.6f} | {d_drop['symbol']['mean']:.6f} | {d_drop['line']['mean']:.6f} | "
            f"{d_rd['file']['mean']:.6f} | {d_rd['symbol']['mean']:.6f} | {d_rd['line']['mean']:.6f} |"
        )
    md_lines.append("")

    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    # Console table (mean drop per file, per dim)
    print("Drop from last step -> final (computed from all.jsonl)")
    print()
    print(
        f"{'file':60s} {'N':>4s} "
        f"{'file_mean':>10s} {'sym_mean':>10s} {'line_mean':>10s} "
        f"{'file_p90':>10s} {'sym_p90':>10s} {'line_p90':>10s} {'err':>4s}"
    )
    print("-" * 130)
    for rel, s in sorted(per_file.items()):
        df = s["drop_last_to_final"]["file"]
        ds = s["drop_last_to_final"]["symbol"]
        dl = s["drop_last_to_final"]["line"]
        print(
            f"{rel:60.60s} {s['num_instances']:4d} "
            f"{df['mean']:10.4f} {ds['mean']:10.4f} {dl['mean']:10.4f} "
            f"{df['p90']:10.4f} {ds['p90']:10.4f} {dl['p90']:10.4f} {s['error_rows']:4d}"
        )
    print()
    print(f"Wrote JSON: {output_path}")
    print(f"Wrote Markdown: {md_path}")


if __name__ == "__main__":
    main()

