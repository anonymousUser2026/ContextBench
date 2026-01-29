#!/usr/bin/env python3
"""
Compute Evidence Drop for miniswe trajectories.

We compute two variants at line granularity:

1) Union-drop (standard "retrieval ≠ use"):
   G_seen_union = (⋃_t C_t^A) ∩ C_G
   Keep_union = |C_final ∩ C_G| / |G_seen_union|
   Drop_union = 1 - Keep_union

2) Last-step-drop (what gets discarded from the last explore step to final):
   G_seen_last = C_T^A ∩ C_G          (C_T^A = last EXPLORE_CONTEXT step)
   Keep_last = |C_final ∩ C_G| / |G_seen_last|
   Drop_last = 1 - Keep_last

Edge case convention:
If |G_seen_*| == 0, Keep_* = 0 and Drop_* = 1.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple


LineKey = Tuple[str, int]  # (normalized_file_path, line_number)
Span = Tuple[str, int, int]  # (file_path, start_line, end_line)


def load_jsonl(path: Path) -> List[dict]:
    items: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def load_gold_contexts(gold_file: Path) -> Dict[str, List[dict]]:
    """Map original_inst_id -> gold_ctx list."""
    gold_data = load_jsonl(gold_file)
    gold_contexts: Dict[str, List[dict]] = {}
    for entry in gold_data:
        original_id = entry.get("original_inst_id")
        if not original_id:
            continue
        gold_contexts[original_id] = entry.get("gold_ctx", []) or []
    return gold_contexts


def normalize_file_path(path: str, base_paths: Sequence[str] | None = None) -> str:
    """
    Normalize to a relative-ish path from the repo root.

    Handles common prefixes found in trajectories / gold:
    - /home/<repo>/...  -> <repo>/...
    - /testbed/...      -> ...
    - /workspace/...    -> ...
    """
    if base_paths is None:
        base_paths = ("/testbed/", "/home/", "/workspace/")
    for base in base_paths:
        if path.startswith(base):
            return path[len(base) :]
    return path


def lines_to_set(file_path: str, start: int, end: int) -> Set[LineKey]:
    return {(file_path, i) for i in range(start, end + 1)}


def _parse_context_blocks_from_text(text: str, tag: str) -> List[Span]:
    """
    Parse blocks like:
      <EXPLORE_CONTEXT>
      File: path
      Lines: 10-20
      ...
      </EXPLORE_CONTEXT>
    """
    spans: List[Span] = []
    if not text:
        return spans

    regex = re.compile(rf"<{re.escape(tag)}>(.*?)</{re.escape(tag)}>", re.DOTALL)
    matches = regex.findall(text)
    for match in matches:
        current_file: str | None = None
        for raw_line in match.strip().split("\n"):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("File:"):
                current_file = line.replace("File:", "", 1).strip()
                continue
            if line.startswith("Lines:") and current_file:
                line_range = line.replace("Lines:", "", 1).strip()
                if "-" not in line_range:
                    continue
                start_s, end_s = line_range.split("-", 1)
                try:
                    start_i = int(start_s.strip())
                    end_i = int(end_s.strip())
                except ValueError:
                    continue
                spans.append((current_file, start_i, end_i))
    return spans


def parse_contexts_grouped_by_assistant_message(messages: Sequence[dict], tag: str) -> List[List[Span]]:
    """
    Return a list of groups; each group corresponds to one assistant message that
    contains at least one <tag>...</tag> block.
    """
    groups: List[List[Span]] = []
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content") or ""
        spans = _parse_context_blocks_from_text(content, tag)
        if spans:
            groups.append(spans)
    return groups


def spans_to_lines(spans: Iterable[Span]) -> Set[LineKey]:
    out: Set[LineKey] = set()
    for file_path, start, end in spans:
        out.update(lines_to_set(normalize_file_path(file_path), start, end))
    return out


def safe_drop(numer_kept: int, denom_seen: int) -> Tuple[float, float]:
    """Return (keep, drop) with the user's convention."""
    if denom_seen <= 0:
        return 0.0, 1.0
    keep = numer_kept / denom_seen
    # Keep should be in [0, 1] for "fraction of seen gold kept".
    if keep > 1.0:
        keep = 1.0
    return keep, 1.0 - keep


def quantile(sorted_vals: Sequence[float], q: float) -> float:
    """
    Simple linear interpolation quantile, q in [0,1].
    Assumes sorted_vals is non-empty and sorted.
    """
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = (len(sorted_vals) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def compute_instance_drops(traj_path: Path, gold_ctx: List[dict]) -> dict:
    if not traj_path.exists():
        return {
            "traj_exists": False,
            "g_seen_union": 0,
            "g_seen_last": 0,
            "kept_gold_in_final": 0,
            "drop_union": 1.0,
            "drop_last": 1.0,
            "keep_union": 0.0,
            "keep_last": 0.0,
            "lost_gold_from_last": 0,
            "num_explore_messages": 0,
            "num_patch_messages": 0,
        }

    traj_data = json.loads(traj_path.read_text(encoding="utf-8"))
    messages = traj_data.get("messages", []) or []

    explore_groups = parse_contexts_grouped_by_assistant_message(messages, "EXPLORE_CONTEXT")
    patch_groups = parse_contexts_grouped_by_assistant_message(messages, "PATCH_CONTEXT")

    gold_lines: Set[LineKey] = set()
    for ctx in gold_ctx:
        file_path = normalize_file_path(ctx["file"])
        gold_lines.update(lines_to_set(file_path, int(ctx["start_line"]), int(ctx["end_line"])))

    if not gold_lines:
        return {
            "traj_exists": True,
            "g_seen_union": 0,
            "g_seen_last": 0,
            "kept_gold_in_final": 0,
            "drop_union": 1.0,
            "drop_last": 1.0,
            "keep_union": 0.0,
            "keep_last": 0.0,
            "lost_gold_from_last": 0,
            "num_explore_messages": len(explore_groups),
            "num_patch_messages": len(patch_groups),
        }

    explored_union_lines = spans_to_lines(span for group in explore_groups for span in group)
    last_explore_lines = spans_to_lines(explore_groups[-1]) if explore_groups else set()
    final_lines = spans_to_lines(span for group in patch_groups for span in group)

    g_seen_union = explored_union_lines & gold_lines
    g_seen_last = last_explore_lines & gold_lines
    kept_in_final = final_lines & gold_lines
    # For "fraction of seen gold kept", the kept numerator should be restricted to seen gold.
    kept_from_union = kept_in_final & g_seen_union
    kept_from_last = kept_in_final & g_seen_last

    keep_union, drop_union = safe_drop(len(kept_from_union), len(g_seen_union))
    keep_last, drop_last = safe_drop(len(kept_from_last), len(g_seen_last))

    lost_from_last = len(g_seen_last - kept_from_last)

    return {
        "traj_exists": True,
        "g_seen_union": len(g_seen_union),
        "g_seen_last": len(g_seen_last),
        "kept_gold_in_final": len(kept_in_final),
        "kept_seen_gold_in_final_union": len(kept_from_union),
        "kept_seen_gold_in_final_last": len(kept_from_last),
        "drop_union": drop_union,
        "drop_last": drop_last,
        "keep_union": keep_union,
        "keep_last": keep_last,
        "lost_gold_from_last": lost_from_last,
        "num_explore_messages": len(explore_groups),
        "num_patch_messages": len(patch_groups),
    }


def infer_traj_dir_name(model_dir_name: str) -> str:
    # Matches existing repo layout (see other scripts).
    mapping = {"claude45": "claude", "gpt5": "gpt", "gemini": "gemini", "mistral": "mistral"}
    return mapping.get(model_dir_name, model_dir_name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_dir",
        type=Path,
        default=Path("/root/lh/Context-Bench"),
        help="Context-Bench base directory",
    )
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=None,
        help="Results dir containing model/benchmark/all.jsonl (defaults to <base_dir>/results/miniswe)",
    )
    parser.add_argument(
        "--traj_base_dir",
        type=Path,
        default=None,
        help="Traj base dir (defaults to <base_dir>/traj/miniswe)",
    )
    parser.add_argument(
        "--gold_file",
        type=Path,
        default=None,
        help="Gold jsonl (defaults to <base_dir>/results/gold/contextbench_full.gold.jsonl)",
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
        help="Output JSON path (defaults to <results_dir>/evidence_drop_laststep_to_final.json)",
    )
    args = parser.parse_args()

    base_dir: Path = args.base_dir
    results_dir: Path = args.results_dir or (base_dir / "results" / "miniswe")
    traj_base_dir: Path = args.traj_base_dir or (base_dir / "traj" / "miniswe")
    gold_file: Path = args.gold_file or (base_dir / "results" / "gold" / "contextbench_full.gold.jsonl")
    output_path: Path = args.output or (results_dir / "evidence_drop_laststep_to_final.json")

    gold_contexts = load_gold_contexts(gold_file)

    all_jsonls = sorted(results_dir.glob("**/all.jsonl"))
    if args.only:
        all_jsonls = [p for p in all_jsonls if args.only.strip("/").lower() in str(p.relative_to(results_dir)).lower()]

    per_file: Dict[str, dict] = {}
    per_instance: Dict[str, dict] = {}

    for jsonl_path in all_jsonls:
        rel = str(jsonl_path.relative_to(results_dir))
        parts = jsonl_path.relative_to(results_dir).parts
        if len(parts) < 3:
            continue
        model_dir = parts[0]
        benchmark = parts[1]
        traj_dir_name = infer_traj_dir_name(model_dir)
        traj_subdir = f"traj_{benchmark.lower()}_{traj_dir_name}"

        instances = load_jsonl(jsonl_path)

        drops_union: List[float] = []
        drops_last: List[float] = []
        denom_union_nonzero = 0
        denom_last_nonzero = 0
        missing_traj = 0

        for inst in instances:
            instance_id = inst["instance_id"]
            traj_path = traj_base_dir / traj_dir_name / traj_subdir / instance_id / f"{instance_id}.traj.json"
            gold_ctx = gold_contexts.get(instance_id, [])
            metrics = compute_instance_drops(traj_path, gold_ctx)

            key = f"{model_dir}/{benchmark}/{instance_id}"
            per_instance[key] = {
                "model": model_dir,
                "benchmark": benchmark,
                "instance_id": instance_id,
                "traj_path": str(traj_path),
                **metrics,
            }

            drops_union.append(float(metrics["drop_union"]))
            drops_last.append(float(metrics["drop_last"]))
            if not metrics["traj_exists"]:
                missing_traj += 1
            if metrics["g_seen_union"] > 0:
                denom_union_nonzero += 1
            if metrics["g_seen_last"] > 0:
                denom_last_nonzero += 1

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

        per_file[rel] = {
            "model": model_dir,
            "benchmark": benchmark,
            "num_instances": len(instances),
            "missing_traj": missing_traj,
            "denom_union_nonzero": denom_union_nonzero,
            "denom_last_nonzero": denom_last_nonzero,
            "drop_union": summarize(drops_union),
            "drop_last": summarize(drops_last),
        }

    by_model_benchmark = defaultdict(lambda: {"drops_union": [], "drops_last": [], "n": 0})
    for rel, s in per_file.items():
        key = f"{s['model']}/{s['benchmark']}"
        by_model_benchmark[key]["n"] += s["num_instances"]
        if s["drop_union"]["mean"] is not None:
            by_model_benchmark[key]["drops_union"].append(s["drop_union"]["mean"])
        if s["drop_last"]["mean"] is not None:
            by_model_benchmark[key]["drops_last"].append(s["drop_last"]["mean"])

    output = {
        "meta": {
            "base_dir": str(base_dir),
            "results_dir": str(results_dir),
            "traj_base_dir": str(traj_base_dir),
            "gold_file": str(gold_file),
            "only": args.only,
            "num_all_jsonl_files": len(all_jsonls),
        },
        "per_file": per_file,
        "per_instance": per_instance,
    }
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    # Console summary table (high signal only)
    print("Evidence Drop summary (line-level)")
    print("  Union-drop: (all explore steps) -> final")
    print("  Last-drop : (last explore step) -> final")
    print()
    print(f"{'file':60s} {'N':>4s} {'U_mean':>8s} {'L_mean':>8s} {'U_p90':>8s} {'L_p90':>8s} {'miss':>5s}")
    print("-" * 110)
    for rel, s in sorted(per_file.items()):
        u = s["drop_union"]
        l = s["drop_last"]
        print(
            f"{rel:60.60s} "
            f"{s['num_instances']:4d} "
            f"{(u['mean'] if u['mean'] is not None else float('nan')):8.4f} "
            f"{(l['mean'] if l['mean'] is not None else float('nan')):8.4f} "
            f"{(u['p90'] if u['p90'] is not None else float('nan')):8.4f} "
            f"{(l['p90'] if l['p90'] is not None else float('nan')):8.4f} "
            f"{s['missing_traj']:5d}"
        )
    print()
    print(f"Wrote JSON: {output_path}")


if __name__ == "__main__":
    main()

