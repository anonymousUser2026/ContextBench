#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


def _as_list_gold_context(v: Any) -> list:
    if v is None:
        return []
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return []
        try:
            parsed = json.loads(s)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []
    return []


def _repo_url_from_repo_field(v: Any) -> str:
    if not v:
        return ""
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return ""
        if s.startswith("http://") or s.startswith("https://"):
            return s
        if s.startswith("git@"):
            return s
        if "/" in s and " " not in s and s.count("/") == 1:
            return f"https://github.com/{s}.git"
        return ""
    return ""


def _pick(example: Dict[str, Any], *keys: str) -> Any:
    for k in keys:
        if k in example and example[k] is not None:
            return example[k]
    return None


def _iter_examples(ds: Any, split_name: str, limit: Optional[int]) -> Iterable[Dict[str, Any]]:
    if split_name not in ds:
        return []
    dsplit = ds[split_name]
    n = len(dsplit) if hasattr(dsplit, "__len__") else None
    take = min(limit, n) if (limit is not None and n is not None) else limit
    if take is not None:
        for i in range(int(take)):
            yield dsplit[i]
    else:
        for ex in dsplit:
            yield ex


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/gold/contextbench_verified.gold.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="If >0, export at most this many per split")
    ap.add_argument("--print_columns", action="store_true")
    args = ap.parse_args()

    try:
        from datasets import load_dataset
    except Exception as e:
        raise SystemExit("datasets is required (pip install datasets)") from e

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("Schwerli/ContextBench", "contextbench_verified")

    if args.print_columns:
        for split in ds.keys():
            cols = list(getattr(ds[split], "column_names", []))
            print(f"{split}: columns={cols}")

    limit = args.limit if args.limit and args.limit > 0 else None

    seen = set()
    written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for split in ds.keys():
            for ex in _iter_examples(ds, split, limit):
                instance_id = _pick(ex, "instance_id", "inst_id", "id")
                original_inst_id = _pick(ex, "original_inst_id", "original_id", "original")
                commit = _pick(ex, "base_commit", "commit")
                repo = _pick(ex, "repo", "repo_url", "repository")
                gold_context = _pick(ex, "gold_context", "gold_ctx")
                patch = _pick(ex, "patch")
                test_patch = _pick(ex, "test_patch")

                if not instance_id and not original_inst_id:
                    continue

                key = str(instance_id or original_inst_id)
                if key in seen:
                    continue
                seen.add(key)

                row = {
                    "inst_id": str(instance_id or ""),
                    "original_inst_id": str(original_inst_id or ""),
                    "commit": str(commit or ""),
                    "repo_url": _repo_url_from_repo_field(repo),
                    "gold_ctx": _as_list_gold_context(gold_context),
                    "patch": str(patch or ""),
                    "test_patch": str(test_patch or ""),
                    "split": str(split),
                }

                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1

    print(f"wrote {written} rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


