#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Any, Set, Tuple


def _key(d: Dict[str, Any]) -> str:
    return str(d.get("inst_id") or d.get("original_inst_id") or d.get("instance_id") or "").strip()


def dedupe_jsonl(in_path: str, out_path: str) -> Tuple[int, int, int, int]:
    """Return (total_lines, kept, dropped, bad_json)."""
    seen: Set[str] = set()
    total = kept = dropped = bad = 0
    with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = (line or "").rstrip("\n")
            if not line.strip():
                continue
            total += 1
            try:
                d = json.loads(line)
            except Exception:
                bad += 1
                continue
            k = _key(d)
            if not k:
                # Keep rows with no key (rare), but treat them as unique by position.
                fout.write(json.dumps(d, ensure_ascii=False) + "\n")
                kept += 1
                continue
            if k in seen:
                dropped += 1
                continue
            seen.add(k)
            fout.write(json.dumps(d, ensure_ascii=False) + "\n")
            kept += 1
    return total, kept, dropped, bad


def main() -> int:
    ap = argparse.ArgumentParser(description="Dedupe a JSONL file by inst_id/original_inst_id.")
    ap.add_argument("--in", dest="inp", required=True, help="Input JSONL path")
    ap.add_argument("--out", dest="out", required=True, help="Output JSONL path")
    args = ap.parse_args()

    inp = args.inp
    out = args.out
    if os.path.abspath(inp) == os.path.abspath(out):
        raise SystemExit("--in and --out must be different paths")

    total, kept, dropped, bad = dedupe_jsonl(inp, out)
    print(f"total={total} kept={kept} dropped={dropped} bad_json={bad}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

