#!/usr/bin/env python3
"""
Split a combined stderr log from `python -m contextbench.evaluate` into per-instance logs.

The evaluator prints markers like:
  [10/22] Evaluating <instance_id>

This script splits the stream into files:
  <out_dir>/<instance_id>.stderr.log
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path


MARKER_RE = re.compile(r"^\[\d+/\d+\]\s+Evaluating\s+(.+?)\s*$")


def sanitize_filename(name: str) -> str:
    # Keep it simple and filesystem-safe.
    name = name.strip()
    name = name.replace("/", "__")
    name = name.replace("\0", "")
    return name


def split_log(log_path: Path, out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)

    current_id = None
    current_fh = None
    created = 0

    def _close_current() -> None:
        nonlocal current_fh
        if current_fh is not None:
            current_fh.close()
            current_fh = None

    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = MARKER_RE.match(line)
            if m:
                _close_current()
                current_id = sanitize_filename(m.group(1))
                if not current_id:
                    current_id = "unknown_instance"
                out_file = out_dir / f"{current_id}.stderr.log"
                current_fh = out_file.open("w", encoding="utf-8")
                created += 1

            if current_fh is not None:
                current_fh.write(line)

    _close_current()
    return created


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="Path to combined stderr log")
    ap.add_argument("--out-dir", required=True, help="Directory to write per-instance logs")
    args = ap.parse_args()

    log_path = Path(args.log)
    out_dir = Path(args.out_dir)

    if not log_path.is_file():
        raise SystemExit(f"ERROR: log file not found: {log_path}")

    created = split_log(log_path, out_dir)
    print(f"created={created} out_dir={out_dir}")


if __name__ == "__main__":
    main()

