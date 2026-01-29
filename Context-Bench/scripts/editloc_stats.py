#!/usr/bin/env python3
"""Compute editloc recall mean, precision mean, and F1 (per-instance harmonic mean, then average)."""

import json
from pathlib import Path

AGENTLESS_ROOT = Path(__file__).resolve().parents[1] / "results" / "agentless"


def harmonic_mean(a: float, b: float) -> float:
    """Harmonic mean of a and b. Returns 0 when both are 0."""
    if a == 0 and b == 0:
        return 0.0
    return 2 * a * b / (a + b)


def main():
    instances = []
    for p in AGENTLESS_ROOT.rglob("*.jsonl"):
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                el = obj.get("editloc")
                if el is None:
                    continue
                r = el.get("recall", 0.0)
                p_val = el.get("precision", 0.0)
                instances.append({"recall": r, "precision": p_val})

    if not instances:
        print("No instances with editloc found.")
        return

    n = len(instances)
    recall_sum = sum(x["recall"] for x in instances)
    precision_sum = sum(x["precision"] for x in instances)
    recall_mean = recall_sum / n
    precision_mean = precision_sum / n

    f1_per_instance = [harmonic_mean(x["recall"], x["precision"]) for x in instances]
    f1_mean = sum(f1_per_instance) / n

    print(f"Instances: {n}")
    print(f"editloc recall mean:   {recall_mean:.6f}")
    print(f"editloc precision mean: {precision_mean:.6f}")
    print(f"editloc F1 mean:       {f1_mean:.6f}")


if __name__ == "__main__":
    main()
