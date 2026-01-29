#!/usr/bin/env python3
"""统计 1.json 测试结果中有多少实例出现在 selected_500 数据集中，以及其中 resolved 的数量。"""

import json
import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
JSON_PATH = REPO_ROOT / "traj" / "prometheus" / "1.json"
CSV_PATH = REPO_ROOT / "selected_500_instances.csv"


def main():
    with open(JSON_PATH) as f:
        data = json.load(f)

    no_gen = set(data.get("no_generation", []))
    no_logs = set(data.get("no_logs", []))
    resolved = set(data.get("resolved", []))

    all_in_json = no_gen | no_logs | resolved

    dataset_ids = set()
    with open(CSV_PATH, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            oid = row.get("original_inst_id", "").strip()
            if oid:
                dataset_ids.add(oid)

    in_dataset = all_in_json & dataset_ids
    in_dataset_resolved = in_dataset & resolved

    print("1.json 中的实例:")
    print(f"  no_generation: {len(no_gen)}, no_logs: {len(no_logs)}, resolved: {len(resolved)}")
    print(f"  合计: {len(all_in_json)}")
    print()
    print("selected_500 数据集中 original_inst_id 数量:", len(dataset_ids))
    print()
    print("出现在数据集中的实例数:", len(in_dataset))
    print("其中 resolved 的数量:", len(in_dataset_resolved))

    if in_dataset:
        not_resolved = in_dataset - resolved
        print("其中 非resolved (no_generation/no_logs) 的数量:", len(not_resolved))
        if not_resolved:
            print("  这些实例:", sorted(not_resolved))


if __name__ == "__main__":
    main()
