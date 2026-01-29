#!/usr/bin/env python3
"""统计 logs 文件夹下有多少个出现在数据集中的实例，以及这些实例的通过情况。"""

import json
import csv
from pathlib import Path
from collections import defaultdict

REPO_ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = REPO_ROOT / "traj" / "prometheus" / "20251130_prometheus_gpt-5" / "logs"
CSV_PATH = REPO_ROOT / "selected_500_instances.csv"


def main():
    # 读取数据集中的 instance IDs
    dataset_ids = set()
    with open(CSV_PATH, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            oid = row.get("original_inst_id", "").strip()
            if oid:
                dataset_ids.add(oid)
    
    print(f"数据集中 original_inst_id 数量: {len(dataset_ids)}")
    print()
    
    # 从 logs 文件夹中提取所有实例 ID
    log_files = list(LOGS_DIR.glob("*_result.json"))
    instance_ids_in_logs = set()
    
    for log_file in log_files:
        # 从文件名提取 instance_id: {instance_id}_result.json
        instance_id = log_file.stem.replace("_result", "")
        instance_ids_in_logs.add(instance_id)
    
    print(f"logs 文件夹中的实例数量: {len(instance_ids_in_logs)}")
    print()
    
    # 找出在数据集中的实例
    in_dataset = instance_ids_in_logs & dataset_ids
    print(f"出现在数据集中的实例数量: {len(in_dataset)}")
    print()
    
    # 统计通过情况
    resolved_count = 0
    not_resolved_count = 0
    no_result_file = 0
    
    resolved_instances = []
    not_resolved_instances = []
    
    for instance_id in sorted(in_dataset):
        result_file = LOGS_DIR / f"{instance_id}_result.json"
        
        if not result_file.exists():
            no_result_file += 1
            continue
        
        try:
            with open(result_file) as f:
                result_data = json.load(f)
                resolved = result_data.get("resolved", False)
                
                if resolved:
                    resolved_count += 1
                    resolved_instances.append(instance_id)
                else:
                    not_resolved_count += 1
                    not_resolved_instances.append(instance_id)
        except Exception as e:
            print(f"警告: 无法读取 {result_file}: {e}")
            not_resolved_count += 1
    
    print("=" * 60)
    print("通过情况统计:")
    print(f"  通过 (resolved=True): {resolved_count}")
    print(f"  未通过 (resolved=False): {not_resolved_count}")
    if no_result_file > 0:
        print(f"  无 result.json 文件: {no_result_file}")
    print(f"  总计: {len(in_dataset)}")
    print()
    
    if resolved_instances:
        print(f"通过的实例 ({len(resolved_instances)}):")
        for inst in resolved_instances[:10]:  # 只显示前10个
            print(f"  - {inst}")
        if len(resolved_instances) > 10:
            print(f"  ... 还有 {len(resolved_instances) - 10} 个")
        print()
    
    if not_resolved_instances:
        print(f"未通过的实例 ({len(not_resolved_instances)}):")
        for inst in not_resolved_instances[:10]:  # 只显示前10个
            print(f"  - {inst}")
        if len(not_resolved_instances) > 10:
            print(f"  ... 还有 {len(not_resolved_instances) - 10} 个")
        print()


if __name__ == "__main__":
    main()
