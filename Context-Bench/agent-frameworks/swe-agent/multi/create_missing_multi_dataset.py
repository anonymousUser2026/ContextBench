#!/usr/bin/env python3
"""
从各个语言的JSONL数据文件中提取missing_multi中指定的实例，
创建一个新的missing_multi.jsonl文件
"""
import json
import os
from pathlib import Path
from glob import glob

# 读取missing_multi文件获取所需的实例ID
missing_multi_file = Path("/root/sweagent_eval/missing_multi")
with open(missing_multi_file, 'r') as f:
    target_instance_ids = set(line.strip() for line in f if line.strip())

print(f"需要提取的实例总数: {len(target_instance_ids)}")
print(f"实例示例: {list(target_instance_ids)[:5]}")

# 数据目录
data_dir = Path("/root/sweagent_eval/MSWE-agent/data")

# 查找所有的 *_dataset.jsonl 文件
dataset_files = []
for lang_dir in ['c', 'cpp', 'go', 'java', 'js', 'ts', 'rust']:
    lang_path = data_dir / lang_dir
    if lang_path.exists() and lang_path.is_dir():
        pattern = str(lang_path / "*_dataset.jsonl")
        dataset_files.extend(glob(pattern))

print(f"\n找到 {len(dataset_files)} 个数据集文件")

# 收集所有匹配的实例
matched_instances = []
found_instance_ids = set()

for dataset_file in sorted(dataset_files):
    file_path = Path(dataset_file)
    print(f"\n正在处理: {file_path.relative_to(data_dir)}")
    count = 0
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # 跳过空行和git-lfs标记
                if not line.strip() or line.startswith('version https://git-lfs'):
                    continue
                
                try:
                    instance = json.loads(line)
                    # 构建instance_id: org__repo-number
                    instance_id = f"{instance['org']}__{instance['repo']}-{instance['number']}"
                    
                    if instance_id in target_instance_ids:
                        matched_instances.append(instance)
                        found_instance_ids.add(instance_id)
                        count += 1
                        print(f"  ✓ 找到: {instance_id}")
                except json.JSONDecodeError:
                    continue  # 跳过无法解析的行
        
        if count > 0:
            print(f"  从此文件中找到 {count} 个实例")
    except Exception as e:
        print(f"处理 {file_path} 时出错: {e}")

# 检查缺失的实例
missing_ids = target_instance_ids - found_instance_ids
if missing_ids:
    print(f"\n警告: 以下 {len(missing_ids)} 个实例未找到:")
    for mid in sorted(missing_ids):
        print(f"  - {mid}")
else:
    print(f"\n✓ 所有 {len(target_instance_ids)} 个实例都已找到!")

# 写入新的JSONL文件
output_file = data_dir / "missing_multi.jsonl"
with open(output_file, 'w', encoding='utf-8') as f:
    for instance in matched_instances:
        f.write(json.dumps(instance, ensure_ascii=False) + '\n')

print(f"\n✓ 已创建文件: {output_file}")
print(f"  包含 {len(matched_instances)} 个实例")
