#!/usr/bin/env python3
"""
将 traj.json 中的补丁转换为 multi-swe-bench 需要的格式
输出 JSONL 格式，每行包含 org, repo, number, fix_patch
"""

import json
import csv
import os
import re
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

# 添加当前目录到路径以便导入
sys.path.insert(0, str(Path(__file__).parent))
from extract_patch_from_traj import extract_fix_patch, find_traj_files


def parse_original_inst_id(original_inst_id: str) -> Optional[Tuple[str, str, str]]:
    """
    解析 Pro.csv 的 original_inst_id 获取 org, repo, version/number
    
    格式：instance_{org}__{repo}-{hash}-{version}
    
    Args:
        original_inst_id: 原始实例 ID
        
    Returns:
        (org, repo, number) 元组，如果解析失败返回 None
    """
    if not original_inst_id:
        return None
    
    # 移除 instance_ 前缀（如果有）
    id_str = original_inst_id
    if id_str.startswith("instance_"):
        id_str = id_str[len("instance_"):]
    
    # 解析 org__repo-hash-version
    # 这里我们把 version 作为 "number" 返回，因为 multi-swe-bench 期望一个标识符
    # 我们可以通过 split 来获取各个部分
    try:
        if "__" not in id_str:
            return None
        
        org, rest = id_str.split("__", 1)
        
        # 处理 rest: repo-hash-version
        # 注意 repo 名字里可能也有杠，但 hash 固定的（或者我们从右往左数）
        # 通常格式是 repo-hash-version
        parts = rest.rsplit("-", 2) # [repo, hash, version]
        if len(parts) == 3:
            repo, commit_hash, version = parts
            return (org, repo, version)
        elif len(parts) == 2:
            # 如果没有 version 或者格式略有不同
            repo, commit_hash = parts
            return (org, repo, commit_hash[:8])
            
    except Exception as e:
        print(f"解析 ID 失败: {original_inst_id}, 错误: {e}")
        
    return None


def convert_traj_to_patch_jsonl(
    csv_file: str,
    results_dir: str,
    output_file: str,
    trajs_dir: Optional[str] = None
) -> int:
    """
    将 traj.json 文件转换为 multi-swe-bench 格式的补丁 JSONL 文件
    
    Args:
        csv_file: Pro.csv 文件路径
        results_dir: Magentless 结果目录
        output_file: 输出的 JSONL 文件路径
        trajs_dir: traj.json 文件目录（如果指定，优先从此目录查找）
        
    Returns:
        成功转换的补丁数量
    """
    # 读取 CSV 文件
    instances = []
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            instances.append(row)
    
    print(f"读取到 {len(instances)} 个实例")
    
    # 转换每个实例
    converted_count = 0
    failed_count = 0
    
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for instance in instances:
            instance_id = instance.get('instance_id', '')
            original_inst_id = instance.get('original_inst_id', '')
            
            if not instance_id or not original_inst_id:
                print(f"跳过: instance_id 或 original_inst_id 为空")
                failed_count += 1
                continue
            
            # 解析 org, repo, number
            parsed = parse_original_inst_id(original_inst_id)
            if not parsed:
                print(f"跳过: 无法解析 original_inst_id: {original_inst_id}")
                failed_count += 1
                continue
            
            org, repo, number = parsed
            
            # 查找 traj.json 文件
            traj_files = []
            
            # 如果指定了 trajs_dir，优先从此目录查找
            if trajs_dir:
                traj_path = Path(trajs_dir) / f"{instance_id}_traj.json"
                if traj_path.exists():
                    traj_files.append(str(traj_path))
            
            # 从 results_dir 中查找
            if not traj_files:
                traj_files = find_traj_files(results_dir, instance_id)
            
            # 如果还是找不到，尝试从 repair_sample 目录查找
            if not traj_files:
                results_path = Path(results_dir)
                for repair_dir in results_path.glob("repair_sample_*"):
                    # 查找所有 traj.json 文件（包括子目录）
                    for traj_file in repair_dir.glob("**/*_traj.json"):
                        # 检查 traj 文件内容是否包含该 instance_id
                        try:
                            with open(traj_file, 'r', encoding='utf-8') as f:
                                traj_data = json.load(f)
                                if traj_data.get('instance_id') == instance_id:
                                    traj_files.append(str(traj_file))
                                    break
                        except Exception:
                            pass
                
                # 如果还是找不到，尝试从 all_preds.jsonl 中查找
                if not traj_files:
                    all_preds_file = results_path / "all_preds.jsonl"
                    if all_preds_file.exists():
                        try:
                            with open(all_preds_file, 'r', encoding='utf-8') as f:
                                for line in f:
                                    if line.strip():
                                        pred_data = json.loads(line)
                                        if pred_data.get('instance_id') == instance_id:
                                            # 从 all_preds.jsonl 中提取补丁
                                            patch = pred_data.get('model_patch', '')
                                            if patch:
                                                # 创建一个临时的 traj 结构用于提取
                                                temp_traj = {
                                                    'instance_id': instance_id,
                                                    '6_final_selected_patch': patch
                                                }
                                                # 保存到临时文件以便提取
                                                temp_file = Path(trajs_dir) / f"{instance_id}_temp_traj.json"
                                                with open(temp_file, 'w', encoding='utf-8') as tf:
                                                    json.dump(temp_traj, tf)
                                                traj_files.append(str(temp_file))
                                                break
                        except Exception as e:
                            print(f"警告: 无法从 all_preds.jsonl 读取: {e}")
            
            if not traj_files:
                print(f"警告: 找不到 {instance_id} 的 traj.json 文件")
                failed_count += 1
                continue
            
            # 从第一个找到的 traj.json 文件提取补丁
            patch = None
            for traj_file in traj_files:
                patch = extract_fix_patch(traj_file)
                if patch:
                    break
            
            if not patch:
                print(f"警告: 无法从 traj.json 提取补丁: {instance_id}")
                failed_count += 1
                continue
            
            # 生成 multi-swe-bench 格式的输出
            output_data = {
                "org": org,
                "repo": repo,
                "number": number,
                "fix_patch": patch
            }
            
            out_f.write(json.dumps(output_data, ensure_ascii=False) + '\n')
            converted_count += 1
            print(f"✓ 转换成功: {instance_id} -> {org}/{repo}#{number}")
    
    print(f"\n转换完成: 成功 {converted_count} 个, 失败 {failed_count} 个")
    return converted_count


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="将 traj.json 转换为 multi-swe-bench 格式")
    parser.add_argument("--csv", type=str, required=True, help="Pro.csv 文件路径")
    parser.add_argument("--results_dir", type=str, required=True, help="Magentless 结果目录")
    parser.add_argument("--output", type=str, required=True, help="输出的 JSONL 文件路径")
    parser.add_argument("--trajs_dir", type=str, default=None, help="traj.json 文件目录（可选）")
    
    args = parser.parse_args()
    
    convert_traj_to_patch_jsonl(
        args.csv,
        args.results_dir,
        args.output,
        args.trajs_dir
    )

