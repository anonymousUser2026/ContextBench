#!/usr/bin/env python3
"""
从 traj.json 文件中提取补丁的工具函数
关键：从 6_final_selected_patch 字段提取最终选中的补丁
文件名格式：{instance_id}_traj.json（例如：SWE-Bench-Pro__javascript__maintenance__bugfix__xxx_traj.json）
"""

import json
import os
from pathlib import Path
from typing import Optional


def extract_fix_patch(traj_json_path: str) -> Optional[str]:
    """
    从 traj.json 文件中提取补丁内容
    traj.json 是最终结果存储格式，必须包含 6_final_selected_patch 字段
    
    Args:
        traj_json_path: traj.json 文件的路径
        
    Returns:
        补丁内容（diff 格式字符串），如果提取失败返回 None
    """
    if not os.path.exists(traj_json_path):
        print(f"警告: traj.json 文件不存在: {traj_json_path}")
        return None
    
    try:
        with open(traj_json_path, 'r', encoding='utf-8') as f:
            traj_data = json.load(f)
        
        # 必须从 6_final_selected_patch 字段提取（这是最终选中的补丁）
        if "6_final_selected_patch" in traj_data:
            patch = traj_data["6_final_selected_patch"]
            if patch and patch.strip():
                return patch.strip()
        
        # 如果 6_final_selected_patch 不存在，尝试从 5_sampled_edit_locs_and_patches 中选择（fallback）
        if "5_sampled_edit_locs_and_patches" in traj_data:
            samples = traj_data["5_sampled_edit_locs_and_patches"]
            if isinstance(samples, list) and len(samples) > 0:
                # 选择第一个样本的第一个补丁
                first_sample = samples[0]
                if "patches" in first_sample and isinstance(first_sample["patches"], list):
                    if len(first_sample["patches"]) > 0:
                        patch = first_sample["patches"][0]
                        if patch and patch.strip():
                            print(f"警告: {traj_json_path} 缺少 6_final_selected_patch，使用 5_sampled_edit_locs_and_patches 中的补丁")
                            return patch.strip()
        
        print(f"错误: traj.json 文件缺少补丁信息: {traj_json_path}")
        print(f"  必须包含 6_final_selected_patch 字段")
        return None
        
    except json.JSONDecodeError as e:
        print(f"错误: traj.json 文件格式错误: {traj_json_path}, {e}")
        return None
    except Exception as e:
        print(f"错误: 读取 traj.json 文件时出错: {traj_json_path}, {e}")
        return None


def find_traj_files(results_dir: str, instance_id: str) -> list:
    """
    在结果目录中查找与 instance_id 相关的所有 traj.json 文件
    
    Args:
        results_dir: 结果目录路径
        instance_id: 实例 ID
        
    Returns:
        traj.json 文件路径列表
    """
    traj_files = []
    results_path = Path(results_dir)
    
    if not results_path.exists():
        return traj_files
    
    # 查找所有可能的 traj.json 文件位置
    patterns = [
        f"**/{instance_id}_traj.json",
        f"**/repair_sample_*/output_*_traj.json",
        f"**/trajs/{instance_id}_traj.json",
    ]
    
    for pattern in patterns:
        for traj_file in results_path.glob(pattern):
            traj_files.append(str(traj_file))
    
    return traj_files


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python extract_patch_from_traj.py <traj_json_path>")
        sys.exit(1)
    
    traj_path = sys.argv[1]
    patch = extract_fix_patch(traj_path)
    
    if patch:
        print(patch)
    else:
        print("无法提取补丁", file=sys.stderr)
        sys.exit(1)

