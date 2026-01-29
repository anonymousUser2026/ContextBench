#!/usr/bin/env python3
"""
轨迹数据提取与 SB-CLI 格式转换工具

使用说明：
=========

基本用法：
    python scripts/trajectory_extractor.py

指定输出路径：
    python scripts/trajectory_extractor.py --output output/predictions.json

按状态过滤（支持 pass, fail, fail2pass）：
    python scripts/trajectory_extractor.py --filter pass fail2pass

只验证文件，不生成：
    python scripts/trajectory_extractor.py --validate output/predictions.json

查看帮助：
    python scripts/trajectory_extractor.py --help

字段映射：
    Trajectory.original_id → SB-CLI.instance_id（字典键）
    Trajectory.6_final_selected_patch → SB-CLI.model_patch

输出格式：
    {
        "django__django-15104": {
            "model_patch": "diff --git a/..."
        }
    }
"""

import json
import csv
import argparse
import os
from pathlib import Path
from typing import Dict, List, Set, Optional


# 默认路径配置
DEFAULT_TRAJS_DIR = "results/agentless/Verified/trajs"
DEFAULT_CSV_PATH = "data/Verified.csv"
DEFAULT_OUTPUT_PATH = "output/predictions.json"


def load_trajectories(trajs_dir: str) -> List[dict]:
    """加载 trajs 目录下所有 JSON 文件"""
    trajs = []
    trajs_path = Path(trajs_dir)
    
    if not trajs_path.exists():
        raise FileNotFoundError(f"Trajectory 目录不存在: {trajs_dir}")
    
    for json_file in trajs_path.glob("*_traj.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                trajs.append(data)
        except json.JSONDecodeError as e:
            print(f"警告: 无法解析文件 {json_file}: {e}")
        except Exception as e:
            print(f"警告: 读取文件 {json_file} 时出错: {e}")
    
    return trajs


def load_csv_metadata(csv_path: str) -> Dict[str, dict]:
    """加载 CSV 元数据，按 instance_id 索引"""
    metadata = {}
    
    if not Path(csv_path).exists():
        print(f"警告: CSV 文件不存在: {csv_path}")
        return metadata
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            instance_id = row.get('instance_id', '')
            if instance_id:
                metadata[instance_id] = row
    
    return metadata


def convert_to_submissions(
    trajectories: List[dict],
    metadata: Optional[Dict[str, dict]] = None,
    statuses: Optional[Set[str]] = None
) -> Dict[str, dict]:
    """
    转换为 SB-CLI 格式
    
    字段映射：
        Trajectory.original_id → SB-CLI.instance_id（字典键）
        Trajectory.6_final_selected_patch → SB-CLI.model_patch
    
    输出格式：
        {
            "django__django-15104": {
                "model_patch": "diff --git a/...",
                "model_name_or_path": "agentless"
            }
        }
    """
    submissions = {}
    metadata = metadata or {}
    model_name = "agentless"  # SB-CLI 要求所有预测使用同一模型
    
    for traj in trajectories:
        # 使用 original_id 作为 SB-CLI 的 instance_id
        sb_instance_id = traj.get('original_id', '')
        
        if not sb_instance_id:
            traj_instance_id = traj.get('instance_id', 'unknown')
            print(f"警告: 跳过缺少 original_id 的 trajectory: {traj_instance_id}")
            continue
        
        # 检查状态过滤
        if statuses and metadata:
            traj_instance_id = traj.get('instance_id', '')
            if traj_instance_id in metadata:
                row = metadata[traj_instance_id]
                status = row.get('status', '')
                if status and status not in statuses:
                    continue
        
        # 获取补丁
        patch = traj.get('6_final_selected_patch', '')
        if not patch:
            print(f"警告: trajectory {sb_instance_id} 缺少 6_final_selected_patch")
            continue
        
        # 构建提交数据（SB-CLI 要求包含 model_name_or_path）
        submissions[sb_instance_id] = {
            'model_patch': patch,
            'model_name_or_path': model_name
        }
    
    return submissions


def save_predictions(submissions: Dict[str, dict], output_path: str):
    """保存为 JSON 文件"""
    output_file = Path(output_path)
    
    # 创建输出目录
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(submissions, f, indent=2, ensure_ascii=False)
    
    print(f"已保存预测文件: {output_file}")
    print(f"包含 {len(submissions)} 条记录")


def validate_predictions(predictions_path: str) -> bool:
    """验证预测文件"""
    is_valid = True
    
    print(f"\n验证文件: {predictions_path}")
    
    if not Path(predictions_path).exists():
        print("错误: 文件不存在")
        return False
    
    try:
        with open(predictions_path, 'r', encoding='utf-8') as f:
            predictions = json.load(f)
    except json.JSONDecodeError as e:
        print(f"错误: JSON 格式错误 - {e}")
        return False
    
    print(f"总条目数: {len(predictions)}")
    
    # 检查必需字段
    for instance_id, data in predictions.items():
        if not isinstance(data, dict):
            print(f"错误: {instance_id} 的值不是对象")
            is_valid = False
            continue
        
        if 'model_patch' not in data:
            print(f"错误: {instance_id} 缺少 model_patch 字段")
            is_valid = False
        
        # 验证 diff 格式
        patch = data.get('model_patch', '')
        if patch and not patch.startswith('diff --git'):
            print(f"警告: {instance_id} 的 model_patch 可能不是有效的 diff 格式")
    
    # 检查重复
    if len(predictions) != len(set(predictions.keys())):
        print("错误: 发现重复的 instance_id")
        is_valid = False
    
    if is_valid:
        print("验证通过!")
    
    return is_valid


def print_statistics(trajectories: List[dict], metadata: Dict[str, dict]):
    """打印统计信息"""
    print(f"\n统计信息:")
    print(f"  Trajectory 文件数: {len(trajectories)}")
    print(f"  CSV 元数据数: {len(metadata)}")
    
    # 按状态统计
    status_counts = {}
    for traj in trajectories:
        traj_instance_id = traj.get('instance_id', '')
        if traj_instance_id in metadata:
            status = metadata[traj_instance_id].get('status', 'unknown')
            status_counts[status] = status_counts.get(status, 0) + 1
    
    if status_counts:
        print("  状态分布:")
        for status, count in sorted(status_counts.items()):
            print(f"    {status}: {count}")


def main():
    parser = argparse.ArgumentParser(
        description="轨迹数据提取与 SB-CLI 格式转换工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/trajectory_extractor.py
  python scripts/trajectory_extractor.py --output output/predictions.json
  python scripts/trajectory_extractor.py --filter pass fail2pass
  python scripts/trajectory_extractor.py --validate output/predictions.json
        """
    )
    
    parser.add_argument(
        '--trajs_dir',
        type=str,
        default=DEFAULT_TRAJS_DIR,
        help=f"Trajectory 文件目录 (默认: {DEFAULT_TRAJS_DIR})"
    )
    
    parser.add_argument(
        '--csv_path',
        type=str,
        default=DEFAULT_CSV_PATH,
        help=f"CSV 元数据文件路径 (默认: {DEFAULT_CSV_PATH})"
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help=f"输出文件路径 (默认: {DEFAULT_OUTPUT_PATH})"
    )
    
    parser.add_argument(
        '--filter',
        type=str,
        nargs='*',
        choices=['pass', 'fail', 'fail2pass'],
        default=None,
        help="按状态过滤 (可选: pass fail fail2pass)"
    )
    
    parser.add_argument(
        '--validate',
        type=str,
        default=None,
        help="只验证指定文件，不生成新文件"
    )
    
    parser.add_argument(
        '--no_stats',
        action='store_true',
        help="不打印统计信息"
    )
    
    args = parser.parse_args()
    
    # 验证模式
    if args.validate:
        validate_predictions(args.validate)
        return
    
    # 加载数据
    print("加载 trajectory 文件...")
    trajectories = load_trajectories(args.trajs_dir)
    print(f"已加载 {len(trajectories)} 个 trajectory 文件")
    
    print("加载 CSV 元数据...")
    metadata = load_csv_metadata(args.csv_path)
    print(f"已加载 {len(metadata)} 条元数据记录")
    
    # 确定过滤状态
    statuses = None
    if args.filter:
        statuses = set(args.filter)
        print(f"过滤状态: {', '.join(args.filter)}")
    
    # 转换格式
    print("转换格式...")
    submissions = convert_to_submissions(trajectories, metadata, statuses)
    print(f"生成 {len(submissions)} 条预测记录")
    
    # 打印统计信息
    if not args.no_stats:
        print_statistics(trajectories, metadata)
    
    # 保存结果
    save_predictions(submissions, args.output)


if __name__ == "__main__":
    main()
