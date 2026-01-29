#!/usr/bin/env python3
"""
Trajectory Resolver Script
==========================
为 trajectory 文件添加 resolved 状态标记。

功能：
- 读取 sb-cli 报告文件中的 resolved_ids 和 unresolved_ids
- 为 trajs 目录下的 trajectory 文件添加 7_resolved 字段
- 输出到 trajs2 目录

使用：
    python scripts/trajectory_resolver.py

输出目录：results/agentless/Verified/trajs2
"""

import json
import os
from pathlib import Path


def load_report(report_path: str) -> tuple[set[str], set[str]]:
    """加载 sb-cli 报告，解析 resolved 和 unresolved ID 列表。"""
    with open(report_path, 'r', encoding='utf-8') as f:
        report = json.load(f)
    
    resolved_ids = set(report.get('resolved_ids', []))
    unresolved_ids = set(report.get('unresolved_ids', []))
    
    return resolved_ids, unresolved_ids


def process_trajectory(
    traj_path: Path,
    resolved_ids: set[str],
    unresolved_ids: set[str]
) -> dict:
    """处理单个 trajectory 文件，添加 7_resolved 字段。"""
    with open(traj_path, 'r', encoding='utf-8') as f:
        traj = json.load(f)
    
    original_id = traj.get('original_id', '')
    
    if original_id in resolved_ids:
        traj['7_resolved'] = True
    elif original_id in unresolved_ids:
        traj['7_resolved'] = False
    else:
        traj['7_resolved'] = None
    
    return traj


def main():
    # 路径配置：动态获取项目根目录（当前文件所在目录向上两级）
    script_dir = Path(__file__).parent.resolve()
    base_dir = script_dir.parent
    trajs_dir = base_dir / 'results/agentless/Verified/trajs'
    trajs2_dir = base_dir / 'results/agentless/Verified/trajs2'
    report_path = base_dir / 'sb-cli/sb-cli-reports/swe-bench_verified__test__agentless_run.json'
    
    print("=" * 60)
    print("Trajectory Resolver - 为 Trajectory 添加 Resolved 状态")
    print("=" * 60)
    
    # 加载报告
    print(f"\n加载报告文件: {report_path}")
    resolved_ids, unresolved_ids = load_report(str(report_path))
    print(f"  - Resolved IDs: {len(resolved_ids)} 个")
    print(f"  - Unresolved IDs: {len(unresolved_ids)} 个")
    
    # 获取所有 trajectory 文件
    traj_files = sorted(trajs_dir.glob('*_traj.json'))
    print(f"\n找到 {len(traj_files)} 个 trajectory 文件")
    
    # 创建输出目录
    trajs2_dir.mkdir(parents=True, exist_ok=True)
    
    # 统计
    stats = {'resolved': 0, 'unresolved': 0, 'unknown': 0}
    
    # 处理每个文件
    for traj_path in traj_files:
        traj = process_trajectory(traj_path, resolved_ids, unresolved_ids)
        
        # 统计
        if traj['7_resolved'] is True:
            stats['resolved'] += 1
        elif traj['7_resolved'] is False:
            stats['unresolved'] += 1
        else:
            stats['unknown'] += 1
        
        # 写入输出目录
        output_path = trajs2_dir / traj_path.name
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(traj, f, indent=2, ensure_ascii=False)
    
    # 输出统计
    print("\n处理完成!")
    print(f"  - Resolved (已解决): {stats['resolved']} 个")
    print(f"  - Unresolved (未解决): {stats['unresolved']} 个")
    print(f"  - Unknown (未知): {stats['unknown']} 个")
    print(f"\n输出目录: {trajs2_dir}")
    print(f"生成文件数: {len(traj_files)} 个")


if __name__ == '__main__':
    main()
