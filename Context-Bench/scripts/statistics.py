#!/usr/bin/env python3
"""
统计 results 目录下各个 agent 的评估结果
统计 file, symbol, span, line 四个 level 的 coverage、precision 和 F1
"""

import json
import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).parent.parent
RESULTS_DIR = REPO_ROOT / "results"
OUTPUT_FILE = RESULTS_DIR / "statistics.json"

# 需要统计的 agent 列表
AGENTS = ["agentless", "miniswe", "openhands", "prometheus", "sweagent"]

# 四个 level
LEVELS = ["file", "symbol", "span", "line"]


def find_all_jsonl_files(results_dir: Path) -> List[Tuple[str, Path]]:
    """
    找到所有 all.jsonl 文件
    返回: [(agent_name, file_path), ...]
    
    路径结构：
    - agentless/default/{Multi,Pro,Poly,Verified}/all.jsonl
    - miniswe/{model}/{Multi,Pro,Poly,Verified}/all.jsonl
    - openhands/gpt-5/{Multi,Pro,Poly,Verified}/all.jsonl
    - prometheus/gpt-5/{Multi,Pro,Poly,Verified}/all.jsonl
    - sweagent/default/{Multi,Pro,Poly,Verified}/all.jsonl
    """
    files = []
    benches = ["Multi", "Pro", "Poly", "Verified"]
    
    # agentless/default/{bench}/all.jsonl
    agentless_dir = results_dir / "agentless" / "default"
    if agentless_dir.exists():
        for bench in benches:
            jsonl_file = agentless_dir / bench / "all.jsonl"
            if jsonl_file.exists():
                files.append(("agentless", jsonl_file))
    
    # miniswe/{model}/{bench}/all.jsonl
    miniswe_dir = results_dir / "miniswe"
    if miniswe_dir.exists():
        for model_dir in miniswe_dir.iterdir():
            if model_dir.is_dir():
                for bench in benches:
                    jsonl_file = model_dir / bench / "all.jsonl"
                    if jsonl_file.exists():
                        files.append(("miniswe", jsonl_file))
    
    # openhands/gpt-5/{bench}/all.jsonl
    openhands_dir = results_dir / "openhands" / "gpt-5"
    if openhands_dir.exists():
        for bench in benches:
            jsonl_file = openhands_dir / bench / "all.jsonl"
            if jsonl_file.exists():
                files.append(("openhands", jsonl_file))
    
    # prometheus/gpt-5/{bench}/all.jsonl
    prometheus_dir = results_dir / "prometheus" / "gpt-5"
    if prometheus_dir.exists():
        for bench in benches:
            jsonl_file = prometheus_dir / bench / "all.jsonl"
            if jsonl_file.exists():
                files.append(("prometheus", jsonl_file))
    
    # sweagent/default/{bench}/all.jsonl
    sweagent_dir = results_dir / "sweagent" / "default"
    if sweagent_dir.exists():
        for bench in benches:
            jsonl_file = sweagent_dir / bench / "all.jsonl"
            if jsonl_file.exists():
                files.append(("sweagent", jsonl_file))
    
    return files


def load_instances(jsonl_file: Path) -> List[Dict]:
    """加载 all.jsonl 文件中的所有实例"""
    instances = []
    try:
        with open(jsonl_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    instance = json.loads(line)
                    instances.append(instance)
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line in {jsonl_file}: {e}")
                    continue
    except Exception as e:
        print(f"Warning: Failed to read {jsonl_file}: {e}")
    
    return instances


def calculate_f1(coverage: float, precision: float) -> float:
    """计算 F1 分数（调和平均）"""
    if coverage == 0 and precision == 0:
        return 0.0
    return 2 * (coverage * precision) / (coverage + precision)


def extract_metrics(instance: Dict) -> Dict[str, Dict[str, float]]:
    """
    从实例中提取四个 level 的 coverage 和 precision
    返回: {
        "file": {"coverage": ..., "precision": ...},
        ...
    }
    """
    metrics = {}
    final = instance.get("final", {})
    
    for level in LEVELS:
        level_data = final.get(level, {})
        coverage = level_data.get("coverage", 0.0)
        precision = level_data.get("precision", 0.0)
        
        metrics[level] = {
            "coverage": float(coverage),
            "precision": float(precision)
        }
    
    return metrics


def calculate_statistics(instances: List[Dict]) -> Dict:
    """计算统计信息"""
    if not instances:
        return {}
    
    # 存储所有实例的指标
    all_metrics = {level: {"coverage": [], "precision": [], "f1": []} for level in LEVELS}
    
    for instance in instances:
        metrics = extract_metrics(instance)
        
        for level in LEVELS:
            if level not in metrics:
                continue
            
            coverage = metrics[level]["coverage"]
            precision = metrics[level]["precision"]
            f1 = calculate_f1(coverage, precision)
            
            all_metrics[level]["coverage"].append(coverage)
            all_metrics[level]["precision"].append(precision)
            all_metrics[level]["f1"].append(f1)
    
    # 计算平均值
    statistics = {}
    for level in LEVELS:
        if not all_metrics[level]["coverage"]:
            continue
        
        statistics[level] = {
            "coverage_mean": sum(all_metrics[level]["coverage"]) / len(all_metrics[level]["coverage"]),
            "precision_mean": sum(all_metrics[level]["precision"]) / len(all_metrics[level]["precision"]),
            "f1_mean": sum(all_metrics[level]["f1"]) / len(all_metrics[level]["f1"])
        }
    
    return statistics


def main():
    """主函数"""
    print(f"Scanning results directory: {RESULTS_DIR}")
    
    # 找到所有 all.jsonl 文件
    jsonl_files = find_all_jsonl_files(RESULTS_DIR)
    print(f"Found {len(jsonl_files)} all.jsonl files")
    
    # 按 agent 分组统计
    agent_statistics = defaultdict(lambda: {"instances": [], "files": []})
    
    for agent, jsonl_file in jsonl_files:
        print(f"Processing {agent}: {jsonl_file.relative_to(RESULTS_DIR)}")
        instances = load_instances(jsonl_file)
        agent_statistics[agent]["instances"].extend(instances)
        agent_statistics[agent]["files"].append(str(jsonl_file.relative_to(RESULTS_DIR)))
    
    # 计算每个 agent 的统计信息
    results = {}
    for agent, data in agent_statistics.items():
        instances = data["instances"]
        if not instances:
            print(f"Warning: No instances found for {agent}")
            continue
        
        print(f"\nCalculating statistics for {agent} ({len(instances)} instances)")
        stats = calculate_statistics(instances)
        
        if stats:
            results[agent] = {
                "statistics": stats,
                "source_files": data["files"]
            }
    
    # 计算总体统计（所有 agent 合并）
    all_instances = []
    for agent, data in agent_statistics.items():
        all_instances.extend(data["instances"])
    
    if all_instances:
        print(f"\nCalculating overall statistics ({len(all_instances)} total instances)")
        overall_stats = calculate_statistics(all_instances)
        if overall_stats:
            results["overall"] = {
                "statistics": overall_stats
            }
    
    # 保存结果
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nStatistics saved to: {OUTPUT_FILE}")
    
    # 打印摘要
    print("\n" + "="*80)
    print("Summary:")
    print("="*80)
    for agent, data in results.items():
        if "statistics" not in data:
            continue
        print(f"\n{agent}:")
        for level in LEVELS:
            if level in data["statistics"]:
                stats = data["statistics"][level]
                print(f"  {level:8s}: Coverage={stats['coverage_mean']:.4f}, "
                      f"Precision={stats['precision_mean']:.4f}, "
                      f"F1={stats['f1_mean']:.4f}")


if __name__ == "__main__":
    main()
