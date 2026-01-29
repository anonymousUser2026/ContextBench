#!/usr/bin/env python3
"""
汇总四个bench的symbol_line_metrics.jsonl文件，计算所有实例的coverage和precision平均数
"""

import json
import os
import sys
import argparse
from pathlib import Path

def load_metrics(file_path):
    """加载JSONL文件中的metrics"""
    results = []
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found", file=sys.stderr)
        return results
    
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    if 'symbol_line' in data:
                        results.append(data['symbol_line'])
                except json.JSONDecodeError as e:
                    print(f"Error parsing line in {file_path}: {e}", file=sys.stderr)
    
    return results

def aggregate_metrics(all_metrics):
    """计算所有metrics的平均数"""
    if not all_metrics:
        return None
    
    total_coverage = 0.0
    total_precision = 0.0
    total_intersection = 0
    total_gold_size = 0
    total_pred_size = 0
    
    for metric in all_metrics:
        total_coverage += metric.get('coverage', 0.0)
        total_precision += metric.get('precision', 0.0)
        total_intersection += metric.get('intersection', 0)
        total_gold_size += metric.get('gold_size', 0)
        total_pred_size += metric.get('pred_size', 0)
    
    n = len(all_metrics)
    
    return {
        'num_instances': n,
        'coverage_mean': total_coverage / n if n > 0 else 0.0,
        'precision_mean': total_precision / n if n > 0 else 0.0,
        'coverage_sum': total_coverage,
        'precision_sum': total_precision,
        'intersection_total': total_intersection,
        'gold_size_total': total_gold_size,
        'pred_size_total': total_pred_size,
        # 也可以计算micro-average（基于总数）
        'coverage_micro': total_intersection / total_gold_size if total_gold_size > 0 else 0.0,
        'precision_micro': total_intersection / total_pred_size if total_pred_size > 0 else 0.0
    }

def main():
    parser = argparse.ArgumentParser(description="汇总symbol_line_metrics.jsonl文件")
    parser.add_argument("--results_dir", required=True, help="结果目录")
    parser.add_argument("--output", default=None, help="输出汇总结果到JSON文件（可选）")
    
    args = parser.parse_args()
    
    benches = ['Multi', 'Pro', 'Verified', 'Poly']
    all_metrics = []
    bench_stats = {}
    
    # 加载每个bench的metrics
    for bench in benches:
        metrics_file = os.path.join(args.results_dir, bench, 'symbol_line_metrics.jsonl')
        metrics = load_metrics(metrics_file)
        all_metrics.extend(metrics)
        
        if metrics:
            bench_stats[bench] = aggregate_metrics(metrics)
            print(f"{bench}: {len(metrics)} instances", file=sys.stderr)
    
    # 计算总体统计
    overall_stats = aggregate_metrics(all_metrics)
    
    # 输出结果
    print("\n" + "="*70, file=sys.stderr)
    print("SYMBOL_LINE METRICS SUMMARY", file=sys.stderr)
    print("="*70, file=sys.stderr)
    
    print(f"\nOverall Statistics:", file=sys.stderr)
    print(f"  Total instances: {overall_stats['num_instances']}", file=sys.stderr)
    print(f"  Coverage (mean):  {overall_stats['coverage_mean']:.6f}", file=sys.stderr)
    print(f"  Precision (mean): {overall_stats['precision_mean']:.6f}", file=sys.stderr)
    print(f"  Coverage (micro): {overall_stats['coverage_micro']:.6f}", file=sys.stderr)
    print(f"  Precision (micro): {overall_stats['precision_micro']:.6f}", file=sys.stderr)
    
    print(f"\nPer-Bench Statistics:", file=sys.stderr)
    for bench in benches:
        if bench in bench_stats:
            stats = bench_stats[bench]
            print(f"  {bench}:", file=sys.stderr)
            print(f"    Instances: {stats['num_instances']}", file=sys.stderr)
            print(f"    Coverage (mean):  {stats['coverage_mean']:.6f}", file=sys.stderr)
            print(f"    Precision (mean): {stats['precision_mean']:.6f}", file=sys.stderr)
    
    # 输出JSON格式
    output_data = {
        'overall': overall_stats,
        'per_bench': bench_stats
    }
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output}", file=sys.stderr)
    else:
        # 输出到stdout
        print("\n" + json.dumps(output_data, indent=2))

if __name__ == '__main__':
    main()
