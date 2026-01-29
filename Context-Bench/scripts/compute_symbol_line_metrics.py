#!/usr/bin/env python3
"""
计算基于symbol行数范围的Coverage和Precision指标（用于agentless）

从stderr.log文件中提取gold_symbols和pred_symbols，然后：
1. 解析symbol格式：file:type@start-end
2. 直接使用@start-end中的数字作为行数范围（不进行转换）
3. 仿照line层级的做法计算coverage和precision
4. 将结果输出到新的JSONL文件中
"""

import json
import re
import os
import sys
from typing import Dict, List, Set, Tuple
from pathlib import Path
from collections import defaultdict

# LineInterval = Tuple[int, int]  # (start_line, end_line) inclusive

def parse_symbol(symbol_str: str) -> Tuple[str, int, int]:
    """
    解析symbol字符串格式：file:type@start-end
    直接使用start和end作为行数范围，不进行转换
    返回：(file_path, start_line, end_line)
    
    示例：
        "src/libponyc/type/alias.c:function_definition@5226-6401"
        -> ("src/libponyc/type/alias.c", 5226, 6401)
    """
    # 格式：file:type@start-end
    # 正则表达式：(.+?)匹配文件路径（非贪婪，到第一个冒号为止）
    #            :[^@]+匹配冒号和类型（直到@符号）
    #            @(\d+)-(\d+)匹配行数范围
    match = re.match(r'^(.+?):[^@]+@(\d+)-(\d+)$', symbol_str)
    if not match:
        return None, 0, 0
    
    file_path, start_line, end_line = match.groups()
    return file_path, int(start_line), int(end_line)

def _merge_line_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """合并重叠或相邻的行区间"""
    if not intervals:
        return []
    sorted_intervals = sorted(intervals)
    merged = [sorted_intervals[0]]
    for current in sorted_intervals[1:]:
        last = merged[-1]
        if current[0] <= last[1] + 1:
            merged[-1] = (last[0], max(last[1], current[1]))
        else:
            merged.append(current)
    return merged

def _line_interval_length(intervals: List[Tuple[int, int]]) -> int:
    """计算区间覆盖的总行数"""
    merged = _merge_line_intervals(intervals)
    return sum(end - start + 1 for start, end in merged)

def _line_interval_intersect(a: List[Tuple[int, int]], b: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """计算两个行区间列表的交集"""
    a_m = _merge_line_intervals(a)
    b_m = _merge_line_intervals(b)
    result = []
    i, j = 0, 0
    while i < len(a_m) and j < len(b_m):
        overlap = (max(a_m[i][0], b_m[j][0]), min(a_m[i][1], b_m[j][1]))
        if overlap[0] <= overlap[1]:
            result.append(overlap)
        if a_m[i][1] < b_m[j][1]:
            i += 1
        elif b_m[j][1] < a_m[i][1]:
            j += 1
        else:
            i += 1
            j += 1
    return result

def _line_interval_intersect_size(a: List[Tuple[int, int]], b: List[Tuple[int, int]]) -> int:
    """计算交集的行数"""
    return _line_interval_length(_line_interval_intersect(a, b))

def line_total_lines(lines_by_file: Dict[str, List[Tuple[int, int]]]) -> int:
    """计算所有文件的总行数"""
    return sum(_line_interval_length(intervals) for intervals in lines_by_file.values())

def line_intersection_lines(a: Dict[str, List[Tuple[int, int]]], b: Dict[str, List[Tuple[int, int]]]) -> int:
    """
    计算两个文件字典的交集行数
    
    重要：只有相同文件路径的symbol才会进行比较。
    遍历所有文件（gold和pred的并集），对每个文件分别计算行数范围的交集。
    
    Args:
        a: gold_lines字典，key为文件路径，value为行数区间列表
        b: pred_lines字典，key为文件路径，value为行数区间列表
    
    Returns:
        所有文件的交集行数总和
    """
    total = 0
    # 遍历所有文件（gold和pred的并集），确保相同文件路径才会比较
    for f in set(a.keys()) | set(b.keys()):
        # 对每个文件，分别计算gold和pred的行数区间交集
        # 如果文件只在一个字典中存在，get会返回空列表，交集为0
        total += _line_interval_intersect_size(a.get(f, []), b.get(f, []))
    return total

def coverage_precision(pred_size: float, gold_size: float, inter_size: float) -> Tuple[float, float]:
    """计算coverage和precision"""
    cov = inter_size / gold_size if gold_size > 0 else 1.0
    prec = inter_size / pred_size if pred_size > 0 else 1.0
    return cov, prec

def parse_stderr_log(stderr_path: str, repo_dir: str = None) -> Dict:
    """
    从stderr.log文件中解析gold_symbols和pred_symbols
    直接使用@start-end中的数字作为行数范围，不进行转换
    返回包含行数范围的字典
    """
    with open(stderr_path, 'r') as f:
        content = f.read()
    
    # 提取gold_symbols
    gold_symbols = []
    gold_match = re.search(r'gold_symbols: n=(\d+)\s*\n((?:\s+- .+\n?)+)', content)
    if gold_match:
        symbols_text = gold_match.group(2)
        for line in symbols_text.strip().split('\n'):
            if line.strip().startswith('- '):
                symbol = line.strip()[2:]  # 移除 '- '
                gold_symbols.append(symbol)
    
    # 提取pred_symbols
    pred_symbols = []
    pred_match = re.search(r'pred_symbols: n=(\d+)\s*\n((?:\s+- .+\n?)+)', content)
    if pred_match:
        symbols_text = pred_match.group(2)
        for line in symbols_text.strip().split('\n'):
            if line.strip().startswith('- '):
                symbol = line.strip()[2:]  # 移除 '- '
                pred_symbols.append(symbol)
    
    # 转换为行数范围（直接使用@start-end中的数字作为行数）
    gold_lines = defaultdict(list)
    pred_lines = defaultdict(list)
    
    for symbol in gold_symbols:
        file_path, start_line, end_line = parse_symbol(symbol)
        if file_path:
            gold_lines[file_path].append((start_line, end_line))
    
    for symbol in pred_symbols:
        file_path, start_line, end_line = parse_symbol(symbol)
        if file_path:
            pred_lines[file_path].append((start_line, end_line))
    
    return {
        'gold_lines': dict(gold_lines),
        'pred_lines': dict(pred_lines),
        'gold_symbols': gold_symbols,
        'pred_symbols': pred_symbols
    }

def compute_symbol_line_metrics(gold_lines: Dict[str, List[Tuple[int, int]]], 
                                pred_lines: Dict[str, List[Tuple[int, int]]]) -> Dict:
    """计算基于symbol行数范围的metrics"""
    gold_total = line_total_lines(gold_lines)
    pred_total = line_total_lines(pred_lines)
    intersection = line_intersection_lines(gold_lines, pred_lines)
    
    coverage, precision = coverage_precision(pred_total, gold_total, intersection)
    
    return {
        "coverage": coverage,
        "precision": precision,
        "intersection": intersection,
        "gold_size": gold_total,
        "pred_size": pred_total
    }

def find_stderr_log(instance_id: str, results_dir: str) -> str:
    """查找对应的stderr.log文件"""
    # 从instance_id中提取bench类型
    if instance_id.startswith('Multi-'):
        bench = 'Multi'
    elif instance_id.startswith('Pro-'):
        bench = 'Pro'
    elif instance_id.startswith('Verified-'):
        bench = 'Verified'
    elif instance_id.startswith('Poly-'):
        bench = 'Poly'
    else:
        bench = 'Multi'  # 默认
    
    stderr_path = os.path.join(results_dir, bench, f"{instance_id}.stderr.log")
    if os.path.exists(stderr_path):
        return stderr_path
    
    # 尝试其他可能的位置
    for subdir in ['Multi', 'Pro', 'Verified', 'Poly']:
        stderr_path = os.path.join(results_dir, subdir, f"{instance_id}.stderr.log")
        if os.path.exists(stderr_path):
            return stderr_path
    
    return None

def get_repo_dir(instance_id: str, selected_csv: str, cache_dir: str) -> str:
    """从selected_500_instances.csv中获取repo信息并构建repo目录路径"""
    import csv
    
    with open(selected_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['instance_id'] == instance_id:
                repo = row.get('repo', '').strip()
                commit = row.get('commit', '').strip()
                if repo and commit:
                    # 构建repo目录路径
                    # 格式：/tmp/contextbench_worktrees/github.com__ponylang__ponyc/8ae0ecd7dcc1...
                    repo_name = repo.replace('https://github.com/', '').replace('/', '__')
                    repo_dir = os.path.join(cache_dir, f"github.com__{repo_name}", commit)
                    return repo_dir
    
    return None

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="计算基于symbol行数范围的Coverage和Precision")
    parser.add_argument("--results_dir", required=True, help="结果目录（包含all.jsonl和stderr.log文件）")
    parser.add_argument("--selected_csv", required=True, help="selected_500_instances.csv文件路径")
    parser.add_argument("--cache_dir", default="/tmp/contextbench_worktrees", help="Repo缓存目录")
    parser.add_argument("--bench", default="Multi", help="Bench类型（Multi/Pro/Verified/Poly）")
    parser.add_argument("--output", required=True, help="输出文件路径（JSONL格式）")
    
    args = parser.parse_args()
    
    all_jsonl_path = os.path.join(args.results_dir, args.bench, "all.jsonl")
    if not os.path.exists(all_jsonl_path):
        print(f"Error: {all_jsonl_path} not found", file=sys.stderr)
        sys.exit(1)
    
    # 读取all.jsonl获取instance_id列表
    instance_ids = []
    with open(all_jsonl_path, 'r') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                instance_id = data.get('instance_id')
                if instance_id:
                    instance_ids.append(instance_id)
    
    # 处理每个实例
    output_results = []
    for instance_id in instance_ids:
        # 查找stderr.log文件
        stderr_path = find_stderr_log(instance_id, args.results_dir)
        if not stderr_path:
            print(f"Warning: stderr.log not found for {instance_id}", file=sys.stderr)
            continue
        
        # 解析stderr.log（不需要repo_dir，直接使用数字）
        try:
            parsed = parse_stderr_log(stderr_path)
            gold_lines = parsed['gold_lines']
            pred_lines = parsed['pred_lines']
            
            # 计算metrics
            symbol_line_metrics = compute_symbol_line_metrics(gold_lines, pred_lines)
            
            # 构建输出结果
            output_result = {
                "instance_id": instance_id,
                "symbol_line": symbol_line_metrics
            }
            output_results.append(output_result)
            
            print(f"Processed {instance_id}: coverage={symbol_line_metrics['coverage']:.4f}, precision={symbol_line_metrics['precision']:.4f}", file=sys.stderr)
        except Exception as e:
            print(f"Error processing {instance_id}: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
    
    # 写入输出文件
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    with open(args.output, 'w') as f:
        for result in output_results:
            f.write(json.dumps(result) + '\n')
    
    print(f"Results written to {args.output} ({len(output_results)} instances)", file=sys.stderr)

if __name__ == '__main__':
    main()
