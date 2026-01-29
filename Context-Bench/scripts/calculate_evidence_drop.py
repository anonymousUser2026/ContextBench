#!/usr/bin/env python3
"""
Calculate Evidence Drop rate for different models.

Evidence Drop (Retrieval ≠ Use):
- G_seen: Gold evidence that was ever seen during exploration
- Drop: Fraction of seen gold that gets discarded in final context
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict

def load_jsonl(filepath: Path) -> List[Dict]:
    """Load data from a JSONL file."""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def load_gold_contexts(gold_file: Path) -> Dict[str, List[Dict]]:
    """
    Load gold contexts indexed by instance_id.
    
    Returns:
        Dict mapping instance_id to list of gold context entries
    """
    gold_data = load_jsonl(gold_file)
    gold_contexts = {}
    
    for entry in gold_data:
        # Map original_inst_id to gold_ctx
        original_id = entry.get('original_inst_id')
        if original_id:
            gold_contexts[original_id] = entry.get('gold_ctx', [])
    
    return gold_contexts

def parse_context_blocks(content: str, pattern: str) -> List[Tuple[str, int, int]]:
    """
    Parse EXPLORE_CONTEXT or PATCH_CONTEXT blocks from traj content.
    
    Returns:
        List of (filepath, start_line, end_line) tuples
    """
    contexts = []
    
    # Find all context blocks
    regex = re.compile(f'<{pattern}>(.*?)</{pattern}>', re.DOTALL)
    matches = regex.findall(content)
    
    for match in matches:
        # Parse File: and Lines: entries
        lines_in_block = match.strip().split('\n')
        
        current_file = None
        for line in lines_in_block:
            line = line.strip()
            if line.startswith('File:'):
                current_file = line.replace('File:', '').strip()
            elif line.startswith('Lines:') and current_file:
                line_range = line.replace('Lines:', '').strip()
                if '-' in line_range:
                    start, end = line_range.split('-')
                    try:
                        contexts.append((current_file, int(start), int(end)))
                    except ValueError:
                        pass
    
    return contexts

def lines_to_set(file_path: str, start: int, end: int) -> Set[Tuple[str, int]]:
    """Convert a line range to a set of (file, line_number) tuples."""
    return {(file_path, line) for line in range(start, end + 1)}

def calculate_drop_for_instance(
    traj_path: Path,
    gold_ctx: List[Dict]
) -> Tuple[int, int, float]:
    """
    Calculate evidence drop for a single instance.
    
    Returns:
        (g_seen_size, final_intersection_size, drop_rate)
    """
    if not traj_path.exists():
        return 0, 0, 1.0
    
    # Load trajectory
    with open(traj_path, 'r') as f:
        traj_data = json.load(f)
    
    # Convert trajectory messages to string for parsing
    traj_content = json.dumps(traj_data)
    
    # Parse all EXPLORE_CONTEXT blocks (all steps)
    explore_contexts = parse_context_blocks(traj_content, 'EXPLORE_CONTEXT')
    
    # Parse PATCH_CONTEXT blocks (final context)
    patch_contexts = parse_context_blocks(traj_content, 'PATCH_CONTEXT')
    
    # Build gold context set (at line granularity)
    gold_lines = set()
    for ctx in gold_ctx:
        file_path = ctx['file']
        start = ctx['start_line']
        end = ctx['end_line']
        gold_lines.update(lines_to_set(file_path, start, end))
    
    if len(gold_lines) == 0:
        return 0, 0, 1.0
    
    # Build explored lines set (union of all EXPLORE_CONTEXT)
    explored_lines = set()
    for file_path, start, end in explore_contexts:
        explored_lines.update(lines_to_set(file_path, start, end))
    
    # Build final context lines set (from PATCH_CONTEXT)
    final_lines = set()
    for file_path, start, end in patch_contexts:
        final_lines.update(lines_to_set(file_path, start, end))
    
    # Calculate G_seen: gold evidence that was seen during exploration
    g_seen = explored_lines & gold_lines
    g_seen_size = len(g_seen)
    
    # Calculate final intersection: gold evidence in final context
    final_intersection = final_lines & gold_lines
    final_intersection_size = len(final_intersection)
    
    # Calculate drop rate
    if g_seen_size == 0:
        drop_rate = 1.0  # Convention: if nothing was seen, drop = 1
    else:
        keep_rate = final_intersection_size / g_seen_size
        drop_rate = 1.0 - keep_rate
    
    return g_seen_size, final_intersection_size, drop_rate

def main():
    # Paths
    base_dir = Path('/root/lh/Context-Bench')
    results_dir = base_dir / 'results' / 'miniswe'
    traj_base_dir = base_dir / 'traj' / 'miniswe'
    gold_file = base_dir / 'results' / 'gold' / 'contextbench_full.gold.jsonl'
    
    # Load gold contexts
    print("Loading gold contexts...")
    gold_contexts = load_gold_contexts(gold_file)
    print(f"  Loaded {len(gold_contexts)} gold contexts")
    print()
    
    # Models and benchmarks
    models = {
        'claude45': 'claude',
        'gemini': 'gemini',
        'gpt5': 'gpt',
        'mistral': 'mistral'
    }
    benchmarks = ['Multi', 'Poly', 'Pro', 'Verified']
    
    # Store results
    results = defaultdict(lambda: defaultdict(dict))
    
    print("=" * 100)
    print("Evidence Drop Analysis")
    print("=" * 100)
    print()
    
    for model_name, traj_dir_name in models.items():
        print(f"Model: {model_name}")
        print("-" * 100)
        
        for benchmark in benchmarks:
            # Load all.jsonl to get instance list
            jsonl_path = results_dir / model_name / benchmark / 'all.jsonl'
            
            if not jsonl_path.exists():
                print(f"  {benchmark:12s}: File not found")
                continue
            
            instances = load_jsonl(jsonl_path)
            
            # Calculate drop for each instance
            drop_rates = []
            g_seen_sizes = []
            final_sizes = []
            
            for instance in instances:
                instance_id = instance['instance_id']
                
                # Find trajectory file
                traj_subdir = f"traj_{benchmark.lower()}_{traj_dir_name}"
                traj_path = traj_base_dir / traj_dir_name / traj_subdir / instance_id / f"{instance_id}.traj.json"
                
                # Get gold context for this instance
                gold_ctx = gold_contexts.get(instance_id, [])
                
                # Calculate drop
                g_seen_size, final_size, drop_rate = calculate_drop_for_instance(traj_path, gold_ctx)
                
                drop_rates.append(drop_rate)
                g_seen_sizes.append(g_seen_size)
                final_sizes.append(final_size)
            
            # Compute statistics
            if drop_rates:
                avg_drop = sum(drop_rates) / len(drop_rates)
                avg_g_seen = sum(g_seen_sizes) / len(g_seen_sizes)
                avg_final = sum(final_sizes) / len(final_sizes)
                
                results[model_name][benchmark] = {
                    'avg_drop': avg_drop,
                    'avg_g_seen': avg_g_seen,
                    'avg_final': avg_final,
                    'num_instances': len(drop_rates)
                }
                
                print(f"  {benchmark:12s}: "
                      f"Instances={len(drop_rates):3d}, "
                      f"Avg Drop={avg_drop:6.4f}, "
                      f"Avg G_seen={avg_g_seen:7.2f} lines, "
                      f"Avg Final={avg_final:7.2f} lines")
            else:
                print(f"  {benchmark:12s}: No data")
        
        print()
    
    # Summary comparison for Multi benchmark
    print("=" * 100)
    print("Summary: Multi Benchmark - Evidence Drop Comparison")
    print("=" * 100)
    print(f"{'Model':<12s} {'Instances':>10s} {'Avg Drop':>10s} {'Keep Rate':>10s} "
          f"{'Avg G_seen':>12s} {'Avg Final':>12s}")
    print("-" * 100)
    
    for model_name in models.keys():
        if model_name in results and 'Multi' in results[model_name]:
            r = results[model_name]['Multi']
            keep_rate = 1.0 - r['avg_drop']
            print(f"{model_name:<12s} {r['num_instances']:>10d} {r['avg_drop']:>10.4f} "
                  f"{keep_rate:>10.4f} {r['avg_g_seen']:>12.2f} {r['avg_final']:>12.2f}")
    
    print()
    print("Note: Lower Drop = Better consolidation (keeps gold evidence it found)")
    print("      Higher Drop = Frequently discards observed gold evidence")
    print()
    
    # Save results
    output_path = results_dir / 'evidence_drop_analysis.json'
    with open(output_path, 'w') as f:
        json.dump(dict(results), f, indent=2)
    
    print(f"Detailed results saved to: {output_path}")

if __name__ == '__main__':
    main()
