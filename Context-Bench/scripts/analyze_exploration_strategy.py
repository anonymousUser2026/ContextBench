#!/usr/bin/env python3
"""
Analyze exploration strategy for different models in Context-Bench.

This script calculates:
1. Total number of context retrieval steps across all benchmark instances
2. Total lines of code viewed
3. Average steps per instance
4. Average lines of code per instance
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

def load_jsonl(filepath: Path) -> List[Dict]:
    """Load data from a JSONL file."""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def analyze_model_exploration(jsonl_path: Path) -> Tuple[int, int, int]:
    """
    Analyze exploration strategy for a single model.
    
    Returns:
        (total_steps, total_lines, num_instances)
    """
    data = load_jsonl(jsonl_path)
    
    total_steps = 0
    total_lines = 0
    num_instances = len(data)
    
    for instance in data:
        # Get number of steps
        total_steps += instance.get('num_steps', 0)
        
        # Get lines of code viewed (pred_size from final line metrics)
        final_line = instance.get('final', {}).get('line', {})
        total_lines += final_line.get('pred_size', 0)
    
    return total_steps, total_lines, num_instances

def main():
    # Base directory for results
    base_dir = Path('/root/lh/Context-Bench/results/miniswe')
    
    # Models and benchmarks to analyze
    models = ['claude45', 'gemini', 'gpt5', 'mistral']
    benchmarks = ['Multi', 'Poly', 'Pro', 'Verified']
    
    # Store results
    results = defaultdict(dict)
    
    print("=" * 80)
    print("Exploration Strategy Analysis")
    print("=" * 80)
    print()
    
    for model in models:
        print(f"Model: {model}")
        print("-" * 80)
        
        for benchmark in benchmarks:
            jsonl_path = base_dir / model / benchmark / 'all.jsonl'
            
            if not jsonl_path.exists():
                print(f"  {benchmark:12s}: File not found")
                continue
            
            total_steps, total_lines, num_instances = analyze_model_exploration(jsonl_path)
            
            if num_instances > 0:
                avg_steps = total_steps / num_instances
                avg_lines = total_lines / num_instances
            else:
                avg_steps = 0
                avg_lines = 0
            
            results[model][benchmark] = {
                'total_steps': total_steps,
                'total_lines': total_lines,
                'num_instances': num_instances,
                'avg_steps': avg_steps,
                'avg_lines': avg_lines
            }
            
            print(f"  {benchmark:12s}: "
                  f"Instances={num_instances:3d}, "
                  f"Total Steps={total_steps:5d}, "
                  f"Total Lines={total_lines:8d}, "
                  f"Avg Steps={avg_steps:6.2f}, "
                  f"Avg Lines={avg_lines:8.2f}")
        
        print()
    
    # Summary comparison across models for Multi benchmark
    print("=" * 80)
    print("Summary: Multi Benchmark Comparison")
    print("=" * 80)
    print(f"{'Model':<12s} {'Instances':>10s} {'Total Steps':>12s} {'Total Lines':>12s} "
          f"{'Avg Steps':>10s} {'Avg Lines':>12s}")
    print("-" * 80)
    
    for model in models:
        if model in results and 'Multi' in results[model]:
            r = results[model]['Multi']
            print(f"{model:<12s} {r['num_instances']:>10d} {r['total_steps']:>12d} "
                  f"{r['total_lines']:>12d} {r['avg_steps']:>10.2f} {r['avg_lines']:>12.2f}")
    
    print()
    
    # Save detailed results to JSON
    output_path = base_dir / 'exploration_analysis.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Detailed results saved to: {output_path}")

if __name__ == '__main__':
    main()
