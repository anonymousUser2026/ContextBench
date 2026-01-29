#!/usr/bin/env python3
"""
Compare exploration efficiency across different models.

This script provides deeper analysis of exploration strategies:
- Lines per step (efficiency metric)
- Distribution analysis
- Strategy characterization
"""

import json
import os
from pathlib import Path
from typing import Dict, List
import math

def load_jsonl(filepath: Path) -> List[Dict]:
    """Load data from a JSONL file."""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def analyze_per_instance_metrics(jsonl_path: Path) -> Dict:
    """
    Get per-instance metrics for detailed analysis.
    """
    data = load_jsonl(jsonl_path)
    
    steps_list = []
    lines_list = []
    lines_per_step = []
    
    for instance in data:
        num_steps = instance.get('num_steps', 0)
        final_line = instance.get('final', {}).get('line', {})
        num_lines = final_line.get('pred_size', 0)
        
        steps_list.append(num_steps)
        lines_list.append(num_lines)
        
        if num_steps > 0:
            lines_per_step.append(num_lines / num_steps)
        else:
            lines_per_step.append(0)
    
    return {
        'steps': steps_list,
        'lines': lines_list,
        'lines_per_step': lines_per_step
    }

def compute_stats(values: List[float]) -> Dict:
    """Compute statistics for a list of values."""
    if not values:
        return {
            'mean': 0,
            'median': 0,
            'std': 0,
            'min': 0,
            'max': 0,
            'p25': 0,
            'p75': 0
        }
    
    sorted_values = sorted(values)
    n = len(sorted_values)
    
    # Mean
    mean_val = sum(values) / n
    
    # Median
    if n % 2 == 0:
        median_val = (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2
    else:
        median_val = sorted_values[n // 2]
    
    # Standard deviation
    if n > 1:
        variance = sum((x - mean_val) ** 2 for x in values) / (n - 1)
        std_val = math.sqrt(variance)
    else:
        std_val = 0
    
    # Percentiles
    def percentile(data, p):
        k = (len(data) - 1) * p
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return data[int(k)]
        d0 = data[int(f)] * (c - k)
        d1 = data[int(c)] * (k - f)
        return d0 + d1
    
    return {
        'mean': mean_val,
        'median': median_val,
        'std': std_val,
        'min': min(values),
        'max': max(values),
        'p25': percentile(sorted_values, 0.25),
        'p75': percentile(sorted_values, 0.75)
    }

def main():
    base_dir = Path('/root/lh/Context-Bench/results/miniswe')
    
    models = ['claude45', 'gemini', 'gpt5', 'mistral']
    benchmark = 'Multi'  # Focus on Multi benchmark
    
    print("=" * 100)
    print("Exploration Efficiency Comparison - Multi Benchmark")
    print("=" * 100)
    print()
    
    results = {}
    
    for model in models:
        jsonl_path = base_dir / model / benchmark / 'all.jsonl'
        
        if not jsonl_path.exists():
            continue
        
        metrics = analyze_per_instance_metrics(jsonl_path)
        
        steps_stats = compute_stats(metrics['steps'])
        lines_stats = compute_stats(metrics['lines'])
        lps_stats = compute_stats(metrics['lines_per_step'])
        
        results[model] = {
            'steps': steps_stats,
            'lines': lines_stats,
            'lines_per_step': lps_stats
        }
    
    # Print detailed comparison
    print("1. STEPS PER INSTANCE")
    print("-" * 100)
    print(f"{'Model':<12s} {'Mean':>8s} {'Median':>8s} {'Std':>8s} {'Min':>8s} {'Max':>8s} {'P25':>8s} {'P75':>8s}")
    print("-" * 100)
    for model in models:
        if model in results:
            s = results[model]['steps']
            print(f"{model:<12s} {s['mean']:>8.2f} {s['median']:>8.2f} {s['std']:>8.2f} "
                  f"{s['min']:>8.0f} {s['max']:>8.0f} {s['p25']:>8.2f} {s['p75']:>8.2f}")
    print()
    
    print("2. LINES OF CODE PER INSTANCE")
    print("-" * 100)
    print(f"{'Model':<12s} {'Mean':>8s} {'Median':>8s} {'Std':>8s} {'Min':>8s} {'Max':>8s} {'P25':>8s} {'P75':>8s}")
    print("-" * 100)
    for model in models:
        if model in results:
            l = results[model]['lines']
            print(f"{model:<12s} {l['mean']:>8.2f} {l['median']:>8.2f} {l['std']:>8.2f} "
                  f"{l['min']:>8.0f} {l['max']:>8.0f} {l['p25']:>8.2f} {l['p75']:>8.2f}")
    print()
    
    print("3. LINES PER STEP (Efficiency Metric)")
    print("-" * 100)
    print(f"{'Model':<12s} {'Mean':>8s} {'Median':>8s} {'Std':>8s} {'Min':>8s} {'Max':>8s} {'P25':>8s} {'P75':>8s}")
    print("-" * 100)
    for model in models:
        if model in results:
            lps = results[model]['lines_per_step']
            print(f"{model:<12s} {lps['mean']:>8.2f} {lps['median']:>8.2f} {lps['std']:>8.2f} "
                  f"{lps['min']:>8.2f} {lps['max']:>8.2f} {lps['p25']:>8.2f} {lps['p75']:>8.2f}")
    print()
    
    # Strategy characterization
    print("=" * 100)
    print("EXPLORATION STRATEGY CHARACTERIZATION")
    print("=" * 100)
    print()
    
    for model in models:
        if model not in results:
            continue
        
        avg_steps = results[model]['steps']['mean']
        avg_lines = results[model]['lines']['mean']
        avg_lps = results[model]['lines_per_step']['mean']
        
        print(f"{model.upper()}")
        print("-" * 100)
        print(f"  Average steps per instance:      {avg_steps:8.2f}")
        print(f"  Average lines per instance:      {avg_lines:8.2f}")
        print(f"  Average lines per step:          {avg_lps:8.2f}")
        
        # Characterize strategy
        if avg_steps < 10 and avg_lps > 100:
            strategy = "AGGRESSIVE: Few steps with large code chunks per step"
        elif avg_steps < 10 and avg_lps < 100:
            strategy = "FOCUSED: Few steps with small code chunks (precise targeting)"
        elif avg_steps > 15 and avg_lps < 50:
            strategy = "ITERATIVE: Many small steps exploring gradually"
        elif avg_steps > 15 and avg_lps > 50:
            strategy = "COMPREHENSIVE: Many steps with moderate code chunks per step"
        else:
            strategy = "BALANCED: Moderate exploration strategy"
        
        print(f"  Strategy:                        {strategy}")
        print()
    
    # Save results
    output_path = base_dir / 'exploration_efficiency.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Detailed statistics saved to: {output_path}")

if __name__ == '__main__':
    main()
