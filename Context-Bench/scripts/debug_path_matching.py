#!/usr/bin/env python3
"""Debug path matching between gold and trajectory contexts."""

import json
import re
from pathlib import Path
from typing import List

def load_jsonl(filepath: Path):
    """Load data from a JSONL file."""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def parse_context_from_messages(messages: List, pattern: str):
    """Parse context blocks from assistant messages."""
    contexts = []
    
    for msg in messages:
        if msg.get('role') != 'assistant':
            continue
        
        content = msg.get('content', '')
        if not content:
            continue
        
        regex = re.compile(f'<{pattern}>(.*?)</{pattern}>', re.DOTALL)
        matches = regex.findall(content)
        
        for match in matches:
            lines_in_block = match.strip().split('\n')
            
            current_file = None
            for line in lines_in_block:
                line = line.strip()
                if not line:
                    continue
                    
                if line.startswith('File:'):
                    current_file = line.replace('File:', '').strip()
                elif line.startswith('Lines:') and current_file:
                    line_range = line.replace('Lines:', '').strip()
                    if '-' in line_range:
                        try:
                            start, end = line_range.split('-')
                            contexts.append((current_file, int(start), int(end)))
                        except ValueError:
                            pass
    
    return contexts

def normalize_file_path(path: str, base_paths: List[str] = None) -> str:
    """Normalize file path to relative path."""
    if base_paths is None:
        base_paths = ['/testbed/', '/home/', '/workspace/']
    
    for base in base_paths:
        if path.startswith(base):
            return path[len(base):]
    
    return path

# Test with one instance
instance_id = 'alibaba__fastjson2-2559'

# Load gold context
gold_file = Path('/root/lh/Context-Bench/results/gold/contextbench_full.gold.jsonl')
gold_data = load_jsonl(gold_file)

gold_ctx = None
for entry in gold_data:
    if entry.get('original_inst_id') == instance_id:
        gold_ctx = entry.get('gold_ctx', [])
        print(f"Gold context for {instance_id}:")
        for ctx in gold_ctx:
            print(f"  File: {ctx['file']}")
            print(f"  Lines: {ctx['start_line']}-{ctx['end_line']}")
            print(f"  Normalized: {normalize_file_path(ctx['file'])}")
        break

# Load trajectory
traj_path = Path('/root/lh/Context-Bench/traj/miniswe/mistral/traj_multi_mistral/alibaba__fastjson2-2559/alibaba__fastjson2-2559.traj.json')

if traj_path.exists():
    with open(traj_path, 'r') as f:
        traj_data = json.load(f)
    
    messages = traj_data.get('messages', [])
    
    print("\n" + "=" * 80)
    print("EXPLORE_CONTEXT entries:")
    print("=" * 80)
    explore_contexts = parse_context_from_messages(messages, 'EXPLORE_CONTEXT')
    
    unique_files = set()
    for file, start, end in explore_contexts:
        normalized = normalize_file_path(file)
        unique_files.add(normalized)
        print(f"  Original: {file}")
        print(f"  Normalized: {normalized}")
        print(f"  Lines: {start}-{end}")
        print()
    
    print(f"Total EXPLORE_CONTEXT entries: {len(explore_contexts)}")
    print(f"Unique normalized files: {unique_files}")
    
    print("\n" + "=" * 80)
    print("PATCH_CONTEXT entries:")
    print("=" * 80)
    patch_contexts = parse_context_from_messages(messages, 'PATCH_CONTEXT')
    
    for file, start, end in patch_contexts:
        normalized = normalize_file_path(file)
        print(f"  Original: {file}")
        print(f"  Normalized: {normalized}")
        print(f"  Lines: {start}-{end}")
        print()
    
    print(f"Total PATCH_CONTEXT entries: {len(patch_contexts)}")
    
    # Now check if paths match
    print("\n" + "=" * 80)
    print("Path Matching Check:")
    print("=" * 80)
    
    gold_file_normalized = normalize_file_path(gold_ctx[0]['file'])
    print(f"Gold file (normalized): {gold_file_normalized}")
    print(f"Gold lines: {gold_ctx[0]['start_line']}-{gold_ctx[0]['end_line']}")
    
    # Build gold lines set
    gold_lines = set()
    for ctx in gold_ctx:
        file_path = normalize_file_path(ctx['file'])
        for line in range(ctx['start_line'], ctx['end_line'] + 1):
            gold_lines.add((file_path, line))
    
    print(f"\nGold lines set size: {len(gold_lines)}")
    print(f"Sample gold lines: {list(gold_lines)[:3]}")
    
    # Build explored lines set
    explored_lines = set()
    for file_path, start, end in explore_contexts:
        normalized_path = normalize_file_path(file_path)
        for line in range(start, end + 1):
            explored_lines.add((normalized_path, line))
    
    print(f"\nExplored lines set size: {len(explored_lines)}")
    
    # Calculate intersection
    g_seen = explored_lines & gold_lines
    print(f"\nG_seen (intersection) size: {len(g_seen)}")
    print(f"G_seen lines: {sorted(g_seen)}")
    
    # Build final lines set
    final_lines = set()
    for file_path, start, end in patch_contexts:
        normalized_path = normalize_file_path(file_path)
        for line in range(start, end + 1):
            final_lines.add((normalized_path, line))
    
    print(f"\nFinal lines set size: {len(final_lines)}")
    
    # Calculate final intersection
    final_intersection = final_lines & gold_lines
    print(f"\nFinal intersection size: {len(final_intersection)}")
    print(f"Final intersection lines: {sorted(final_intersection)}")
    
    # Calculate drop
    if len(g_seen) > 0:
        drop_rate = 1.0 - (len(final_intersection) / len(g_seen))
        keep_rate = len(final_intersection) / len(g_seen)
        print(f"\nDrop rate: {drop_rate:.4f}")
        print(f"Keep rate: {keep_rate:.4f}")
    else:
        print(f"\nNo gold evidence seen during exploration!")
