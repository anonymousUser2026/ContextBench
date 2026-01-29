#!/usr/bin/env python3
"""Debug script to check evidence drop calculation."""

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
    """Parse EXPLORE_CONTEXT or PATCH_CONTEXT blocks from assistant messages."""
    contexts = []
    
    for msg in messages:
        # Only look at assistant messages
        if msg.get('role') != 'assistant':
            continue
        
        content = msg.get('content', '')
        if not content:
            continue
        
        # Find all context blocks in this message
        regex = re.compile(f'<{pattern}>(.*?)</{pattern}>', re.DOTALL)
        matches = regex.findall(content)
        
        for match in matches:
            # Parse File: and Lines: entries
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
        break

# Load trajectory
traj_path = Path('/root/lh/Context-Bench/traj/miniswe/mistral/traj_multi_mistral/alibaba__fastjson2-2559/alibaba__fastjson2-2559.traj.json')

print(f"\nLoading trajectory from: {traj_path}")
print(f"File exists: {traj_path.exists()}")

if traj_path.exists():
    with open(traj_path, 'r') as f:
        traj_data = json.load(f)
    
    messages = traj_data.get('messages', [])
    print(f"\nTotal messages in trajectory: {len(messages)}")
    
    print("\n" + "=" * 80)
    print("EXPLORE_CONTEXT parsing:")
    print("=" * 80)
    explore_contexts = parse_context_from_messages(messages, 'EXPLORE_CONTEXT')
    print(f"\nTotal EXPLORE_CONTEXT entries: {len(explore_contexts)}")
    for i, (file, start, end) in enumerate(explore_contexts[:5]):
        print(f"  {i+1}. {file} Lines: {start}-{end}")
    
    print("\n" + "=" * 80)
    print("PATCH_CONTEXT parsing:")
    print("=" * 80)
    patch_contexts = parse_context_from_messages(messages, 'PATCH_CONTEXT')
    print(f"\nTotal PATCH_CONTEXT entries: {len(patch_contexts)}")
    for i, (file, start, end) in enumerate(patch_contexts):
        print(f"  {i+1}. {file} Lines: {start}-{end}")
