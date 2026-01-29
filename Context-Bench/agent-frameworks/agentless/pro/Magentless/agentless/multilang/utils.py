import json
import os
from pathlib import Path

from agentless.multilang.const import LANGUAGE, LANG_EXT


def process(raw_data):
    raw = json.loads(raw_data)
    
    # 支持两种数据格式：
    # 1. 原始格式: {"instance_id": "...", "repo": "org/repo", "base_commit": "...", ...}
    # 2. SWE-Bench 格式: {"instance_id": "...", "org": "...", "repo": "...", "base": {"sha": "..."}, ...}
    
    # 提取 instance_id
    instance_id = raw.get('instance_id', '')
    
    # 提取 repo
    if 'org' in raw and 'repo' in raw:
        repo = f'{raw["org"]}/{raw["repo"]}'
    else:
        repo = raw.get('repo', '')
    
    # 提取 base_commit
    if 'base' in raw and isinstance(raw['base'], dict) and 'sha' in raw['base']:
        base_commit = raw['base']['sha']
    else:
        base_commit = raw.get('base_commit', '')
    
    # 提取 problem_statement
    problem_statement = ''
    if 'resolved_issues' in raw and isinstance(raw['resolved_issues'], list) and len(raw['resolved_issues']) > 0:
        issue = raw['resolved_issues'][0]
        title = issue.get('title', '')
        body = issue.get('body', '')
        problem_statement = f"{title}\n{body}" if title or body else ''
    elif 'problem_statement' in raw:
        problem_statement = raw.get('problem_statement', '')
    
    data = {
        'repo': repo,
        'instance_id': instance_id,
        'base_commit': base_commit,
        'problem_statement': problem_statement,
    }
    return data


def load_local_json():
    dataset = []

    # 首先检查是否有自定义数据文件（通过环境变量传递）
    custom_data_file = os.environ.get('CUSTOM_DATA_FILE')
    if custom_data_file and Path(custom_data_file).exists():
        print(f"[INFO] 使用自定义数据文件: {custom_data_file}")
        lines = []
        for line in Path(custom_data_file).read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if line:
                lines.append(line)

        for line in lines:
            try:
                processed = process(line)
                dataset.append(processed)
            except Exception as e:
                print(f"WARNING: 处理数据行失败: {e}")
                continue

        print(f"[INFO] 从自定义文件加载了 {len(dataset)} 个实例")
        return dataset

    # 原有逻辑：读取 data/{lang}/ 目录
    if LANGUAGE == 'javascript':
        lang = 'js'
    elif LANGUAGE == 'typescript':
        lang = 'ts'
    else:
        lang = LANGUAGE
    # 尝试多个可能的路径（相对于 MagentLess 目录）
    paths_to_try = [
        Path(f'data/{lang}'),  # 符号链接指向 ../data/multi-swe-bench/{lang}
        Path(f'data/multi-swe-bench/{lang}'),
        Path(f'../data/multi-swe-bench/{lang}'),
        Path(f'../data/{lang}'),
    ]
    path = None
    for p in paths_to_try:
        if p.exists():
            path = p
            break
    
    if path is None:
        raise FileNotFoundError(f"找不到数据目录，尝试了: {paths_to_try}")
    
    lines = []
    for file in sorted(path.iterdir()):
        if file.is_file() and file.suffix == '.jsonl':
            try:
                file_lines = file.read_text(encoding='utf-8').splitlines()
                lines.extend(file_lines)
            except Exception as e:
                print(f"WARNING: 读取文件 {file} 失败: {e}")
                continue
    
    if not lines:
        raise ValueError(f"数据目录 {path} 中没有找到有效的 JSONL 数据")
    
    dataset = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            processed = process(line)
            dataset.append(processed)
        except Exception as e:
            print(f"WARNING: 处理数据行失败: {e}")
            continue
    
    if not dataset:
        raise ValueError(f"处理数据后没有有效结果，处理了 {len(lines)} 行")
    
    return dataset


def end_with_ext(file_name):
    for ext in LANG_EXT:
        if file_name.endswith(f'.{ext}'):
            return True
    return False
