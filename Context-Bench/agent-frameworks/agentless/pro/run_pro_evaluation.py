#!/usr/bin/env python3
"""
Poly/Pro 全量评测启动脚本
针对 Pro.csv 及其背后的 external_hf.csv 数据集进行评测
"""

import os
import sys
import csv
import json
import argparse
import subprocess
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Set

# 项目路径
PROJECT_ROOT = Path(__file__).parent.absolute()
MAGENTLESS_DIR = PROJECT_ROOT / "Magentless"
VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
PRO_CSV_FILE = PROJECT_ROOT / "data" / "Pro.csv"
HF_DATA_FILE = PROJECT_ROOT / "data" / "SWE-bench_Pro" / "external_hf.csv"
TEMP_DATA_DIR = PROJECT_ROOT / "temp_data" / "pro"
RESULTS_DIR = PROJECT_ROOT / "results" / "Pro"
TRAJS_DIR = RESULTS_DIR / "trajs"

# 增加 CSV 字段限制，防止 patch 等大字段导致报错
csv.field_size_limit(sys.maxsize)

def is_instance_completed(original_inst_id: str, language: str) -> bool:
    """
    检查实例是否已有完整的 traj.json
    """
    if not TRAJS_DIR.exists():
        return False

    # 查找对应的 traj.json 文件
    # 兼容新的命名：{instance_id}_traj.json 或者带语言信息的
    for traj_file in TRAJS_DIR.glob(f"*{original_inst_id}*_traj.json"):
        try:
            with open(traj_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            traj_original_id = data.get('original_id', data.get('instance_id', ''))
            if traj_original_id != original_inst_id:
                continue

            stage1 = data.get('1_model_selected_files', [])
            stage2 = data.get('2_embedding_selected_files', [])
            stage3 = data.get('3_final_combined_files', [])
            stage4 = data.get('4_related_elements', {})
            stage5 = data.get('5_sampled_edit_locs_and_patches', [])
            stage6 = data.get('6_final_selected_patch', '').strip()

            if not stage1 or not stage2:
                return True

            stage5_has_content = any(
                p.get('model_patch', '').strip()
                for p in stage5
            ) if stage5 else False
            if not stage5_has_content and stage3: # 已有前面的阶段但后面是空的，说明定位到了但修不出
                return True

            if stage3 and stage4 and stage5 and stage6:
                return True

            return False
        except:
            continue
    return False

def load_pro_instances(language: Optional[str] = None, instance_id: Optional[str] = None) -> Dict[str, List[Dict]]:
    """
    加载 Pro.csv 并匹配 external_hf.csv 中的详细信息
    """
    print(f"读取 Pro.csv: {PRO_CSV_FILE}")
    pro_rows = []
    with open(PRO_CSV_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pro_rows.append(row)
    
    if instance_id:
        # 同时匹配 instance_id 和 original_inst_id
        pro_rows = [row for row in pro_rows if row['instance_id'].strip() == instance_id or row['original_inst_id'].strip() == instance_id]
        if not pro_rows:
            print(f"警告: 没有找到 ID 为 {instance_id} 的实例")
            return {}
        target_ids = {row['original_inst_id'].strip() for row in pro_rows}
    else:
        target_ids = {row['original_inst_id'].strip() for row in pro_rows}

    # 从 HF 数据集中匹配详细信息
    print(f"从 {HF_DATA_FILE.name} 匹配 {len(target_ids)} 个实例的信息...")
    matched_data = {}
    with open(HF_DATA_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            inst_id = row['instance_id'].strip()
            if inst_id in target_ids:
                matched_data[inst_id] = row
                if len(matched_data) == len(target_ids):
                    break
    
    instances_by_lang = defaultdict(list)
    for row in pro_rows:
        orig_id = row['original_inst_id'].strip()
        hf_info = matched_data.get(orig_id)
        if not hf_info:
            print(f"警告: 找不到实例 {orig_id} 的详细信息")
            continue
        
        # 使用 HF 中的真实语言
        real_lang = hf_info.get('repo_language', 'unknown')
        if language and real_lang != language:
            continue
        
        # 合并信息
        full_instance = row.copy()
        full_instance.update(hf_info)
        # 统一 commit 字段
        full_instance['commit'] = row.get('commit', '').strip()
        full_instance['base_commit'] = hf_info.get('base_commit', full_instance['commit'])
        
        instances_by_lang[real_lang].append(full_instance)
        
    return dict(instances_by_lang)

def generate_temp_dataset(instances_by_lang: Dict[str, List[Dict]], force_rerun: bool = False) -> Dict[str, Path]:
    """
    为每种语言生成临时 JSONL 文件
    """
    TEMP_DATA_DIR.mkdir(parents=True, exist_ok=True)
    temp_files = {}

    for language, instances in instances_by_lang.items():
        matched_json_lines = []
        skipped_count = 0
        
        for inst in instances:
            orig_id = inst['original_inst_id']
            if not force_rerun and is_instance_completed(orig_id, language):
                skipped_count += 1
                continue
            
            # 构造 Magentless 期望的格式
            data = {
                'instance_id': orig_id,
                'repo': inst.get('repo', ''),
                'base_commit': inst.get('base_commit', ''),
                'problem_statement': inst.get('problem_statement', ''),
                'test_patch': inst.get('test_patch', ''),
                'patch': inst.get('patch', ''),
            }
            matched_json_lines.append(json.dumps(data, ensure_ascii=False))
        
        print(f"  {language}: 共 {len(instances)} 个, 跳过 {skipped_count} 个, 需要处理 {len(matched_json_lines)} 个")
        
        if not matched_json_lines:
            continue
            
        temp_file = TEMP_DATA_DIR / f"{language}.jsonl"
        temp_file.write_text('\n'.join(matched_json_lines) + '\n', encoding='utf-8')
        temp_files[language] = temp_file
        
    return temp_files

def setup_environment(language: str, temp_file: Path):
    env = os.environ.copy()
    env['PYTHONPATH'] = str(MAGENTLESS_DIR)
    env['CUSTOM_DATA_FILE'] = str(temp_file.absolute())
    env['SWEBENCH_LANG'] = language
    return env

def run_evaluation(language: str, instances: List[Dict], temp_file: Path, num_threads: int = 1, skip_evaluation: bool = False):
    print(f"\n{'='*60}")
    print(f"开始处理语言: {language}, 实例数: {len(instances)}")
    print(f"{'='*60}\n")

    env = setup_environment(language, temp_file)
    
    cmd = [
        str(VENV_PYTHON),
        str(PROJECT_ROOT / "evaluate_poly" / "evaluate_poly.py"),
        "--language", language,
        "--num_threads", str(num_threads),
        "--csv", str(PRO_CSV_FILE),
    ]
    
    if skip_evaluation:
        cmd.append("--skip_evaluation")

    result = subprocess.run(
        cmd,
        env=env,
        cwd=PROJECT_ROOT,
        capture_output=False,
        text=True
    )
    return result.returncode == 0

def cleanup_incomplete_trajs():
    if not TRAJS_DIR.exists():
        return
    # 这里可以添加清理逻辑，Pro 场景下可能暂时不需要那么激进的清理
    pass

def main():
    parser = argparse.ArgumentParser(description="Pro.csv 全量评测启动脚本")
    parser.add_argument("--language", "-l", help="只运行指定语言")
    parser.add_argument("--instance", "-i", help="只运行指定单个实例")
    parser.add_argument("--num_threads", "-t", type=int, default=1, help="线程数")
    parser.add_argument("--force_rerun", "-f", action="store_true", help="强制重新运行")
    parser.add_argument("--skip_evaluation", "-s", action="store_true", help="跳过评测步骤")
    
    args = parser.parse_args()
    
    instances_by_lang = load_pro_instances(args.language, args.instance)
    if not instances_by_lang:
        print("错误: 没有找到匹配的实例")
        return

    print("\n实例统计:")
    for lang, insts in sorted(instances_by_lang.items()):
        print(f"  {lang}: {len(insts)} 个实例")

    temp_files = generate_temp_dataset(instances_by_lang, args.force_rerun)
    if not temp_files:
        print("所有实例已完成或无任务。")
        return

    cleanup_incomplete_trajs()
    
    success_count = 0
    fail_count = 0
    for lang in sorted(temp_files.keys()):
        success = run_evaluation(lang, instances_by_lang[lang], temp_files[lang], args.num_threads, args.skip_evaluation)
        if success:
            success_count += 1
        else:
            fail_count += 1

    print(f"\n处理完成: 成功 {success_count}, 失败 {fail_count}")

if __name__ == "__main__":
    main()
