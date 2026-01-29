#!/usr/bin/env python3
"""
Pro.csv 评测主脚本
对 Pro.csv 中的实例进行评测，使用 Magentless 生成补丁，然后使用 multi-swe-bench 进行评测
"""

import os
import sys
import csv
import json
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
MAGENTLESS_DIR = PROJECT_ROOT / "Magentless"
RESULTS_DIR = PROJECT_ROOT / "results" / "Pro"
DATA_DIR = PROJECT_ROOT / "data"
VENV_DIR = PROJECT_ROOT / ".venv"

# 确保使用虚拟环境
if VENV_DIR.exists():
    venv_python = VENV_DIR / "bin" / "python"
    if venv_python.exists():
        # 如果虚拟环境存在，使用虚拟环境的 Python
        if sys.executable != str(venv_python):
            print(f"注意: 请使用虚拟环境运行: {venv_python} {__file__}")
            print(f"或者运行: source {VENV_DIR}/bin/activate")


def load_instances_from_csv(csv_file: str) -> List[Dict]:
    """从 CSV 文件加载实例列表，并从 external_hf.csv 补充真实语言"""
    instances = []
    hf_data_file = PROJECT_ROOT / "data" / "SWE-bench_Pro" / "external_hf.csv"
    
    # 首先读取 Pro.csv
    pro_rows = []
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pro_rows.append(row)
            
    if not hf_data_file.exists():
        print(f"警告: {hf_data_file} 不存在，将使用 Pro.csv 中的语言字段")
        return pro_rows

    # 获取需要匹配的 ID
    target_ids = {row['original_inst_id'].strip() for row in pro_rows}
    
    # 增加字段限制
    csv.field_size_limit(sys.maxsize)
    
    # 从 HF 数据中获取真实语言
    matched_langs = {}
    with open(hf_data_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            inst_id = row['instance_id'].strip()
            if inst_id in target_ids:
                matched_langs[inst_id] = row.get('repo_language', 'unknown')
                if len(matched_langs) == len(target_ids):
                    break
    
    # 更新 Pro.csv 任务的语言
    for row in pro_rows:
        orig_id = row['original_inst_id'].strip()
        if orig_id in matched_langs:
            row['language'] = matched_langs[orig_id]
        instances.append(row)
        
    return instances


def group_instances_by_language(instances: List[Dict]) -> Dict[str, List[Dict]]:
    """按语言分组实例"""
    grouped = defaultdict(list)
    for instance in instances:
        lang = instance.get('language', 'unknown')
        grouped[lang].append(instance)
    return dict(grouped)


def setup_magentless_env(proxy_url: str = "http://localhost:18888", proxy_port: int = 18888):
    """设置 Magentless 环境变量，指向 proxy"""
    env = os.environ.copy()
    env['OPENAI_BASE_URL'] = proxy_url
    env['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'dummy-key')
    env['OPENAI_EMBED_URL'] = os.getenv('OPENAI_EMBED_URL', f'{proxy_url}/v1')
    env['PYTHONPATH'] = str(MAGENTLESS_DIR)
    
    # 如果存在自定义数据文件，设置其路径
    custom_data_file = os.getenv('CUSTOM_DATA_FILE')
    if custom_data_file:
        env['CUSTOM_DATA_FILE'] = custom_data_file
        print(f"[ENV] CUSTOM_DATA_FILE={custom_data_file}")
    
    # 确保 PATH 包含虚拟环境的 bin 目录
    venv_bin = str(PROJECT_ROOT / ".venv" / "bin")
    if venv_bin not in env.get('PATH', ''):
        env['PATH'] = f"{venv_bin}:{env.get('PATH', '')}"
    
    return env


def run_magentless_for_language(
    language: str,
    instances: List[Dict],
    folder_name: str,
    num_sets: int = 2,
    num_samples_per_set: int = 2,
    num_threads: int = 10,
    target_id: Optional[str] = None
) -> bool:
    """
    为指定语言的实例运行 Magentless 流程
    
    Returns:
        是否成功
    """
    print(f"\n{'='*60}")
    print(f"开始处理语言: {language}, 实例数: {len(instances)}")
    print(f"{'='*60}\n")
    
    # 设置环境变量
    env = setup_magentless_env()
    env['FOLDER_NAME'] = folder_name
    env['SWEBENCH_LANG'] = language
    # 不设置 PROJECT_FILE_LOC，让 Magentless 自动生成结构（不依赖预处理的 structure 文件）
    # 如果设置了但文件不存在会报错，所以不设置环境变量
    if 'PROJECT_FILE_LOC' in env:
        del env['PROJECT_FILE_LOC']
    env['DATASET'] = 'local_json'
    env['SPLIT'] = 'test'
    env['NUM_SETS'] = str(num_sets)
    env['NUM_SAMPLES_PER_SET'] = str(num_samples_per_set)
    env['NUM_REPRODUCTION'] = '0'
    env['NJ'] = str(num_threads)
    # 只有当 target_id 非空时才设置 TARGET_ID，避免空字符串导致的错误过滤
    if target_id:
        env['TARGET_ID'] = target_id
    
    # 切换到 Magentless 目录
    original_cwd = os.getcwd()
    try:
        os.chdir(MAGENTLESS_DIR)
        
        # 运行 localization 步骤
        print("运行 localization...")
        scripts = [
            ('script/localization1.1.sh', 'results/{}/file_level/loc_outputs.jsonl'),
            ('script/localization1.2.sh', 'results/{}/file_level_irrelevant/loc_outputs.jsonl'),
            ('script/localization1.3.sh', 'results/{}/retrievel_embedding/retrieve_locs.jsonl'),
            ('script/localization1.4.sh', 'results/{}/file_level_combined/combined_locs.jsonl'),
            ('script/localization2.1.sh', 'results/{}/related_elements/loc_outputs.jsonl'),
            ('script/localization3.1.sh', 'results/{}/edit_location_samples/loc_outputs.jsonl'),
            ('script/localization3.2.sh', 'results/{}/edit_location_individual/loc_merged_0-0_outputs.jsonl'),
        ]
        
        for script, expected_output in scripts:
            script_path = MAGENTLESS_DIR / script
            if script_path.exists():
                print(f"  [DEBUG] 执行: {script}")
                print(f"  [DEBUG] 工作目录: {MAGENTLESS_DIR}")
                print(f"  [DEBUG] FOLDER_NAME: {folder_name}")
                
                result = subprocess.run(
                    ['bash', str(script_path)],
                    env=env,
                    cwd=MAGENTLESS_DIR,
                    capture_output=True,
                    text=True
                )
                
                print(f"  [DEBUG] 返回码: {result.returncode}")
                
                if result.returncode != 0:
                    print(f"  [ERROR] {script} 执行失败")
                    if result.stderr:
                        print(f"  [ERROR] stderr (前1000字符):\n{result.stderr[:1000]}")
                    if result.stdout:
                        print(f"  [ERROR] stdout (前1000字符):\n{result.stdout[:1000]}")
                else:
                    print(f"  [DEBUG] {script} 执行成功")
                
                # 检查预期输出文件是否存在
                expected_file = MAGENTLESS_DIR / expected_output.format(folder_name)
                if expected_file.exists():
                    file_size = expected_file.stat().st_size
                    print(f"  [DEBUG] ✓ 输出文件存在: {expected_file} (大小: {file_size} 字节)")
                    # 检查文件是否有内容
                    if file_size > 0:
                        with open(expected_file, 'r', encoding='utf-8') as f:
                            lines = f.readlines()
                            print(f"  [DEBUG]   文件行数: {len(lines)}")
                            if lines:
                                print(f"  [DEBUG]   第一行预览: {lines[0][:100]}")
                    else:
                        print(f"  [WARNING] 输出文件为空!")
                else:
                    print(f"  [WARNING] ✗ 输出文件不存在: {expected_file}")
                
                # 针对 retrieve_locs.jsonl 的特殊检查
                if 'retrieve_locs.jsonl' in str(expected_file):
                    print(f"  [DEBUG] === 详细诊断 retrieve_locs.jsonl ===")
                    filter_file = MAGENTLESS_DIR / f'results/{folder_name}/file_level_irrelevant/loc_outputs.jsonl'
                    if filter_file.exists():
                        import json
                        with open(filter_file, 'r') as f:
                            all_lines = f.readlines()
                            print(f"  [DEBUG]   输入文件: {filter_file} (共 {len(all_lines)} 行)")
                            
                            # 统计 found_files 为空的数量
                            empty_count = 0
                            non_empty_count = 0
                            sample_instances = []
                            for i, line in enumerate(all_lines[:10]):
                                try:
                                    data = json.loads(line)
                                    found_files = data.get('found_files', [])
                                    if found_files:
                                        non_empty_count += 1
                                        if len(sample_instances) < 3:
                                            sample_instances.append((data['instance_id'], len(found_files)))
                                    else:
                                        empty_count += 1
                                except:
                                    pass
                            
                            print(f"  [DEBUG]   前10个实例: {non_empty_count} 个有文件, {empty_count} 个为空")
                            if sample_instances:
                                print(f"  [DEBUG]   有文件的实例示例: {sample_instances}")
                            
                            # 检查所有实例
                            total_empty = sum(1 for line in all_lines if not json.loads(line).get('found_files', []))
                            print(f"  [DEBUG]   全部 {len(all_lines)} 个实例中，{total_empty} 个 found_files 为空")
                            
                            # 检查是否有实例在 prev_o 中（会被跳过）
                            output_dir = MAGENTLESS_DIR / f'results/{folder_name}/retrievel_embedding'
                            if output_dir.exists():
                                prev_file = output_dir / 'retrieve_locs.jsonl'
                                if prev_file.exists():
                                    with open(prev_file, 'r') as f:
                                        prev_lines = f.readlines()
                                        print(f"  [DEBUG]   已有 {len(prev_lines)} 个实例在 prev_o 中（会被跳过）")
                                else:
                                    print(f"  [DEBUG]   prev_o 文件不存在（所有实例都是新的）")
                            
                            # 检查日志文件
                            log_dir = output_dir / 'retrieval_logs'
                            if log_dir.exists():
                                log_files = list(log_dir.glob('*.log'))
                                print(f"  [DEBUG]   日志文件数量: {len(log_files)}")
                                if log_files:
                                    # 检查最后一个日志文件
                                    latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
                                    with open(latest_log, 'r') as f:
                                        log_content = f.read()
                                        if 'Total number of documents: 0' in log_content:
                                            print(f"  [DEBUG]   ⚠️ 日志显示: Total number of documents: 0")
                                        if 'Total number of considered files: 0' in log_content:
                                            print(f"  [DEBUG]   ⚠️ 日志显示: Total number of considered files: 0")
                                        if 'Error' in log_content or 'Exception' in log_content:
                                            print(f"  [DEBUG]   ⚠️ 日志中有错误信息")
                                            error_lines = [l for l in log_content.split('\n') if 'Error' in l or 'Exception' in l][:3]
                                            for err in error_lines:
                                                print(f"  [DEBUG]      {err[:150]}")
                    else:
                        print(f"  [DEBUG]   ⚠️ 输入文件不存在: {filter_file}")
                    print(f"  [DEBUG] === 诊断结束 ===")
                
                print()  # 空行分隔
        
        # 运行 repair 步骤
        print("运行 repair...")
        repair_script = MAGENTLESS_DIR / 'script/repair.sh'
        if repair_script.exists():
            print(f"  [DEBUG] 执行: {repair_script}")
            result = subprocess.run(
                ['bash', str(repair_script)],
                env=env,
                cwd=MAGENTLESS_DIR,
                capture_output=True,
                text=True
            )
            print(f"  [DEBUG] 返回码: {result.returncode}")
            
            if result.returncode != 0:
                print(f"  [ERROR] repair 执行失败")
                if result.stderr:
                    print(f"  [ERROR] stderr (前1000字符):\n{result.stderr[:1000]}")
                if result.stdout:
                    print(f"  [ERROR] stdout (前1000字符):\n{result.stdout[:1000]}")
            else:
                print(f"  [DEBUG] repair 执行成功")
            
            # 检查 repair 输出
            print(f"  [DEBUG] === 详细检查 repair 输出 ===")
            for sample_idx in range(num_sets):
                repair_dir = MAGENTLESS_DIR / f'results/{folder_name}/repair_sample_{sample_idx + 1}'
                if repair_dir.exists():
                    print(f"  [DEBUG] ✓ repair_sample_{sample_idx + 1} 目录存在")
                    
                    # 检查 processed 文件
                    output_files = list(repair_dir.glob('output_*_processed.jsonl'))
                    print(f"  [DEBUG]   processed.jsonl 文件数量: {len(output_files)}")
                    for f in output_files[:3]:  # 只显示前3个
                        size = f.stat().st_size
                        print(f"  [DEBUG]   - {f.name}: {size} 字节")
                    
                    # 检查原始输出文件
                    raw_files = list(repair_dir.glob('output_*.jsonl'))
                    raw_processed = [f for f in raw_files if '_processed' not in f.name and '_normalized' not in f.name]
                    print(f"  [DEBUG]   原始输出文件数量: {len(raw_processed)}")
                    for f in raw_processed[:3]:
                        size = f.stat().st_size
                        print(f"  [DEBUG]   - {f.name}: {size} 字节")
                    
                    # 检查 normalized 文件
                    norm_files = list(repair_dir.glob('output_*_normalized.jsonl'))
                    print(f"  [DEBUG]   normalized 文件数量: {len(norm_files)}")
                else:
                    print(f"  [WARNING] ✗ repair_sample_{sample_idx + 1} 目录不存在")
                    
                    # 检查 repair 的输入文件
                    edit_loc_file = MAGENTLESS_DIR / f'results/{folder_name}/edit_location_individual/loc_merged_{sample_idx}-{sample_idx}_outputs.jsonl'
                    if edit_loc_file.exists():
                        size = edit_loc_file.stat().st_size
                        print(f"  [DEBUG]   输入文件存在: {edit_loc_file.name} ({size} 字节)")
                        # 检查输入文件内容
                        import json
                        try:
                            with open(edit_loc_file, 'r') as f:
                                lines = f.readlines()
                                print(f"  [DEBUG]   输入文件行数: {len(lines)}")
                                if lines:
                                    first_data = json.loads(lines[0])
                                    print(f"  [DEBUG]   第一个实例: {first_data.get('instance_id', 'unknown')}")
                        except Exception as e:
                            print(f"  [DEBUG]   ⚠️ 无法读取输入文件: {e}")
                    else:
                        print(f"  [DEBUG]   ⚠️ 输入文件不存在: {edit_loc_file.name}")
            print(f"  [DEBUG] === repair 检查结束 ===\n")
        
        # 运行 selection 步骤
        print("运行 selection...")
        print(f"  [DEBUG] === 详细检查 selection 输入 ===")
        # 检查 selection 的输入文件
        for sample_idx in range(num_sets):
            repair_dir = MAGENTLESS_DIR / f'results/{folder_name}/repair_sample_{sample_idx + 1}'
            if repair_dir.exists():
                processed_files = list(repair_dir.glob('output_*_processed.jsonl'))
                print(f"  [DEBUG]   repair_sample_{sample_idx + 1}: {len(processed_files)} 个 processed 文件")
                if not processed_files:
                    print(f"  [DEBUG]   ⚠️ repair_sample_{sample_idx + 1} 没有 processed 文件，selection 可能会失败")
                    # 检查是否有原始文件
                    raw_files = list(repair_dir.glob('output_*.jsonl'))
                    print(f"  [DEBUG]     但有 {len(raw_files)} 个原始文件")
            else:
                print(f"  [DEBUG]   ⚠️ repair_sample_{sample_idx + 1} 目录不存在")
        print(f"  [DEBUG] === selection 输入检查结束 ===")
        
        selection_script = MAGENTLESS_DIR / 'script/selection3.1.sh'
        if selection_script.exists():
            print(f"  [DEBUG] 执行: {selection_script}")
            result = subprocess.run(
                ['bash', str(selection_script)],
                env=env,
                cwd=MAGENTLESS_DIR,
                capture_output=True,
                text=True
            )
            print(f"  [DEBUG] 返回码: {result.returncode}")
            
            if result.returncode != 0:
                print(f"  [ERROR] selection 执行失败")
                if result.stderr:
                    print(f"  [ERROR] stderr (前1000字符):\n{result.stderr[:1000]}")
                if result.stdout:
                    print(f"  [ERROR] stdout (前1000字符):\n{result.stdout[:1000]}")
            else:
                print(f"  [DEBUG] selection 执行成功")
            
            # 检查 all_preds.jsonl
            all_preds_file = MAGENTLESS_DIR / f'results/{folder_name}/all_preds.jsonl'
            if all_preds_file.exists():
                size = all_preds_file.stat().st_size
                print(f"  [DEBUG] ✓ all_preds.jsonl 存在: {size} 字节")
                if size > 0:
                    with open(all_preds_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        print(f"  [DEBUG]   包含 {len(lines)} 个补丁")
                        if lines:
                            import json
                            try:
                                first_pred = json.loads(lines[0])
                                print(f"  [DEBUG]   第一个补丁 instance_id: {first_pred.get('instance_id', 'N/A')}")
                            except:
                                print(f"  [DEBUG]   第一行预览: {lines[0][:100]}")
                else:
                    print(f"  [WARNING] all_preds.jsonl 为空!")
            else:
                print(f"  [WARNING] ✗ all_preds.jsonl 不存在: {all_preds_file}")
                print(f"  [DEBUG] === 诊断 selection 失败原因 ===")
                # 检查是否有 repair 输出
                for sample_idx in range(num_sets):
                    repair_dir = MAGENTLESS_DIR / f'results/{folder_name}/repair_sample_{sample_idx + 1}'
                    if repair_dir.exists():
                        processed_files = list(repair_dir.glob('output_*_processed.jsonl'))
                        print(f"  [DEBUG]   repair_sample_{sample_idx + 1}: {len(processed_files)} 个 processed 文件")
                        if processed_files:
                            # 检查文件内容
                            import json
                            try:
                                with open(processed_files[0], 'r') as f:
                                    lines = f.readlines()
                                    print(f"  [DEBUG]     第一个文件有 {len(lines)} 行")
                                    if lines:
                                        first_data = json.loads(lines[0])
                                        print(f"  [DEBUG]     第一个实例: {first_data.get('instance_id', 'unknown')}")
                                        if 'model_patch' in first_data:
                                            patch_len = len(first_data.get('model_patch', ''))
                                            print(f"  [DEBUG]     第一个补丁长度: {patch_len} 字符")
                            except Exception as e:
                                print(f"  [DEBUG]     ⚠️ 无法读取文件: {e}")
                        else:
                            print(f"  [DEBUG]     ⚠️ 没有 processed 文件")
                            # 检查原始文件
                            raw_files = list(repair_dir.glob('output_*.jsonl'))
                            raw_processed = [f for f in raw_files if '_processed' not in f.name and '_normalized' not in f.name]
                            print(f"  [DEBUG]     但有 {len(raw_processed)} 个原始文件")
                    else:
                        print(f"  [DEBUG]   ⚠️ repair_sample_{sample_idx + 1} 目录不存在")
                print(f"  [DEBUG] === 诊断结束 ===")
            print()
        
        return True
        
    except Exception as e:
        print(f"错误: 运行 Magentless 时出错: {e}")
        return False
    finally:
        os.chdir(original_cwd)


def generate_traj_files(magentless_results_dir: str, output_trajs_dir: str, csv_file: str):
    """
    从 Magentless 的输出生成 traj.json 文件
    traj.json 是最终结果存储格式，包含 6_final_selected_patch 字段
    """
    print(f"\n生成 traj.json 文件...")
    
    # 使用 generate_traj_json.py 生成
    generate_cmd = [
        sys.executable,
        str(PROJECT_ROOT / 'evaluate_poly' / 'generate_traj_json.py'),
        '--csv', csv_file,
        '--results_dir', magentless_results_dir,
        '--output_dir', output_trajs_dir
    ]
    
    result = subprocess.run(generate_cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(result.stdout)
        return True
    else:
        print(f"生成 traj.json 失败: {result.stderr}")
        return False


def run_evaluation(
    patch_jsonl_file: str,
    dataset_files: List[str],
    workdir: str,
    output_dir: str,
    log_dir: str,
    max_workers: int = 8,
    repo_dir: Optional[str] = None
) -> bool:
    """
    运行 multi-swe-bench 评测
    
    Returns:
        是否成功
    """
    print(f"\n{'='*60}")
    print("开始运行 multi-swe-bench 评测")
    print(f"{'='*60}\n")
    
    if not dataset_files:
        print("错误: 未指定数据集文件，无法运行评测")
        print("提示: 请使用 --dataset_files 参数指定数据集文件路径")
        return False
    
    # 构建命令
    cmd = [
        sys.executable,
        '-m', 'multi_swe_bench.harness.run_evaluation',
        '--mode', 'evaluation',
        '--workdir', workdir,
        '--patch_files', patch_jsonl_file,
    ]
    
    # 添加数据集文件
    for dataset_file in dataset_files:
        cmd.extend(['--dataset_files', dataset_file])
    
    cmd.extend([
        '--output_dir', output_dir,
        '--log_dir', log_dir,
        '--max_workers', str(max_workers),
        '--log_level', 'INFO',
        '--need_clone', 'false'  # 假设仓库已克隆
    ])
    
    if repo_dir:
        cmd.extend(['--repo_dir', repo_dir])
    
    print(f"执行命令: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("评测完成")
            if result.stdout:
                print("输出:", result.stdout[-500:])  # 只显示最后500字符
            return True
        else:
            print(f"评测失败: {result.stderr}")
            if result.stdout:
                print("输出:", result.stdout[-500:])
            return False
    except Exception as e:
        print(f"错误: 运行评测时出错: {e}")
        return False


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Pro.csv 评测程序")
    parser.add_argument(
        '--csv', type=str,
        default=str(DATA_DIR / 'Pro.csv'),
        help='Pro.csv 文件路径'
    )
    parser.add_argument(
        '--proxy_url', type=str,
        default='http://localhost:18888',
        help='Proxy URL'
    )
    parser.add_argument(
        '--proxy_port', type=int,
        default=18888,
        help='Proxy 端口'
    )
    parser.add_argument(
        '--num_sets', type=int,
        default=2,
        help='Magentless NUM_SETS 参数'
    )
    parser.add_argument(
        '--num_samples_per_set', type=int,
        default=2,
        help='Magentless NUM_SAMPLES_PER_SET 参数'
    )
    parser.add_argument(
        '--num_threads', type=int,
        default=10,
        help='线程数'
    )
    parser.add_argument(
        '--target_id', type=str,
        default=None,
        help='只处理指定的 instance_id（用于测试）'
    )
    parser.add_argument(
        '--skip_magentless', action='store_true',
        help='跳过 Magentless 步骤（直接使用已有结果）'
    )
    parser.add_argument(
        '--skip_evaluation', action='store_true',
        help='跳过评测步骤（只生成补丁）'
    )
    parser.add_argument(
        '--language', type=str,
        default=None,
        help='只处理指定语言（用于测试）'
    )
    parser.add_argument(
        '--dataset_files', type=str,
        nargs='+',
        default=None,
        help='multi-swe-bench 数据集文件路径（JSONL 格式）'
    )
    parser.add_argument(
        '--repo_dir', type=str,
        default=None,
        help='仓库目录路径（如果已克隆）'
    )
    
    args = parser.parse_args()
    
    # 检查虚拟环境
    if VENV_DIR.exists():
        venv_python = VENV_DIR / "bin" / "python"
        if venv_python.exists() and sys.executable != str(venv_python):
            print(f"警告: 建议使用虚拟环境: source {VENV_DIR}/bin/activate")
    
    # 创建结果目录
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    trajs_dir = RESULTS_DIR / "trajs"
    details_dir = RESULTS_DIR / "details"
    trajs_dir.mkdir(parents=True, exist_ok=True)
    details_dir.mkdir(parents=True, exist_ok=True)
    
    # 读取 CSV
    print(f"读取 CSV 文件: {args.csv}")
    instances = load_instances_from_csv(args.csv)
    print(f"共 {len(instances)} 个实例")
    
    # 如果指定了 target_id，只处理该实例
    if args.target_id:
        instances = [inst for inst in instances if inst.get('instance_id') == args.target_id or inst.get('original_inst_id') == args.target_id]
        print(f"筛选后: {len(instances)} 个实例")
    
    # 按语言分组
    grouped = group_instances_by_language(instances)
    print(f"\n语言分组: {list(grouped.keys())}")
    for lang, insts in grouped.items():
        print(f"  {lang}: {len(insts)} 个实例")
    
    # 如果指定了语言，只处理该语言
    if args.language:
        if args.language in grouped:
            grouped = {args.language: grouped[args.language]}
        else:
            print(f"错误: 语言 {args.language} 不存在")
            return 1
    
    # 处理每个语言组
    all_patches_jsonl = []
    
    for language, lang_instances in grouped.items():
        folder_name = f"Pro_{language}"
        magentless_results_dir = MAGENTLESS_DIR / "results" / folder_name
        
        # 确定 target_id：如果 args.target_id 未设置，但 CSV 中只有一个实例，使用该实例的 original_inst_id
        effective_target_id = args.target_id
        if effective_target_id is None and len(lang_instances) == 1:
            effective_target_id = lang_instances[0].get('original_inst_id', '')
            print(f"  [DEBUG] 自动设置 target_id: {effective_target_id}")
        
        # 运行 Magentless
        if not args.skip_magentless:
            success = run_magentless_for_language(
                language=language,
                instances=lang_instances,
                folder_name=folder_name,
                num_sets=args.num_sets,
                num_samples_per_set=args.num_samples_per_set,
                num_threads=args.num_threads,
                target_id=effective_target_id
            )
            if not success:
                print(f"警告: {language} 的 Magentless 执行可能有问题，继续处理...")
        else:
            print(f"跳过 Magentless 步骤（使用已有结果）")
        
        # 生成 traj.json 文件（从 Magentless 输出生成）
        print(f"\n生成 {language} 的 traj.json 文件...")
        generate_traj_files(
            str(magentless_results_dir),
            str(trajs_dir),
            args.csv
        )
        
        # 转换补丁格式
        print(f"\n转换 {language} 的补丁格式...")
        patch_jsonl_file = RESULTS_DIR / f"patches_{language}.jsonl"
        
        # 使用 convert_patches.py 转换
        convert_cmd = [
            sys.executable,
            str(PROJECT_ROOT / 'evaluate_poly' / 'convert_patches.py'),
            '--csv', args.csv,
            '--results_dir', str(magentless_results_dir),
            '--output', str(patch_jsonl_file),
            '--trajs_dir', str(trajs_dir)
        ]
        
        result = subprocess.run(convert_cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"补丁转换成功: {patch_jsonl_file}")
            all_patches_jsonl.append(str(patch_jsonl_file))
        else:
            print(f"补丁转换失败: {result.stderr}")
    
    # 合并所有补丁文件
    if all_patches_jsonl:
        merged_patch_file = RESULTS_DIR / "patches_all.jsonl"
        print(f"\n合并所有补丁文件到: {merged_patch_file}")
        with open(merged_patch_file, 'w', encoding='utf-8') as out_f:
            for patch_file in all_patches_jsonl:
                if os.path.exists(patch_file):
                    with open(patch_file, 'r', encoding='utf-8') as in_f:
                        for line in in_f:
                            if line.strip():
                                out_f.write(line)
        
        # 运行评测
        if not args.skip_evaluation:
            # 获取数据集文件
            if args.dataset_files:
                dataset_files = args.dataset_files
            else:
                # 尝试从环境变量或默认位置查找
                dataset_files = []
                # 首先检查 CUSTOM_DATA_FILE 环境变量
                custom_data = os.getenv('CUSTOM_DATA_FILE')
                if custom_data and os.path.exists(custom_data):
                    dataset_files.append(str(Path(custom_data).absolute()))
                else:
                    default_dataset_dir = PROJECT_ROOT / "temp_data" / "pro"
                    if default_dataset_dir.exists():
                        # 查找所有 JSONL 文件，使用绝对路径
                        for jsonl_file in default_dataset_dir.glob("**/*.jsonl"):
                            # 使用绝对路径，确保 multi-swe-bench 能找到文件
                            dataset_files.append(str(jsonl_file.absolute()))
                
                if not dataset_files:
                    print("警告: 未找到数据集文件，请使用 --dataset_files 参数指定")
                    print("跳过评测步骤")
                    return 0
            
            workdir = RESULTS_DIR / "workdir"
            log_dir = RESULTS_DIR / "logs"
            workdir.mkdir(parents=True, exist_ok=True)
            log_dir.mkdir(parents=True, exist_ok=True)
            
            repo_dir = args.repo_dir or str(RESULTS_DIR / "repos")
            
            success = run_evaluation(
                patch_jsonl_file=str(merged_patch_file.absolute()),
                dataset_files=dataset_files,
                workdir=str(workdir.absolute()),
                output_dir=str(details_dir.absolute()),
                log_dir=str(log_dir.absolute()),
                max_workers=args.num_threads,
                repo_dir=str(Path(repo_dir).absolute()) if repo_dir else None
            )
            
            if success:
                print("\n✓ 评测完成！")
            else:
                print("\n✗ 评测失败，请检查日志")
                return 1
        else:
            print("\n跳过评测步骤")
    
    print(f"\n结果存储在: {RESULTS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

