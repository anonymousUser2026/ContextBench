#!/usr/bin/env python3
"""
测试单个 instance 的完整流程
用于快速验证修复是否有效
"""

import os
import sys
import csv
import json
import subprocess
from pathlib import Path
from typing import Dict, Optional

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
MAGENTLESS_DIR = PROJECT_ROOT / "MagentLess"
EVALUATE_DIR = PROJECT_ROOT / "evaluate" / "multi-swe-bench"
RESULTS_DIR = PROJECT_ROOT / "results" / "Multi"
DATA_DIR = PROJECT_ROOT / "data"
VENV_DIR = PROJECT_ROOT / ".venv"

# 确保使用虚拟环境
if VENV_DIR.exists():
    venv_python = VENV_DIR / "bin" / "python"
    if venv_python.exists() and sys.executable != str(venv_python):
        print(f"注意: 请使用虚拟环境运行: {venv_python} {__file__}")


def find_instance_in_csv(csv_file: str, instance_id: str) -> Optional[Dict]:
    """在 CSV 中查找指定的 instance"""
    # 如果 csv_file 是相对路径，尝试相对于 PROJECT_ROOT 查找
    if not Path(csv_file).is_absolute():
        csv_path = PROJECT_ROOT / csv_file
        if csv_path.exists():
            csv_file = str(csv_path)
        else:
            # 尝试在 data 目录下查找
            csv_path = PROJECT_ROOT / "data" / csv_file
            if csv_path.exists():
                csv_file = str(csv_path)
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('instance_id') == instance_id:
                return row
    return None


def setup_magentless_env(proxy_url: str = "http://localhost:18888", proxy_port: int = 18888):
    """设置 MagentLess 环境变量，指向 proxy"""
    env = os.environ.copy()
    env['OPENAI_BASE_URL'] = proxy_url
    env['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'dummy-key')
    env['OPENAI_EMBED_URL'] = os.getenv('OPENAI_EMBED_URL', f'{proxy_url}/v1')
    env['PYTHONPATH'] = str(MAGENTLESS_DIR)
    
    # 修复：确保 PATH 包含虚拟环境，以便 bash 脚本能使用正确的 Python
    venv_bin = str(PROJECT_ROOT / ".venv" / "bin")
    if venv_bin not in env['PATH']:
        env['PATH'] = f"{venv_bin}:{env['PATH']}"
    
    return env


def test_single_instance(
    instance_id: str,
    csv_file: str = "data/Multi.csv",
    proxy_url: str = "http://localhost:18888",
    num_threads: int = 1,
    num_sets: int = 1,
    num_samples_per_set: int = 1,
):
    """
    测试单个 instance 的完整流程
    
    Args:
        instance_id: 要测试的 instance_id（例如：Multi-SWE-Bench__javascript__maintenance__bugfix__72488b59）
        csv_file: CSV 文件路径
        proxy_url: Proxy URL
        num_threads: 线程数
        num_sets: 集合数
        num_samples_per_set: 每个集合的样本数
    """
    print("=" * 60)
    print(f"测试单个 Instance: {instance_id}")
    print("=" * 60)
    
    # 查找 instance
    print(f"\n1. 查找 instance 信息...")
    instance = find_instance_in_csv(csv_file, instance_id)
    if not instance:
        print(f"✗ 错误: 在 {csv_file} 中找不到 instance_id: {instance_id}")
        return False
    
    language = instance.get('language', '')
    original_inst_id = instance.get('original_inst_id', '')
    print(f"  ✓ 找到 instance:")
    print(f"    语言: {language}")
    print(f"    original_inst_id: {original_inst_id}")
    
    # 设置环境
    env = setup_magentless_env(proxy_url)
    folder_name = f"Test_{language}_{instance_id.split('__')[-1]}"
    env['FOLDER_NAME'] = folder_name
    env['SWEBENCH_LANG'] = language
    env['DATASET'] = 'local_json'
    env['SPLIT'] = 'test'
    env['NUM_SETS'] = str(num_sets)
    env['NUM_SAMPLES_PER_SET'] = str(num_samples_per_set)
    env['NUM_REPRODUCTION'] = '0'
    env['NJ'] = str(num_threads)
    env['TARGET_ID'] = original_inst_id  # 使用 original_inst_id 作为 target_id
    
    # 不设置 PROJECT_FILE_LOC，让 MagentLess 自动生成结构
    if 'PROJECT_FILE_LOC' in env:
        del env['PROJECT_FILE_LOC']
    
    print(f"\n2. 环境配置:")
    print(f"  FOLDER_NAME: {folder_name}")
    print(f"  SWEBENCH_LANG: {language}")
    print(f"  TARGET_ID: {original_inst_id}")
    print(f"  NUM_THREADS: {num_threads}")
    
    # 切换到 MagentLess 目录
    original_cwd = os.getcwd()
    try:
        os.chdir(MAGENTLESS_DIR)
        
        # 运行各个步骤
        steps = [
            ('localization1.1.sh', 'file_level/loc_outputs.jsonl', '文件级别定位（相关文件）'),
            ('localization1.2.sh', 'file_level_irrelevant/loc_outputs.jsonl', '文件级别定位（无关文件）'),
            ('localization1.3.sh', 'retrievel_embedding/retrieve_locs.jsonl', '嵌入检索定位'),
            ('localization1.4.sh', 'file_level_combined/combined_locs.jsonl', '合并定位结果'),
            ('localization2.1.sh', 'related_elements/loc_outputs.jsonl', '相关元素定位'),
            ('localization3.1.sh', 'edit_location_samples/loc_outputs.jsonl', '编辑位置采样'),
            ('localization3.2.sh', 'edit_location_individual/loc_merged_0-0_outputs.jsonl', '编辑位置细化'),
            ('repair.sh', 'repair_sample_1/output_0_processed.jsonl', '修复生成'),
            ('selection3.1.sh', 'all_preds.jsonl', '补丁选择'),
        ]
        
        results_dir = MAGENTLESS_DIR / "results" / folder_name
        
        for i, (script, expected_file, description) in enumerate(steps, 1):
            print(f"\n{i}. {description} ({script})...")
            script_path = MAGENTLESS_DIR / "script" / script
            
            if not script_path.exists():
                print(f"  ⚠ 脚本不存在，跳过: {script_path}")
                continue
            
            print(f"  执行: {script}")
            
            # 修复：使用虚拟环境中的 Python，并确保 PYTHONPATH 正确
            venv_python = MAGENTLESS_DIR / ".venv" / "bin" / "python"
            if not venv_python.exists():
                # 尝试项目根目录的虚拟环境
                venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"
            
            result = subprocess.run(
                ['bash', str(script_path)],
                env=env,
                cwd=MAGENTLESS_DIR,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                print(f"  ✗ 执行失败 (返回码: {result.returncode})")
                if result.stderr:
                    print(f"  stderr (前500字符):\n{result.stderr[:500]}")
                if result.stdout:
                    print(f"  stdout (前500字符):\n{result.stdout[:500]}")
                return False
            
            # 检查输出文件
            expected_path = results_dir / expected_file
            if expected_path.exists():
                file_size = expected_path.stat().st_size
                if file_size > 0:
                    # 检查是否包含目标 instance
                    with open(expected_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        has_target = any(original_inst_id in line for line in lines)
                    print(f"  ✓ 成功: {expected_path} ({file_size} 字节, {len(lines)} 行)")
                    if has_target:
                        print(f"    ✓ 包含目标 instance")
                    else:
                        print(f"    ⚠ 不包含目标 instance（可能使用了不同的 ID 格式）")
                else:
                    print(f"  ⚠ 文件存在但为空: {expected_path}")
            else:
                print(f"  ⚠ 输出文件不存在: {expected_path}")
                # 继续执行，可能某些步骤是可选的
        
        # 检查最终结果
        print(f"\n最终检查:")
        all_preds_file = results_dir / "all_preds.jsonl"
        if all_preds_file.exists() and all_preds_file.stat().st_size > 0:
            print(f"  ✓ all_preds.jsonl 存在且有内容")
            with open(all_preds_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                print(f"    行数: {len(lines)}")
                # 检查是否包含目标 instance
                for line in lines:
                    try:
                        data = json.loads(line)
                        if original_inst_id in data.get('instance_id', ''):
                            print(f"    ✓ 找到目标 instance 的补丁")
                            print(f"      补丁长度: {len(data.get('model_patch', ''))} 字符")
                            return True
                    except:
                        pass
            print(f"    ⚠ 未找到目标 instance 的补丁")
        else:
            print(f"  ✗ all_preds.jsonl 不存在或为空")
        
        return False
        
    finally:
        os.chdir(original_cwd)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="测试单个 instance 的完整流程")
    parser.add_argument(
        'instance_id',
        type=str,
        help='要测试的 instance_id（例如：Multi-SWE-Bench__javascript__maintenance__bugfix__72488b59）'
    )
    parser.add_argument(
        '--csv',
        type=str,
        default='data/Multi.csv',
        help='CSV 文件路径（默认: data/Multi.csv）'
    )
    parser.add_argument(
        '--proxy-url',
        type=str,
        default='http://localhost:18888',
        help='Proxy URL（默认: http://localhost:18888）'
    )
    parser.add_argument(
        '--num-threads',
        type=int,
        default=1,
        help='线程数（默认: 1）'
    )
    parser.add_argument(
        '--num-sets',
        type=int,
        default=1,
        help='集合数（默认: 1）'
    )
    parser.add_argument(
        '--num-samples-per-set',
        type=int,
        default=1,
        help='每个集合的样本数（默认: 1）'
    )
    
    args = parser.parse_args()
    
    success = test_single_instance(
        instance_id=args.instance_id,
        csv_file=args.csv,
        proxy_url=args.proxy_url,
        num_threads=args.num_threads,
        num_sets=args.num_sets,
        num_samples_per_set=args.num_samples_per_set,
    )
    
    if success:
        print("\n" + "=" * 60)
        print("✓ 测试成功！")
        print("=" * 60)
        sys.exit(0)
    else:
        print("\n" + "=" * 60)
        print("✗ 测试失败！")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()

