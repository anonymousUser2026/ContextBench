#!/usr/bin/env python3
"""
Multi.csv 全量评测启动脚本
支持多语言并行处理，自动生成临时数据集（只包含 CSV 中未完成的实例）

用法:
    # 全量运行（所有语言，只处理未完成的实例）
    python run_full_evaluation.py

    # 单语言运行
    python run_full_evaluation.py --language javascript

    # 单实例测试
    python run_full_evaluation.py --language javascript --instance iamkun__dayjs-734

    # 多线程加速（4线程）
    python run_full_evaluation.py --num_threads 4

    # 强制重新运行所有实例（即使已完成）
    python run_full_evaluation.py --force_rerun
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
MAGENTLESS_DIR = PROJECT_ROOT / "MagentLess"
VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
CSV_FILE = PROJECT_ROOT / "data" / "Pro.csv"
HF_DATA_FILE = PROJECT_ROOT / "data" / "SWE-bench_Pro" / "test.csv"
TEMP_DATA_DIR = PROJECT_ROOT / "temp_data"
TRAJS_DIR = PROJECT_ROOT / "results" / "Pro" / "trajs"


def is_instance_completed(original_inst_id: str, language: str) -> bool:
    """
    检查实例是否已有完整的 traj.json

    框架限制（不重新运行）：
    - 阶段1或2缺失（LLM 定位不到文件）
    - 阶段5补丁全为空（LLM Repair 返回空补丁）

    Args:
        original_inst_id: 原始实例 ID（如 "facebook__zstd-1532"）
        language: 语言

    Returns:
        True 表示已有结果（完整或框架限制），跳过；False 表示需要重新处理
    """
    if not TRAJS_DIR.exists():
        return False

    # 查找对应的 traj.json 文件
    for traj_file in TRAJS_DIR.glob(f"*__{language}__*_traj.json"):
        try:
            with open(traj_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 比较 original_id（不是 instance_id）
            traj_original_id = data.get('original_id', '')
            if traj_original_id != original_inst_id:
                continue

            # 提取各阶段数据
            stage1 = data.get('1_model_selected_files', [])
            stage2 = data.get('2_embedding_selected_files', [])
            stage3 = data.get('3_final_combined_files', [])
            stage4 = data.get('4_related_elements', {})
            stage5 = data.get('5_sampled_edit_locs_and_patches', [])
            stage6 = data.get('6_final_selected_patch', '').strip()

            # 框架限制1：阶段1或2缺失 = LLM 定位不到文件 = 保留
            if not stage1 or not stage2:
                return True

            # 框架限制2：阶段5补丁全为空 = LLM Repair 返回空 = 保留
            stage5_has_content = any(
                p.get('model_patch', '').strip()
                for p in stage5
            ) if stage5 else False
            if not stage5_has_content:
                return True

            # 完整结果：阶段3-6 都必须有数据
            if stage3 and stage4 and stage5 and stage6:
                return True

            return False

        except (json.JSONDecodeError, Exception):
            continue

    return False


def cleanup_incomplete_trajs():
    """
    删除不完整的 traj.json，只保留符合 is_instance_completed() 标准的文件

    框架限制（保留）：
    - 阶段1或2缺失（LLM 定位不到文件）
    - 阶段5补丁全为空（LLM Repair 返回空补丁）
    """
    if not TRAJS_DIR.exists():
        return

    deleted_count = 0
    for traj_file in TRAJS_DIR.glob("*_traj.json"):
        try:
            with open(traj_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 提取各阶段数据
            stage1 = data.get('1_model_selected_files', [])
            stage2 = data.get('2_embedding_selected_files', [])
            stage3 = data.get('3_final_combined_files', [])
            stage4 = data.get('4_related_elements', {})
            stage5 = data.get('5_sampled_edit_locs_and_patches', [])
            stage6 = data.get('6_final_selected_patch', '').strip()

            # 框架限制1：阶段1或2缺失 = 保留
            if not stage1 or not stage2:
                continue

            # 框架限制2：阶段5补丁全为空 = LLM 返回空 = 保留
            stage5_has_content = any(
                p.get('model_patch', '').strip()
                for p in stage5
            ) if stage5 else False
            if not stage5_has_content:
                print(f"  保留(LLM空补丁): {traj_file.name}")
                continue

            # 完整结果：阶段3-6 都必须有数据才保留
            if not (stage3 and stage4 and stage5 and stage6):
                print(f"  删除不完整: {traj_file.name}")
                traj_file.unlink()
                deleted_count += 1

        except (json.JSONDecodeError, Exception):
            # 无法解析的文件也删除
            print(f"  删除损坏: {traj_file.name}")
            traj_file.unlink()
            deleted_count += 1

    if deleted_count > 0:
        print(f"  共删除 {deleted_count} 个不完整的 traj.json")


def load_instances(csv_file: Path, language: Optional[str] = None, instance_id: Optional[str] = None) -> Dict[str, List[Dict]]:
    """
    加载 CSV 实例，按语言分组。
    针对 Pro.csv，需要从 HF_DATA_FILE (test.csv) 中补充完整信息。

    Returns:
        {language: [instances...]} 按语言分组的实例列表
    """
    instances_by_lang = defaultdict(list)
    
    # 1. 加载 Pro.csv 索引
    pro_rows = []
    target_ids = set()
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            orig_id = row.get('original_inst_id', '').strip()
            if not orig_id:
                continue
            
            # 过滤条件 (初步过滤 ID)
            if instance_id and orig_id != instance_id:
                continue
                
            pro_rows.append(row)
            target_ids.add(orig_id)

    if not pro_rows:
        return {}

    # 2. 从 HF_DATA_FILE (test.csv) 加载完整信息 (流式处理以处理超大数据集)
    print(f"  正在从 {HF_DATA_FILE.name} 补充完整数据 (匹配 {len(target_ids)} 个实例)...")
    
    # 提高 CSV 限制
    csv.field_size_limit(sys.maxsize)
    
    hf_data_map = {}
    with open(HF_DATA_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            hf_id = row.get('instance_id', '').strip()
            if hf_id in target_ids:
                hf_data_map[hf_id] = row
                if len(hf_data_map) == len(target_ids):
                    break

    # 3. 合并数据并分组
    for pro_row in pro_rows:
        orig_id = pro_row['original_inst_id'].strip()
        hf_row = hf_data_map.get(orig_id)
        
        if not hf_row:
            print(f"    警告: 在 {HF_DATA_FILE.name} 中找不到实例 {orig_id}")
            continue
            
        # 使用 repo_language 细分
        lang = hf_row.get('repo_language', 'unknown')
        
        # 补全信息
        instance_detail = pro_row.copy()
        instance_detail['language'] = lang  # 覆盖 'mixed'
        instance_detail['repo'] = hf_row.get('repo', '')
        instance_detail['base_commit'] = hf_row.get('base_commit', '').strip()
        instance_detail['problem_statement'] = hf_row.get('problem_statement', '')
        # 显式处理 commit strip
        if 'commit' in instance_detail:
            instance_detail['commit'] = instance_detail['commit'].strip()

        # 过滤语言
        if language and lang != language:
            continue

        instances_by_lang[lang].append(instance_detail)

    return dict(instances_by_lang)


def generate_temp_dataset(instances_by_lang: Dict[str, List[Dict]], force_rerun: bool = False) -> Dict[str, Path]:
    """
    为每种语言生成临时 JSONL 文件，只包含 CSV 中未完成的实例。
    针对 Pro.csv 适配：数据已在 load_instances 中加载。

    Args:
        instances_by_lang: 按语言分组的实例
        force_rerun: 是否强制重新处理所有实例

    Returns:
        {language: temp_jsonl_path}
    """
    temp_dir = TEMP_DATA_DIR / "pro-swe-bench"
    temp_dir.mkdir(parents=True, exist_ok=True)

    language_map = {
        'javascript': 'js',
        'typescript': 'ts',
        'go': 'go',
        'rust': 'rust',
        'c': 'c',
        'cpp': 'cpp',
        'java': 'java',
        'python': 'python',
    }

    temp_files = {}

    for language, instances in instances_by_lang.items():
        # 映射到 data 目录的语言
        data_lang = language_map.get(language, language)

        # 统计信息
        total_csv = len(instances)
        skipped_count = 0

        # 过滤未完成的实例
        matched_instances = []
        for inst in instances:
            orig_id = inst['original_inst_id']
            if not force_rerun and is_instance_completed(orig_id, language):
                skipped_count += 1
                continue
            
            # 构建 MagentLess 期望的 JSON 结构
            json_data = {
                'instance_id': orig_id,
                'repo': inst.get('repo', ''),
                'base_commit': inst.get('base_commit', inst.get('commit', '')),
                'problem_statement': inst.get('problem_statement', ''),
                'repo_language': language
            }
            matched_instances.append(json.dumps(json_data, ensure_ascii=False))

        print(f"  {language}: CSV {total_csv} 个, 跳过 {skipped_count} 个, 需要处理 {len(matched_instances)} 个")

        if not matched_instances:
            print(f"  {language}: 所有实例已完成，跳过")
            continue

        # 生成临时 JSONL 文件
        temp_file = temp_dir / f"{data_lang}.jsonl"
        temp_file.write_text('\n'.join(matched_instances) + '\n', encoding='utf-8')
        temp_files[language] = temp_file
        print(f"  {language}: 生成临时文件: {temp_file} ({temp_file.stat().st_size} 字节)")

    return temp_files


def cleanup():
    """清理环境（保留临时数据文件）"""
    print("\n清理环境...")

    # 清理 MagentLess 结果
    results_dir = MAGENTLESS_DIR / "results"
    if results_dir.exists():
        for item in results_dir.iterdir():
            if item.is_dir() and item.name.startswith("Multi_"):
                print(f"  删除: {item}")
                subprocess.run(["rm", "-rf", str(item)], check=False)

    # 清理 playground
    playground_dir = MAGENTLESS_DIR / "playground"
    if playground_dir.exists():
        for item in playground_dir.iterdir():
            if item.is_dir():
                print(f"  删除: {item}")
                subprocess.run(["rm", "-rf", str(item)], check=False)

    # 注意：保留临时数据文件 temp_data/，因为它们需要用于评测

    print("清理完成\n")


def setup_environment(language: str, temp_file: Optional[Path] = None):
    """设置环境变量"""
    env = os.environ.copy()

    # 核心环境变量
    env['SWEBENCH_LANG'] = language
    env['PYTHONPATH'] = str(MAGENTLESS_DIR)

    # 如果有临时数据文件，设置其路径
    if temp_file and temp_file.exists():
        env['CUSTOM_DATA_FILE'] = str(temp_file.absolute())
        print(f"  设置自定义数据文件: {temp_file}")
    elif 'CUSTOM_DATA_FILE' in env:
        del env['CUSTOM_DATA_FILE']

    # 虚拟环境路径
    venv_bin = str(PROJECT_ROOT / ".venv" / "bin")
    if venv_bin not in env.get('PATH', ''):
        env['PATH'] = f"{venv_bin}:{env.get('PATH', '')}"

    # 确保代理设置
    env['OPENAI_BASE_URL'] = os.getenv('OPENAI_BASE_URL', 'http://localhost:18888')
    env['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'dummy-key')
    env['OPENAI_EMBED_URL'] = os.getenv('OPENAI_EMBED_URL', 'http://localhost:18888/v1')

    return env


def run_evaluation(language: str, instances: List[Dict], temp_file: Optional[Path] = None, num_threads: int = 1):
    """
    运行指定语言的评测

    Args:
        language: 语言
        instances: 该语言的实例列表
        temp_file: 临时数据文件路径
        num_threads: 线程数
    """
    # 如果没有临时文件，说明该语言的所有实例都已被跳过，直接返回成功
    if temp_file is None or not temp_file.exists():
        print(f"\n{'='*60}")
        print(f"开始处理语言: {language}")
        print(f"实例数: {len(instances)}")
        print(f"所有实例已完成，跳过")
        print(f"{'='*60}\n")
        return True

    folder_name = f"Multi_{language}"

    # 设置环境变量（传入临时文件）
    env = setup_environment(language, temp_file)

    # 构建 evaluate_multi.py 的参数
    cmd = [
        str(VENV_PYTHON),
        str(PROJECT_ROOT / "evaluate_multi" / "evaluate_multi.py"),
        "--language", language,
        "--num_threads", str(num_threads),
    ]

    print(f"\n{'='*60}")
    print(f"开始处理语言: {language}")
    print(f"实例数: {len(instances)}")
    print(f"{'='*60}\n")

    result = subprocess.run(
        cmd,
        env=env,
        cwd=PROJECT_ROOT,
        capture_output=False,
        text=True
    )

    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="Multi.csv 全量评测启动脚本（自动跳过已完成的实例）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 全量运行（自动跳过已有补丁的实例）
  python run_full_evaluation.py

  # 只运行 javascript
  python run_full_evaluation.py --language javascript

  # 单实例测试
  python run_full_evaluation.py --language javascript --instance iamkun__dayjs-734

  # 多线程加速（4线程）
  python run_full_evaluation.py --num_threads 4

  # 只清理，不运行
  python run_full_evaluation.py --cleanup

  # 强制重新处理所有实例（即使已完成）
  python run_full_evaluation.py --force_rerun
        """
    )

    parser.add_argument(
        "--language", "-l",
        type=str,
        help="只运行指定语言的实例（如 javascript, typescript, go, rust, c, cpp, java）"
    )

    parser.add_argument(
        "--instance", "-i",
        type=str,
        help="只运行指定 original_inst_id 的单个实例（需要同时指定 --language）"
    )

    parser.add_argument(
        "--num_threads", "-t",
        type=int,
        default=1,
        help="线程数（默认: 1）"
    )

    parser.add_argument(
        "--cleanup", "-c",
        action="store_true",
        help="只执行清理，不运行评测"
    )

    parser.add_argument(
        "--skip_evaluation", "-s",
        action="store_true",
        help="跳过评测，只生成 traj.json 和补丁转换"
    )

    parser.add_argument(
        "--force_rerun", "-f",
        action="store_true",
        help="强制重新处理所有实例（即使已完成）"
    )

    args = parser.parse_args()

    # 验证参数
    if args.instance and not args.language:
        print("错误: --instance 必须与 --language 一起使用")
        sys.exit(1)

    # 如果只清理
    if args.cleanup:
        cleanup()
        sys.exit(0)

    # 加载实例
    print(f"\n读取 CSV 文件: {CSV_FILE}")
    instances_by_lang = load_instances(CSV_FILE, args.language, args.instance)

    if not instances_by_lang:
        print("错误: 没有找到匹配的实例")
        sys.exit(1)

    # 显示统计信息
    print(f"\n实例统计:")
    for lang, instances in sorted(instances_by_lang.items()):
        print(f"  {lang}: {len(instances)} 个实例")

    # 生成临时数据集（自动跳过已完成的实例）
    print(f"\n生成临时数据集（自动跳过已完成的实例）...")
    temp_files = generate_temp_dataset(instances_by_lang, force_rerun=args.force_rerun)

    if not temp_files:
        print("错误: 无法生成临时数据集（所有实例都已完成或无匹配）")
        print("如果需要重新处理，请使用: python run_full_evaluation.py --force_rerun")
        sys.exit(0)

    # 清理环境（防止 MagentLess 缓存影响）
    cleanup()

    # 删除所有不完整的 traj.json，只保留完美的
    cleanup_incomplete_trajs()

    # 按语言运行评测
    success_count = 0
    fail_count = 0

    for language in sorted(instances_by_lang.keys()):
        instances = instances_by_lang[language]
        temp_file = temp_files.get(language)

        success = run_evaluation(
            language=language,
            instances=instances,
            temp_file=temp_file,
            num_threads=args.num_threads
        )

        if success:
            success_count += 1
            print(f"✓ {language} 处理完成")
        else:
            fail_count += 1
            print(f"✗ {language} 处理失败")

    # 总结
    print(f"\n{'='*60}")
    print(f"评测完成")
    print(f"成功: {success_count} 种语言")
    print(f"失败: {fail_count} 种语言")
    print(f"{'='*60}\n")

    # 返回退出码
    sys.exit(0 if fail_count == 0 else 1)


if __name__ == "__main__":
    main()
