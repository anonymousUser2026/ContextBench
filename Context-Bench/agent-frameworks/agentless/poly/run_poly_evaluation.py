#!/usr/bin/env python3
"""
Poly.csv 全量评测启动脚本
支持多语言并行处理，自动生成临时数据集（只包含 CSV 中未完成的实例）
从 instance_id 中提取语言（格式: SWE-PolyBench__python__...）

用法:
    # 全量运行（所有语言，只处理未完成的实例）
    python run_poly_evaluation.py

    # 单语言运行
    python run_poly_evaluation.py --language typescript

    # 单实例测试
    python run_poly_evaluation.py --language python --instance keras-team__keras-18553

    # 多线程加速（4线程）
    python run_poly_evaluation.py --num_threads 4

    # 强制重新运行所有实例（即使已完成）
    python run_poly_evaluation.py --force_rerun
"""

import os
import sys
import csv
import json
import argparse
import subprocess
import logging
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Set

# 增加 CSV 字段大小限制（test.csv 的 patch 字段可能很大）
csv.field_size_limit(sys.maxsize)


# Poly 数据集支持的语言映射
POLY_LANGUAGE_MAP = {
    'python': 'python',
    'javascript': 'javascript',
    'typescript': 'typescript',
    'java': 'java',
}


def extract_language_from_instance_id(instance_id: str) -> str:
    """
    从 instance_id 中提取语言
    Poly 数据集格式: SWE-PolyBench__python__maintenance__bugfix__8c189fda

    Args:
        instance_id: 实例 ID

    Returns:
        语言名称 (python, javascript, typescript, java)
    """
    parts = instance_id.split('__')
    if len(parts) >= 2:
        lang = parts[1].lower()
        if lang in POLY_LANGUAGE_MAP:
            return lang
    return 'unknown'

# 项目路径
PROJECT_ROOT = Path(__file__).parent.absolute()
MAGENTLESS_DIR = PROJECT_ROOT / "Magentless"
VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
TEMP_DATA_DIR = PROJECT_ROOT / "temp_data"
TRAJS_DIR = PROJECT_ROOT / "results" / "Poly" / "trajs"

# 数据文件路径
POLY_CSV_FILE = PROJECT_ROOT / "data" / "Poly.csv"
TEST_CSV_FILE = PROJECT_ROOT / "data" / "SWE-PolyBench" / "test.csv"

# 配置日志
LOG_FILE = PROJECT_ROOT / "evaluation.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def log_and_flush(msg: str):
    """打印日志并立即刷新缓冲区"""
    print(msg)
    sys.stdout.flush()


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
    加载 CSV 实例，按语言分组

    支持两种数据集格式:
    - Poly.csv: 子集格式，instance_id 格式为 SWE-PolyBench__python__...
    - test.csv: 全集格式，instance_id 格式为 owner__repo-PR-number

    Args:
        csv_file: CSV 文件路径
        language: 可选，按语言过滤
        instance_id: 可选，按 original_inst_id 过滤

    Returns:
        {language: [instances...]} 按语言分组的实例列表
    """
    instances_by_lang = defaultdict(list)
    is_poly_format = csv_file.name == "Poly.csv"

    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            instance_id_val = row.get('instance_id', '')
            original_inst_id = row.get('original_inst_id', '')

            # 从 instance_id 提取语言
            if is_poly_format:
                # Poly 格式: SWE-PolyBench__python__maintenance__bugfix__8c189fda
                lang = extract_language_from_instance_id(instance_id_val)
            else:
                # test.csv 格式: 直接使用 language 字段
                lang = row.get('language', 'unknown').lower()
                if lang not in POLY_LANGUAGE_MAP:
                    lang = extract_language_from_instance_id(instance_id_val)

            # 过滤条件
            if language and lang != language:
                continue
            if instance_id and original_inst_id != instance_id:
                continue

            # 将语言信息添加到 row 中，方便后续使用
            row['_extracted_language'] = lang
            instances_by_lang[lang].append(row)

    return dict(instances_by_lang)


def csv_row_to_jsonl(poly_row: Dict, test_row: Optional[Dict], language: str) -> Optional[Dict]:
    """
    将 Poly.csv 行和 test.csv 行合并转换为 JSONL 格式

    Args:
        poly_row: Poly.csv 的行数据
        test_row: test.csv 的行数据（可能为 None）
        language: 提取的语言

    Returns:
        JSONL 格式的字典，如果转换失败则返回 None
    """
    try:
        # 使用 Poly.csv 的 instance_id 格式
        instance_id = poly_row.get('instance_id', '')
        original_inst_id = poly_row.get('original_inst_id', '')

        # 优先从 test.csv 获取完整数据，回退到 Poly.csv
        repo = test_row.get('repo', '') if test_row else poly_row.get('repo', '')
        base_commit = test_row.get('base_commit', '') if test_row else poly_row.get('commit', '')

        jsonl_data = {
            'instance_id': instance_id,
            'original_id': original_inst_id,
            'repo': repo,
            'base_commit': base_commit,
            # 从 test.csv 获取关键字段
            'patch': test_row.get('patch', '') if test_row else '',
            'problem_statement': test_row.get('problem_statement', '') if test_row else '',
            'test_patch': test_row.get('test_patch', '') if test_row else '',
            'hints_text': test_row.get('hints_text', '') if test_row else '',
            # 其他字段
            'language': language,
            'task_category': test_row.get('task_category', '') if test_row else '',
            'created_at': test_row.get('created_at', '') if test_row else '',
            # Poly.csv 特有的字段
            'gold_context_length': poly_row.get('gold_context_length', ''),
            'patch_files': poly_row.get('patch_files', ''),
            'patch_blocks': poly_row.get('patch_blocks', ''),
            'patch_span': poly_row.get('patch_span', ''),
            'num_agents': poly_row.get('num_agents', ''),
            'status': poly_row.get('status', ''),
        }

        return jsonl_data
    except Exception:
        return None


def generate_temp_dataset(instances_by_lang: Dict[str, List[Dict]],
                          force_rerun: bool = False,
                          use_full_data: bool = False) -> Dict[str, Path]:
    """
    为每种语言生成临时 JSONL 文件，只包含 CSV 中未完成的实例

    核心逻辑：
    1. 从 Poly.csv 读取实例列表（按语言分组）
    2. 加载完整的 test.csv 到内存（按 instance_id 建立索引）
    3. 对于每个 Poly 实例，用 original_inst_id 查找 test.csv 的完整数据
    4. 合并生成完整的 JSONL 文件

    Args:
        instances_by_lang: 按语言分组的实例
        force_rerun: 是否强制重新处理所有实例
        use_full_data: 是否直接使用 test.csv（不推荐）

    Returns:
        {language: temp_jsonl_path}
    """
    temp_dir = TEMP_DATA_DIR / "poly-swe-bench"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # 加载完整的 test.csv 到内存（按 instance_id 建立索引）
    # 这是实现 Poly.csv + test.csv 关联的关键
    test_data_index = {}
    if TEST_CSV_FILE.exists():
        print(f"  加载 test.csv 索引...")
        with open(TEST_CSV_FILE, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                instance_id = row.get('instance_id', '')
                test_data_index[instance_id] = row
        print(f"  test.csv 索引完成: {len(test_data_index)} 条记录")
    else:
        print(f"  警告: test.csv 不存在: {TEST_CSV_FILE}")

    temp_files = {}

    for language, instances in instances_by_lang.items():
        # 统计信息
        total_csv = len(instances)
        skipped_count = 0

        # 获取需要处理的 instance_id 集合
        if force_rerun:
            target_instance_ids: Set[str] = set(inst['original_inst_id'] for inst in instances)
        else:
            target_instance_ids: Set[str] = set()
            for inst in instances:
                inst_id = inst['original_inst_id']
                if not is_instance_completed(inst_id, language):
                    target_instance_ids.add(inst_id)
                else:
                    skipped_count += 1

        print(f"  {language}: CSV {total_csv} 个, 跳过 {skipped_count} 个, 需要处理 {len(target_instance_ids)} 个")

        if not target_instance_ids:
            print(f"  {language}: 所有实例已完成，跳过")
            continue

        # 生成 JSONL 数据
        matched_instances = []
        for inst in instances:
            inst_id = inst['original_inst_id']

            # 关键：从 test.csv 中查找完整数据
            # 关联条件：Poly.csv.original_inst_id = test.csv.instance_id
            test_row = test_data_index.get(inst_id)

            # 生成 JSONL 数据
            jsonl_data = csv_row_to_jsonl(inst, test_row, language)
            if jsonl_data:
                matched_instances.append(jsonl_data)

                # 调试输出：显示数据关联情况
                if test_row:
                    patch_preview = test_row.get('patch', '')[:50] if test_row.get('patch') else '(无)'
                    print(f"    ✓ {inst_id} -> 找到完整数据, patch: {patch_preview}...")
                else:
                    print(f"    ✗ {inst_id} -> 未在 test.csv 中找到数据")

        print(f"  {language}: 转换 {len(matched_instances)} 个实例为 JSONL 格式")

        # 生成临时 JSONL 文件
        temp_file = temp_dir / f"{language}.jsonl"
        if matched_instances:
            lines = [json.dumps(data, ensure_ascii=False) for data in matched_instances]
            temp_file.write_text('\n'.join(lines) + '\n', encoding='utf-8')
            temp_files[language] = temp_file
            print(f"  {language}: 生成临时文件: {temp_file} ({temp_file.stat().st_size} 字节)")
        else:
            print(f"  {language}: 警告: 没有可处理的实例")

    return temp_files


def cleanup():
    """清理环境（保留临时数据文件）"""
    print("\n清理环境...")

    # 清理 MagentLess 结果
    results_dir = MAGENTLESS_DIR / "results"
    if results_dir.exists():
        for item in results_dir.iterdir():
            if item.is_dir() and item.name.startswith("Poly_"):
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

    folder_name = f"Poly_{language}"

    # 设置环境变量（传入临时文件）
    env = setup_environment(language, temp_file)

    # 构建 evaluate_poly.py 的参数
    cmd = [
        str(VENV_PYTHON),
        str(PROJECT_ROOT / "evaluate_poly" / "evaluate_multi.py"),
        "--language", language,
        "--num_threads", str(num_threads),
        "--csv", str(POLY_CSV_FILE),
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
        description="Poly/SWE-PolyBench 全量评测启动脚本（自动跳过已完成的实例）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用 Poly.csv 子集运行（默认）
  python run_poly_evaluation.py

  # 使用 test.csv 全集运行
  python run_poly_evaluation.py --dataset test

  # 只运行 python
  python run_poly_evaluation.py --language python

  # 单实例测试
  python run_poly_evaluation.py --language python --instance keras-team__keras-18553

  # 多线程加速（4线程）
  python run_poly_evaluation.py --num_threads 4

  # 只清理，不运行
  python run_poly_evaluation.py --cleanup

  # 强制重新处理所有实例（即使已完成）
  python run_poly_evaluation.py --force_rerun
        """
    )

    parser.add_argument(
        "--dataset", "-d",
        type=str,
        choices=['poly', 'test'],
        default='poly',
        help="选择数据集: poly=Poly.csv 子集, test=test.csv 全集（默认: poly）"
    )

    parser.add_argument(
        "--language", "-l",
        type=str,
        help="只运行指定语言的实例（如 python, javascript, typescript, java）"
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

    parser.add_argument(
        "--unclean", "-u",
        action="store_true",
        help="跳过 cleanup 步骤，保留 Magentless/results/ 下的中间结果（用于断点续传）"
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

    # 选择数据集文件
    if args.dataset == 'test':
        csv_file = TEST_CSV_FILE
        dataset_name = "test.csv (全集)"
    else:
        csv_file = POLY_CSV_FILE
        dataset_name = "Poly.csv (子集)"

    if not csv_file.exists():
        print(f"错误: 数据文件不存在: {csv_file}")
        sys.exit(1)

    # 加载实例
    print(f"\n读取数据集: {csv_file} ({dataset_name})")
    instances_by_lang = load_instances(csv_file, args.language, args.instance)

    if not instances_by_lang:
        print("错误: 没有找到匹配的实例")
        sys.exit(1)

    # 显示统计信息
    print(f"\n实例统计:")
    for lang, instances in sorted(instances_by_lang.items()):
        print(f"  {lang}: {len(instances)} 个实例")

    # 生成临时数据集（自动跳过已完成的实例）
    print(f"\n生成临时数据集（自动跳过已完成的实例）...")
    use_full_data = (args.dataset == 'test')
    temp_files = generate_temp_dataset(
        instances_by_lang,
        force_rerun=args.force_rerun,
        use_full_data=use_full_data
    )

    if not temp_files:
        print("错误: 无法生成临时数据集（所有实例都已完成或无匹配）")
        print("如果需要重新处理，请使用: python run_poly_evaluation.py --force_rerun")
        sys.exit(0)

    # 清理环境（防止 MagentLess 缓存影响），除非指定 --unclean
    if not args.unclean:
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
