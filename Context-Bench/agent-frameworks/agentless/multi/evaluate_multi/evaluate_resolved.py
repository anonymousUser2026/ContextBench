#!/usr/bin/env python3
"""
对 traj.json 文件进行评测，在每个 traj.json 中添加 7_resolved 字段

流程：
1. 读取所有 traj.json，提取 6_final_selected_patch
2. 创建 patch 文件和 dataset 文件
3. 运行 multi-swe-bench 的 --mode instance 评测所有实例
4. 使用 ReportBuilder 生成 final_report.json
5. 从 final_report.json 中读取 resolved_ids
6. 更新 traj.json，添加 7_resolved 字段

用法:
    python evaluate_resolved.py
    python evaluate_resolved.py --threads 4
    python evaluate_resolved.py --skip_done
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Set

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
TRAJS_DIR = PROJECT_ROOT / "results" / "Multi" / "trajs"
CSV_FILE = PROJECT_ROOT / "data" / "Multi.csv"
DATASET_DIR = PROJECT_ROOT / "data" / "multi-swe-bench"
WORKDIR = PROJECT_ROOT / "results" / "Multi" / "workdir_eval"
OUTPUT_DIR = PROJECT_ROOT / "results" / "Multi" / "eval_output"
LOG_DIR = PROJECT_ROOT / "results" / "Multi" / "eval_logs"
REPO_DIR = PROJECT_ROOT / "results" / "Multi" / "repos_eval"

LANG_MAP = {
    'javascript': 'js',
    'typescript': 'ts',
    'go': 'go',
    'rust': 'rust',
    'c': 'c',
    'cpp': 'cpp',
    'java': 'java',
}


def load_csv_instances(csv_file: Path) -> Dict[str, dict]:
    """加载 CSV 中的实例信息"""
    instances = {}
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = __import__('csv').DictReader(f)
        for row in reader:
            instances[row['original_inst_id']] = row
    return instances


def find_dataset_file(original_id: str, language: str) -> Optional[Path]:
    """
    查找包含指定实例的数据集文件

    根据 original_id 在所有 dataset 文件中查找，返回包含该实例的文件
    """
    data_lang = LANG_MAP.get(language, language)
    dataset_lang_dir = DATASET_DIR / data_lang

    if not dataset_lang_dir.exists():
        return None

    # 遍历所有 dataset 文件查找 original_id
    for df in sorted(dataset_lang_dir.glob("*_dataset.jsonl")):
        with open(df, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if data.get('instance_id') == original_id:
                        return df
                except json.JSONDecodeError:
                    continue

    return None


def create_dataset_file(original_id: str, language: str, dataset_file: Path) -> Optional[dict]:
    """从原始数据集文件中提取单个实例的信息"""
    data_lang = LANG_MAP.get(language, language)
    dataset_lang_dir = DATASET_DIR / data_lang

    if not dataset_lang_dir.exists():
        return None

    for df in dataset_lang_dir.glob("*_dataset.jsonl"):
        with open(df, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if data.get('instance_id') == original_id:
                        return data
                except json.JSONDecodeError:
                    continue
    return None


def create_patch_file(original_id: str, stage6: str, dataset_info: dict, output_path: str) -> bool:
    """创建 patch 文件"""
    try:
        patch_data = {
            "org": dataset_info['org'],
            "repo": dataset_info['repo'],
            "number": dataset_info['number'],
            "base_commit": dataset_info['base']['sha'] if isinstance(dataset_info.get('base'), dict) else dataset_info.get('base', {}).get('sha', ''),
            "fix_patch": stage6
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(patch_data, ensure_ascii=False) + '\n')

        return True
    except Exception as e:
        print(f"  错误: 创建 patch 文件失败: {e}")
        return False


def check_number_duplicate(original_id: str, language: str) -> bool:
    """检查 original_id 的 number 是否在其他仓库中重复（跨所有语言目录）"""
    data_lang = LANG_MAP.get(language, language)
    dataset_lang_dir = DATASET_DIR / data_lang

    if not dataset_lang_dir.exists():
        return False

    # 在当前语言目录中查找 original_id 及其 number
    target_number = None
    for df in dataset_lang_dir.glob("*_dataset.jsonl"):
        with open(df, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    if data.get('instance_id') == original_id:
                        target_number = data.get('number')
                        break
                except json.JSONDecodeError:
                    continue
        if target_number is not None:
            break

    if target_number is None:
        return False

    # 在所有语言目录中查找具有相同 number 的其他实例
    for lang_dir in DATASET_DIR.iterdir():
        if not lang_dir.is_dir():
            continue
        for dataset_file in lang_dir.glob("*_dataset.jsonl"):
            with open(dataset_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                        # 找到其他具有相同 number 的实例
                        if data.get('number') == target_number and data.get('instance_id') != original_id:
                            org = data.get('org')
                            repo = data.get('repo')
                            print(f"    发现重复: {original_id} (number={target_number}) 与 {org}/{repo} 重复")
                            return True
                    except json.JSONDecodeError:
                        continue

    return False


def run_magentless_evaluation(patch_files: List[str], dataset_files: List[str], workdir: Path, output_dir: Path, max_workers: int) -> bool:
    """
    运行 multi-swe-bench 评测

    使用 --mode evaluation，会自动生成 final_report.json
    """
    try:
        result = subprocess.run(
            [
                sys.executable, "-m", "multi_swe_bench.harness.run_evaluation",
                "--mode", "evaluation",
                "--workdir", str(workdir),
                "--patch_files", *patch_files,
                "--dataset_files", *dataset_files,
                "--output_dir", str(output_dir),
                "--max_workers", str(max_workers),
                "--log_level", "INFO",
                "--log_dir", str(LOG_DIR),
                "--repo_dir", str(REPO_DIR),
            ],
            cwd=PROJECT_ROOT / "evaluate/multi-swe-bench",
            capture_output=True,
            text=True,
            timeout=86400  # 24小时超时
        )

        if result.returncode != 0:
            print(f"评测失败:")
            print(f"stdout: {result.stdout[-1000:]}")
            print(f"stderr: {result.stderr[-1000:]}")
            return False

        return True

    except subprocess.TimeoutExpired:
        print("  错误: 评测超时")
        return False
    except Exception as e:
        print(f"  错误: 评测失败: {e}")
        return False


def parse_final_report(output_dir: Path) -> Set[str]:
    """
    从 final_report.json 中读取 resolved_ids

    返回已解决实例的 ID 集合
    """
    final_report_file = output_dir / "final_report.json"

    if not final_report_file.exists():
        print(f"  警告: final_report.json 不存在: {final_report_file}")
        return set()

    try:
        with open(final_report_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        resolved_ids = set(data.get('resolved_ids', []))
        print(f"  找到 {len(resolved_ids)} 个已解决的实例")

        # 打印统计信息
        total = data.get('total_instances', 0)
        resolved = data.get('resolved_instances', 0)
        unresolved = data.get('unresolved_instances', 0)
        error = data.get('error_instances', 0)
        print(f"  总计: {total}, 已解决: {resolved}, 未解决: {unresolved}, 错误: {error}")

        return resolved_ids

    except Exception as e:
        print(f"  错误: 解析 final_report.json 失败: {e}")
        return set()


def update_traj_files(resolved_ids: Set[str]):
    """
    更新 traj.json 文件，添加 7_resolved 字段

    resolved_ids 格式: org/repo:pr-number (如 "facebook/zstd:pr-1532")
    traj.json 使用 original_id: org__repo-number (如 "facebook__zstd-1532")

    需要进行格式转换
    """
    # 创建 ID 格式转换函数
    def to_original_id(report_id: str) -> str:
        """将 'org/repo:pr-number' 转换为 'org__repo-number'"""
        # 移除 ":pr-" 后缀
        if ":pr-" in report_id:
            org_repo = report_id.replace(":pr-", "-")
        else:
            org_repo = report_id
        # 将 "/" 替换为 "__"
        return org_repo.replace("/", "__")

    # 将 resolved_ids 转换为 traj.json 使用的格式
    original_ids_resolved = {to_original_id(rid) for rid in resolved_ids}

    updated_count = 0
    skipped_count = 0
    error_count = 0

    for traj_file in sorted(TRAJS_DIR.glob("*_traj.json")):
        try:
            with open(traj_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            original_id = data.get('original_id', '')

            # 检查是否已有 7_resolved 字段
            if '7_resolved' in data:
                skipped_count += 1
                continue

            # 判断是否 resolved（需要转换格式）
            is_resolved = original_id in original_ids_resolved

            # 添加 7_resolved 字段
            data['7_resolved'] = is_resolved

            with open(traj_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            status = "✓ resolved" if is_resolved else "✗ not resolved"
            print(f"  {original_id}: {status}")
            updated_count += 1

        except Exception as e:
            print(f"  错误: 更新 {traj_file.name} 失败: {e}")
            error_count += 1

    print(f"\n更新完成: {updated_count} 个已更新, {skipped_count} 个已跳过, {error_count} 个错误")


def main():
    parser = argparse.ArgumentParser(
        description="对 traj.json 文件进行评测，添加 7_resolved 字段",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python evaluate_resolved.py
  python evaluate_resolved.py --threads 4
  python evaluate_resolved.py --skip_done
  python evaluate_resolved.py --skip_done --retry_duplicates  # 重试 number 重复的实例
  python evaluate_resolved.py --force_retry_failed  # 强制重试之前失败的实例
        """
    )

    parser.add_argument("--threads", "-t", type=int, default=1, help="并行线程数（默认: 1）")
    parser.add_argument("--skip_done", action="store_true", help="跳过已有 7_resolved 字段的实例")
    parser.add_argument("--retry_duplicates", action="store_true", help="仅重试 number 重复的实例（需要先运行正常实例）")
    parser.add_argument("--force_retry_failed", action="store_true", help="强制重试之前失败（7_resolved=False）的实例（用于修复因换行符等问题导致的失败）")

    args = parser.parse_args()

    print("=" * 60)
    print("traj.json 评测脚本")
    print("=" * 60)

    # 1. 加载 CSV
    print(f"\n[1/5] 读取 CSV 文件: {CSV_FILE}")
    csv_instances = load_csv_instances(CSV_FILE)
    print(f"  加载了 {len(csv_instances)} 个实例")

    # 2. 收集需要评测的 traj 文件
    print(f"\n[2/5] 收集 traj.json 文件")
    traj_files = sorted(TRAJS_DIR.glob("*_traj.json"))
    print(f"  找到 {len(traj_files)} 个 traj.json 文件")

    if args.force_retry_failed:
        # 重试之前失败的实例（7_resolved=False）
        remaining = []
        force_retry_count = 0
        for traj_file in traj_files:
            with open(traj_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if data.get('7_resolved') is False:
                # 移除 7_resolved 字段，重新评测
                del data['7_resolved']
                with open(traj_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                remaining.append(traj_file)
                force_retry_count += 1
            elif '7_resolved' not in data:
                remaining.append(traj_file)
        traj_files = remaining
        print(f"  强制重试 {force_retry_count} 个之前失败的实例，剩余 {len(traj_files)} 个文件")
    elif args.skip_done:
        remaining = []
        for traj_file in traj_files:
            with open(traj_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if '7_resolved' not in data:
                remaining.append(traj_file)
        traj_files = remaining
        print(f"  跳过已有结果的实例，剩余 {len(traj_files)} 个文件")

    if args.retry_duplicates:
        # 只处理 number 重复的实例
        duplicate_files = []
        other_files = []
        for traj_file in traj_files:
            with open(traj_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            original_id = data.get('original_id', '')
            csv_row = csv_instances.get(original_id)
            if csv_row:
                language = csv_row.get('language', 'unknown')
                if check_number_duplicate(original_id, language):
                    duplicate_files.append(traj_file)
                else:
                    other_files.append(traj_file)
        
        if other_files:
            print(f"  警告: 发现 {len(other_files)} 个非重复实例，使用 --skip_done 跳过它们")
        traj_files = duplicate_files
        print(f"  将处理 {len(traj_files)} 个 number 重复的实例")

    if not traj_files:
        print("  没有需要评测的实例")
        return

    # 3. 创建 patch 文件并单独评测每个实例
    print(f"\n[3/5] 创建 patch 文件并单独评测每个实例")
    processed_count = 0
    skipped_count = 0
    error_count = 0

    for traj_file in traj_files:
        try:
            with open(traj_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            original_id = data.get('original_id', '')
            stage6 = data.get('6_final_selected_patch', '').strip()

            # 检查是否已有 7_resolved
            if '7_resolved' in data:
                skipped_count += 1
                continue

            # 检查 stage6 是否为空
            if not stage6:
                print(f"  跳过 {original_id}: stage6 为空")
                error_count += 1
                continue

            # 从 CSV 获取 language
            csv_row = csv_instances.get(original_id)
            if not csv_row:
                print(f"  跳过 {original_id}: CSV 中找不到")
                error_count += 1
                continue

            language = csv_row.get('language', 'unknown')

            # 检查是否 number 重复
            is_duplicate = check_number_duplicate(original_id, language)

            # 创建 dataset 文件
            dataset_info = create_dataset_file(original_id, language, None)
            if not dataset_info:
                print(f"  跳过 {original_id}: 找不到数据集")
                error_count += 1
                continue

            # 创建 patch 文件
            with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as tmp:
                patch_file = tmp.name

            if not create_patch_file(original_id, stage6, dataset_info, patch_file):
                try:
                    os.unlink(patch_file)
                except:
                    pass
                error_count += 1
                continue

            # 方案 C: 所有实例都单独运行评测（故障隔离，进度保存）
            print(f"  {original_id} ({language}): 单独评测")
            dataset_file = find_dataset_file(original_id, language)
            dataset_files_single = [str(dataset_file)] if dataset_file else []
            
            # 确保目录存在
            WORKDIR.mkdir(parents=True, exist_ok=True)
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            REPO_DIR.mkdir(parents=True, exist_ok=True)
            
            # 为每个实例创建独立的工作目录（彻底隔离，避免历史结果累积）
            instance_name = original_id.replace("__", "_")
            instance_workdir = WORKDIR / instance_name
            instance_output_dir = OUTPUT_DIR / instance_name
            instance_workdir.mkdir(parents=True, exist_ok=True)
            instance_output_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"    单独运行: patch_file={patch_file}")
            # 单独运行 (max_workers=1)，使用独立 workdir 和 output 目录
            success = run_magentless_evaluation([patch_file], dataset_files_single, instance_workdir, instance_output_dir, 1)
            print(f"    运行结果: success={success}")
            
            # 清理临时 patch 文件
            try:
                os.unlink(patch_file)
            except:
                pass
            
            # 更新结果（从独立目录读取结果）
            if success:
                resolved_ids = parse_final_report(instance_output_dir)
                
                # 将 resolved_ids 转换为 original_id 格式进行比较
                def to_original_id(report_id: str) -> str:
                    """将 'org/repo:pr-number' 转换为 'org__repo-number'"""
                    if ":pr-" in report_id:
                        org_repo = report_id.replace(":pr-", "-")
                    else:
                        org_repo = report_id
                    return org_repo.replace("/", "__")
                
                original_ids_resolved = {to_original_id(rid) for rid in resolved_ids}
                
                if original_id in original_ids_resolved:
                    data['7_resolved'] = True
                    print(f"    ✓ resolved")
                else:
                    data['7_resolved'] = False
                    print(f"    ✗ not resolved")
            else:
                data['7_resolved'] = False
                print(f"    ✗ 评测失败")
            
            with open(traj_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            processed_count += 1

        except Exception as e:
            print(f"  错误: 处理 {traj_file.name} 失败: {e}")
            error_count += 1

    print(f"  处理完成: {processed_count} 个, 跳过: {skipped_count}, 错误: {error_count}")

    # 方案 C: 所有实例都已单独评测完成，无需批量处理
    print(f"\n" + "=" * 60)
    print("所有实例单独评测完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
