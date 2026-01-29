#!/usr/bin/env python3
"""
从 MagentLess 的输出生成 traj.json 文件
traj.json 是最终结果存储格式，包含完整的六个阶段信息
文件名格式：{instance_id}_traj.json（例如：Multi-SWE-Bench__javascript__maintenance__bugfix__xxx_traj.json）
"""

import json
import os
import logging
from pathlib import Path
from typing import Dict, Optional, List

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 错误日志文件
ERROR_LOG_FILE = Path(__file__).parent.parent.parent / "results" / "Multi" / "logs" / "error_analysis.log"


def load_jsonl_file(jsonl_path: str) -> List[Dict]:
    """加载 JSONL 文件"""
    if not os.path.exists(jsonl_path):
        return []
    
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f if line.strip()]
    except Exception as e:
        print(f"警告: 无法读取 {jsonl_path}: {e}")
        return []


def find_instance_data(data_list: List[Dict], instance_id: str) -> Optional[Dict]:
    """在数据列表中查找指定 instance_id 的数据"""
    for data in data_list:
        if data.get('instance_id') == instance_id:
            return data
    return None


def load_all_preds(all_preds_file: str) -> Dict[str, str]:
    """
    从 all_preds.jsonl 加载所有补丁（第6阶段：最终选中的补丁）
    
    Returns:
        {instance_id: model_patch} 字典
    """
    patches = {}
    if not os.path.exists(all_preds_file):
        return patches
    
    try:
        with open(all_preds_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    instance_id = data.get('instance_id', '')
                    model_patch = data.get('model_patch', '')
                    if instance_id and model_patch:
                        patches[instance_id] = model_patch
    except Exception as e:
        print(f"警告: 无法读取 all_preds.jsonl: {e}")
    
    return patches


def collect_stage_data(results_dir: str, instance_id: str, error_logger: Optional[Dict] = None) -> Dict:
    """
    从 MagentLess 结果目录收集六个阶段的数据
    
    Args:
        results_dir: 结果目录
        instance_id: 实例 ID
        error_logger: 用于记录错误的字典

    Returns:
        包含六个阶段数据的字典
    """
    results_path = Path(results_dir)
    stage_data = {}
    errors = []
    
    # 阶段1: 1_model_selected_files - 从 file_level/loc_outputs.jsonl
    file_level_file = results_path / "file_level" / "loc_outputs.jsonl"
    if file_level_file.exists():
        file_level_data = load_jsonl_file(str(file_level_file))
        instance_data = find_instance_data(file_level_data, instance_id)
        if instance_data:
            stage_data['1_model_selected_files'] = instance_data.get('found_files', [])
        else:
            errors.append("阶段1: loc_outputs.jsonl 中找不到实例数据")
    else:
        errors.append(f"阶段1: 文件不存在 {file_level_file}")
    
    # 阶段2: 2_embedding_selected_files - 从 retrievel_embedding/retrieve_locs.jsonl
    retrieve_file = results_path / "retrievel_embedding" / "retrieve_locs.jsonl"
    if retrieve_file.exists():
        retrieve_data = load_jsonl_file(str(retrieve_file))
        instance_data = find_instance_data(retrieve_data, instance_id)
        if instance_data:
            stage_data['2_embedding_selected_files'] = instance_data.get('found_files', [])
        else:
            errors.append("阶段2: retrieve_locs.jsonl 中找不到实例数据")
    else:
        errors.append(f"阶段2: 文件不存在 {retrieve_file}")
    
    # 阶段3: 3_final_combined_files - 从 file_level_combined/combined_locs.jsonl
    combined_file = results_path / "file_level_combined" / "combined_locs.jsonl"
    if combined_file.exists():
        combined_data = load_jsonl_file(str(combined_file))
        instance_data = find_instance_data(combined_data, instance_id)
        if instance_data:
            stage_data['3_final_combined_files'] = instance_data.get('found_files', [])
        else:
            errors.append("阶段3: combined_locs.jsonl 中找不到实例数据")
    else:
        errors.append(f"阶段3: 文件不存在 {combined_file}")
    
    # 阶段4: 4_related_elements - 从 related_elements/loc_outputs.jsonl
    related_file = results_path / "related_elements" / "loc_outputs.jsonl"
    if related_file.exists():
        related_data = load_jsonl_file(str(related_file))
        instance_data = find_instance_data(related_data, instance_id)
        if instance_data:
            stage_data['4_related_elements'] = instance_data.get('found_related_locs', {})
        else:
            errors.append("阶段4: related_elements/loc_outputs.jsonl 中找不到实例数据")
    else:
        errors.append(f"阶段4: 文件不存在 {related_file}")
    
    # 阶段5: 5_sampled_edit_locs_and_patches
    edit_locs_file = results_path / "edit_location_samples" / "loc_outputs.jsonl"
    edit_locs_data = []
    if edit_locs_file.exists():
        edit_locs_data = load_jsonl_file(str(edit_locs_file))
    else:
        errors.append(f"阶段5: 文件不存在 {edit_locs_file}")
    
    # 从 repair_sample 目录查找补丁
    repair_patches_by_sample = {}
    for repair_dir in results_path.glob("repair_sample_*"):
        for normalized_file in repair_dir.glob("output_*_normalized.jsonl"):
            try:
                patches_data = load_jsonl_file(str(normalized_file))
                for patch_data in patches_data:
                    if patch_data.get('instance_id') == instance_id:
                        file_name = normalized_file.stem
                        parts = file_name.split('_')
                        if len(parts) >= 2 and parts[0] == 'output':
                            try:
                                sample_idx = int(parts[1])
                                if sample_idx not in repair_patches_by_sample:
                                    repair_patches_by_sample[sample_idx] = []
                                patch = patch_data.get('model_patch', '') or patch_data.get('normalized_patch', '')
                                if patch:
                                    repair_patches_by_sample[sample_idx].append(patch)
                            except ValueError:
                                pass
            except Exception as e:
                errors.append(f"阶段5: 读取 {normalized_file} 失败: {e}")
    
    # 合并编辑位置和补丁
    instance_edit_data = find_instance_data(edit_locs_data, instance_id)
    if instance_edit_data and 'found_edit_locs' in instance_edit_data:
        found_edit_locs = instance_edit_data['found_edit_locs']
        sampled_data = []
        for sample_idx, edit_loc_sample in enumerate(found_edit_locs):
            sample_data = {
                'sample_index': sample_idx,
                'edit_locs': edit_loc_sample,
                'patches': repair_patches_by_sample.get(sample_idx, []),
            }
            sampled_data.append(sample_data)
        if sampled_data:
            stage_data['5_sampled_edit_locs_and_patches'] = sampled_data
    elif edit_locs_file.exists():
        errors.append("阶段5: edit_location_samples/loc_outputs.jsonl 中找不到实例数据")

    # 阶段6: 6_final_selected_patch - 从 all_preds.jsonl
    all_preds_file = results_path / "all_preds.jsonl"
    if all_preds_file.exists():
        if instance_id in load_all_preds(str(all_preds_file)):
            stage_data['6_final_selected_patch'] = load_all_preds(str(all_preds_file))[instance_id]
        else:
            errors.append("阶段6: all_preds.jsonl 中找不到补丁")
    else:
        errors.append(f"阶段6: 文件不存在 {all_preds_file}")

    # 记录错误
    if errors and error_logger is not None:
        error_logger[instance_id] = errors
    
    return stage_data


def generate_traj_json(
    instance_id: str,
    original_inst_id: str,
    all_preds: Dict[str, str],
    results_dir: str,
    output_file: str,
    error_logger: Optional[Dict] = None
) -> bool:
    """
    生成 traj.json 文件，包含完整的六个阶段信息
    
    Args:
        instance_id: 实例 ID（完整格式，如 SWE-PolyBench__python__maintenance__bugfix__xxx）
        original_inst_id: 原始实例 ID（原始格式，如 keras-team__keras-18553）
        all_preds: 从 all_preds.jsonl 加载的补丁字典（第6阶段）
        results_dir: MagentLess 结果目录
        output_file: 输出的 traj.json 文件路径
        error_logger: 用于记录错误的字典
        
    Returns:
        是否成功生成
    """
    # 【关键修复】使用 instance_id（PolyBench 格式）查找数据
    # MagentLess 输出文件中的 instance_id 是 PolyBench 格式
    # all_preds 的 key 也是 PolyBench 格式
    final_patch = all_preds.get(instance_id, '')
    
    # 收集前五个阶段的数据（使用 instance_id 去 MagentLess 结果目录中查找）
    stage_data = collect_stage_data(results_dir, instance_id, error_logger)
    
    # 如果没有任何数据可保存，直接返回
    if not final_patch and not stage_data:
        print(f"警告: 找不到 {instance_id} (原始ID: {original_inst_id}) 的任何阶段数据")
        return False
    
    # 构建 traj.json 结构
    traj_json = {
        "instance_id": instance_id,
        "original_id": original_inst_id,
    }
    
    # 添加各个阶段的数据
    if '1_model_selected_files' in stage_data:
        traj_json["1_model_selected_files"] = stage_data['1_model_selected_files']
    
    if '2_embedding_selected_files' in stage_data:
        traj_json["2_embedding_selected_files"] = stage_data['2_embedding_selected_files']
    
    if '3_final_combined_files' in stage_data:
        traj_json["3_final_combined_files"] = stage_data['3_final_combined_files']
    
    if '4_related_elements' in stage_data:
        traj_json["4_related_elements"] = stage_data['4_related_elements']
    
    if '5_sampled_edit_locs_and_patches' in stage_data:
        traj_json["5_sampled_edit_locs_and_patches"] = stage_data['5_sampled_edit_locs_and_patches']
    
    # 关键：添加最终选中的补丁（第6阶段）
    traj_json["6_final_selected_patch"] = final_patch
    
    # 保存到文件
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(traj_json, f, indent=4, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"错误: 无法写入 traj.json 文件: {e}")
        return False


def generate_all_traj_files(
    csv_file: str,
    magentless_results_dir: str,
    output_trajs_dir: str
) -> int:
    """
    为所有实例生成 traj.json 文件
    
    Args:
        csv_file: Multi.csv 文件路径
        magentless_results_dir: MagentLess 结果目录
        output_trajs_dir: 输出的 trajs 目录
        
    Returns:
        成功生成的 traj.json 文件数量
    """
    import csv
    
    # 读取 CSV（用于获取原始ID等信息）
    instances = []
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            instances.append(row)
    
    # 加载 all_preds.jsonl（第6阶段）
    all_preds_file = Path(magentless_results_dir) / "all_preds.jsonl"
    all_preds = load_all_preds(str(all_preds_file))
    print(f"从 {all_preds_file} 加载了 {len(all_preds)} 个补丁（第6阶段）")
    
    # 【关键修改】只处理 all_preds.jsonl 中存在的实例
    # MagentLess 可能只处理了部分实例（取决于临时数据文件）
    # 我们需要根据 instance_id 建立映射
    instance_id_to_row = {inst.get('instance_id', ''): inst for inst in instances}
    processed_instances = []
    
    for instance_id in all_preds.keys():
        if instance_id in instance_id_to_row:
            processed_instances.append(instance_id_to_row[instance_id])
        else:
            # 尝试从 original_inst_id 查找（如果 CSV 中的 instance_id 不同）
            print(f"警告: 在 CSV 中找不到 instance_id={instance_id} 的实例数据")
    
    print(f"CSV 中共有 {len(instances)} 个实例，将为其中的 {len(processed_instances)} 个生成 traj.json")
    
    # 错误日志收集器
    error_logger: Dict[str, List[str]] = {}
    
    # 为每个已处理的实例生成 traj.json
    generated_count = 0
    failed_count = 0
    
    for instance in processed_instances:
        instance_id = instance.get('instance_id', '')
        original_inst_id = instance.get('original_inst_id', '')
        
        if not instance_id:
            continue
        
        # 生成 traj.json
        output_file = Path(output_trajs_dir) / f"{instance_id}_traj.json"
        
        success = generate_traj_json(
            instance_id=instance_id,
            original_inst_id=original_inst_id,
            all_preds=all_preds,
            results_dir=magentless_results_dir,
            output_file=str(output_file),
            error_logger=error_logger
        )
        
        if success:
            generated_count += 1
            print(f"✓ 生成: {output_file.name}")
        else:
            failed_count += 1
            print(f"✗ 失败: {instance_id}")
    
    print(f"\n生成完成: 成功 {generated_count} 个, 失败 {failed_count} 个")

    # 生成错误汇总报告
    if error_logger:
        error_report = {
            "total_failed": len(error_logger),
            "errors_by_instance": error_logger,
            "errors_by_stage": {},
            "errors_by_reason": {}
        }

        # 按阶段和原因统计错误
        for inst_id, errors in error_logger.items():
            for error in errors:
                # 提取阶段号
                stage_match = error.split(":")[0] if ":" in error else "未知"
                # 提取原因
                reason = error.split(":")[-1].strip() if ":" in error else error

                if stage_match not in error_report["errors_by_stage"]:
                    error_report["errors_by_stage"][stage_match] = []
                error_report["errors_by_stage"][stage_match].append(inst_id)

                if reason not in error_report["errors_by_reason"]:
                    error_report["errors_by_reason"][reason] = []
                error_report["errors_by_reason"][reason].append(inst_id)

        # 保存错误报告
        error_report_file = Path(output_trajs_dir).parent / "error_report.json"
        with open(error_report_file, 'w', encoding='utf-8') as f:
            json.dump(error_report, f, indent=2, ensure_ascii=False)
        print(f"\n错误报告已保存到: {error_report_file}")

    return generated_count


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="从 MagentLess 输出生成 traj.json 文件")
    parser.add_argument("--csv", type=str, required=True, help="Multi.csv 文件路径")
    parser.add_argument("--results_dir", type=str, required=True, help="MagentLess 结果目录")
    parser.add_argument("--output_dir", type=str, required=True, help="输出的 trajs 目录")
    
    args = parser.parse_args()
    
    generate_all_traj_files(
        args.csv,
        args.results_dir,
        args.output_dir
    )
