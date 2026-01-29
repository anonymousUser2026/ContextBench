import argparse
import concurrent.futures
import json
import os
from threading import Lock

from datasets import load_dataset
from tqdm import tqdm

from agentless.fl.FL import LLMFL
from agentless.multilang.utils import load_local_json
from agentless.util.preprocess_data import (
    check_contains_valid_loc,
    filter_none_python,
    filter_out_test_files,
    get_full_file_paths_and_classes_and_functions,
    get_repo_structure,
)
from agentless.util.utils import load_existing_instance_ids, load_jsonl, setup_logger

MAX_RETRIES = 5


def localize_irrelevant_instance(
    bug, args, swe_bench_data, existing_instance_ids, write_lock=None
):
    instance_id = bug["instance_id"]
    log_file = os.path.join(
        args.output_folder, "localization_logs", f"{instance_id}.log"
    )
    if args.target_id is not None:
        if args.target_id != bug["instance_id"]:
            return

    logger = setup_logger(log_file)
    logger.info(f"Processing bug {instance_id}")

    if bug["instance_id"] in existing_instance_ids:
        logger.info(f"Skipping existing instance_id: {bug['instance_id']}")
        return

    logger.info(f"================ localize {instance_id} ================")

    try:
        logger.info(f"查找 bench_data for {instance_id}...")
        bench_data = [x for x in swe_bench_data if x["instance_id"] == instance_id][0]
        problem_statement = bench_data["problem_statement"]
        logger.info(f"✓ 找到 bench_data: repo={bench_data.get('repo')}, base_commit={bench_data.get('base_commit')[:20]}...")
    except (IndexError, KeyError) as e:
        logger.error(f"ERROR: Cannot find bench_data for {instance_id}: {e}")
        # 写入错误结果 - 确保一定会写入
        if write_lock is not None:
            write_lock.acquire()
        try:
            # 确保输出目录存在 - 使用绝对路径
            output_dir = os.path.dirname(os.path.abspath(args.output_file))
            os.makedirs(output_dir, exist_ok=True)
            # 确保文件可以写入
            with open(args.output_file, "a") as f:
                error_result = {
                    "instance_id": instance_id,
                    "found_files": [],
                    "additional_artifact_loc_file": {"error": f"Cannot find bench_data: {e}"},
                    "file_traj": {},
                }
                f.write(json.dumps(error_result) + "\n")
                f.flush()  # 确保立即写入
            logger.info(f"✓ 写入错误结果到 {args.output_file}")
        except Exception as write_error:
            logger.error(f"ERROR: Failed to write error result: {write_error}")
            import traceback
            logger.error(traceback.format_exc())
            # 即使写入失败，也打印到控制台
            print(f"CRITICAL: Failed to write error result for {instance_id}: {write_error}")
        finally:
            if write_lock is not None:
                write_lock.release()
        return

    try:
        logger.info(f"获取仓库结构 for {instance_id}...")
        logger.info(f"  repo: {bug['repo']}, base_commit: {bug['base_commit']}")
        structure = get_repo_structure(
            instance_id, bug["repo"], bug["base_commit"], "playground"
        )
        logger.info(f"✓ 获取仓库结构成功: type={type(structure)}, len={len(structure) if isinstance(structure, (dict, list)) else 'N/A'}")
    except Exception as e:
        logger.error(f"ERROR: Failed to get repo structure for {instance_id}: {e}")
        import traceback
        error_traceback = traceback.format_exc()
        logger.error(error_traceback)
        # 写入错误结果 - 确保一定会写入
        if write_lock is not None:
            write_lock.acquire()
        try:
            # 确保输出目录存在 - 使用绝对路径
            output_dir = os.path.dirname(os.path.abspath(args.output_file))
            os.makedirs(output_dir, exist_ok=True)
            # 确保文件可以写入
            with open(args.output_file, "a") as f:
                error_result = {
                    "instance_id": instance_id,
                    "found_files": [],
                    "additional_artifact_loc_file": {"error": f"Failed to get repo structure: {e}", "traceback": error_traceback},
                    "file_traj": {},
                }
                f.write(json.dumps(error_result) + "\n")
                f.flush()  # 确保立即写入
            logger.info(f"✓ 写入错误结果到 {args.output_file}")
        except Exception as write_error:
            logger.error(f"ERROR: Failed to write error result: {write_error}")
            logger.error(traceback.format_exc())
            # 即使写入失败，也打印到控制台
            print(f"CRITICAL: Failed to write error result for {instance_id}: {write_error}")
        finally:
            if write_lock is not None:
                write_lock.release()
        return
    
    # 检查 structure 是否为空
    logger.info(f"检查 structure 是否为空...")
    if not structure or len(structure) == 0:
        logger.error(f"ERROR: Repository structure is empty for {instance_id}!")
        logger.error(f"Repo: {bug['repo']}, Commit: {bug['base_commit']}")
        # 即使结构为空，也写入空结果 - 确保一定会写入
        if write_lock is not None:
            write_lock.acquire()
        try:
            # 确保输出目录存在 - 使用绝对路径
            output_dir = os.path.dirname(os.path.abspath(args.output_file))
            os.makedirs(output_dir, exist_ok=True)
            # 确保文件可以写入
            with open(args.output_file, "a") as f:
                error_result = {
                    "instance_id": instance_id,
                    "found_files": [],
                    "additional_artifact_loc_file": {"error": "Empty repository structure"},
                    "file_traj": {},
                }
                f.write(json.dumps(error_result) + "\n")
                f.flush()  # 确保立即写入
            logger.info(f"✓ 写入空结果到 {args.output_file}")
        except Exception as write_error:
            logger.error(f"ERROR: Failed to write empty result: {write_error}")
            import traceback
            logger.error(traceback.format_exc())
            # 即使写入失败，也打印到控制台
            print(f"CRITICAL: Failed to write empty result for {instance_id}: {write_error}")
        finally:
            if write_lock is not None:
                write_lock.release()
        return

    logger.info(f"开始过滤 structure...")
    try:
        filter_none_python(structure)  # some basic filtering steps
        logger.info(f"✓ filter_none_python 完成")
    except Exception as e:
        logger.error(f"ERROR: filter_none_python failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise
    
    try:
        filter_out_test_files(structure)
        logger.info(f"✓ filter_out_test_files 完成")
    except Exception as e:
        logger.error(f"ERROR: filter_out_test_files failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise
    
    # 检查过滤后的结构
    logger.info(f"获取文件路径...")
    try:
        files, _, _ = get_full_file_paths_and_classes_and_functions(structure)
        logger.info(f"✓ Repository structure has {len(files)} files after filtering")
        if len(files) == 0:
            logger.warning(f"WARNING: No files remaining after filtering for {instance_id}!")
    except Exception as e:
        logger.error(f"ERROR: get_full_file_paths_and_classes_and_functions failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

    found_files = []
    found_related_locs = {}
    found_edit_locs = {}

    additional_artifact_loc_file = None
    additional_artifact_loc_related = None
    additional_artifact_loc_edit_location = None
    file_traj, related_loc_traj, edit_loc_traj = {}, {}, {}

    # file level localization
    logger.info(f"开始 file level localization...")
    try:
        if args.file_level:
            logger.info(f"创建 LLMFL 实例...")
            fl = LLMFL(
                instance_id,
                structure,
                problem_statement,
                args.model,
                args.backend,
                logger,
            )
            logger.info(f"调用 fl.localize_irrelevant...")
            found_files, additional_artifact_loc_file, file_traj = fl.localize_irrelevant(
                mock=args.mock
            )
            logger.info(f"✓ localize_irrelevant 完成: found_files={len(found_files)}")
        else:
            raise NotImplementedError
    except Exception as e:
        logger.error(f"ERROR: Failed to localize_irrelevant for {instance_id}: {e}")
        import traceback
        error_traceback = traceback.format_exc()
        logger.error(error_traceback)
        # 写入错误结果
        found_files = []
        additional_artifact_loc_file = {"error": str(e), "traceback": error_traceback}
        file_traj = {}

    logger.info(f"准备写入结果...")
    if write_lock is not None:
        write_lock.acquire()
    try:
        # 确保输出目录存在 - 使用绝对路径
        output_dir = os.path.dirname(os.path.abspath(args.output_file))
        logger.info(f"输出目录: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"✓ 输出目录已创建")
        # 确保文件可以写入
        logger.info(f"写入文件: {args.output_file}")
        with open(args.output_file, "a") as f:
            result = {
                "instance_id": instance_id,
                "found_files": found_files,
                "additional_artifact_loc_file": additional_artifact_loc_file,
                "file_traj": file_traj,
                "found_related_locs": found_related_locs,
                "additional_artifact_loc_related": additional_artifact_loc_related,
                "related_loc_traj": related_loc_traj,
                "found_edit_locs": found_edit_locs,
                "additional_artifact_loc_edit_location": additional_artifact_loc_edit_location,
                "edit_loc_traj": edit_loc_traj,
            }
            f.write(json.dumps(result) + "\n")
            f.flush()  # 确保立即写入
        logger.info(f"✓ 写入结果到 {args.output_file} 成功")
    except Exception as write_error:
        logger.error(f"ERROR: Failed to write result for {instance_id}: {write_error}")
        import traceback
        logger.error(traceback.format_exc())
        # 即使写入失败，也打印到控制台
        print(f"CRITICAL: Failed to write result for {instance_id}: {write_error}")
    finally:
        if write_lock is not None:
            write_lock.release()
    logger.info(f"✓ localize_irrelevant_instance 完成")


def localize_instance(
    bug, args, swe_bench_data, start_file_locs, existing_instance_ids, write_lock=None
):
    instance_id = bug["instance_id"]
    log_file = os.path.join(
        args.output_folder, "localization_logs", f"{instance_id}.log"
    )
    if args.target_id is not None:
        if args.target_id != bug["instance_id"]:
            return

    logger = setup_logger(log_file)
    logger.info(f"Processing bug {instance_id}")

    if instance_id in existing_instance_ids:
        logger.info(f"Skipping existing instance_id: {bug['instance_id']}")
        return

    structure = get_repo_structure(
        instance_id, bug["repo"], bug["base_commit"], "playground"
    )

    logger.info(f"================ localize {instance_id} ================")

    bench_data = [x for x in swe_bench_data if x["instance_id"] == instance_id][0]
    problem_statement = bench_data["problem_statement"]

    filter_none_python(structure)  # some basic filtering steps
    filter_out_test_files(structure)

    found_files = []
    found_related_locs = {}
    found_edit_locs = {}
    additional_artifact_loc_file = None
    additional_artifact_loc_related = None
    additional_artifact_loc_edit_location = None
    file_traj, related_loc_trajs, edit_loc_traj = {}, [], {}

    # file level localization
    if args.file_level:
        fl = LLMFL(
            instance_id,
            structure,
            problem_statement,
            args.model,
            args.backend,
            logger,
        )
        found_files, additional_artifact_loc_file, file_traj = fl.localize(
            mock=args.mock
        )
    else:
        # assume start_file is provided
        for locs in start_file_locs:
            if locs["instance_id"] == instance_id:
                found_files = locs["found_files"]
                if "additional_artifact_loc_file" in locs:
                    additional_artifact_loc_file = locs["additional_artifact_loc_file"]
                    file_traj = locs["file_traj"]
                if "found_related_locs" in locs:
                    found_related_locs = locs["found_related_locs"]
                    additional_artifact_loc_related = locs[
                        "additional_artifact_loc_related"
                    ]
                    related_loc_trajs = locs["related_loc_traj"]
                break

        if len(found_files) == 0:
            return

    # related class, functions, global var localization
    if args.related_level:
        if len(found_files) != 0:
            trying_temp = 0  # related always try with temp 0
            related_loc_trajs = []
            for _ in range(MAX_RETRIES):
                pred_files = found_files[: args.top_n]
                fl = LLMFL(
                    instance_id,
                    structure,
                    problem_statement,
                    args.model,
                    args.backend,
                    logger,
                )
                additional_artifact_loc_related = []
                found_related_locs = {}
                related_loc_traj = {}
                if args.compress and not args.related_level_separate_file:
                    (
                        found_related_locs,
                        additional_artifact_loc_related,
                        related_loc_traj,
                    ) = fl.localize_function_from_compressed_files(
                        pred_files,
                        mock=args.mock,
                        temperature=trying_temp,
                        keep_old_order=args.keep_old_order,
                        compress_assign=args.compress_assign,
                        total_lines=args.compress_assign_total_lines,
                        prefix_lines=args.compress_assign_prefix_lines,
                        suffix_lines=args.compress_assign_suffix_lines,
                    )
                    additional_artifact_loc_related = [additional_artifact_loc_related]
                    related_loc_trajs.append(related_loc_traj)

                    if check_contains_valid_loc(
                        found_related_locs, structure=structure
                    ):
                        break

                    logger.info(
                        f"No valid related locations found ... retrying with higher temperature ..."
                    )
                    trying_temp = 1.0  # set trying temp to 1 to get valid locs

                elif args.compress and args.related_level_separate_file:
                    additional_artifact_loc_related = []
                    found_related_locs = {fn: [] for fn in pred_files}
                    related_loc_traj = []
                    for i, pred_file in enumerate(pred_files):
                        (
                            found_related_locs_i,
                            additional_artifact_loc_related_i,
                            related_loc_traj_i,
                        ) = fl.localize_function_from_compressed_files(
                            [pred_file],
                            mock=args.mock,
                            temperature=trying_temp,
                            compress_assign=args.compress_assign,
                            total_lines=args.compress_assign_total_lines,
                            prefix_lines=args.compress_assign_prefix_lines,
                            suffix_lines=args.compress_assign_suffix_lines,
                        )
                        found_related_locs[pred_file] = found_related_locs_i[pred_file]
                        additional_artifact_loc_related.append(
                            additional_artifact_loc_related_i
                        )
                        related_loc_traj.append(related_loc_traj_i)
                    related_loc_trajs.append(related_loc_traj)

                    if check_contains_valid_loc(
                        found_related_locs, structure=structure
                    ):
                        break
                    trying_temp = 1.0  # set trying temp to 1 to get valid locs
                else:
                    # directly use raw code file instead of skeleton format as ablation
                    (
                        found_related_locs,
                        additional_artifact_loc_related,
                        related_loc_traj,
                    ) = fl.localize_function_from_raw_text(
                        pred_files,
                        mock=args.mock,
                        temperature=trying_temp,
                        keep_old_order=args.keep_old_order,
                    )
                    additional_artifact_loc_related = [additional_artifact_loc_related]
                    related_loc_trajs.append(related_loc_traj)

                    if check_contains_valid_loc(
                        found_related_locs, structure=structure
                    ):
                        break

                    logger.info(
                        f"No valid related locations found ... retrying with higher temperature ..."
                    )
                    trying_temp = 1.0  # set trying temp to 1 to get valid locs

    if args.fine_grain_line_level:
        if len(found_files) != 0:
            # Only supports the following args for now
            trying_temp = args.temperature
            for _ in range(MAX_RETRIES):
                pred_files = found_files[: args.top_n]
                fl = LLMFL(
                    instance_id,
                    structure,
                    problem_statement,
                    args.model,
                    args.backend,
                    logger,
                )
                if not args.direct_edit_loc:
                    coarse_found_locs = found_related_locs
                    (
                        found_edit_locs,
                        additional_artifact_loc_edit_location,
                        edit_loc_traj,
                    ) = fl.localize_line_from_coarse_function_locs(
                        pred_files,
                        coarse_found_locs,
                        context_window=args.context_window,
                        add_space=args.add_space,
                        no_line_number=args.no_line_number,
                        sticky_scroll=args.sticky_scroll,
                        mock=args.mock,
                        temperature=trying_temp,
                        num_samples=args.num_samples,
                        keep_old_order=args.keep_old_order,
                    )
                    additional_artifact_loc_edit_location = [
                        additional_artifact_loc_edit_location
                    ]

                    sample_valid = False
                    if args.num_samples > 1:
                        for found_edit_loc in found_edit_locs:
                            if check_contains_valid_loc(
                                found_edit_loc, structure=structure
                            ):
                                # at least one set of location contains valid edit locs is okay
                                sample_valid = True
                                break
                    else:
                        if check_contains_valid_loc(
                            found_edit_locs, structure=structure
                        ):
                            break

                    if sample_valid:
                        break

                    logger.info(
                        f"No valid edit locations found ... retrying with higher temperature ..."
                    )
                    trying_temp = 1.0  # set trying temp to 1 to get valid locs
                else:
                    # directly use the raw source code to get to edit location, used as ablation setting
                    (
                        found_edit_locs,
                        additional_artifact_loc_edit_location,
                        edit_loc_traj,
                    ) = fl.localize_line_from_raw_text(
                        pred_files,
                        mock=args.mock,
                        temperature=trying_temp,
                        num_samples=args.num_samples,
                        keep_old_order=args.keep_old_order,
                    )
                    additional_artifact_loc_edit_location = [
                        additional_artifact_loc_edit_location
                    ]

                    if check_contains_valid_loc(
                        found_related_locs, structure=structure
                    ):
                        break

                    sample_valid = False
                    if args.num_samples > 1:
                        for found_edit_loc in found_edit_locs:
                            if check_contains_valid_loc(
                                found_edit_loc, structure=structure
                            ):
                                # at least one set of location contains valid edit locs is okay
                                sample_valid = True
                                break
                    else:
                        if check_contains_valid_loc(
                            found_edit_locs, structure=structure
                        ):
                            break

                    if sample_valid:
                        break

                    logger.info(
                        f"No valid edit locations found ... retrying with higher temperature ..."
                    )
                    trying_temp = 1.0  # set trying temp to 1 to get valid locs

    if write_lock is not None:
        write_lock.acquire()
    with open(args.output_file, "a") as f:
        f.write(
            json.dumps(
                {
                    "instance_id": instance_id,
                    "found_files": found_files,
                    "additional_artifact_loc_file": additional_artifact_loc_file,
                    "file_traj": file_traj,
                    "found_related_locs": found_related_locs,
                    "additional_artifact_loc_related": additional_artifact_loc_related,
                    "related_loc_traj": related_loc_trajs,
                    "found_edit_locs": found_edit_locs,
                    "additional_artifact_loc_edit_location": additional_artifact_loc_edit_location,
                    "edit_loc_traj": edit_loc_traj,
                }
            )
            + "\n"
        )
    if write_lock is not None:
        write_lock.release()


def localize_irrelevant(args):
    if args.dataset == 'local_json':
        swe_bench_data = load_local_json()
    else:
        swe_bench_data = load_dataset(args.dataset, split=args.split)
    existing_instance_ids = (
        load_existing_instance_ids(args.output_file) if args.skip_existing else set()
    )
    if args.num_threads == 1:
        for bug in tqdm(swe_bench_data, colour="MAGENTA"):
            try:
                localize_irrelevant_instance(
                    bug, args, swe_bench_data, existing_instance_ids
                )
            except Exception as e:
                # 记录异常但不中断整个流程
                import traceback
                print(f"ERROR: Exception in localize_irrelevant_instance for {bug.get('instance_id', 'unknown')}: {e}")
                traceback.print_exc()
    else:
        write_lock = Lock()
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.num_threads
        ) as executor:
            futures = [
                executor.submit(
                    localize_irrelevant_instance,
                    bug,
                    args,
                    swe_bench_data,
                    existing_instance_ids,
                    write_lock,
                )
                for bug in swe_bench_data
            ]
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(swe_bench_data),
                colour="MAGENTA",
            ):
                try:
                    future.result()
                except Exception as e:
                    # 记录异常但不中断整个流程
                    import traceback
                    print(f"ERROR: Exception in localize_irrelevant_instance: {e}")
                    traceback.print_exc()


def localize(args):
    if args.dataset == 'local_json':
        swe_bench_data = load_local_json()
    else:
        swe_bench_data = load_dataset(args.dataset, split=args.split)
    start_file_locs = load_jsonl(args.start_file) if args.start_file else None
    existing_instance_ids = (
        load_existing_instance_ids(args.output_file) if args.skip_existing else set()
    )

    if args.num_threads == 1:
        for bug in tqdm(swe_bench_data, colour="MAGENTA"):
            try:
                localize_instance(
                    bug, args, swe_bench_data, start_file_locs, existing_instance_ids
                )
            except Exception as e:
                # 记录异常但不中断整个流程
                import traceback
                print(f"ERROR: Exception in localize_instance for {bug.get('instance_id', 'unknown')}: {e}")
                traceback.print_exc()
    else:
        write_lock = Lock()
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.num_threads
        ) as executor:
            futures = [
                executor.submit(
                    localize_instance,
                    bug,
                    args,
                    swe_bench_data,
                    start_file_locs,
                    existing_instance_ids,
                    write_lock,
                )
                for bug in swe_bench_data
            ]
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(swe_bench_data),
                colour="MAGENTA",
            ):
                try:
                    future.result()
                except Exception as e:
                    # 记录异常但不中断整个流程
                    import traceback
                    print(f"ERROR: Exception in localize_irrelevant_instance: {e}")
                    traceback.print_exc()


def merge(args):
    """Merge predicted locations."""
    start_file_locs = load_jsonl(args.start_file)

    def merge_locs(sample_found_locs: list[dict]):
        merged_found_locs = {}
        for locs in sample_found_locs:
            for fn, file_found_locs in locs.items():
                # 过滤空值
                if isinstance(file_found_locs, str):
                    if not file_found_locs.strip():
                        # 如果是空字符串，使用占位符
                        merged_found_locs.setdefault(fn, [""])
                        continue
                    merged_found_locs.setdefault(fn, [""])[0] += "\n" + file_found_locs
                elif isinstance(file_found_locs, list):
                    # 过滤空字符串，只保留非空内容
                    non_empty_locs = [loc for loc in file_found_locs if loc and loc.strip()]
                    if not non_empty_locs:
                        # 如果所有 locs 都为空，使用占位符
                        merged_found_locs.setdefault(fn, [""])
                    else:
                        merged_found_locs.setdefault(fn, [""])[0] += "\n" + "\n".join(non_empty_locs)
        return merged_found_locs
    # Dump each location sample.
    for st_id in range(args.num_samples):
        en_id = st_id
        merged_locs = []
        for locs in start_file_locs:
            merged_found_locs = {}
            if "found_edit_locs" in locs:
                edit_locs = locs["found_edit_locs"]
                # 处理不同的格式：可能是列表或字典
                if isinstance(edit_locs, list) and len(edit_locs) > 0:
                    # 如果是列表，取第 st_id 个元素
                    if st_id < len(edit_locs):
                        merged_found_locs = merge_locs([edit_locs[st_id]])
                elif isinstance(edit_locs, dict) and len(edit_locs) > 0:
                    # 如果是字典，直接使用（已经是合并后的格式）
                    merged_found_locs = edit_locs
            merged_locs.append({**locs, "found_edit_locs": merged_found_locs})
        with open(
            f"{args.output_folder}/loc_merged_{st_id}-{en_id}_outputs.jsonl", "w"
        ) as f:
            for data in merged_locs:
                f.write(json.dumps(data) + "\n")


def check_valid_args(args):
    assert (
        not os.path.exists(args.output_file) or args.skip_existing
    ), "Output file already exists and not set to skip existing localizations"

    assert not (
        args.file_level and args.start_file
    ), "Cannot use both file_level and start_file"

    assert not (
        args.file_level and args.fine_grain_line_level and not args.related_level
    ), "Cannot use both file_level and fine_grain_line_level without related_level"

    assert not (
        (not args.file_level) and (not args.start_file)
    ), "Must use either file_level or start_file"

    assert (not "deepseek" in args.model) or (
        args.backend == "deepseek"
    ), "Must specify `--backend deepseek` if using a DeepSeek model"


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="loc_outputs.jsonl")
    parser.add_argument(
        "--start_file",
        type=str,
        help="""previous output file to start with to reduce
        the work, should use in combination without --file_level""",
    )
    parser.add_argument("--file_level", action="store_true")
    parser.add_argument("--related_level", action="store_true")
    parser.add_argument("--fine_grain_line_level", action="store_true")
    parser.add_argument("--top_n", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--compress", action="store_true")
    parser.add_argument("--compress_assign", action="store_true")
    parser.add_argument("--compress_assign_total_lines", type=int, default=30)
    parser.add_argument("--compress_assign_prefix_lines", type=int, default=10)
    parser.add_argument("--compress_assign_suffix_lines", type=int, default=10)
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--add_space", action="store_true")
    parser.add_argument("--no_line_number", action="store_true")
    parser.add_argument("--sticky_scroll", action="store_true")
    parser.add_argument("--related_level_separate_file", action="store_true")
    parser.add_argument("--context_window", type=int, default=10)
    parser.add_argument("--keep_old_order", action="store_true")
    parser.add_argument("--irrelevant", action="store_true")
    parser.add_argument("--direct_edit_loc", action="store_true")
    parser.add_argument(
        "--num_threads",
        type=int,
        default=1,
        help="Number of threads to use for creating API requests",
    )
    parser.add_argument("--target_id", type=str)
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip localization of instance id's which already contain a localization in the output file.",
    )
    parser.add_argument(
        "--mock", action="store_true", help="Mock run to compute prompt tokens."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-2024-05-13",
        choices=[
            "gpt-4o-2024-05-13",
            "deepseek-coder",
            "gpt-4o-mini-2024-07-18",
            "claude-3-5-sonnet-20241022",
        ],
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="openai",
        choices=["openai", "deepseek", "anthropic"],
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="princeton-nlp/SWE-bench_Lite",
        choices=["princeton-nlp/SWE-bench_Lite", "princeton-nlp/SWE-bench_Verified", "Daoguang/Multi-SWE-bench", "local_json"],
        help="Current supported dataset for evaluation",
    )

    args = parser.parse_args()
    # 如果 output_file 不是绝对路径且不包含 output_folder，才拼接
    if not os.path.isabs(args.output_file) and args.output_folder not in args.output_file:
        args.output_file = os.path.join(args.output_folder, args.output_file)
    check_valid_args(args)

    os.makedirs(os.path.join(args.output_folder, "localization_logs"), exist_ok=True)
    os.makedirs(args.output_folder, exist_ok=True)

    # write the arguments
    with open(f"{args.output_folder}/args.json", "w") as f:
        json.dump(vars(args), f, indent=4)

    if args.merge:
        merge(args)
    elif args.irrelevant:
        localize_irrelevant(args)
    else:
        localize(args)


if __name__ == "__main__":
    main()
