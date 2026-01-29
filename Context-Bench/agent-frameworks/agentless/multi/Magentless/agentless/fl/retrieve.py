import argparse
import concurrent.futures
import json
import os
from threading import Lock

from datasets import load_dataset
from tqdm import tqdm

from agentless.fl.Index import EmbeddingIndex
from agentless.multilang.utils import load_local_json
from agentless.util.preprocess_data import (
    filter_none_python,
    filter_out_test_files,
    get_repo_structure,
)
from agentless.util.utils import load_json, load_jsonl, setup_logger


def retrieve_locs(bug, args, swe_bench_data, found_files, prev_o, write_lock=None):

    instance_id = bug["instance_id"]

    # 提前检查，避免不必要的处理
    if args.target_id is not None:
        if args.target_id != instance_id:
            return None

    found = False
    for o in prev_o:
        if o["instance_id"] == instance_id:
            found = True
            break

    # 创建 logger（需要在检查 found 之前，因为可能需要记录）
    log_file = os.path.join(args.output_folder, "retrieval_logs", f"{instance_id}.log")
    logger = setup_logger(log_file)
    
    if found:
        logger.info(f"skipping {instance_id} since patch already generated")
        return None

    logger.info(f"Processing bug {instance_id}")

    bench_data = [x for x in swe_bench_data if x["instance_id"] == instance_id][0]
    problem_statement = bench_data["problem_statement"]
    structure = get_repo_structure(
        instance_id, bug["repo"], bug["base_commit"], "playground"
    )

    filter_none_python(structure)
    filter_out_test_files(structure)

    if args.filter_file:
        kwargs = {  # build retrieval kwargs
            "given_files": [x for x in found_files if x["instance_id"] == instance_id][
                0
            ]["found_files"],
            "filter_top_n": args.filter_top_n,
        }
    else:
        kwargs = {}

    # main retrieval
    retriever = EmbeddingIndex(
        instance_id,
        structure,
        problem_statement,
        persist_dir=args.persist_dir,
        filter_type=args.filter_type,
        index_type=args.index_type,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        logger=logger,
        **kwargs,
    )

    try:
        file_names, meta_infos, traj = retriever.retrieve(mock=args.mock)
    except Exception as e:
        logger.error(f"Error in retriever.retrieve() for {instance_id}: {type(e).__name__}: {e}")
        # 即使出错，也写入空结果
        file_names, meta_infos, traj = [], [], {"error": str(e)}

    if write_lock is not None:
        write_lock.acquire()
    try:
        # 确保输出目录存在
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        with open(args.output_file, "a") as f:
            f.write(
                json.dumps(
                    {
                        "instance_id": instance_id,
                        "found_files": file_names,
                        "node_info": meta_infos,
                        "traj": traj,
                    }
                )
                + "\n"
            )
        logger.info(f"Successfully wrote result for {instance_id} to {args.output_file}")
    except Exception as e:
        logger.error(f"Error writing to {args.output_file} for {instance_id}: {type(e).__name__}: {e}")
        raise
    finally:
        if write_lock is not None:
            write_lock.release()


def retrieve(args):
    if args.filter_file:
        # 确保 filter_file 路径正确
        original_filter_file = args.filter_file
        if not os.path.isabs(args.filter_file):
            # 尝试多个可能的路径
            possible_paths = [
                args.filter_file,  # 原始路径（相对于当前工作目录）
                os.path.join(os.path.dirname(args.output_folder), args.filter_file),  # 相对于 output_folder 的父目录
            ]
            # 如果 filter_file 包含 results/，尝试相对于 MagentLess 目录
            if 'results/' in args.filter_file:
                # 提取 results/ 之后的部分
                results_part = args.filter_file.split('results/', 1)[-1] if 'results/' in args.filter_file else args.filter_file
                possible_paths.append(os.path.join('results', results_part))
            
            # 尝试每个可能的路径
            for path in possible_paths:
                if os.path.exists(path):
                    args.filter_file = path
                    break
            else:
                # 如果所有路径都不存在，使用原始路径（让 load_jsonl 抛出错误）
                args.filter_file = original_filter_file
        
        try:
            found_files = load_jsonl(args.filter_file)
        except FileNotFoundError as e:
            # 如果文件不存在，打印所有尝试的路径
            print(f"ERROR: Cannot find filter_file: {original_filter_file}")
            print(f"  Tried paths:")
            for path in possible_paths if not os.path.isabs(original_filter_file) else [original_filter_file]:
                print(f"    - {path} (exists: {os.path.exists(path)})")
            raise
    else:
        found_files = []

    if args.dataset == 'local_json':
        swe_bench_data = load_local_json()
    else:
        swe_bench_data = load_dataset(args.dataset, split=args.split)
    prev_o = load_jsonl(args.output_file) if os.path.exists(args.output_file) else []

    if args.num_threads == 1:
        for bug in tqdm(swe_bench_data, colour="MAGENTA"):
            retrieve_locs(
                bug, args, swe_bench_data, found_files, prev_o, write_lock=None
            )
    else:
        write_lock = Lock()
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.num_threads
        ) as executor:
            futures = [
                executor.submit(
                    retrieve_locs,
                    bug,
                    args,
                    swe_bench_data,
                    found_files,
                    prev_o,
                    write_lock,
                )
                for bug in swe_bench_data
            ]
            for _ in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(swe_bench_data),
                colour="MAGENTA",
            ):
                pass


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="retrieve_locs.jsonl")
    parser.add_argument(
        "--index_type", type=str, default="simple", choices=["simple", "complex"]
    )
    parser.add_argument(
        "--filter_type", type=str, default="none", choices=["none", "given_files"]
    )
    parser.add_argument("--filter_top_n", type=int, default=None)
    parser.add_argument("--filter_file", type=str, default="")
    parser.add_argument("--chunk_size", type=int, default=1024)
    parser.add_argument("--chunk_overlap", type=int, default=0)
    parser.add_argument("--persist_dir", type=str)
    parser.add_argument("--target_id", type=str)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument(
        "--num_threads",
        type=int,
        default=1,
        help="Number of threads to use for creating API requests (WARNING, embedding token counts are only accurate when thread=1)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="princeton-nlp/SWE-bench_Lite",
        choices=["princeton-nlp/SWE-bench_Lite", "princeton-nlp/SWE-bench_Verified", "Daoguang/Multi-SWE-bench", "local_json"],
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
    )

    args = parser.parse_args()

    args.output_file = os.path.join(args.output_folder, args.output_file)
    assert (
        not args.filter_type == "given_files" or args.filter_file != ""
    ), "Need to provide a filtering file"

    os.makedirs(args.output_folder, exist_ok=True)
    os.makedirs(os.path.join(args.output_folder, "retrieval_logs"), exist_ok=True)

    # dump argument
    with open(os.path.join(args.output_folder, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    retrieve(args)


if __name__ == "__main__":
    main()
