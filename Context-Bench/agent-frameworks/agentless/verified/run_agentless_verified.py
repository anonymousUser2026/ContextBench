import pandas as pd
import json
import os
import subprocess
import shutil
import argparse
from tqdm import tqdm

# === 路径与环境 ===
# 动态获取项目根目录（当前文件所在目录）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AGENT_DIR = os.path.join(BASE_DIR, "agent/Agentless")
DATA_FILE = os.path.join(BASE_DIR, "data/Verified.csv")
OUTPUT_ROOT = os.path.join(BASE_DIR, "results/agentless/Verified")
DETAILS_DIR = os.path.join(OUTPUT_ROOT, "details")
TRAJS_DIR = os.path.join(OUTPUT_ROOT, "trajs-fix")

env = os.environ.copy()
env["PYTHONPATH"] = f"{AGENT_DIR}:{env.get('PYTHONPATH', '')}"
env["OPENAI_BASE_URL"] = "http://127.0.0.1:5000/v1"
env["OPENAI_API_KEY"] = "sk-proxy-is-working-properly"

def run_cmd(cmd, name, log_file, timeout=1200):
    """
    运行命令，带超时机制
    
    Args:
        cmd: 要执行的命令列表
        name: 步骤名称
        log_file: 日志文件路径
        timeout: 超时时间（秒），默认20分钟
    """
    import signal
    from datetime import datetime
    
    print(f"  >> [STEP] {name} (超时: {timeout//60}分钟)", flush=True)
    print(f"  >> [TIME] 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    
    with open(log_file, "a") as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting: {name}\n")
        f.write(f"Command: {' '.join(cmd)}\n")
        f.write(f"{'='*80}\n")
        f.flush()
        
        try:
            # 使用 subprocess.run 的 timeout 参数
            res = subprocess.run(
                cmd, 
                env=env, 
                cwd=AGENT_DIR, 
                stdout=f, 
                stderr=subprocess.STDOUT,
                timeout=timeout
            )
            success = res.returncode == 0
        except subprocess.TimeoutExpired:
            print(f"  !! [ERROR] {name} 超时（超过 {timeout//60} 分钟）", flush=True)
            f.write(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] TIMEOUT: {name} exceeded {timeout} seconds\n")
            f.flush()
            success = False
        except Exception as e:
            print(f"  !! [ERROR] {name} 执行出错: {e}", flush=True)
            f.write(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ERROR: {name} failed with exception: {e}\n")
            f.flush()
            success = False
    
    end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"  >> [TIME] 结束时间: {end_time}", flush=True)
    
    return success

def get_jsonl_last(fpath):
    """读取 JSONL 文件的最后一行有效数据"""
    if not os.path.exists(fpath): return None
    try:
        with open(fpath, "r") as f:
            lines = [l.strip() for l in f if l.strip()]
            return json.loads(lines[-1]) if lines else None
    except:
        return None

def process_instance(row):
    """处理单个实例的完整流程"""
    inst_id = row['instance_id']
    orig_id = row['original_inst_id']
    
    # 目录规划
    inst_path = os.path.join(DETAILS_DIR, inst_id)
    traj_file = os.path.join(TRAJS_DIR, f"{inst_id}_traj.json")
    
    # 清除该实例的所有缓存，确保重新生成
    print(f"  >> [CLEAN] Clearing all cache for instance {inst_id}", flush=True)
    
    loc_path = os.path.join(inst_path, "localization")
    rep_path = os.path.join(inst_path, "repairs")
    tst_path = os.path.join(inst_path, "tests")
    final_preds_file = os.path.join(inst_path, "all_preds.jsonl")
    
    # 清除所有缓存目录和文件
    if os.path.exists(loc_path):
        shutil.rmtree(loc_path)
        print(f"  >> [CLEAN] Removed localization cache: {loc_path}", flush=True)
    if os.path.exists(rep_path):
        shutil.rmtree(rep_path)
        print(f"  >> [CLEAN] Removed repairs cache: {rep_path}", flush=True)
    if os.path.exists(tst_path):
        shutil.rmtree(tst_path)
        print(f"  >> [CLEAN] Removed tests cache: {tst_path}", flush=True)
    if os.path.exists(final_preds_file):
        os.remove(final_preds_file)
        print(f"  >> [CLEAN] Removed final predictions file: {final_preds_file}", flush=True)
    if os.path.exists(traj_file):
        os.remove(traj_file)
        print(f"  >> [CLEAN] Removed existing traj file: {traj_file}", flush=True)
    
    # 清除日志文件（可选，如果需要完全重新开始）
    log_file = os.path.join(inst_path, "workflow_full.log")
    if os.path.exists(log_file):
        os.remove(log_file)
        print(f"  >> [CLEAN] Removed log file: {log_file}", flush=True)
    
    # 持久化索引目录 (修复 EmbeddingIndex 的 NoneType 报错)
    index_dir = os.path.join(loc_path, "retrieval", "index")
    os.makedirs(index_dir, exist_ok=True)

    os.makedirs(loc_path, exist_ok=True)
    os.makedirs(rep_path, exist_ok=True)
    os.makedirs(tst_path, exist_ok=True)
    log = os.path.join(inst_path, "workflow_full.log")
    
    # 初始化变量，用于 traj 记录
    f_loc_file = None
    ret_file = None
    comb_file_path = None
    r_loc_file = None
    m_out = None
    
    # 使用 try-finally 确保无论成功或失败都记录 traj
    try:

        # --- 1. 故障定位 ---
        # 1.1 模型文件定位
        f_out = os.path.join(loc_path, "file_level")
        f_loc_file = None
        if run_cmd(["python", "agentless/fl/localize.py", "--file_level", "--output_folder", f_out, "--dataset", "princeton-nlp/SWE-bench_Verified", "--target_id", orig_id, "--skip_existing"], "File-Level Loc", log):
            f_loc_file = os.path.join(f_out, "loc_outputs.jsonl")
        
        # 1.2 过滤
        irr_out = os.path.join(loc_path, "irrelevant")
        irr_file = None
        if run_cmd(["python", "agentless/fl/localize.py", "--file_level", "--irrelevant", "--output_folder", irr_out, "--dataset", "princeton-nlp/SWE-bench_Verified", "--target_id", orig_id, "--skip_existing"], "Irrelevant Filter", log):
            irr_file = os.path.join(irr_out, "loc_outputs.jsonl")
        
        # 1.3 检索
        ret_out = os.path.join(loc_path, "retrieval")
        ret_file = None
        if irr_file and os.path.exists(irr_file):
            # 补充 --persist_dir 修复报错，增加 --chunk_size 2048 修复元数据过长问题
            # 增加超时时间到 40 分钟（2400秒），因为处理大量 chunks 时速度会变慢
            if run_cmd(["python", "agentless/fl/retrieve.py", "--index_type", "simple", "--filter_type", "given_files", "--filter_file", irr_file, "--output_folder", ret_out, "--target_id", orig_id, "--dataset", "princeton-nlp/SWE-bench_Verified", "--persist_dir", index_dir, "--chunk_size", "2048", "--chunk_overlap", "100"], "Embedding Retrieval", log, timeout=2400):
                ret_file = os.path.join(ret_out, "retrieve_locs.jsonl")
        else:
            print(f"  !! Skipping Embedding Retrieval (irr_file not found)", flush=True)
        
        # 1.4 合并
        comb_out = os.path.join(loc_path, "combined")
        comb_file_path = None
        if f_out:
            f_loc_file = os.path.join(f_out, "loc_outputs.jsonl")
            if os.path.exists(f_loc_file) and ret_file and os.path.exists(ret_file):
                # 绕过 combine.py 的 output_file already exists 断言
                comb_file_path = os.path.join(comb_out, "combined_locs.jsonl")
                if os.path.exists(comb_file_path): os.remove(comb_file_path)
                if not run_cmd(["python", "agentless/fl/combine.py", "--retrieval_loc_file", ret_file, "--model_loc_file", f_loc_file, "--top_n", "3", "--output_folder", comb_out], "Combine Results", log):
                    comb_file_path = None
            else:
                print(f"  !! Skipping Combine Results (missing input files)", flush=True)
        else:
            print(f"  !! Skipping Combine Results (f_out not available)", flush=True)
        
        # 1.5 元素定位
        r_out = os.path.join(loc_path, "related_elements")
        r_loc_file = None
        if comb_file_path and os.path.exists(comb_file_path):
            if run_cmd(["python", "agentless/fl/localize.py", "--related_level", "--output_folder", r_out, "--top_n", "3", "--compress", "--start_file", comb_file_path, "--dataset", "princeton-nlp/SWE-bench_Verified", "--target_id", orig_id, "--skip_existing"], "Related-Level Loc", log):
                r_loc_file = os.path.join(r_out, "loc_outputs.jsonl")
        else:
            print(f"  !! Skipping Related-Level Loc (comb_file not found)", flush=True)
        
        # 1.6 行采样
        e_out = os.path.join(loc_path, "edit_samples")
        e_loc_file = None
        if r_loc_file and os.path.exists(r_loc_file):
            if run_cmd(["python", "agentless/fl/localize.py", "--fine_grain_line_level", "--output_folder", e_out, "--top_n", "3", "--compress", "--num_samples", "4", "--temperature", "0.8", "--start_file", r_loc_file, "--dataset", "princeton-nlp/SWE-bench_Verified", "--target_id", orig_id, "--skip_existing"], "Line-Level Sampling", log):
                e_loc_file = os.path.join(e_out, "loc_outputs.jsonl")
        else:
            print(f"  !! Skipping Line-Level Sampling (r_loc_file not found)", flush=True)
        
        # 1.7 拆分
        m_out = os.path.join(loc_path, "merged_sets")
        if e_loc_file and os.path.exists(e_loc_file):
            if not run_cmd(["python", "agentless/fl/localize.py", "--merge", "--output_folder", m_out, "--top_n", "3", "--num_samples", "4", "--start_file", e_loc_file], "Merge Samples", log):
                m_out = None
        else:
            print(f"  !! Skipping Merge Samples (e_loc_file not found)", flush=True)
            m_out = None

        # --- 2. 修复阶段 ---
        repair_success = True
        for i in range(4):
            sample_loc = os.path.join(m_out, f"loc_merged_{i}-{i}_outputs.jsonl")
            sample_rep = os.path.join(rep_path, f"sample_{i+1}")
            if not run_cmd(["python", "agentless/repair/repair.py", "--loc_file", sample_loc, "--output_folder", sample_rep, "--loc_interval", "--top_n", "3", "--max_samples", "10", "--cot", "--diff_format", "--gen_and_process", "--dataset", "princeton-nlp/SWE-bench_Verified"], f"Repair Sample {i+1}", log):
                repair_success = False
        
        if not repair_success:
            print(f"  !! Repair failed for {inst_id}. Skipping verification.")
            return False

        # --- 3. 补丁验证阶段 ---
        # 3.1 识别原始通过测试 (回归测试基准)
        pass_tests_file = os.path.join(tst_path, "passing_tests.jsonl")
        # 缩短 run_id 以防文件系统限制
        short_id = inst_id[-8:]
        print(f"  >> [DEBUG] Running regression tests for original_id: {orig_id}", flush=True)
        
        # 标记测试阶段是否成功
        test_phase_success = True
        
        if not os.path.exists(pass_tests_file):
            # Find Passing Tests 步骤：设置20分钟超时，减少并行数避免资源竞争
            if not run_cmd(
                ["python", "agentless/test/run_regression_tests.py", 
                 "--run_id", f"reg_{short_id}", 
                 "--output_file", pass_tests_file, 
                 "--dataset", "princeton-nlp/SWE-bench_Verified", 
                 "--target_id", str(orig_id),
                 "--num_workers", "4"],  # 减少并行数从12到4，避免资源竞争
                "Find Passing Tests", 
                log,
                timeout=1200  # 20分钟超时
            ):
                print(f"  !! Failed to find passing tests for {inst_id}. Will skip test phase and use fallback.", flush=True)
                test_phase_success = False
        
        # 3.2 LLM 筛选回归测试
        repro_test_success = False
        repro_out = os.path.join(tst_path, "reproduction_samples")  # 提前定义，避免未定义错误
        repro_final = os.path.join(repro_out, "reproduction_tests.jsonl")  # 提前定义
        
        if test_phase_success:
            reg_select_out = os.path.join(tst_path, "select_regression")
            reg_tests_file = os.path.join(reg_select_out, "output.jsonl")
            if not os.path.exists(reg_tests_file):
                if not run_cmd(["python", "agentless/test/select_regression_tests.py", "--passing_tests", pass_tests_file, "--output_folder", reg_select_out, "--dataset", "princeton-nlp/SWE-bench_Verified", "--target_id", orig_id], "Select Regression Tests", log):
                    test_phase_success = False
            
            # 3.3 LLM 生成重现测试
            if test_phase_success:
                repro_test_success = run_cmd(["python", "agentless/test/generate_reproduction_tests.py", "--max_samples", "40", "--output_folder", repro_out, "--dataset", "princeton-nlp/SWE-bench_Verified", "--target_id", orig_id], "Generate Repro Tests", log)
                
                if not repro_test_success:
                    print(f"  !! Generate Repro Tests failed for {inst_id}. Will continue with fallback patch selection.", flush=True)
        
        # --- 补全：验证生成的重现测试 ---
        if repro_test_success:
            print(f"  >> [STEP] Verify generated repro tests for {inst_id}", flush=True)
            for i in range(1): # 只验证第 0 个 sample，因为 Agentless --select 默认找 output_0
                repro_test_sample = os.path.join(repro_out, f"output_{i}_processed_reproduction_test.jsonl")
                if os.path.exists(repro_test_sample):
                    run_cmd(["python", "agentless/test/run_reproduction_tests.py", "--test_jsonl", repro_test_sample, "--run_id", f"v_{short_id}", "--dataset", "princeton-nlp/SWE-bench_Verified", "--instance_ids", str(orig_id), "--testing"], f"Verify Repro Test {i}", log)

        # 3.4 LLM 筛选最终重现测试
        if repro_test_success:
            if not run_cmd(["python", "agentless/test/generate_reproduction_tests.py", "--max_samples", "40", "--output_folder", repro_out, "--output_file", "reproduction_tests.jsonl", "--select", "--dataset", "princeton-nlp/SWE-bench_Verified", "--target_id", orig_id], "Select Final Repro Test", log):
                repro_test_success = False

        # 3.5 执行测试并验证补丁（优化：减少测试数量+并行）
        # 只有在 repro_test_success 时才执行测试
        if repro_test_success:
            test_success = True
            reg_tests_file = os.path.join(reg_select_out, "output.jsonl")
            
            # 收集所有需要测试的补丁路径 - 只测试前5个×4样本=20个（原40个）
            test_tasks = []
            for i in range(4):
                folder = os.path.join(rep_path, f"sample_{i+1}")
                for num in range(min(5, 10)):  # 只测试前5个
                    pred_path = os.path.join(folder, f"output_{num}_processed.jsonl")
                    if os.path.exists(pred_path) and os.path.getsize(pred_path) > 0:
                        test_tasks.append((i, num, pred_path))
            
            print(f"  >> [INFO] Testing {len(test_tasks)} patches with max 8 parallel", flush=True)
            
            # 并行执行测试
            import time
            
            max_parallel = 8  # 8个并行
            running_processes = []
            
            for i, num, pred_path in test_tasks:
                # 等待直到有空槽位
                while len(running_processes) >= max_parallel:
                    time.sleep(0.5)
                    running_processes = [p for p in running_processes if p.poll() is None]
                
                # 启动回归测试
                if os.path.exists(reg_tests_file) and os.path.getsize(reg_tests_file) > 0:
                    reg_log = os.path.join(inst_path, f"test_reg_{i}_{num}.log")
                    cmd = ["python", "agentless/test/run_regression_tests.py", "--regression_tests", reg_tests_file, "--predictions_path", pred_path, "--run_id", f"r_{short_id}_{i}_{num}", "--dataset", "princeton-nlp/SWE-bench_Verified", "--target_id", str(orig_id)]
                    with open(reg_log, "w") as f:
                        p = subprocess.Popen(cmd, env=env, cwd=AGENT_DIR, stdout=f, stderr=subprocess.STDOUT)
                        running_processes.append(p)
                
                # 启动重现测试  
                if os.path.exists(repro_final) and os.path.getsize(repro_final) > 0:
                    repro_log = os.path.join(inst_path, f"test_repro_{i}_{num}.log")
                    cmd = ["python", "agentless/test/run_reproduction_tests.py", "--test_jsonl", repro_final, "--predictions_path", pred_path, "--run_id", f"p_{short_id}_{i}_{num}", "--dataset", "princeton-nlp/SWE-bench_Verified", "--instance_ids", str(orig_id)]
                    with open(repro_log, "w") as f:
                        p = subprocess.Popen(cmd, env=env, cwd=AGENT_DIR, stdout=f, stderr=subprocess.STDOUT)
                        running_processes.append(p)
            
            # 等待所有测试完成
            if running_processes:
                print(f"  >> [INFO] Waiting for all tests to complete...", flush=True)
                max_wait_time = 3600  # 最大等待1小时
                start_wait = time.time()
                check_interval = 5  # 每5秒检查一次
                last_progress_time = start_wait
                
                while running_processes:
                    # 检查是否超时
                    elapsed = time.time() - start_wait
                    if elapsed > max_wait_time:
                        print(f"  !! [WARNING] Max wait time ({max_wait_time}s) exceeded. {len(running_processes)} processes may still be running.", flush=True)
                        break
                    
                    # 检查进程状态
                    still_running = []
                    for p in running_processes:
                        # 检查进程是否已退出
                        if p.poll() is not None:
                            continue  # 进程已退出，跳过
                        
                        # 进程还在运行，继续等待
                        still_running.append(p)
                    
                    running_processes = still_running
                    
                    if running_processes:
                        time.sleep(check_interval)
                        # 每30秒打印一次进度
                        current_time = time.time()
                        if current_time - last_progress_time >= 30:
                            print(f"  >> [INFO] Still waiting for {len(running_processes)} processes (elapsed: {int(elapsed)}s)...", flush=True)
                            last_progress_time = current_time
                
                print(f"  >> [INFO] All tests completed", flush=True)

        # --- 4. 最终重排序 (基于测试结果) ---
        rep_folders = ",".join([os.path.join(rep_path, f"sample_{i+1}") for i in range(4)])
        final_preds_src = os.path.join(AGENT_DIR, "all_preds.jsonl")
        final_preds_file = os.path.join(inst_path, "all_preds.jsonl")
        
        # 强制清理旧结果，确保"所见即最新"
        if os.path.exists(final_preds_src): os.remove(final_preds_src)
        if os.path.exists(final_preds_file): os.remove(final_preds_file)

        # 只有在 repro_test_success 时才尝试带验证的重排序
        rerank_success = False
        if repro_test_success:
            # 尝试带验证的重排序（20分钟超时）
            rerank_success = run_cmd(
                ["python", "agentless/repair/rerank.py", "--patch_folder", rep_folders, "--num_samples", "40", "--deduplicate", "--regression", "--reproduction"], 
                "Final Reranking (w/ Test)", 
                log,
                timeout=1200  # 20分钟超时
            )

            if not rerank_success or not os.path.exists(final_preds_src):
                print(f"  !! Reranking with test failed for {inst_id}. Falling back to simple reranking.")
                # 降级模式：不带 --regression 和 --reproduction（20分钟超时）
                rerank_success = run_cmd(
                    ["python", "agentless/repair/rerank.py", "--patch_folder", rep_folders, "--num_samples", "40", "--deduplicate"], 
                    "Fallback Simple Reranking", 
                    log,
                    timeout=1200  # 20分钟超时
                )

        # 如果rerank失败或repro_test失败，fallback为默认选择第一个可用的diff
        if not rerank_success or not os.path.exists(final_preds_src):
            print(f"  !! Reranking failed or skipped for {inst_id}. Falling back to default patch selection.")
            # 从repair阶段的输出中选择第一个可用的patch
            fallback_patch = None
            fallback_found = False
            
            # 按顺序查找第一个可用的patch：sample_1/output_0, sample_1/output_1, ..., sample_4/output_9
            for i in range(4):
                folder = os.path.join(rep_path, f"sample_{i+1}")
                for num in range(10):
                    proc_file = os.path.join(folder, f"output_{num}_processed.jsonl")
                    data = get_jsonl_last(proc_file)
                    if data:
                        patch = data.get("model_patch", "")
                        # 检查 patch 是否非空
                        if patch and patch.strip():
                            fallback_patch = patch
                            fallback_found = True
                            print(f"  >> [FALLBACK] Selected patch from sample_{i+1}/output_{num}", flush=True)
                            break
                if fallback_found:
                    break
            
            # 创建fallback的all_preds.jsonl（即使没有找到patch也要创建，记录失败原因）
            fallback_reason = "rerank_timeout_or_failed" if (rerank_success == False and repro_test_success) else ("repro_test_failed_or_timeout" if not repro_test_success else "test_phase_failed")
            fallback_result = {
                "model_patch": fallback_patch if fallback_patch else "",
                "fallback_reason": fallback_reason,
                "patch_found": fallback_found
            }
            with open(final_preds_file, "w") as f:
                f.write(json.dumps(fallback_result) + "\n")
            if fallback_patch:
                print(f"  >> [FALLBACK] Created fallback all_preds.jsonl with default patch", flush=True)
            else:
                print(f"  !! [WARNING] No valid patches found for fallback. Created all_preds.jsonl with empty patch.", flush=True)
        else:
            # 如果rerank成功，移动结果文件
            shutil.move(final_preds_src, final_preds_file)
    
    finally:
        # --- 5. 轨迹聚合 (Traj) - 无论成功或失败都记录 ---
        print(f"  >> [TRAJ] Saving trajectory data for {inst_id}", flush=True)
        traj = {
            "instance_id": inst_id,
            "original_id": orig_id,
            "1_model_selected_files": (get_jsonl_last(f_loc_file) or {}).get("found_files", []) if f_loc_file else [],
            "2_embedding_selected_files": (get_jsonl_last(ret_file) or {}).get("found_files", []) if ret_file else [],
            "3_final_combined_files": (get_jsonl_last(comb_file_path) or {}).get("found_files", []) if comb_file_path else [],
            "4_related_elements": (get_jsonl_last(r_loc_file) or {}).get("found_related_locs", {}) if r_loc_file else {},
            "5_sampled_edit_locs_and_patches": [],
            "6_final_selected_patch": None
        }
        
        # 收集采样补丁映射 (从 processed 文件中读取)
        if m_out:
            for i in range(4):
                folder = os.path.join(rep_path, f"sample_{i+1}")
                all_patches_for_sample = []
                edit_locs_for_sample = []
                
                # 尝试从该 sample 的所有 10 个 processed 文件中收集补丁
                # 保持列表长度为 10，用 None 表示缺失或空的补丁
                for num in range(10):
                    proc_file = os.path.join(folder, f"output_{num}_processed.jsonl")
                    data = get_jsonl_last(proc_file)
                    if data:
                        patch = data.get("model_patch", "")
                        # 只添加非空的补丁，空字符串用 None 表示
                        all_patches_for_sample.append(patch if patch and patch.strip() else None)
                    else:
                        # 文件不存在，用 None 表示
                        all_patches_for_sample.append(None)
                
                # 获取该 sample 的 edit_locs
                sample_loc_file = os.path.join(m_out, f"loc_merged_{i}-{i}_outputs.jsonl")
                loc_data = get_jsonl_last(sample_loc_file)
                if loc_data:
                    edit_locs_for_sample = loc_data.get("found_edit_locs", [])

                traj["5_sampled_edit_locs_and_patches"].append({
                    "sample_index": i, 
                    "edit_locs": edit_locs_for_sample, 
                    "patches": all_patches_for_sample
                })

        # 最终选中的补丁
        final_res = get_jsonl_last(final_preds_file)
        if final_res:
            traj["6_final_selected_patch"] = final_res.get("model_patch")

        # 确保目录存在
        os.makedirs(TRAJS_DIR, exist_ok=True)
        with open(traj_file, "w") as f:
            json.dump(traj, f, indent=4)
        
        print(f"  >> [TRAJ] Trajectory saved to {traj_file}", flush=True)
    
    return True

def main():
    parser = argparse.ArgumentParser(description="运行 Agentless Verified 工作流")
    parser.add_argument(
        "--instance_id", 
        type=str, 
        default=None,
        help="指定要运行的单个实例ID（例如: SWE-Bench-Verified__python__maintenance__bugfix__27320d49）。如果不指定，则运行所有实例。"
    )
    parser.add_argument(
        "--original_id",
        type=str,
        default=None,
        help="指定要运行的原始实例ID（例如: scikit-learn__scikit-learn-25232）。如果指定了 instance_id，此参数会被忽略。"
    )
    
    args = parser.parse_args()
    
    os.makedirs(DETAILS_DIR, exist_ok=True)
    os.makedirs(TRAJS_DIR, exist_ok=True)
    df = pd.read_csv(DATA_FILE)
    
    # 如果指定了 instance_id，只处理该实例
    if args.instance_id:
        filtered_df = df[df['instance_id'] == args.instance_id]
        if len(filtered_df) == 0:
            print(f"❌ 错误: 找不到实例ID '{args.instance_id}'")
            print(f"可用的实例ID示例:")
            print(df['instance_id'].head(5).to_string(index=False))
            return
        print(f"🎯 单实例运行模式: {args.instance_id}")
        for _, row in filtered_df.iterrows():
            process_instance(row)
    # 如果指定了 original_id，只处理该实例
    elif args.original_id:
        filtered_df = df[df['original_inst_id'] == args.original_id]
        if len(filtered_df) == 0:
            print(f"❌ 错误: 找不到原始实例ID '{args.original_id}'")
            print(f"可用的原始实例ID示例:")
            print(df['original_inst_id'].head(5).to_string(index=False))
            return
        print(f"🎯 单实例运行模式 (通过 original_id): {args.original_id}")
        for _, row in filtered_df.iterrows():
            process_instance(row)
    # 否则处理所有实例
    else:
        print("📋 批量运行模式: 处理所有实例")
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Verified Full Workflow"):
            process_instance(row)

if __name__ == "__main__":
    main()
