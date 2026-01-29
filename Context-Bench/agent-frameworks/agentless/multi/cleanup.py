#!/usr/bin/env python3
"""
清理项目缓存和残留进程
"""

import os
import sys
import subprocess
import shutil
import json
import signal
from pathlib import Path
from typing import List, Tuple

PROJECT_ROOT = Path(__file__).parent.absolute()


def kill_processes(process_names: List[str], exclude_patterns: List[str] = None) -> int:
    """查找并终止指定名称的进程（排除指定模式）"""
    if exclude_patterns is None:
        exclude_patterns = []
    
    killed_count = 0
    killed_pids = set()  # 避免重复终止同一个进程
    
    try:
        # 使用 ps 和 grep 查找进程
        result = subprocess.run(
            ['ps', 'aux'],
            capture_output=True,
            text=True,
            check=False
        )
        
        for line in result.stdout.split('\n'):
            if not line.strip():
                continue
            
            # 排除 proxy 相关进程
            if any(exclude in line for exclude in exclude_patterns):
                continue
            
            # 排除 grep 自身
            if 'grep' in line.lower():
                continue
                
            for proc_name in process_names:
                # 检查进程命令行是否包含目标名称
                if proc_name in line:
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            pid = int(parts[1])
                            
                            # 避免重复终止
                            if pid in killed_pids:
                                continue
                            
                            # 检查进程是否还在运行
                            try:
                                os.kill(pid, 0)  # 检查进程是否存在
                                # 获取进程的完整命令行用于显示
                                cmd_line = ' '.join(parts[10:]) if len(parts) > 10 else proc_name
                                print(f"  终止进程 {pid}: {cmd_line[:80]}")
                                try:
                                    os.kill(pid, signal.SIGTERM)
                                except:
                                    os.kill(pid, signal.SIGKILL)
                                killed_pids.add(pid)
                                killed_count += 1
                            except ProcessLookupError:
                                pass  # 进程已不存在
                        except (ValueError, IndexError):
                            pass
    except Exception as e:
        print(f"  警告: 查找进程时出错: {e}")
    
    return killed_count


def kill_magentless_scripts() -> int:
    """专门查找并终止 MagentLess 脚本进程（更彻底）"""
    exclude_patterns = ["openai_proxy", "proxy", "grep"]
    
    # MagentLess 相关的脚本名称
    script_patterns = [
        "agentless/fl/localize.py",
        "agentless/fl/retrieve.py",
        "agentless/fl/combine.py",
        "agentless/repair/repair.py",
        "agentless/repair/rerank.py",
        "localize.py",
        "retrieve.py",
        "combine.py",
        "repair.py",
        "rerank.py",
    ]
    
    killed_count = 0
    killed_pids = set()
    
    try:
        result = subprocess.run(
            ['ps', 'aux'],
            capture_output=True,
            text=True,
            check=False
        )
        
        for line in result.stdout.split('\n'):
            if not line.strip():
                continue
            
            # 排除 proxy 和 grep
            if any(exclude in line.lower() for exclude in exclude_patterns):
                continue
            
            # 检查是否包含 MagentLess 脚本
            for pattern in script_patterns:
                if pattern in line:
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            pid = int(parts[1])
                            if pid in killed_pids:
                                continue
                            
                            try:
                                os.kill(pid, 0)
                                cmd_line = ' '.join(parts[10:]) if len(parts) > 10 else pattern
                                print(f"  终止 MagentLess 脚本进程 {pid}: {cmd_line[:80]}")
                                try:
                                    os.kill(pid, signal.SIGTERM)
                                except:
                                    os.kill(pid, signal.SIGKILL)
                                killed_pids.add(pid)
                                killed_count += 1
                            except ProcessLookupError:
                                pass
                        except (ValueError, IndexError):
                            pass
                    break  # 找到一个匹配就足够了
    except Exception as e:
        print(f"  警告: 查找 MagentLess 脚本进程时出错: {e}")
    
    return killed_count


def get_dir_size(path: Path) -> int:
    """获取目录大小（字节）"""
    total = 0
    try:
        for entry in path.rglob('*'):
            if entry.is_file():
                total += entry.stat().st_size
    except Exception:
        pass
    return total


def format_size(size_bytes: int) -> str:
    """格式化文件大小"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f}TB"


def clean_magentless_results() -> Tuple[int, int]:
    """清理 MagentLess 结果目录"""
    results_dir = PROJECT_ROOT / "MagentLess" / "results"
    if not results_dir.exists():
        return 0, 0
    
    total_size = 0
    count = 0
    
    # 只清理 Multi_* 开头的目录
    for multi_dir in results_dir.glob("Multi_*"):
        if multi_dir.is_dir():
            size = get_dir_size(multi_dir)
            total_size += size
            count += 1
            print(f"  删除: {multi_dir.name} ({format_size(size)})")
            shutil.rmtree(multi_dir, ignore_errors=True)
    
    return count, total_size


def clean_playground() -> Tuple[int, int]:
    """清理 playground 临时仓库目录"""
    playground_dir = PROJECT_ROOT / "MagentLess" / "playground"
    if not playground_dir.exists():
        return 0, 0
    
    dirs = [d for d in playground_dir.iterdir() if d.is_dir()]
    if not dirs:
        return 0, 0
    
    size = get_dir_size(playground_dir)
    print(f"  删除 {len(dirs)} 个临时仓库目录 ({format_size(size)})")
    
    # 清空目录但保留目录本身
    for item in playground_dir.iterdir():
        if item.is_dir():
            shutil.rmtree(item, ignore_errors=True)
        else:
            item.unlink(missing_ok=True)
    
    return len(dirs), size


def clean_magentless_logs() -> int:
    """清理 MagentLess 运行时的临时日志文件"""
    count = 0
    
    # 只清理 MagentLess/results 目录下的日志文件
    results_dir = PROJECT_ROOT / "MagentLess" / "results"
    if results_dir.exists():
        for log_file in results_dir.rglob("*.log"):
            log_file.unlink(missing_ok=True)
            count += 1
    
    return count




def clean_incomplete_trajs() -> int:
    """清理不完整的 traj.json 文件"""
    traj_dir = PROJECT_ROOT / "results" / "Multi" / "trajs"
    if not traj_dir.exists():
        return 0
    
    count = 0
    for traj_file in traj_dir.glob("*_traj.json"):
        if not traj_file.is_file():
            continue
        
        # 检查文件大小
        size = traj_file.stat().st_size
        if size < 100:  # 小于 100 字节可能不完整
            print(f"  删除不完整的文件: {traj_file.name} ({size} 字节)")
            traj_file.unlink()
            count += 1
            continue
        
        # 检查 JSON 格式
        try:
            with open(traj_file, 'r', encoding='utf-8') as f:
                json.load(f)
        except (json.JSONDecodeError, Exception):
            print(f"  删除格式错误的文件: {traj_file.name}")
            traj_file.unlink()
            count += 1
    
    return count


def clean_python_cache() -> Tuple[int, int]:
    """清理 Python 缓存文件（__pycache__ 和 .pyc）"""
    magentless_dir = PROJECT_ROOT / "MagentLess"
    if not magentless_dir.exists():
        return 0, 0
    
    pycache_count = 0
    pyc_count = 0
    total_size = 0
    
    # 清理 __pycache__ 目录
    for pycache_dir in magentless_dir.rglob("__pycache__"):
        if pycache_dir.is_dir():
            size = get_dir_size(pycache_dir)
            total_size += size
            pycache_count += 1
            shutil.rmtree(pycache_dir, ignore_errors=True)
    
    # 清理 .pyc 文件
    for pyc_file in magentless_dir.rglob("*.pyc"):
        if pyc_file.is_file():
            size = pyc_file.stat().st_size
            total_size += size
            pyc_count += 1
            pyc_file.unlink(missing_ok=True)
    
    return pycache_count + pyc_count, total_size


def main():
    print("=" * 50)
    print("开始清理项目缓存和残留进程")
    print("=" * 50)
    print()
    
    # 1. 清理 MagentLess 残留进程（排除 proxy）
    print("1. 查找并终止 MagentLess 残留进程（保留 proxy）...")
    
    # 排除 proxy 相关进程
    exclude_patterns = ["openai_proxy.py", "proxy", "openai_proxy"]
    
    # 1.1 查找评估脚本进程
    process_names = [
        "evaluate_multi.py",
        "generate_traj_json.py",
        "convert_patches.py",
        "extract_patch_from_traj.py",
    ]
    killed_count = kill_processes(process_names, exclude_patterns)
    
    # 1.2 查找 MagentLess 目录相关进程
    magentless_count = kill_processes(["MagentLess"], exclude_patterns)
    killed_count += magentless_count
    
    # 1.3 专门查找 MagentLess 脚本进程（更彻底）
    script_count = kill_magentless_scripts()
    killed_count += script_count
    
    if killed_count == 0:
        print("  ✓ 无残留进程")
    else:
        print(f"  ✓ 已终止 {killed_count} 个进程")
    
    # 等待进程退出
    import time
    time.sleep(2)
    print()
    
    # 2. 清理 MagentLess 结果目录
    print("2. 清理 MagentLess 结果目录...")
    multi_count, multi_size = clean_magentless_results()
    if multi_count == 0:
        print("  ✓ 无 Multi_* 结果目录")
    else:
        print(f"  ✓ 已清理 {multi_count} 个目录 ({format_size(multi_size)})")
    print()
    
    # 3. 清理 playground
    print("3. 清理 playground 临时仓库目录...")
    pg_count, pg_size = clean_playground()
    if pg_count == 0:
        print("  ✓ playground 目录为空")
    else:
        print(f"  ✓ 已清理 {pg_count} 个临时仓库 ({format_size(pg_size)})")
    print()
    
    # 4. 清理 MagentLess 运行时日志
    print("4. 清理 MagentLess 运行时日志文件...")
    log_count = clean_magentless_logs()
    if log_count == 0:
        print("  ✓ 无 MagentLess 日志文件")
    else:
        print(f"  ✓ 已清理 {log_count} 个日志文件")
    print()
    
    # 5. 清理不完整的 traj.json
    print("5. 检查不完整的 traj.json 文件...")
    traj_count = clean_incomplete_trajs()
    if traj_count == 0:
        print("  ✓ 无 incomplete traj.json 文件")
    else:
        print(f"  ✓ 已清理 {traj_count} 个不完整的 traj.json 文件")
    print()
    
    # 6. 清理 Python 缓存
    print("6. 清理 Python 缓存文件...")
    cache_count, cache_size = clean_python_cache()
    if cache_count == 0:
        print("  ✓ 无 Python 缓存文件")
    else:
        print(f"  ✓ 已清理 {cache_count} 个缓存文件 ({format_size(cache_size)})")
    print()
    
    # 总结
    print("=" * 50)
    print("清理完成！")
    print("=" * 50)
    print()
    print("已清理内容：")
    print(f"  - MagentLess 残留进程: {killed_count} 个（proxy 已保留）")
    print(f"  - MagentLess 不完整结果目录: {multi_count} 个")
    print(f"  - playground 临时仓库: {pg_count} 个")
    print(f"  - MagentLess 运行时日志: {log_count} 个")
    print(f"  - 不完整的 traj.json: {traj_count} 个")
    print(f"  - Python 缓存文件: {cache_count} 个")
    print()
    print("注意：")
    print("  - proxy 进程未清理")
    print("  - results/Multi/ 目录中的完整 traj.json 文件已保留")
    print("  - 数据文件（data/）未清理")
    print("  - 符号链接（MagentLess/data/）未清理")
    print()


if __name__ == "__main__":
    main()

