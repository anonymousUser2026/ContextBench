#!/usr/bin/env python3
"""将 run_batch_exit_statuses.yaml 中所有 submitted* 的实例目录从 gpt-5__missing_pro 移动到 gpt-5__no-context。"""

import shutil
import yaml
from pathlib import Path

# 使用脚本所在目录作为基准，支持相对路径
SCRIPT_DIR = Path(__file__).parent
SRC = SCRIPT_DIR / "trajectories" / "gpt-5__missing_pro"
DST = SCRIPT_DIR / "trajectories" / "gpt-5__no-context"
STATUS_FILE = SRC / "run_batch_exit_statuses.yaml"

def main():
    data = yaml.safe_load(STATUS_FILE.read_text())
    by_status = data.get("instances_by_exit_status") or {}

    # 收集所有 submitted* 键下的实例名
    to_move = []
    for k, v in by_status.items():
        if v and isinstance(v, list) and "submitted" in k.lower():
            to_move.extend(v)

    to_move = sorted(set(to_move))
    print(f"共 {len(to_move)} 个 submitted 实例待移动")

    DST.mkdir(parents=True, exist_ok=True)
    moved = 0
    for name in to_move:
        src_dir = SRC / name
        dst_dir = DST / name
        if not src_dir.is_dir():
            print(f"  [SKIP] 源目录不存在: {src_dir}")
            continue
        if dst_dir.exists():
            print(f"  [SKIP] 目标已存在，先删除: {dst_dir}")
            shutil.rmtree(dst_dir, ignore_errors=True)
        shutil.move(str(src_dir), str(dst_dir))
        print(f"  moved: {name}")
        moved += 1

    print(f"已移动 {moved} 个实例到 {DST}")

if __name__ == "__main__":
    main()
