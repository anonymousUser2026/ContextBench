#!/usr/bin/env python3
"""清理 .traj 文件中 info 里没有 patch_context 的实例（删除整个实例目录），并从 preds.json 移除对应记录。"""

import json
import shutil
import sys
from pathlib import Path


def has_patch_context(traj_path: Path) -> bool:
    """检查 .traj 的 info 中是否有非空的 patch_context。"""
    try:
        data = json.loads(traj_path.read_text())
        info = data.get("info") or {}
        pc = info.get("patch_context")
        return bool(pc and isinstance(pc, str) and pc.strip())
    except Exception as e:
        print(f"  [WARN] 读取失败 {traj_path}: {e}", file=sys.stderr)
        return False


def main():
    # 使用脚本所在目录作为基准，支持相对路径
    script_dir = Path(__file__).parent
    base = script_dir / "trajectories" / "gpt-5__missing_pro"
    traj_files = list(base.rglob("*.traj"))
    to_remove: list[Path] = []  # 实例目录（.traj 的父目录）

    for p in traj_files:
        if has_patch_context(p):
            continue
        # 实例目录 = .traj 所在目录
        inst_dir = p.parent
        to_remove.append(inst_dir)

    # 去重（同一目录下理论上只有一个 .traj）
    to_remove = sorted(set(to_remove))

    print(f"共 {len(traj_files)} 个 .traj，其中 {len(to_remove)} 个实例的 info 无 patch_context，将被删除：")
    for d in to_remove:
        print(f"  {d.relative_to(base)}")

    if not to_remove:
        print("无需删除。")
        return

    # 实例 id = 目录名，用于从 preds.json 删除
    instance_ids = [d.name for d in to_remove]

    # 删除实例目录
    for d in to_remove:
        shutil.rmtree(d, ignore_errors=True)
        print(f"已删除目录: {d}")

    # 从 preds.json 移除对应记录
    preds_path = base / "preds.json"
    if preds_path.exists():
        try:
            preds = json.loads(preds_path.read_text())
            if isinstance(preds, dict):
                removed = 0
                for iid in instance_ids:
                    if iid in preds:
                        del preds[iid]
                        removed += 1
                if removed:
                    preds_path.write_text(json.dumps(preds, indent=4, ensure_ascii=False))
                    print(f"已从 preds.json 移除 {removed} 条记录。")
        except Exception as e:
            print(f"  [WARN] 更新 preds.json 失败: {e}", file=sys.stderr)

    print(f"已清理 {len(to_remove)} 个实例目录。")


if __name__ == "__main__":
    main()
