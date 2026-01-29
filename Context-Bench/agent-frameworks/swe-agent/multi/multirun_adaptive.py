#!/usr/bin/env python3
"""
自适应多实例运行脚本
- 动态检测容器占用
- 遇到冲突自动跳过，换下一个实例
- 直到所有实例都完成或无法运行
"""

from __future__ import annotations

import logging
import json
import os
import re
import subprocess
import traceback
import time
import threading
import copy
from typing import Any, Set, Dict
from dataclasses import dataclass
from getpass import getuser
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml
import docker
from rich.markdown import Markdown

try:
    from rich_argparse import RichHelpFormatter
except ImportError:
    msg = "Please install the rich_argparse package with `pip install rich_argparse`."
    raise ImportError(msg)

from simple_parsing import parse
from simple_parsing.helpers.flatten import FlattenedAccess
from simple_parsing.helpers.serialization.serializable import FrozenSerializable
from swebench import KEY_INSTANCE_ID, KEY_MODEL, KEY_PREDICTION
from multi_swe_bench.harness.build_dataset import CliArgs
from unidiff import PatchSet

from sweagent import CONFIG_DIR
from sweagent.utils.log import get_logger
from sweagent.agent.agents import Agent, AgentArguments
from sweagent.agent.models import ModelArguments
from sweagent.environment.swe_env import EnvironmentArguments, SWEEnv
from sweagent.environment.utils import get_instances

logger = get_logger("swe-agent-run-adaptive")
logging.getLogger("simple_parsing").setLevel(logging.WARNING)

INSTANCE_LOG_DIR = 'logs/'

# 全局锁和状态管理
_lock = threading.Lock()
_running_images: Set[str] = set()  # 正在运行的镜像
_completed_instances: Set[str] = set()  # 已完成的实例
_failed_instances: Dict[str, str] = {}  # 失败的实例及原因
_skipped_count = 0  # 因冲突跳过的次数


def get_image_name_from_instance_id(instance_id: str, all_datas: dict) -> str:
    """从实例ID获取镜像名"""
    if instance_id in all_datas:
        record = all_datas[instance_id]
        # 镜像名格式: org/repo:pr-xxx
        org_repo = instance_id.rsplit('-', 1)[0].replace('__', '/')
        # 从 record 中获取 base_commit 或 pr 信息
        if hasattr(record, 'instance') and hasattr(record.instance, 'pr'):
            pr = record.instance.pr
            if hasattr(pr, 'base_commit'):
                return f"{org_repo}:pr-{pr.base_commit[:7]}"
        # 回退：使用实例ID推断
        parts = instance_id.rsplit('-', 1)
        if len(parts) == 2:
            return f"{org_repo}:pr-{parts[1]}"
    return instance_id


def is_image_in_use(image_name: str) -> bool:
    """检查镜像是否正在被使用（有运行中的容器）"""
    try:
        client = docker.from_env(timeout=10)
        containers = client.containers.list()
        for container in containers:
            if container.image.tags:
                for tag in container.image.tags:
                    if image_name in tag or tag in image_name:
                        return True
            # 也检查容器名是否包含镜像名的特征
            container_name = container.name
            image_sanitized = image_name.replace("/", "-").replace(":", "-")
            if image_sanitized in container_name:
                return True
        return False
    except Exception as e:
        logger.warning(f"检查镜像使用状态失败: {e}")
        return False  # 失败时假设不在使用


def is_instance_available(instance_id: str, all_datas: dict) -> bool:
    """检查实例是否可用（没有被占用）"""
    global _running_images, _completed_instances
    
    with _lock:
        # 已完成
        if instance_id in _completed_instances:
            return False
        
        # 已失败
        if instance_id in _failed_instances:
            return False
    
    # 获取镜像名
    image_name = get_image_name_from_instance_id(instance_id, all_datas)
    
    with _lock:
        # 检查本进程是否正在使用这个镜像
        if image_name in _running_images:
            return False
    
    # 检查外部进程是否正在使用这个镜像
    if is_image_in_use(image_name):
        return False
    
    return True


def mark_instance_running(instance_id: str, all_datas: dict):
    """标记实例为运行中"""
    global _running_images
    image_name = get_image_name_from_instance_id(instance_id, all_datas)
    with _lock:
        _running_images.add(image_name)


def mark_instance_done(instance_id: str, all_datas: dict, success: bool, error: str = ""):
    """标记实例为完成"""
    global _running_images, _completed_instances, _failed_instances
    image_name = get_image_name_from_instance_id(instance_id, all_datas)
    with _lock:
        _running_images.discard(image_name)
        if success:
            _completed_instances.add(instance_id)
        else:
            _failed_instances[instance_id] = error


@dataclass(frozen=True)
class ActionsArguments(FlattenedAccess, FrozenSerializable):
    open_pr: bool = False
    apply_patch_locally: bool = False
    skip_if_commits_reference_issue: bool = True
    push_gh_repo_url: str = ""

    def __post_init__(self):
        if self.push_gh_repo_url:
            msg = "push_gh_repo_url is obsolete. Use repo_path instead"
            raise ValueError(msg)


@dataclass(frozen=True)
class ScriptArguments(FlattenedAccess, FrozenSerializable):
    environment: EnvironmentArguments
    agent: AgentArguments
    actions: ActionsArguments
    instance_filter: str = ".*"
    skip_existing: bool = True
    suffix: str = ""
    raise_exceptions: bool = False
    print_config: bool = True

    @property
    def run_name(self):
        model_name = self.agent.model.model_name.replace(":", "-")
        from sweagent.environment.utils import get_data_path_name
        data_stem = get_data_path_name(str(self.environment.cli_args.pr_file))
        config_stem = Path(self.agent.config_file).stem
        temp = self.agent.model.temperature
        top_p = self.agent.model.top_p
        per_instance_cost_limit = self.agent.model.per_instance_cost_limit
        install_env = self.environment.install_environment
        return (
            f"{model_name}__{data_stem}__{config_stem}__t-{temp:.2f}__p-{top_p:.2f}"
            + f"__c-{per_instance_cost_limit:.2f}__install-{int(install_env)}"
            + (f"__{self.suffix}" if self.suffix else "")
        )


class _ContinueLoop(Exception):
    pass


class Main:
    def __init__(self, args: ScriptArguments, filter_instance: str):
        self.args = args
        self.instance_id = filter_instance
        self.traj_dir = Path("trajectories") / Path(getuser()) / args.run_name
        self.traj_dir.mkdir(parents=True, exist_ok=True)
        if self.should_skip(self.instance_id):
            raise _ContinueLoop
        log_dir = Path(INSTANCE_LOG_DIR) / args.run_name / self.instance_id
        if log_dir.exists():
            file_path = log_dir / "log"
            file_path.unlink(missing_ok=True)
        self.agent = Agent("primary", args.agent, log_dir)
        self.env = SWEEnv(args.environment, log_dir)

    def run(self):
        assert isinstance(self.instance_id, str)
        if self.should_skip(self.instance_id):
            raise _ContinueLoop
        logger.info("▶️  Beginning task " + self.instance_id)
        observation, info = self.env.reset(self.instance_id)
        if info is None:
            raise _ContinueLoop

        issue = getattr(self.env, "query", None)
        files = []
        if self.env.record.instance.pr.fix_patch:
            files = "\n".join([f"- {x.path}" for x in PatchSet(self.env.record.instance.pr.fix_patch).modified_files])
        test_files = []
        if self.env.record.instance.pr.test_patch:
            test_patch_obj = PatchSet(self.env.record.instance.pr.test_patch)
            test_files = "\n".join([f"- {x.path}" for x in test_patch_obj.modified_files + test_patch_obj.added_files])
        tests = ""

        setup_args = {"issue": issue, "files": files, "test_files": test_files, "tests": tests}
        info, trajectory = self.agent.run(
            setup_args=setup_args,
            env=self.env,
            observation=observation,
            traj_dir=self.traj_dir,
            return_type="info_trajectory",
        )
        self._save_predictions(self.instance_id, info)
        self._save_patch(self.instance_id, info)

    def main(self):
        logger.info(f'running the instance id {self.instance_id} now!')
        try:
            self.run()
        except _ContinueLoop:
            logger.info("Skipping instance")
        except KeyboardInterrupt:
            logger.info("Exiting...")
            self.env.close()
        except SystemExit:
            logger.critical("❌ Exiting because SystemExit was called")
            self.env.close()
            raise
        except Exception as e:
            traceback.print_exc()
            if self.args.raise_exceptions:
                self.env.close()
                raise e
            if self.env.record:
                logger.warning(f"❌ Failed on {self.env.record.data['instance_id']}: {e}")
            else:
                logger.warning("❌ Failed on unknown instance")
            raise

    def should_skip(self, instance_id: str) -> bool:
        if re.match(self.args.instance_filter, instance_id) is None:
            return True
        if not self.args.skip_existing:
            return False
        log_path = self.traj_dir / (instance_id + ".traj")
        if log_path.exists():
            with log_path.open("r") as f:
                data = json.load(f)
            exit_status = data["info"].get("exit_status", None)
            if exit_status == "early_exit" or exit_status is None:
                os.remove(log_path)
            else:
                logger.info(f"⏭️ Skipping existing trajectory: {log_path}")
                return True
        return False

    def _save_predictions(self, instance_id: str, info):
        output_file = self.traj_dir / "all_preds.jsonl"
        model_patch = info["submission"] if "submission" in info else None
        datum = {
            KEY_MODEL: Path(self.traj_dir).name,
            KEY_INSTANCE_ID: instance_id,
            KEY_PREDICTION: model_patch,
        }
        with open(output_file, "a+") as fp:
            print(json.dumps(datum), file=fp, flush=True)
        logger.info(f"Saved predictions to {output_file}")

    def _save_patch(self, instance_id: str, info):
        patch_output_dir = self.traj_dir / "patches"
        patch_output_dir.mkdir(exist_ok=True, parents=True)
        patch_output_file = patch_output_dir / f"{instance_id}.patch"
        if info.get("submission"):
            patch_output_file.write_text(info["submission"])
            logger.info(f"💾 Trajectory saved for {instance_id}")


def get_args(args=None) -> ScriptArguments:
    defaults = ScriptArguments(
        suffix="",
        environment=EnvironmentArguments(
            cli_args=CliArgs(
                workdir=Path("data_files"),
                repo_dir=None,
                pr_file='data/',
                need_clone=True,
                max_workers_build_image=64,
                max_workers_run_instance=64,
                clear_env=False,
                global_env=[],
            ),
            verbose=True,
            install_environment=True,
            cache_task_images=False,
        ),
        skip_existing=True,
        agent=AgentArguments(
            model=ModelArguments(
                model_name="gpt4",
                total_cost_limit=0.0,
                per_instance_cost_limit=3.0,
                temperature=0.0,
                top_p=0.95,
            ),
            config_file=CONFIG_DIR / "default.yaml",
        ),
        actions=ActionsArguments(open_pr=False, skip_if_commits_reference_issue=True),
    )

    yaml.add_representer(str, lambda dumper, data: 
        dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|") 
        if data.count("\n") > 0 else dumper.represent_scalar("tag:yaml.org,2002:str", data))

    return parse(
        ScriptArguments,
        default=defaults,
        add_config_path_arg=False,
        args=args,
        formatter_class=RichHelpFormatter,
        description=Markdown("Adaptive multi-instance runner"),
    )


def run_single_adaptive(scripts, instance_id: str, all_datas: dict, max_retries: int = 3):
    """运行单个实例，支持冲突检测和重试"""
    global _skipped_count
    
    for attempt in range(max_retries):
        # 检查是否可用
        if not is_instance_available(instance_id, all_datas):
            with _lock:
                _skipped_count += 1
            logger.info(f"⏳ 实例 {instance_id} 被占用，跳过 (attempt {attempt + 1})")
            time.sleep(2)  # 等待一下再检查
            continue
        
        # 标记为运行中
        mark_instance_running(instance_id, all_datas)
        
        try:
            copy_args = copy.deepcopy(scripts)
            handler = Main(copy_args, instance_id)
            handler.main()
            mark_instance_done(instance_id, all_datas, success=True)
            logger.info(f"✅ 完成实例 {instance_id}")
            return True
        except _ContinueLoop:
            mark_instance_done(instance_id, all_datas, success=True, error="skipped")
            logger.info(f"⏭️ 实例 {instance_id} 已跳过")
            return True
        except Exception as e:
            error_msg = str(e)
            # 检查是否是容器冲突错误
            if "container" in error_msg.lower() or "conflict" in error_msg.lower():
                mark_instance_done(instance_id, all_datas, success=False, error="conflict")
                logger.warning(f"🔄 实例 {instance_id} 遇到容器冲突，将重试")
                time.sleep(5)
                continue
            else:
                mark_instance_done(instance_id, all_datas, success=False, error=error_msg[:100])
                logger.error(f"❌ 实例 {instance_id} 失败: {error_msg[:100]}")
                return False
    
    # 重试次数用完
    mark_instance_done(instance_id, all_datas, success=False, error="max_retries")
    return False


def main(args: ScriptArguments):
    global _completed_instances, _failed_instances, _skipped_count
    
    running_threads = int(os.environ.get('RUNNING_THREADS', '50'))
    logger.info(f"🚀 启动自适应多实例运行器，并发数: {running_threads}")
    
    cli_args = args.environment.cli_args
    all_datas = get_instances(
        cli_args.pr_file,
        cli_args,
        prebuild=args.environment.pre_build_all_images,
    )
    instance_ids = list(all_datas.keys())
    total_instances = len(instance_ids)
    logger.info(f"📊 总实例数: {total_instances}")
    
    post_args = parse(
        ScriptArguments,
        default=args,
        add_config_path_arg=False,
        args=['--pre_build_all_images=False'],
        formatter_class=RichHelpFormatter,
        description=Markdown("Adaptive runner"),
    )
    
    executor = ThreadPoolExecutor(max_workers=running_threads)
    futures = {
        executor.submit(run_single_adaptive, post_args, instance_id, all_datas): instance_id 
        for instance_id in instance_ids
    }
    
    # 定期打印进度
    start_time = time.time()
    completed = 0
    
    for future in as_completed(futures):
        instance_id = futures[future]
        try:
            result = future.result()
            completed += 1
            elapsed = time.time() - start_time
            remaining = total_instances - completed
            rate = completed / elapsed if elapsed > 0 else 0
            eta = remaining / rate if rate > 0 else 0
            
            logger.info(
                f"📈 进度: {completed}/{total_instances} "
                f"({100*completed/total_instances:.1f}%) "
                f"| 成功: {len(_completed_instances)} "
                f"| 失败: {len(_failed_instances)} "
                f"| 跳过: {_skipped_count} "
                f"| ETA: {eta/60:.1f}min"
            )
        except Exception as e:
            logger.error(f"❌ 实例 {instance_id} 异常: {e}")
    
    # 最终统计
    logger.info("=" * 50)
    logger.info(f"🏁 运行完成!")
    logger.info(f"   总实例: {total_instances}")
    logger.info(f"   成功: {len(_completed_instances)}")
    logger.info(f"   失败: {len(_failed_instances)}")
    logger.info(f"   冲突跳过次数: {_skipped_count}")
    
    if _failed_instances:
        logger.info("失败实例:")
        for inst, err in list(_failed_instances.items())[:10]:
            logger.info(f"  - {inst}: {err}")


if __name__ == "__main__":
    main(get_args())
















