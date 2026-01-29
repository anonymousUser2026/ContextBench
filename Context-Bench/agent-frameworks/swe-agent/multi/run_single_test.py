#!/usr/bin/env python
"""
MSWE-agent 单实例测试脚本
用法:
  # 使用 Azure OpenAI:
  python run_single_test.py --model_name "azure/gpt-4o" --deployment "your-deployment-name"
  
  # 使用 OpenAI:
  OPENAI_API_KEY=sk-xxx python run_single_test.py --model_name "gpt4o"
  
  # 使用测试模型（不需要 API）:
  python run_single_test.py --model_name "instant_empty_submit"
"""
import os
import sys
import argparse

# 设置工作目录
os.chdir("/home/lih/sweagent-eval/MSWE-agent")
sys.path.insert(0, "/home/lih/sweagent-eval/MSWE-agent")

from pathlib import Path
from sweagent import CONFIG_DIR
from sweagent.agent.agents import AgentArguments
from sweagent.agent.models import ModelArguments
from sweagent.environment.swe_env import EnvironmentArguments
from multi_swe_bench.harness.build_dataset import CliArgs
from run import ScriptArguments, ActionsArguments, Main


def main():
    parser = argparse.ArgumentParser(description="Run MSWE-agent on a single instance")
    parser.add_argument("--model_name", default="instant_empty_submit", 
                       help="Model name (e.g., azure/gpt-4o, gpt4o, instant_empty_submit)")
    parser.add_argument("--deployment", default=None,
                       help="Azure deployment name (overrides AZURE_OPENAI_DEPLOYMENT)")
    parser.add_argument("--data_file", default="data/c_test_single.jsonl",
                       help="Data file path")
    parser.add_argument("--config_file", default="config/default_from_url.yaml",
                       help="Config file path")
    parser.add_argument("--cost_limit", type=float, default=3.0,
                       help="Per instance cost limit")
    args = parser.parse_args()
    
    # 如果指定了 deployment，更新环境变量
    if args.deployment:
        os.environ["AZURE_OPENAI_DEPLOYMENT"] = args.deployment
        print(f"Set AZURE_OPENAI_DEPLOYMENT to: {args.deployment}")
    
    print(f"CONFIG_DIR: {CONFIG_DIR}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Model: {args.model_name}")
    print(f"Data file: {args.data_file}")
    
    # 创建配置
    script_args = ScriptArguments(
        suffix="single_test",
        environment=EnvironmentArguments(
            cli_args=CliArgs(
                workdir=Path("data_files"),
                repo_dir=None,
                pr_file=args.data_file,
                need_clone=True,
                max_workers_build_image=4,
                max_workers_run_instance=4,
                clear_env=False,
                global_env=[],
            ),
            verbose=True,
            install_environment=True,
            cache_task_images=False,
        ),
        skip_existing=False,
        raise_exceptions=True,
        agent=AgentArguments(
            model=ModelArguments(
                model_name=args.model_name,
                total_cost_limit=0.0,
                per_instance_cost_limit=args.cost_limit,
                temperature=0.0,
                top_p=0.95,
            ),
            config_file=Path(args.config_file) if args.config_file.startswith("/") else CONFIG_DIR / args.config_file,
        ),
        actions=ActionsArguments(open_pr=False, skip_if_commits_reference_issue=True),
    )
    
    print(f"\n--- Starting run ---")
    try:
        main_runner = Main(script_args)
        main_runner.main()
        print("\n--- Run completed successfully! ---")
    except Exception as e:
        print(f"\n--- Run failed with error: {e} ---")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

