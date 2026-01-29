#!/bin/bash
# 运行 missing_multi 中指定的实例测试
INSTANCE_FILTER=$(cat missing_multi | tr '\n' '|' | sed 's/|$//')
python3 multirun.py \
   --model_name gpt-5 \
   --cache_task_images True \
   --pre_build_all_images False \
   --remove_image False \
   --pr_file data/missing_multi.jsonl \
   --config_file config/azure_gpt5_context.yaml \
   --skip_existing=True \
   --print_config=False \
   --per_instance_cost_limit 0 \
   --max_workers_build_image 4 \
   --instance_filter "^($INSTANCE_FILTER)$"
