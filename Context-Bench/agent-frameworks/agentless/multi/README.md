# Multi.csv 评测

## 环境准备

### 1. 虚拟环境

```bash
# 进入项目根目录
cd <项目根目录>
source .venv/bin/activate
```

### 2. 安装依赖

```bash
# MagentLess 依赖
cd MagentLess
pip install -r requirements.txt
export PYTHONPATH=$PYTHONPATH:$(pwd)
cd ..
```

### 3. 数据准备

```bash
# 1. 将语言文件夹放入 data/multi-swe-bench/ 目录
mkdir -p data/multi-swe-bench
# 将语言文件夹（js/, java/, ts/ 等）移动到 data/multi-swe-bench/ 下
# 或创建符号链接：ln -s ../js data/multi-swe-bench/js

# 2. 为 MagentLess 创建数据目录的符号链接（重要！）
# MagentLess 在运行时会在 MagentLess/ 目录下查找 data/js/ 等目录
mkdir -p MagentLess/data
cd MagentLess/data
for lang in js java ts c cpp go rust python; do
    if [ -d "../../data/multi-swe-bench/$lang" ]; then
        ln -sf "../../data/multi-swe-bench/$lang" "$lang"
    fi
done
cd ../..

# MagentLess 需要 playground 目录用于克隆仓库（会自动创建）
mkdir -p MagentLess/playground
```

### 4. Proxy 配置

```bash
# 启动 proxy（另一个终端）
cd proxy
python openai_proxy.py  # 运行在端口 18888
```

### 5. MagentLess 配置

```bash
# 设置环境变量
export OPENAI_BASE_URL=http://localhost:18888
export OPENAI_EMBED_URL=http://localhost:18888/v1
export OPENAI_API_KEY=dummy-key  # 如果需要

export OPENAI_MODEL=gpt-5  # 根据实际情况

# 生成 MagentLess 配置文件
bash evaluate_multi/setup_magentless_config.sh
```

这会创建 `MagentLess/script/api_key.sh` 文件，包含：
- `OPENAI_API_KEY`
- `OPENAI_BASE_URL` (指向 proxy)
- `OPENAI_MODEL`
- `OPENAI_EMBED_URL` (指向 proxy)

## 运行评测

### 阶段1: MagentLess 生成补丁

```bash
python evaluate_multi/evaluate_multi.py \
    --csv data/Multi.csv \
    --skip_evaluation  # 只运行 MagentLess，不运行评测
```

### 阶段2: Multi-swe-bench 评测

```bash
python evaluate_multi/evaluate_multi.py \
    --csv data/Multi.csv \
    --skip_magentless  # 跳过 MagentLess，只运行评测
    --dataset_files data/multi-swe-bench/*/*.jsonl
```

### 完整流程

```bash
python evaluate_multi/evaluate_multi.py \
    --csv data/Multi.csv \
    --dataset_files data/multi-swe-bench/*/*.jsonl
```

**注意**：如果数据文件直接在 `data/` 下的语言文件夹中（如 `data/js/`, `data/java/`），可以使用：
```bash
--dataset_files data/*/*.jsonl  # 匹配所有语言
# 或指定特定语言
--dataset_files data/js/*.jsonl data/java/*.jsonl
```

## 结果

- `results/Multi/trajs/` - traj.json 文件（文件名格式：`{instance_id}_traj.json`，包含六个阶段）
- `results/Multi/patches_*.jsonl` - 补丁文件
- `results/Multi/details/` - 评测结果
