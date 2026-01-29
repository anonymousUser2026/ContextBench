# mini-SWE-agent Extensions

这个仓库包含 mini-SWE-agent 的扩展功能和配置文件。

## 目录结构

```
mini-swe-agent-extensions/
├── configs/              # 配置文件
│   ├── swebench_context_aware.yaml
│   └── swebench_following_context.yaml
├── agents/               # Agent 实现
│   ├── context_aware.py
│   └── swebench_context_aware_runner.py
├── scripts/              # 运行和工具脚本
│   ├── run_verified_500.sh
│   ├── run_poly_116.sh
│   ├── retry_failed_instances.sh
│   ├── extract_poly_subset.py
│   ├── analyze_failed_instances.py
│   ├── pull_poly_images_top30.py
│   └── *.txt (数据文件)
└── docs/                 # 文档
    ├── CONTEXT_AWARE_GUIDE.md
    ├── poly_extraction_report.md
    └── failed_instances_report.md
```

## 主要功能

### 1. Context-Aware Agent

新增的 context-aware agent 在提交 patch 前会要求 agent 提供代码上下文信息。

- **配置文件**: `configs/swebench_context_aware.yaml`
- **Agent 实现**: `agents/context_aware.py`
- **运行器**: `agents/swebench_context_aware_runner.py`

### 2. Following-Context Agent

扩展的 following-context agent 支持 `<EXPLORE_CONTEXT>` 标记来跟踪代码探索过程。

- **配置文件**: `configs/swebench_following_context.yaml`

### 3. 批量测试脚本

#### SWE-Bench-Verified (174 个实例)

```bash
cd scripts
./run_verified_500.sh
```

#### SWE-PolyBench (116 个实例)

```bash
cd scripts
./run_poly_116.sh
```

### 4. 失败实例分析

```bash
cd scripts
python analyze_failed_instances.py
./retry_failed_instances.sh
```

## 配置说明

### 关键配置项

- `pull_timeout: 6000` - Docker 镜像拉取超时（100 分钟）
- `timeout: 60` - 命令执行超时（60 秒）
- `step_limit: 250` - 最大步数限制
- `cost_limit: 3.0` - 成本限制（美元）

### 多语言支持

配置文件已包含多语言环境变量：
- Python: `PIP_PROGRESS_BAR`, `TQDM_DISABLE`
- JavaScript/TypeScript: `NPM_CONFIG_LOGLEVEL`, `NODE_ENV`
- Java: `MAVEN_OPTS`
- Rust: `CARGO_TERM_QUIET`
- Go: `GO111MODULE`

## 使用方法

### 运行单个实例测试

```bash
python -m minisweagent.run.extra.swebench_single \
  --subset verified \
  --split test \
  --instance "astropy__astropy-13398" \
  --model openai/gpt-5 \
  --config configs/swebench_context_aware.yaml \
  --output ./test_output
```

### 批量运行

```bash
python -m minisweagent.run.extra.swebench_context_aware \
  --subset verified \
  --split test \
  --filter "^(instance1|instance2|instance3)$" \
  --workers 4 \
  --model openai/gpt-5 \
  --config configs/swebench_context_aware.yaml \
  --output ./batch_output
```

## 数据集

### SWE-Bench-Verified

- 174 个精选实例
- 实例列表: `scripts/verified_500_instance_ids.txt`
- CSV 来源: `scripts/selected_500_instances.csv`

### SWE-PolyBench

- 116 个多语言实例
  - Python: 52
  - JavaScript: 33  
  - TypeScript: 31
- 实例列表: `scripts/poly_instance_ids.txt`
- 子数据集: `scripts/poly_subset.jsonl`

## 注意事项

1. **Docker 要求**: 需要 Docker 运行，并且有足够的磁盘空间
2. **API Key**: 需要配置 OpenAI 或 Anthropic API key
3. **超时设置**: 大型项目（如 matplotlib）需要较长的 pull_timeout
4. **成本估算**: 每个实例约 $0.5-2，具体取决于复杂度

## 相关链接

- [mini-SWE-agent 原仓库](https://github.com/your-org/mini-swe-agent)
- [SWE-Bench-Verified](https://huggingface.co/datasets/princeton-nlp/SWE-Bench_Verified)
- [SWE-PolyBench](https://huggingface.co/datasets/AmazonScience/SWE-PolyBench)

## License

遵循 mini-SWE-agent 原项目的 License
