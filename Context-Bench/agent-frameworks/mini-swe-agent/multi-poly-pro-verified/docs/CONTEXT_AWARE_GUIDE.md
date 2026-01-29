# Context-Aware Agent 使用指南

## 🎯 功能概述

这个修改让 mini-swe-agent 在生成补丁前**强制要求模型输出它用于补丁生成的上下文**。

## 📋 文件清单

### 新增文件
1. **核心实现**
   - `src/minisweagent/agents/context_aware.py` - ContextAwareAgent 类
   
2. **配置文件**
   - `src/minisweagent/config/extra/swebench_context_aware.yaml` - 包含上下文提示的配置
   
3. **运行脚本**
   - `src/minisweagent/run/extra/swebench_context_aware.py` - 单实例运行脚本
   
4. **文档和示例**
   - `docs/context_aware_agent.md` - 详细文档
   - `examples/context_aware_example.py` - 使用示例
   - `test_context_aware.py` - 测试脚本

### 修改文件
1. `src/minisweagent/run/utils/save.py` - 添加上下文保存逻辑

## 🚀 快速开始

### 1. 安装包（如果还没安装）

```bash
# 进入 mini-swe-agent 目录（示例：从仓库根目录进入）
cd mini-swe-agent-extension/mini-swe-agent
pip install -e .
```

### 2. 运行测试验证安装

```bash
python test_context_aware.py
```

预期输出：
```
✓ Context extracted: 98 chars
✓ Context data structure correct
✅ All tests passed!
✓ Loaded config from ...
✓ Template 'system_template' found
...
✅ Workflow configuration validated!
```

### 3. 在单个 SWE-bench 实例上测试

```bash
# 确保设置了 API key
export ANTHROPIC_API_KEY="your-key-here"

# 运行第一个实例
python -m minisweagent.run.extra.swebench_context_aware \
  --subset lite \
  --instance 0 \
  -m anthropic/claude-sonnet-4-5-20250929 \
  -o test_output.traj.json
```

### 4. 查看提取的上下文

```bash
python -c "
import json
data = json.load(open('test_output.traj.json'))
ctx = data['info']['patch_context_data']
print(f'Context length: {ctx[\"context_length\"]} chars')
print(f'Total steps: {ctx[\"total_steps\"]}')
print(f'Total cost: \${ctx[\"total_cost\"]:.2f}')
print(f'\nContext preview:\n{ctx[\"patch_context\"][:500]}...')
"
```

## 📊 批量运行（针对你的 500 个实例）

### 方法 1: 修改现有的 swebench.py

在 `src/minisweagent/run/extra/swebench.py` 中修改：

```python
# 在文件开头添加导入
from minisweagent.agents.context_aware import ContextAwareAgent

# 在 process_instance 函数中替换 agent 创建
def process_instance(instance_id, ...):
    # 原来: agent = DefaultAgent(model, env, **config.get("agent", {}))
    # 改为:
    agent = ContextAwareAgent(model, env, **config.get("agent", {}))
    
    # 其余代码不变
```

然后运行：
```bash
mini-e swebench \
  --subset ./selected_500_instances.csv \
  -c ./src/minisweagent/config/extra/swebench_context_aware.yaml \
  -o ./results_with_context/ \
  -m anthropic/claude-sonnet-4-5-20250929
```

### 方法 2: 创建自定义批量脚本

```python
# my_batch_runner.py
import pandas as pd
from minisweagent.agents.context_aware import ContextAwareAgent
# ... 其他导入

# 读取你的 500 个实例
df = pd.read_csv('selected_500_instances.csv')

for _, row in df.iterrows():
    instance_id = row['instance_id']
    # 创建 ContextAwareAgent 并运行
    # ... 保存结果和上下文
```

## 📈 分析上下文数据

### 提取所有实例的上下文

```python
import json
from pathlib import Path
import pandas as pd

results_dir = Path('results_with_context')
contexts = []

for traj_file in results_dir.glob('*.traj.json'):
    data = json.load(open(traj_file))
    if 'patch_context_data' in data['info']:
        ctx = data['info']['patch_context_data']
        contexts.append({
            'instance_id': traj_file.stem,
            'context_length': ctx['context_length'],
            'total_steps': ctx['total_steps'],
            'total_cost': ctx['total_cost'],
            'context': ctx['patch_context']
        })

df = pd.DataFrame(contexts)
print(df.describe())
df.to_csv('contexts_analysis.csv', index=False)
```

### 统计分析

```python
# 上下文长度分布
print(f"平均上下文长度: {df['context_length'].mean():.0f} 字符")
print(f"中位数: {df['context_length'].median():.0f} 字符")

# 按语言分组（如果有语言信息）
# 分析不同语言任务的上下文使用情况
```

## 🔍 工作原理

### 交互流程

```
Agent 循环:
┌─────────────────────────────────────────────┐
│ 1. 模型执行任务                              │
│    ... 多轮交互 ...                          │
└─────────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────────┐
│ 2. 模型尝试提交:                             │
│    echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT│
└─────────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────────┐
│ 3. ContextAwareAgent 拦截                   │
│    → 抛出 ContextRequested 异常              │
│    → 返回提示: "请提供上下文"                │
└─────────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────────┐
│ 4. 模型响应并提供上下文:                     │
│    <PATCH_CONTEXT>                          │
│    详细的上下文信息...                       │
│    </PATCH_CONTEXT>                         │
│    echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT│
└─────────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────────┐
│ 5. Agent 提取并保存上下文                    │
│    → 正则提取 <PATCH_CONTEXT> 内容          │
│    → 保存到 trajectory.json                 │
│    → 正常完成提交                            │
└─────────────────────────────────────────────┘
```

## ⚙️ 配置选项

在 YAML 配置文件中：

```yaml
agent:
  # 新增字段
  context_request_template: |
    请提供你用于生成补丁的上下文...
  
  context_confirmation_template: |
    ✓ 上下文已接收 ({{context_length}} 字符)
  
  context_regex: r"<PATCH_CONTEXT>(.*?)</PATCH_CONTEXT>"
  
  save_context_to_file: true
  
  # 保留原有字段
  step_limit: 250
  cost_limit: 3.0
  # ...
```

## 💡 最佳实践

1. **首次测试**: 先在 1-2 个实例上测试，确认上下文格式符合预期
2. **监控成本**: ContextAwareAgent 会多一次 API 调用，注意预算
3. **上下文质量**: 检查模型是否提供了有意义的上下文（而非随意填充）
4. **保存原始数据**: 保留完整的 trajectory.json 以便后续分析

## 🐛 故障排除

### 问题 1: 模型没有提供上下文
- **现象**: `patch_context` 为 `null`
- **原因**: 模型没有使用 `<PATCH_CONTEXT>` 标签
- **解决**: 检查配置中的 `context_request_template` 是否清晰

### 问题 2: 提取的上下文为空
- **原因**: 正则表达式不匹配
- **解决**: 检查 `context_regex` 配置

### 问题 3: 成本过高
- **原因**: 每个任务多一次完整的模型调用
- **解决**: 考虑使用 cache control 或调整 `cost_limit`

## 📞 需要帮助？

- 查看详细文档: `docs/context_aware_agent.md`
- 运行测试: `python test_context_aware.py`
- 查看示例: `examples/context_aware_example.py`

## ✅ 验证修改成功

运行以下检查：

```bash
# 1. 文件存在性检查
ls -la src/minisweagent/agents/context_aware.py
ls -la src/minisweagent/config/extra/swebench_context_aware.yaml

# 2. 运行测试
python test_context_aware.py

# 3. 尝试单个实例（需要 API key）
python -m minisweagent.run.extra.swebench_context_aware --instance 0
```

全部成功则修改完成！🎉


