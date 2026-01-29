## 背景

本项目旨在通过agentless和magentless，获取data中的评测的traj数据。

## 项目架构
```
├── agent/
│   ├── AgentLess/
│   └── MagentLess/
├── data/
├── README.md
├── .gitignore
```

- `agent/`：包含AgentLess和MagentLess两种agent的实现。
- `data/`：存放评测所用的traj数据。

## 运行方式

### 单实例运行

运行单个实例（推荐用于测试或调试）：

```bash
# 方式1: 通过 instance_id 运行
python run_agentless_verified.py --instance_id SWE-Bench-Verified__python__maintenance__bugfix__497d4650

# 方式2: 通过 original_inst_id 运行
python run_agentless_verified.py --original_id scikit-learn__scikit-learn-25232
```

### 批量运行

运行所有实例：

```bash
python run_agentless_verified.py
```

### 查看可用实例

查看数据文件中的实例ID：

```bash
head -5 data/Verified.csv
```

## 环境要求

- 确保 OpenAI 代理服务运行在 `http://127.0.0.1:5000/v1`
- 确保已安装所有依赖包
- 确保 `agent/Agentless` 目录存在且包含必要的代码
