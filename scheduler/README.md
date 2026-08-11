# 实验调度中心 (Experiment Scheduling Center)

一个纯 Python 后端，把已批准的实验计划拆分成可独立执行的任务，排入 4 个固定
槽位（本地 2 + 服务器 2），并监控、更新 MLflow 与结果表。**不需要 ChatGPT**。

## 资源规则（固定）

```
本地槽位 1：最多 24 小时
本地槽位 2：最多 24 小时
服务器槽位 1：最多 24 小时
服务器槽位 2：最多 24 小时
```

## 调度逻辑

1. 批准实验计划后，系统先生成“实验任务清单”（manifest），从选定的 `.sh`
   解析可独立实验（方法 × 数据集 × 参数组 × seed）。
2. 使用“最长任务优先、尽量负载均衡、每项 ≤24 小时”的策略，把 runs 分到 4 个槽位。
3. 若某批预计超过 24 小时，拆成两个 ≤24 小时的批次，前一批完成后自动排后续。
4. 每个分片附带 `DASHBOARD_RUN_ID / MLFLOW_EXPERIMENT_NAME`，使 MLflow 结果
   准确回流到对应槽位。
5. 服务器槽位通过 PBS（MetaCentrum）`qsub` 提交，`qstat` 轮询；未配置 host/user
   时**不会自动提交**。
6. 实现“表完整性 + 心跳 + 报告”：显示真实状态、剩余估时、结果文件路径与完整性。

## 安装

```bash
cd /Users/xiaoyuhe/Joint-Problem
uv sync   # 或 pip install -e .
```

需要额外依赖：`fastapi`, `uvicorn`, `pydantic`。

## 运行

```bash
# 启动 Web UI（默认 http://127.0.0.1:8000）
python -m scheduler serve

# 只解析脚本为 manifest（dry run）
python -m scheduler parse /path/to/run_experiments.sh

# 解析 + 生成分片计划（dry run）
python -m scheduler plan /path/to/run_experiments.sh
```

## 环境变量

| 变量 | 说明 |
|---|---|
| `SCHEDULER_BASE` | 数据/报告/日志的根目录（默认当前目录） |
| `SCHEDULER_LOCAL1_CWD` / `SCHEDULER_LOCAL2_CWD` | 本地槽位工作目录 |
| `SCHEDULER_SERVER1_HOST` / `SCHEDULER_SERVER1_USER` | 服务器 1 的 SSH 主机/用户 |
| `SCHEDULER_SERVER2_HOST` / `SCHEDULER_SERVER2_USER` | 服务器 2 的 SSH 主机/用户 |
| `SCHEDULER_SERVER1_PROJECT` | 服务器上项目目录（默认 MetaCentrum 路径） |
| `SCHEDULER_SERVER1_LICENSE` | Gurobi 许可证路径 |
| `MLFLOW_TRACKING_URI` | MLflow 追踪地址 |
| `MLFLOW_EXPERIMENT_NAME` | 默认 MLflow 实验名 |

## 目录结构

```
scheduler/
  config.py        # 固定资源规则与槽位配置
  manifest.py      # 实验计划 manifest 与 .sh 解析
  estimator.py     # 历史/启发式估时
  sharding.py      # 分片与调度
  local_worker.py  # 本地子进程槽位（PID）
  pbs_worker.py    # PBS 服务器槽位（qsub/qstat）
  integrity.py     # 心跳 + 结果表完整性
  reporting.py     # 任务报告生成
  orchestrator.py  # 调度编排
  api.py           # FastAPI 后端
  static/index.html # Web UI
  __main__.py      # CLI 入口
```

## 服务器提交确认

开始第 5 步（接入服务器提交）前，需要确认：

1. MetaCentrum 使用的是 **PBS**（本仓库现有 `.pbs` 脚本用 `#PBS` 指令与 `qsub`）。
2. 现有 `.sh` 能否按 seed/数据集/方法切分成独立运行单元（本实现已提供启发式解析，
   无法解析的脚本会在界面标记为“未知”，需要你补充任务定义）。
3. 服务器槽位需配置 `SCHEDULER_SERVER1_HOST/USER` 等环境变量，否则不会自动提交。
