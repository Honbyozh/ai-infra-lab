# 启动前准备计划

执行周期：2026-07-23 至 2026-07-26。正式开课：2026-07-27。

目标：在不提前完成 W01-D1 的前提下，打通学习端、GitHub 和实验端，消除 Docker 与 Python 环境阻塞，并完成知识热身和一次端到端演练。

## 1. 当前决策

- 学习端 Mac 不运行正式实验，只承担阅读、设计、编辑、记录和结果整理。
- 实验端使用 WSL2 Ubuntu，承担 Linux、Docker、模型和性能实验。
- 当前实验环境没有可用 NVIDIA/CUDA 链路，按 CPU/Mock 路线启动。
- 真实 GPU、CUDA、NCCL 和 vLLM GPU 验收不伪造结果，统一延后补验。
- 双机通过 GitHub 仓库同步；模型、缓存、密钥和大型文件不进入 Git。

## 2. 每日执行看板

### 2026-07-23：仓库与双机链路

负责人：学习端为主，实验端配合。计划用时：60～90 分钟。

任务：

- [ ] 阅读本计划、`docs/pre-study-preparation.md` 和 `docs/lab-preflight.md`
- [ ] 在实验端确认仓库位于 WSL Linux 文件系统而非 `/mnt/c`
- [ ] 实验端克隆或更新 `git@github.com:Honbyozh/ai-infra-lab.git`
- [ ] 验证学习端提交/推送、实验端拉取的单向闭环
- [ ] 确认两台机器不同时编辑同一个每日记录
- [ ] 确认 `.gitignore` 能排除 `.venv`、密钥、模型、缓存和大型产物
- [ ] 将 Windows 与 WSL 版本补入 `docs/env-baseline.md`

验收证据：

- 两端 `git status --short` 输出
- 实验端仓库绝对路径
- 一次成功的 push/pull 对应提交号
- 更新后的环境基线

通过条件：代码同步无需手工复制，仓库中没有密钥、模型权重或缓存。

### 2026-07-24：实验端运行时

负责人：实验端。计划用时：90～120 分钟。

任务：

- [x] 选择并安装一种容器方案：WSL 内 Docker Engine
- [x] 验证普通容器以及 CPU/内存限制
- [x] 使用 `uv` 创建 Python 3.12 `.venv`
- [x] 安装 `procps`、`iproute2`、`lsof`、`sysstat`、`curl`、`jq`、`stress-ng`
- [x] 运行 `scripts/check_lab_preflight.sh`
- [x] 将检查输出保存到 `artifacts/preflight/2026-07-24.txt`，该目录只保留在实验机；在环境基线中记录摘要

核心命令：

```bash
docker version
docker run --rm hello-world
docker run --rm --memory=256m --cpus=1 alpine sh -c 'cat /proc/meminfo | head'

uv python install 3.12
uv venv --python 3.12 .venv
source .venv/bin/activate
python --version

bash scripts/check_lab_preflight.sh
```

通过条件：Docker 两类容器测试成功；`.venv/bin/python` 为 3.12；必需 Linux 命令无缺项。

### 2026-07-25：知识热身与首周材料

负责人：学习端为主。计划用时：60～90 分钟。

任务：

- [ ] 复习进程、线程、进程状态、上下文切换和 load average
- [ ] 复习虚拟内存、RSS、VSZ、Page Cache、Swap 与 OOM
- [ ] 复习文件描述符、Socket、监听端口和基础 I/O 指标
- [ ] 复习 Tokenizer、Embedding、Attention、MLP、Logits、Prefill、Decode 与 KV Cache
- [ ] 熟悉 `ps`、`top`、`free`、`vmstat`、`iostat`、`ss`、`lsof`
- [ ] 建立 `docs/w01-linux-llm-basics.md` 骨架，只写标题和待填项
- [ ] 选择一个无需授权的 100M～500M 参数小模型作为首周候选，不提前宣称可运行

通过条件：能脱稿解释至少 8 个上述概念；首周笔记只有结构和问题，没有预填实验结论。

### 2026-07-26：端到端彩排与启动门禁

负责人：双端。计划用时：90～120 分钟。

任务：

- [ ] 学习端确认 Git 工作区，推送最新准备材料
- [ ] 实验端从最新提交开始，运行预检脚本
- [ ] 在实验端运行一个受限容器并保存命令与退出状态
- [ ] 在 Python 3.12 环境导入 `torch`、`transformers`、`pytest`、`psutil`
- [ ] 做一次公开小模型下载或最小加载冒烟测试；失败时保存可诊断日志
- [ ] 将小型证据提交并推送，学习端拉取并完成核对
- [ ] 逐项检查本文件第 3 节启动门禁
- [ ] 未完成项写入阻塞记录或 `docs/deferred-validations.md`

通过条件：完成“学习端准备 → Git 推送 → 实验端执行 → 证据回传 → 学习端复核”的完整闭环。

## 3. 7 月 27 日启动门禁

必须通过：

- [ ] 双机 Git 同步闭环可用
- [ ] 环境基线包含学习端、Windows/WSL 和实验端信息
- [ ] Docker 可以运行普通容器和资源受限容器
- [ ] 实验端仓库内存在 Python 3.12 `.venv`
- [ ] 第一周 Linux 诊断工具无缺项
- [ ] 实验结果有约定的保存与回传方式
- [ ] 仓库中没有密钥、模型权重、缓存或超大文件
- [ ] 已明确 CPU/Mock 路线及其结论边界
- [ ] GPU 专属验收已进入延后补验流程
- [ ] 已阅读 Week 1 任务和每日验收规则

允许带到 W01-D1 的事项：

- PyTorch/Transformers 的具体版本锁定，可在首次模型实验时完成。
- 小模型最终选择，可在 W01-D5 前确认。
- CMake、Kubernetes、vLLM、CUDA 和监控组件按对应阶段安装。

不允许带入开课的阻塞：Docker 不可用、Python 3.12 环境无法创建、Git 无法同步、实验机基础 Linux 命令缺失。

## 4. 状态记录

| 日期 | 状态 | 实际用时 | 证据 | 阻塞/下一步 |
|---|---|---:|---|---|
| 2026-07-23 | 进行中 | 待填写 | `docs/env-baseline.md`、`.gitignore`、`.env.example`、`scripts/check_lab_preflight.sh` | 待完成双机 push/pull 和实验端路径确认 |
| 2026-07-24 | 通过 | 待补记 | `artifacts/preflight/2026-07-24.txt`、`docs/env-baseline.md` | Docker Hub 首次拉取出现一次 EOF，重试成功；Hugging Face 网络待 7 月 26 日冒烟验证 |
| 2026-07-25 | 未开始 | 0 min | - | - |
| 2026-07-26 | 未开始 | 0 min | - | - |

状态只能填写：未开始、进行中、部分完成、通过、阻塞。准备任务不计入 32 周正式学习时长。
