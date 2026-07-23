# 学习前准备指南

准备周期：2026-07-22 至 2026-07-26。正式学习从 2026-07-27 开始，准备工作不计入 W01-D1 完成度。

本项目采用双机模式：当前 Mac 作为学习与管理端，另一台电脑作为实验端。

## 1. 双机职责

| 设备 | 主要职责 | 不要求承担 |
|---|---|---|
| 本机 Mac（学习端） | 阅读、笔记、画图、实验设计、代码编辑、数据整理、进度与 Git 管理 | Docker、模型推理、压测、CUDA/NCCL 实验 |
| 另一台电脑（实验端） | Linux 观察、Docker、模型运行、测试、压测、GPU/CUDA/NCCL 实验和数据采集 | 长篇阅读与最终文档整理 |
| GitHub 仓库 | 双机同步代码、文档、配置和小型实验结果 | 模型权重、大型缓存、密钥和超大原始文件 |

实验可以在另一台电脑执行，但证据必须回到仓库或在日志中留下稳定链接。只在实验机终端中看过结果、不保存证据，不算通过验收。

## 2. 本机学习端准备

### 2.1 当前状态

2026-07-22 检查结果：

- Apple Silicon Mac，48 GB 内存，磁盘空间充足。
- Git、Homebrew、uv、Clang、curl、jq 已安装。
- 仓库已配置 GitHub 远端 `origin`。
- 本机不计划运行正式实验，因此 Docker、PyTorch、Transformers 和 NVIDIA GPU 都不是启动前必备项。

系统 Python 3.14 可以继续作为普通工具使用，但不要用它生成实验机的依赖锁定结果。模型和服务的运行时版本以实验机为准。

### 2.2 必备能力

本机需要能够完成：

```bash
git status
git diff
git pull --ff-only
git push
ssh <experiment-host>
```

启动前确认：

- Markdown 编辑、预览和 Mermaid 图能够正常使用。
- 能打开主计划、每日模板和进度表。
- GitHub SSH 凭据可用，能够拉取和推送当前仓库。
- 可以通过 SSH 或其他远程工具访问实验机。
- 清楚实验结果从实验机回传到仓库的方式。

### 2.3 本机可选项

以下工具只有确实需要时再安装：

- Python 3.12：运行数据汇总、文档生成或画图脚本。
- Docker Desktop：临时验证容器配置，不作为正式实验环境。
- PyTorch/Transformers：轻量代码检查，不作为性能结论来源。

不要为了“环境完整”在本机重复搭建实验机已有的环境。

## 3. 实验机准备

实验机的具体硬件目前未知，先完成基线采集，再决定 CPU、单 GPU 或多 GPU 路线。

### 3.1 采集环境基线

在实验机运行并将结果整理到后续创建的 `docs/env-baseline.md`：

```bash
uname -a
cat /etc/os-release
lscpu
free -h
df -h
python3 --version
git --version
docker version
docker compose version
nvidia-smi
```

无法运行的命令要记录“未安装”或“不适用”，不要删除该字段。至少确认：

- 操作系统与 CPU 架构
- CPU 核数、内存和可用磁盘
- NVIDIA GPU 型号、数量和显存；没有则明确写无
- NVIDIA Driver 与 CUDA 版本（如适用）
- Python、Git、Docker 与 Compose 版本
- 网络是否可以访问 GitHub、Docker Hub 和 Hugging Face

### 3.2 获取仓库

实验机首次准备：

```bash
git clone git@github.com:Honbyozh/ai-infra-lab.git
cd ai-infra-lab
```

如果已经克隆：

```bash
git status
git pull --ff-only
```

实验前先拉取，实验结束后再同步；两台机器不要同时修改同一个日志文件。

### 3.3 Linux 系统工具

实验机应具备第一周会用到的工具。Ubuntu/Debian 可以执行：

```bash
sudo apt update
sudo apt install -y procps iproute2 lsof sysstat curl stress-ng
```

验证：

```bash
ps --version
free -h
vmstat 1 2
iostat
ss -lnt
lsof -v
```

非 Ubuntu 系统使用对应包管理器，不要为了统一命令强行更换操作系统。

### 3.4 Docker 与 GPU 容器

Docker 安装在实验机，而不是本机。启动前至少验证：

```bash
docker version
docker compose version
docker run --rm hello-world
docker run --rm ubuntu:24.04 uname -a
```

如果实验机有 NVIDIA GPU，再验证容器 GPU 访问：

```bash
docker run --rm --gpus all nvidia/cuda:<compatible-tag> nvidia-smi
```

CUDA 镜像标签要依据实验机驱动兼容性选择，不在准备文档中固定版本。

### 3.5 Python 与模型环境

建议实验机使用独立的 Python 3.12 环境：

```bash
uv python install 3.12
uv venv --python 3.12
source .venv/bin/activate
python --version
```

首周需要时再安装：

```bash
uv pip install torch transformers accelerate pytest psutil
```

验证：

```bash
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
python -c "import transformers; print(transformers.__version__)"
```

依赖版本、CUDA 状态和设备信息以实验机输出为准，并写入环境基线。

### 3.6 模型、缓存和凭据

- 确认实验机能访问 Hugging Face。
- 首周选择无需申请权限的 100M～500M 参数公开小模型。
- 模型缓存预留至少 10～20 GB；后期根据实验机容量重新规划。
- 模型权重和缓存不提交 Git。
- Hugging Face Token、SSH 私钥和其他凭据不写入仓库或日志。

## 4. 双机学习与实验流程

每项任务按以下顺序执行：

1. 在本机阅读当天内容，用每日模板写下目标、验收项和实验命令。
2. 提交并推送本机的计划或代码，保持工作区清晰。
3. 在实验机拉取最新版本，执行实验并保存原始输出。
4. 将代码、配置、小型原始数据和测试输出提交到仓库；大型数据保存到约定位置，并在日志中写明路径和校验值。
5. 回到本机拉取结果，完成解释、自测、边界说明和学习记录。
6. 全部验收通过后更新进度表。

建议实验输出使用统一结构：

```text
reports/raw/Wxx-Dx/<run-id>/
├── environment.txt
├── command.txt
├── stdout.log
├── stderr.log
├── result.csv
└── README.md
```

模型权重、虚拟环境、容器缓存和大型 trace 文件不放进这个目录。

## 5. 知识储备

准备阶段只建立最小心智模型，不提前学完整个课程。

### 5.1 Linux 与操作系统

能够简单解释进程与线程、虚拟内存与物理内存、文件描述符与 Socket、CPU 使用率与 load average、容器与虚拟机的区别。

理解管道、重定向和退出码：

```bash
command | tee output.log
command > output.txt
echo $?
```

### 5.2 Python 与 Git

Python 重点复习函数、异常、列表/字典、文件和 JSON/CSV 读写、虚拟环境、命令行参数与 `time.perf_counter()`。

Git 至少能够使用：

```bash
git status
git diff
git pull --ff-only
git add
git commit
git push
git log
```

### 5.3 Transformer 最小心智模型

启动前能够按顺序说出：

```text
文本
  → Tokenizer
  → Token ID
  → Embedding
  → Transformer Layers
  → Logits
  → 采样下一个 Token
  → 重复生成
```

提前认识 Tokenizer、Embedding、Self-Attention、MLP、Logits、Temperature、top-p、Prefill、Decode 和 KV Cache。数学只需复习向量、矩阵乘法、点积、Softmax 和基本概率。

## 6. 准备周安排

| 日期 | 准备内容 | 建议时长 |
|---|---|---:|
| 2026-07-22 | 阅读计划和验收规则，明确双机职责 | 45～60 分钟 |
| 2026-07-23 | 测试本机 GitHub 推送和实验机远程访问；采集实验机基线 | 60～90 分钟 |
| 2026-07-24 | 在实验机验证 Linux 工具、Docker、Python 和 GPU 状态 | 60～90 分钟 |
| 2026-07-25 | 复习 Linux、Git、Python 与 Transformer 基础；设计结果回传流程 | 60～90 分钟 |
| 2026-07-26 | 在实验机做最小冒烟测试，在本机完成记录与启动检查 | 60～90 分钟 |

准备周控制在约 5～7 小时，不提前透支正式课程时间。

## 7. 启动验收清单

### 本机学习端

- [ ] 能打开计划、每日模板、进度表并完成 Markdown 记录
- [ ] 当前仓库可以正常拉取和推送 GitHub
- [ ] 能远程访问实验机，或已确定其他可靠操作方式
- [ ] 已约定实验结果回传目录和大文件保存方式
- [ ] 已完成 Linux、Git、Python 和 Transformer 的最小知识热身

### 实验端

- [ ] 已记录操作系统、CPU、内存、磁盘、Python、Docker 和 GPU 基线
- [ ] 第一周 Linux 诊断命令可以执行
- [ ] Docker 和 Compose 可用，或已明确首周不用容器的替代方案
- [ ] Python 独立环境可以创建
- [ ] PyTorch 与 Transformers 可以导入，或已有明确安装计划
- [ ] 模型下载通道正常
- [ ] 实验结果可以安全回传到仓库

### 项目管理

- [ ] 已阅读 [Week 1 启动清单](week-01-kickoff.md)
- [ ] 已了解 [每日学习记录模板](templates/daily-log.md) 的填写方式
- [ ] 实验机缺失的 GPU 或其他能力已登记到 [延后补验台账](deferred-validations.md)

全部完成后，环境状态记为“启动就绪”。实验机临时不可用时可以继续阅读和写设计，但涉及实验的当天任务不能标记“通过”。
