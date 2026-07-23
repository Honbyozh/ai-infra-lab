# 实验机启动前准备

> 目标：在正式执行 32 周计划前，把实验机整理成可复现、可回滚、可观测的环境。本文中的“启动前”包括一次性初始化和每次实验前检查。
>
> 范围：本文只记录承担代码运行、模型服务、容器和性能实验的主实验机。另一台学习机用于源码阅读与文档记录；两端通过同一个 GitHub 仓库同步代码、文档和小型证据，不采用手工合并。
>
> 基线采集日期：2026-07-22。版本和资源会变化，开始实验时应重新执行检查命令，不要把本文记录当作永久事实。

## 1. 当前基线与结论

| 项目 | 当前状态 | 结论 |
|---|---|---|
| 系统 | WSL2，Ubuntu 26.04 LTS，x86_64 | 可进行 Linux、容器、服务和 CPU 实验 |
| CPU | AMD Ryzen 7 7840H，8 核 16 线程 | 足够完成开发、Mock 服务和小规模 CPU 压测 |
| 内存 | 15 GiB，Swap 4 GiB | 只适合小模型；启动模型前必须估算峰值内存 |
| 工作区磁盘 | 约 953 GiB 可用 | 容量充足，但模型缓存和容器镜像仍需定期盘点 |
| Python | 3.14.4；`uv` 已安装 | 不把系统 Python 当项目环境；另建 3.11/3.12 虚拟环境以提高 PyTorch/vLLM 兼容性 |
| 基础编译工具 | Git、GCC、G++、Make 已安装 | 可完成普通源码构建 |
| 容器 | 未发现 Docker | 第 1 周容器实验前必须补齐 |
| NVIDIA GPU | 未发现 `nvidia-smi`、CUDA/NVIDIA 容器工具 | 当前按“无 NVIDIA GPU”路线执行，GPU 项登记延后补验 |
| 其他缺项 | `pip3`、CMake 未发现 | `pip3` 可由 `uv` 替代；CMake 在 CUDA/本地扩展实验前安装 |

当前开课前的两个阻塞项是：安装并验证容器运行时、建立独立且锁定版本的 Python 环境。当前按无 NVIDIA GPU 的 CPU/Mock 路线开课，GPU 专属验收登记为延后补验。

## 2. P0：开课前必须完成的一次性准备

### 2.1 明确宿主机与 WSL2 边界

- 仓库、虚拟环境、模型缓存和实验结果都放在 WSL 的 Linux 文件系统中，不放在 `/mnt/c`；大量小文件和模型加载通常会更稳定。
- 在 Windows 侧确认 WSL2 可正常关闭、重启和更新，知道如何执行 `wsl --shutdown`。
- 为 WSL2 保留足够资源。当前可见内存只有 15 GiB，不要同时运行多个模型、Kubernetes 集群和监控全家桶。
- 记录 Windows 宿主机版本、WSL 版本以及是否有独立 NVIDIA GPU，补到 `docs/env-baseline.md`。

如果宿主机确有 NVIDIA GPU，先在 Windows 安装适用于 WSL 的驱动，再在 WSL 内验证 `nvidia-smi`。不要在尚未确认驱动链路时盲目安装多套 CUDA。若没有 NVIDIA GPU，则继续 CPU/Mock 路线，并把真实 GPU 验收写入 `docs/deferred-validations.md`。

### 2.2 安装并验证容器运行时

选择一种方式并固定下来：

1. Docker Desktop + WSL integration：管理直观，适合本机实验。
2. WSL 内 Docker Engine：依赖更少，但需要自己维护 daemon、权限和开机状态。

安装后必须同时通过以下检查：

```bash
docker version
docker info
docker run --rm hello-world
docker run --rm --memory=256m --cpus=1 alpine sh -c 'cat /proc/meminfo | head'
```

同时确认当前用户运行 Docker 不需要长期使用 `sudo`。若采用 `docker` 用户组，要知道该组近似拥有 root 权限，不在多人共用机器上随意授权。

### 2.3 建立隔离的 Python 环境

系统自带 Python 3.14，部分 AI 依赖可能还没有对应 wheel。项目优先使用 Python 3.11 或 3.12，并由 `uv` 管理，不修改系统 Python：

```bash
uv python install 3.12
uv venv --python 3.12 .venv
source .venv/bin/activate
python --version
```

随后遵守以下规则：

- 项目依赖写入 `pyproject.toml`，锁文件提交到 Git。
- 安装后保存 `uv lock`，避免只记录一串无法复现的手工命令。
- PyTorch、Transformers、vLLM、CUDA 相关包先确认版本兼容，再成组安装。
- `.venv/`、模型权重、缓存、密钥和原始大体积结果必须加入 `.gitignore`。

### 2.4 补齐基础工具

至少准备：`curl`、`jq`、`lsof`、`procps`、`sysstat`、`iproute2`、`build-essential`、`git-lfs`。CMake 在编译本地扩展前补齐；Kubernetes CLI、kind/minikube、Helm 等到对应阶段再安装，避免一开始堆积未使用的软件。

安装后用命令存在性做验收：

```bash
for cmd in git curl jq lsof ps top vmstat iostat ss gcc g++ make; do
  command -v "$cmd" >/dev/null || echo "MISSING: $cmd"
done
```

### 2.5 准备目录、缓存与磁盘预算

建议先创建计划中会用到的目录，并约定大文件位置：

```text
docs/          环境基线、实验笔记、决策记录
serving/       模型与推理服务
benchmark/     压测程序；小型结果可提交
data/          本地数据集，不默认提交
models/        本地权重或软链接，不提交
artifacts/     日志、profile、截图和临时产物
```

建议至少保留 100 GiB 空闲空间作为模型、镜像和实验缓冲。下载模型前检查模型大小、文件系统剩余空间以及内存/显存能否加载；不要用“磁盘放得下”代替“模型跑得动”。

### 2.6 建立安全与恢复边界

- 只使用实验专用 API Key；密钥放 `.env` 或本机密钥管理工具，仓库只提交 `.env.example`。
- 服务默认只监听 `127.0.0.1`。确需局域网访问时，再明确防火墙、鉴权和暴露端口。
- 不运行来源不明的模型 `remote code`、镜像或一键脚本；先看来源、固定版本/摘要。
- 建立 Git 远端或其他备份，保证代码和小型报告可恢复；模型权重通常可重新下载，不作为首要备份对象。
- 压测前确认目标是本机或明确授权的实验环境，并设置并发、超时和资源上限。

## 3. P1：开始 GPU 与容器编排阶段前完成

### 3.1 NVIDIA GPU 链路（有 GPU 时）

必须依次验证：宿主机驱动 → WSL 中 `nvidia-smi` → 容器 GPU 运行时 → PyTorch → vLLM。不要跳层排障。

```bash
nvidia-smi
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
```

容器测试应使用与项目锁定版本匹配的 CUDA/PyTorch 镜像，并验证容器内能看到 GPU。验收记录至少包含 GPU 型号、显存、驱动版本、PyTorch 版本、CUDA runtime 版本和一次张量运算结果。

### 3.2 Kubernetes 与监控

- 容器基础实验通过后再安装 kind/minikube、`kubectl` 和 Helm。
- 给本地集群设 CPU、内存上限；15 GiB 内存下不要与本地模型服务同时满配运行。
- Prometheus、Grafana、DCGM Exporter 等按阶段启用，不设置为无条件常驻。
- 把端口、数据卷、资源限制和停止/清理命令写入各阶段 README。

## 4. 每次开机后的 2 分钟检查

开始实验前依次确认：

```bash
date --iso-8601=seconds
uname -a
free -h
df -h .
git status --short
uv --version
docker info --format 'Docker={{.ServerVersion}} Cgroup={{.CgroupVersion}}'
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv
```

没有 Docker 或 NVIDIA GPU 时，对应命令失败是预期状态，但必须确认这与当天任务的环境分级一致。随后检查：

- 当天实验所需端口未被占用：`ss -lntp`。
- 没有上次遗留的容器、模型进程或占用 GPU 的任务。
- 虚拟环境已激活，`which python` 指向仓库内 `.venv`。
- Git 工作区中的既有改动已识别，不覆盖尚未归档的实验结果。
- 本次实验的配置、输入数据、随机种子、预热次数和输出目录已经确定。

## 5. 每次实验结束时

1. 保存实际命令、版本、配置、原始数据和异常日志。
2. 停止不再使用的服务和容器，确认端口、CPU、内存/GPU 占用回落。
3. 将结果分成“事实、推断、待验证”，性能数据至少重复 3 次并保留原始值。
4. 更新学习计划中的完成项；有环境缺口时更新 `docs/deferred-validations.md`。
5. 检查 `git status`，确认没有密钥、权重、缓存或超大文件将被提交。

## 6. 启动门禁清单

以下各项完成后，才正式开始 W01-D1：

- [x] 当前选定无 NVIDIA GPU 的 CPU/Mock 路线；GPU 专属项目延后补验。
- [ ] Docker 可运行普通容器和带 CPU/内存限制的容器。
- [ ] 仓库内 `.venv` 使用 Python 3.11/3.12，且可从锁文件重建。
- [ ] Linux 诊断基础工具检查无缺项。
- [ ] 已创建 `docs/env-baseline.md`，记录宿主机、WSL、CPU、内存、磁盘和工具版本。
- [ ] 已准备 `.gitignore`、`.env.example` 和延后验收文档。
- [ ] 已明确模型/数据/实验产物目录与磁盘告警线。
- [ ] 已验证 Git 备份或远端可用，且仓库中没有密钥和大模型权重。

未满足的门禁不要靠口头记忆跳过；要么先补齐，要么在延后验收文档中写明原因、替代方案、补验条件和日期。
