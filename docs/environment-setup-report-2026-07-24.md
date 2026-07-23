# 实验机环境准备报告

报告日期：2026-07-24

适用机器：WSL2 Ubuntu 实验端

最终状态：启动前环境阻塞项已解除，仓库预检结果为 `RESULT=PASS`

## 1. 本次完成了什么

### 1.1 采集并固化环境基线

记录了实验机的宿主机、WSL、操作系统和硬件资源：

| 项目 | 结果 |
|---|---|
| Windows | 10.0.26200.8875 |
| WSL | 2.7.10.0 |
| Linux | Ubuntu 26.04 LTS，Kernel 6.18.33.2，x86_64 |
| CPU | AMD Ryzen 7 7840H，8 核 16 线程 |
| 内存 | 15 GiB，Swap 4 GiB |
| 工作区磁盘 | 约 953 GiB 可用 |
| NVIDIA/CUDA | 当前不可用，按 CPU/Mock 路线启动 |

用途：后续所有性能结论都能关联到明确环境，避免把 WSL2 CPU 实验结果误写成原生 Linux 或 GPU 结论。

### 1.2 安装容器环境

安装并验证：

- Docker Engine 29.1.3
- Docker Compose 2.40.3
- containerd 与 runc
- Docker daemon 已设置为 enabled/active
- 用户 `zhb` 已加入 `docker` 组

用途：

- Docker 用于隔离依赖、固定运行环境和设置 CPU/内存限制。
- Compose 用于后续同时启动模型服务、Gateway、Prometheus、Grafana 等多组件环境。
- containerd 负责容器生命周期管理，runc 负责创建实际的 Linux 容器进程。

完成了两类验收：

1. `hello-world` 成功拉取并运行，证明客户端、daemon、Docker Hub 和容器运行链路连通。
2. Alpine 容器设置 256 MiB 内存和 1 CPU，容器内读到：

```text
memory.max=268435456
cpu.max=100000 100000
```

这证明 cgroup 资源限制实际生效，而不只是 Docker 命令接受了参数。

### 1.3 建立 Python 独立环境

使用 `uv` 安装 Python 3.12.13，并在仓库内创建：

```text
.venv/
```

系统 Python 3.14.4 保持不变，不用于项目运行时。

用途：

- 避免修改 Ubuntu 系统 Python。
- Python 3.12 对 PyTorch、Transformers 和后续 AI 依赖通常比刚发布的 Python 3.14 有更好的兼容性。
- `.venv` 将本项目依赖与其他项目隔离，后续可以结合 `pyproject.toml` 和锁文件复现环境。

常用命令：

```bash
source .venv/bin/activate
which python
python --version
deactivate
```

### 1.4 补齐 Linux 诊断和压力工具

新增：

- sysstat 12.7.7，提供 `iostat`
- stress-ng 0.20.01

现有的 `ps`、`top`、`vmstat`、`ss`、`lsof`、GCC、G++、Make、curl、jq 等工具也通过预检。

用途：

- `iostat`：观察磁盘吞吐、设备利用率和 I/O 等待。
- `stress-ng`：制造可控 CPU、内存或 I/O 压力，用于观察系统和容器资源限制。
- `vmstat`：联合观察 CPU、运行队列、内存、Swap 和 I/O。
- `ss`、`lsof`：定位监听端口、Socket 和文件描述符。

### 1.5 建立自动预检

运行：

```bash
bash scripts/check_lab_preflight.sh
```

最终检查项全部通过：

- 工作区位于 WSL Linux 文件系统，不在 `/mnt/c`
- 第一周必需命令无缺项
- 项目 Python 为 3.12
- Docker daemon 可访问
- 内存、磁盘和 Git 状态可采集

本地证据保存在：

```text
artifacts/preflight/2026-07-24.txt
```

该目录被 `.gitignore` 排除，只在实验机保留；可提交的摘要已写入环境基线。

## 2. 安装过程中遇到的问题

### 2.1 sudo 无法由自动任务交互认证

沙箱内 `sudo` 受限制，普通提升权限调用又要求终端密码。最终使用 WSL 提供的 root 启动方式，只执行明确的 `apt-get`、服务检查和用户组配置命令，没有读取或保存用户密码。

需要理解：应用沙箱权限、Linux 用户权限和 WSL 宿主控制是三个不同层次，某一层“允许执行”不等于自动获得 Linux root。

### 2.2 组合安装长时间卡在依赖求解/下载

一次性安装 Docker、Compose、sysstat 和 stress-ng 时，`apt-get` 长时间占用 CPU且包缓存没有增长。处理方式：

1. 确认具体 apt PID、dpkg 锁和下载缓存。
2. 对卡住的进程发送温和中断，没有使用强制删除锁文件。
3. 改为逐包安装：sysstat → stress-ng → Docker Engine → Compose。
4. 设置 IPv4、连接超时和自动重试。

需要理解：不能看到 apt 锁就直接删除锁文件；应先确认持锁进程是否仍然存在，避免破坏 dpkg 状态。

### 2.3 Docker Hub 首次拉取 Alpine 返回 EOF

第一次获取 Alpine manifest 时出现临时 EOF；单独执行 `docker pull alpine:latest` 后成功，随后资源受限容器验收通过。

需要理解：一次网络失败不等于 Docker 安装失败。应区分客户端/daemon 权限问题、DNS/网络问题、镜像仓库问题和容器运行问题。

## 3. 目前具备的能力

实验机现在可以：

- 执行第一周 Linux 进程、内存、I/O、端口和文件描述符观察。
- 运行普通容器以及带 CPU/内存限制的容器。
- 在 Python 3.12 独立环境中安装和运行项目依赖。
- 用 GitHub 与学习端同步代码、文档和小型实验证据。
- 按 CPU/Mock 路线运行小模型、服务和基础压测。

当前不能据此宣称：

- 已具备 NVIDIA GPU、CUDA、vLLM GPU 或 NCCL 实验环境。
- CPU/WSL2 性能数据可以代表 GPU 或原生 Linux 服务器。
- PyTorch、Transformers 和模型依赖已经安装完成。

## 4. 建议重点回顾的内容

### 4.1 必须理解

1. WSL2 与 Windows 的边界
   - 为什么仓库和虚拟环境放在 Linux 文件系统。
   - `wsl --shutdown`、WSL 内 systemd 和 Windows 宿主机之间的关系。

2. Docker 基本链路
   - Docker CLI → Docker daemon → containerd → runc → 容器进程。
   - 容器不是虚拟机，它主要依赖 namespace 和 cgroup。
   - `docker` 用户组接近 root 权限，不应随意授予。

3. cgroup 资源限制
   - `memory.max=268435456` 为什么等于 256 MiB。
   - `cpu.max=100000 100000` 如何表示一个 CPU 配额。
   - 容器达到内存上限时与宿主机整体 OOM 的区别。

4. Python 环境隔离
   - 系统 Python 与项目 `.venv` 的区别。
   - `uv python install` 与 `uv venv` 分别做什么。
   - 如何用 `which python` 判断当前环境是否正确。

5. 实验可复现性
   - 环境版本、命令、原始输出和失败记录为什么都要保存。
   - 为什么模型、缓存、密钥和大型产物不能提交 Git。

### 4.2 建议亲手复跑

```bash
cd /home/zhb/ai-infra-lab
source .venv/bin/activate
which python
python --version

docker version
docker compose version
docker run --rm hello-world
docker run --rm --memory=256m --cpus=1 alpine sh -c \
  'cat /sys/fs/cgroup/memory.max; cat /sys/fs/cgroup/cpu.max'

iostat -c 1 2
vmstat 1 5
stress-ng --cpu 1 --timeout 10s --metrics-brief

bash scripts/check_lab_preflight.sh
```

复跑时应能解释每条命令要回答什么问题，而不只是观察到“命令成功”。

## 5. 后续未完成事项

- 验证 Hugging Face 网络和公开小模型下载，计划在 7 月 26 日冒烟测试完成。
- PyTorch、Transformers、pytest、psutil 尚未按项目版本安装；应结合首次模型实验确定版本并生成锁文件。
- CMake 当前不是启动阻塞项，在编译本地扩展或 CUDA/Triton 阶段按需安装。
- NVIDIA GPU、CUDA、vLLM GPU 和 NCCL 项目继续保留在延后补验台账。
- 当前三份状态文档和本报告尚未提交、推送。

## 6. 结论

本次工作不是简单“把软件装上”，而是完成了安装、权限、服务、网络、资源限制和自动预检的闭环。实验机已经满足 7 月 24 日运行时准备要求，可以进入知识热身与首周材料准备阶段。
