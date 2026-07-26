# 启动前准备工作

状态：进行中
适用环境：当前 Windows/WSL2 Ubuntu 实验机
生效日期：2026-07-27

> 本文件是项目唯一有效的启动准备清单。2026-07-27 的实际执行、耗时和证据记录在 `docs/prestart/2026-07-27-startup-preparation.md`。启动准备不计入 32 周正式学习时长；今天门禁通过后，可以在同一天继续执行 W01-D1。

## 1. 当前起点

当前只按实际可观察状态判断：

- 仓库位于 WSL Linux 文件系统 `/home/zhb/ai-infra-lab`，Git 仓库仍在。
- 系统为 Ubuntu 26.04 LTS / WSL2，CPU 为 AMD Ryzen 7 7840H，WSL 可见约 15 GiB 内存和 4 GiB Swap。
- 本机只有 AMD Radeon 780M，没有可用的 NVIDIA/CUDA 链路。
- Git、curl、系统 Python 3.14 和部分 Linux 观察命令可用。
- Docker、Compose、`uv`、项目 Python 3.12、GCC/G++/Make、`jq`、`iostat`、`stress-ng` 当前不可用。
- 仓库内残留 `.venv` 不含 Python 可执行文件，不得继续使用。
- GitHub SSH、Docker Hub、Hugging Face 的当前网络与认证状态均未验证。

因此项目当前处于“启动前准备中”，不能沿用任何历史 PASS 结论。

## 2. 准备原则

1. 不修改 Ubuntu 的系统 Python，不全局安装项目依赖。
2. 使用 `uv` 管理 Python 3.12，并在仓库内创建可重建的独立 `.venv`。
3. 用 `pyproject.toml`、`.python-version` 和锁文件记录环境，不把 `.venv` 当作备份。
4. 只准备 W01～W02 必需能力；vLLM、Kubernetes、Helm、CUDA、NCCL、Nsight、Triton 到对应阶段再安装。
5. 本机采用 CPU/Mock 路线；需要 NVIDIA GPU 的验收必须登记并在外部环境补做。
6. 所有秘密只放本地 `.env` 或密钥管理工具，仓库只保留 `.env.example`。
7. 每一项准备都以当前命令输出为证据，不引用旧截图、旧版本号或旧预检结果。

## 3. P0 工作清单

### 3.1 仓库、认证与备份

- [ ] 备份当前仓库和 WSL 发行版，确认恢复位置与恢复步骤。
- [ ] 复核 Git 工作区已有改动，不覆盖未提交内容。
- [ ] 配置 Git 用户身份。
- [ ] 在 WSL 内配置 GitHub SSH 或其他明确的认证方式。
- [ ] 验证 `origin` 的 fetch、pull 和 push 闭环。
- [ ] 确认模型、缓存、虚拟环境、密钥和大型 trace 均不会进入 Git。

### 3.2 Linux 基础工具

启动前至少准备并记录版本：

- [ ] `ca-certificates`、`curl`、`git`
- [ ] `procps`、`iproute2`、`lsof`
- [ ] `sysstat`（提供 `iostat`）、`jq`、`stress-ng`
- [ ] `build-essential`（GCC、G++、Make）与 `pkg-config`
- [ ] `git-lfs`

CMake 当前为可选项；进入本地扩展、vLLM 源码构建或 CUDA 阶段前再升级为必需项。

### 3.3 Python 3.12 项目环境

- [ ] 按 [uv 官方安装说明](https://docs.astral.sh/uv/getting-started/installation/) 安装 `uv` 并记录版本。
- [ ] 用 `uv` 安装和固定 Python 3.12，不替换系统 Python 3.14。
- [ ] 移除或隔离不可用的旧 `.venv`，重新创建 `.venv`。
- [ ] 新增 `.python-version`、`pyproject.toml` 和锁文件。
- [ ] 为 W01～W02 锁定 CPU 版 PyTorch、Transformers、Accelerate、pytest、psutil。
- [ ] 到 W02 再确认 FastAPI、Uvicorn、指标库和异步 HTTP 客户端的具体依赖。
- [ ] 从新终端验证 Python 路径、版本和核心依赖导入。

vLLM 不放入基础环境。进入 W03 时再按 [vLLM CPU 安装说明](https://docs.vllm.ai/en/latest/getting_started/installation/cpu/) 建独立环境或容器；本机结果只用于功能冒烟，不代表 GPU 性能。

### 3.4 Docker 与容器

- [ ] 确认采用“WSL 内 Docker Engine + systemd”，不与另一套 Docker daemon 混用。
- [ ] 按 [Docker Ubuntu 官方文档](https://docs.docker.com/engine/install/ubuntu/) 配置官方 apt 仓库。
- [ ] 安装 Engine、CLI、containerd、Buildx 和 Compose plugin。
- [ ] 确认 daemon 启动与重启策略。
- [ ] 决定是否授予当前用户 `docker` 组权限，并记录其接近 root 的安全边界。
- [ ] 验证 Docker/Compose 版本、`hello-world` 和一个受 CPU/内存限制的容器。
- [ ] 验证 Docker Hub 拉取失败时能区分认证、DNS、网络、daemon 和镜像问题。

### 3.5 模型、缓存、网络与秘密

- [ ] 在 Linux 文件系统中确定模型缓存目录，首周预留 10～20 GiB。
- [ ] 选择一个无需授权、100M～500M 参数的公开小模型。
- [ ] 记录模型 ID、revision、许可证和预计磁盘/内存占用。
- [ ] 分别验证 GitHub、Docker Hub、Hugging Face；三条链路单独留证据。
- [ ] 只在需要 gated 模型时配置 `HF_TOKEN`，日志不得输出真实 Token。
- [ ] 服务默认监听 `127.0.0.1`；需要局域网访问时另行评审端口、鉴权和防火墙。

### 3.6 GPU 与后续阶段路线

- [ ] 在 `docs/deferred-validations.md` 保留“本机无 NVIDIA/CUDA”的补验项。
- [ ] W01～W14 使用 CPU、小模型和 Mock backend 完成允许的替代验收。
- [ ] W03 前确定单 GPU 外部资源方案，用于 vLLM GPU 功能和性能补验。
- [ ] W21 前确定至少双 GPU 方案，用于 Tensor Parallel 与 NCCL。
- [ ] W27 前冻结 NVIDIA 驱动、CUDA、PyTorch、编译器、Nsight 和 Triton 的兼容矩阵。
- [ ] 外部 GPU 方案包含预算、卡型、镜像、秘密注入、结果回传和关机检查。

## 4. 新预检与证据

环境恢复后重新建立预检脚本。新脚本至少检查：

- 时间、OS、kernel、工作区路径、CPU、内存、Swap 和磁盘；
- 必需命令及其版本；
- `.venv/bin/python` 的绝对路径与 Python 3.12；
- 锁文件与核心 Python 依赖导入；
- Docker daemon、Compose、cgroup、普通容器和资源受限容器；
- Git 状态；
- NVIDIA 状态，并把“本机无 NVIDIA”标记为 EXPECTED/DEFERRED；
- 可选的 GitHub、Docker Hub、Hugging Face 网络测试，避免只读预检意外下载大文件。

新证据必须使用实际执行日期。环境基线应重新创建为 `docs/env-baseline.md`，只记录当前事实和不可用项，不复制历史 PASS。

## 5. 启动验收

以下全部通过后，才可开始 W01-D1。若在 2026-07-27 当天通过，可当天继续正式学习；若未通过，则顺延并重排后续日期：

- [ ] GitHub 同步闭环可用，工作区已有改动已妥善处理。
- [ ] 第一周所需 Linux 工具无缺项。
- [ ] Python 3.12 独立环境可从声明与锁文件重建。
- [ ] Docker、Compose、daemon、普通容器和资源受限容器通过。
- [ ] 公开小模型下载/加载通道与缓存目录通过最小冒烟。
- [ ] 新预检脚本与本文件一致，并生成新的 PASS 证据。
- [ ] CPU/Mock、本地 vLLM CPU 和外部 NVIDIA GPU 的结论边界清楚。
- [ ] `docs/progress.md` 已按实际启动日整体重排。
- [ ] 已阅读主计划 W01-D1 和每日验收规则。

## 6. 本阶段明确不做

- 不提前安装 Kubernetes、kind/minikube、Helm 和完整监控栈。
- 不在本机安装 NVIDIA CUDA/NCCL 并尝试伪造 GPU 验收。
- 不把 vLLM、Gateway、Platform、Benchmark、CUDA/Triton 全部依赖装进一个环境。
- 不在启动门禁通过前执行 W01-D1 的学习与实验内容；门禁通过后允许当天继续。
- 不把准备工作时长计入 384 小时正式计划。

## 7. 准备完成后的收尾

1. 将每项实际版本和证据写入新建的 `docs/env-baseline.md`。
2. 在新预检产物中记录 PASS/DEFERRED，不覆盖历史文件名。
3. 更新本文件状态与所有复选框。
4. 确认或重排 `docs/progress.md` 的 32 周日期。
5. 更新 `docs/prestart/2026-07-27-startup-preparation.md` 的最终状态和证据。
6. 启用正式启动日的 W01-D1 日志，再开始正式学习计时。
