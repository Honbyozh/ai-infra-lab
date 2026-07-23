# 双机环境基线

基线日期：2026-07-22。正式实验前应重新运行检查命令；本文件记录的是可追溯快照，不是永久事实。

## 1. 学习端

| 项目 | 当前值 |
|---|---|
| 角色 | 阅读、笔记、代码编辑、实验设计、结果整理与 Git 管理 |
| 设备 | Apple Silicon MacBook Pro |
| 芯片 | Apple M4 Pro，14 核 |
| 内存 | 48 GB |
| 操作系统 | macOS 26.5.2，arm64 |
| Python | 3.14.6，仅作本机辅助工具，不作为实验运行时 |
| 基础工具 | Git、Homebrew、uv、Clang、curl、jq |
| Docker / NVIDIA GPU | 不作为学习端启动要求 |

## 2. 实验端

| 项目 | 当前值 | 启动判断 |
|---|---|---|
| 角色 | Linux、容器、模型服务、测试与性能实验 |
| 系统 | WSL2，Ubuntu 26.04 LTS，x86_64 | 可执行 Linux、服务与 CPU 实验 |
| CPU | AMD Ryzen 7 7840H，8 核 16 线程 | 足够开发、Mock 服务和小规模 CPU 压测 |
| 内存 | 15 GiB，Swap 4 GiB | 仅运行小模型，避免多个重型组件并存 |
| 工作区磁盘 | 约 953 GiB 可用 | 充足；仍需管理镜像与模型缓存 |
| 系统 Python | 3.14.4 | 不作为项目环境 |
| Python 管理 | `uv` 已安装 | 待创建 Python 3.12 `.venv` |
| 编译工具 | Git、GCC、G++、Make 已安装 | CMake 后续按需安装 |
| Docker | 未发现 | 开课前阻塞项 |
| NVIDIA/CUDA | `nvidia-smi`、CUDA 与 NVIDIA 容器工具未发现 | 当前采用 CPU/Mock 路线，GPU 验收延后 |

## 3. 当前路线与资源边界

- W01～W02 以 Linux、CPU、小模型和 Mock 后端完成可执行验收。
- W03 起所有 vLLM/GPU 专属项按主计划的无 GPU 替代方案执行，并登记补验。
- 实验机同时只运行一个主要重型工作负载；15 GiB 内存下不并行运行模型、Kubernetes 和完整监控栈。
- 模型权重、虚拟环境和缓存留在实验机，不提交 Git。
- 性能结论必须标注 WSL2、CPU 路线和内存边界，不能外推为 NVIDIA GPU 结果。

## 4. 待完成项

- [ ] 记录 Windows 宿主机版本与 WSL 版本
- [ ] 安装并验证 Docker
- [ ] 创建 Python 3.12 `.venv`
- [ ] 补齐 Linux 诊断工具
- [ ] 验证 GitHub、Docker Hub 与 Hugging Face 网络访问
- [ ] 运行 `scripts/check_lab_preflight.sh` 并保存输出
