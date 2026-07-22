# AI Infra 32 周逐日学习计划

> 适用背景：已有后端开发经验，了解一些 Agent；目标转向 AI Platform、Model Serving 与 LLM Inference，并逐步深入 GPU、NCCL、CUDA/Triton。
>
> 时间基准：共 32 周、224 天、384 小时。Day 1 固定为周一；可从任意一个周一启动。

## 1. 路线与最终目标

```text
后端与分布式系统
  → 模型服务与 AI Platform
  → vLLM 推理
  → Kubernetes / GPU 调度
  → 性能压测与优化
  → 分布式推理 / NCCL
  → CUDA / Triton
```

完成 32 周后，应能够：

- 独立部署一个 OpenAI 兼容的开源大模型服务，并支持流式输出、取消、限流和鉴权。
- 使用 vLLM 管理并发、批处理、KV Cache、量化与张量并行等关键配置。
- 使用 Kubernetes 部署、升级、监控和扩缩容 GPU 模型服务。
- 建设具备模型模板、配额、状态管理和可观测性的 Mini LLM Platform。
- 自己编写压测工具，测量 TTFT、TPOT、ITL、吞吐、错误率及 P50/P95/P99。
- 用数据定位 CPU、GPU 计算、显存容量、显存带宽、通信、网络或存储瓶颈。
- 解释并实验 Tensor Parallel、NCCL 集合通信及单机多卡扩展效率。
- 编写并验证基础 CUDA Kernel 和 Triton Kernel，完成一次可复现的算子优化。

## 2. 每周固定时间结构

| 星期 | 时长 | 默认形式 | 每周累计 |
|---|---:|---|---:|
| 周一 | 1 小时 | 概念与原理 | 1 小时 |
| 周二 | 1 小时 | 文档、源码或机制 | 2 小时 |
| 周三 | 1 小时 | 设计、推导与笔记 | 3 小时 |
| 周四 | 1 小时 | 小实验 | 4 小时 |
| 周五 | 1 小时 | 小实验与测试 | 5 小时 |
| 周六 | 5 小时 | 主项目实现 | 10 小时 |
| 周日 | 2 小时 | 测试、复盘、归档 | 12 小时 |

执行规则：

1. 工作日严格控制在 1 小时；未完成的“扩展项”不能挤占睡眠或下一阶段。
2. 周六使用 `50 分钟专注 + 10 分钟休息`，完成 5 个专注单元。
3. 周日必须留下可检查的证据：代码、测试输出、截图、指标数据或学习笔记。
4. 每个实验都记录环境：操作系统、CPU、内存、GPU、驱动、CUDA、Python、框架与模型版本。
5. 所有性能结论至少重复 3 次，报告中写中位数，并保留原始数据。

## 3. 项目与目录建议

建议用一个 Git 仓库贯穿全程：

```text
ai-infra-lab/
├── docs/                 # 概念笔记、架构决策、周报
├── gateway/              # LLM Gateway
├── serving/              # Transformers / vLLM 启动与配置
├── platform/             # Mini LLM Platform
├── deploy/               # Docker、Compose、Kubernetes、Helm
├── benchmark/            # 负载生成、指标统计、原始数据
├── monitoring/           # Prometheus、Grafana、告警规则
├── distributed/          # NCCL / torch.distributed 实验
├── kernels/              # CUDA / Triton 实验
└── reports/              # 阶段报告与最终作品集
```

贯穿全程的三个项目：

- **LLM Gateway**：OpenAI 兼容、鉴权、限流、配额、流式转发、超时、取消、重试、熔断、路由、计量与追踪。
- **Mini LLM Platform**：模型部署、GPU 配置、Kubernetes 资源生成、状态管理、日志、监控、升级、扩缩容和权限。
- **LLM Benchmark Lab**：并发负载、长度分布、TTFT/TPOT/ITL、吞吐、分位数、GPU 指标、配置对比和自动报告。

## 4. 环境分级与替代方案

- **无 NVIDIA GPU**：第 1～14 周可先用小模型、CPU、Mock 推理后端、Docker Desktop 和 kind/minikube；GPU 专属验收标注为“延后补验”。
- **单张 NVIDIA GPU**：可完成绝大多数模型服务、监控、性能分析和 CUDA/Triton 内容。
- **两张及以上 NVIDIA GPU**：完成 Tensor Parallel、NCCL、拓扑与扩展效率的完整验收。
- **没有长期多卡资源**：第 21～26 周集中租用短时云 GPU；先在本地写好脚本、测试与数据模板，云上只执行实验。

“环境不可用”不能直接算完成。允许使用替代验收，但必须在 `docs/deferred-validations.md` 记录原验收、替代证据和计划补验日期。

## 5. 统一验收规则

每项任务只有同时满足以下条件才标记为完成：

- **能复现**：从全新终端按 README 操作可以得到相同结果。
- **有证据**：至少保存代码提交、测试输出、指标文件、截图或笔记中的一种。
- **能解释**：不看资料，用自己的话回答任务中的关键“为什么”。
- **有边界**：写出至少一个限制、失败场景或仍未解决的问题。

建议在每日结束时追加：

```markdown
- [x] Wxx-Dx
  - 用时：55 min
  - 产物：docs/linux-process-memory.md
  - 证据：commit abc123 / test log / screenshot
  - 自测：3/3
  - 未决问题：……
```

## 6. 32 周逐日计划

### 阶段一：AI Infra 必需基础与最小模型服务（第 1～4 周）

#### 第 1 周：Linux 系统诊断与 LLM 推理基础

**本周目标：** 建立进程、内存、I/O、容器资源和 Transformer 推理的共同心智模型，能观察一个小模型从文本输入到逐 Token 输出的过程。  
**本周交付物：** `docs/w01-linux-llm-basics.md`、`serving/run_local_model.py`、`docs/env-baseline.md`、一组可复现的系统观察记录。

| 天 | 时长 | 学习任务 | 具体操作步骤 | 检测与验收 |
|---|---:|---|---|---|
| W01-D1（第 1 天，周一） | 1h | 进程、线程与调度基础 | 1. 阅读 Linux 进程、线程、上下文切换和 load average 的资料。<br>2. 用 `ps -eo pid,ppid,stat,nlwp,comm`、`top` 观察一个现有后端进程。<br>3. 将进程、线程、协程的关系画成一张小图。 | 1. `docs/w01-linux-llm-basics.md` 中有图和命令输出摘要。<br>2. 能脱稿解释 load average 不等于 CPU 使用率，并举出一个高 load、低 CPU 的场景。<br>3. 保存至少一个目标进程的 PID、线程数和状态作为证据。 |
| W01-D2（第 2 天，周二） | 1h | 虚拟内存、Page Cache 与 OOM | 1. 学习 RSS、VSZ、匿名页、文件页和 swap。<br>2. 运行一个逐步分配内存的小程序，前后记录 `free -h`、`vmstat 1 5` 和进程 RSS。<br>3. 阅读 `/proc/<PID>/status` 中 `VmRSS`、`VmSize`、`Threads`。 | 1. 笔记中包含“虚拟内存、物理内存、显存”三者区别表。<br>2. 提交实验前后数据，并能解释 RSS 与 VSZ 不一致的原因。<br>3. 能回答 OOM Killer 与容器 cgroup OOM 的触发边界有何不同；不确定项明确标注待验证。 |
| W01-D3（第 3 天，周三） | 1h | 磁盘、网络 I/O 与文件描述符 | 1. 用 `iostat` 或等价工具观察磁盘吞吐与等待。<br>2. 启动本地 HTTP 服务，用 `ss -lntp` 定位监听端口，用 `lsof -p <PID>` 查看文件描述符。<br>3. 总结延迟可能落在 CPU、磁盘、网络、排队中的哪一段。 | 1. 保存监听端口、socket 状态和 FD 数量。<br>2. 笔记列出至少 5 个排障命令及各自能回答的问题。<br>3. 口述：连接超时、读取超时、服务端排队分别应先查什么指标，三题至少答对两题。 |
| W01-D4（第 4 天，周四） | 1h | cgroup、容器资源限制小实验 | 1. 查看本机 cgroup 版本。<br>2. 给容器设置 CPU 与内存限制，容器内运行 CPU/内存压力程序。<br>3. 对比宿主机与容器内看到的 CPU、内存和退出状态。 | 1. 保存 `docker inspect` 中资源限制和容器退出码。<br>2. 内存超限实验能观察到非零退出或 OOM 事件；若环境不支持，记录到 `docs/deferred-validations.md` 并用 cgroup 文件截图替代。<br>3. 能解释 requests、limits 与实际用量不是同一概念。 |
| W01-D5（第 5 天，周五） | 1h | Transformer、Tokenizer 与 Attention 推理链 | 1. 复习 Tokenizer、Embedding、Self-Attention、MLP、LM Head。<br>2. 选一个小模型，对同一句中英文分别 tokenize。<br>3. 记录 token ID、token 数和 decode 后文本。 | 1. `serving/run_local_model.py` 可打印 token ID、输入长度及解码结果。<br>2. 连续运行两次结果可复现。<br>3. 能按顺序口述一次 next-token prediction 的数据流，并说明 tokenizer 不是模型本体。 |
| W01-D6（第 6 天，周六） | 5h | 主实验：跑通小模型并观察 Prefill/Decode | 1. 记录 OS、CPU、内存、Python、PyTorch、Transformers、模型版本；有 GPU 时追加驱动、CUDA 和 GPU。<br>2. 加载可承受的小型 causal LM，完成非流式生成。<br>3. 使用 streamer 或逐步生成方式观察首 Token 与后续 Token。<br>4. 分别测试短提示词和长提示词，记录输入/输出 Token、首 Token 时间、总耗时和峰值内存；每组至少 3 次。<br>5. 在笔记中解释 Prefill、Decode、KV Cache、上下文长度及精度格式。 | 1. `python serving/run_local_model.py --prompt "hello"` 正常退出并生成文本。<br>2. `docs/env-baseline.md` 字段完整，不可获取项写明原因。<br>3. 原始数据至少 6 条，报告中采用中位数；长输入的 Prefill 变化有数据佐证。<br>4. 能画出请求生命周期，并正确指出 KV Cache 随层数、序列长度和并发增加而增长。<br>5. 无 GPU 时完成 CPU 实验，并把 GPU 显存观察列为延后补验。 |
| W01-D7（第 7 天，周日） | 2h | 测试、复盘与本周归档 | 1. 清理脚本参数和依赖说明。<br>2. 从新终端按 README 重跑关键命令。<br>3. 写 Week 1 Review，整理本周未决问题。<br>4. 进行 10 分钟脱稿自测。 | 1. `python serving/run_local_model.py --help` 可用，README 能让自己从零复现。<br>2. `docs/w01-linux-llm-basics.md` 回答 CPU 100%、内存满、GPU 显存满但利用率低各至少 2 个可能原因。<br>3. 周报列出产物、实际用时、3 个掌握点、2 个薄弱点、下周动作。 |

#### 第 2 周：Transformers 最小模型服务

**本周目标：** 把本地模型封装为具备流式输出、取消、并发限制、Token 统计和健康检查的 HTTP 服务。  
**本周交付物：** `serving/transformers_service/`、Dockerfile、API/并发测试、可复现 README。

| 天 | 时长 | 学习任务 | 具体操作步骤 | 检测与验收 |
|---|---:|---|---|---|
| W02-D1（第 8 天，周一） | 1h | 整理模型加载与生成模块 | 1. 把上周脚本拆分为模型加载、请求校验、生成和 Token 统计函数。<br>2. 增加模型路径、最大输入长度、最大输出 Token 等配置。<br>3. 为纯函数补最小单元测试。 | 1. 模型只在进程启动时加载一次，可通过日志计数证明。<br>2. `pytest -q serving/transformers_service/tests` 至少 3 个测试通过。<br>3. 超长输入得到明确错误而不是进程崩溃。 |
| W02-D2（第 9 天，周二） | 1h | FastAPI 接口与生命周期设计 | 1. 定义 `/healthz`、`/readyz`、`/v1/generate` 的请求响应模型。<br>2. 使用应用 lifespan 加载模型。<br>3. 区分存活、就绪与业务错误。 | 1. OpenAPI 页面能显示三类接口和字段约束。<br>2. 未加载完成时 readiness 非 200，加载后返回 200。<br>3. 无效参数返回 4xx 且包含稳定错误码。 |
| W02-D3（第 10 天，周三） | 1h | 流式协议与背压设计 | 1. 比较 SSE、分块 JSON 和 WebSocket，选定一种流式协议。<br>2. 定义 chunk、结束、错误事件格式。<br>3. 写出客户端慢读、断连时的资源释放策略。 | 1. `docs/streaming-protocol.md` 含协议示例和选择理由。<br>2. 文档明确每个事件至少含 request ID、文本增量和结束原因。<br>3. 能解释“服务端生成快于客户端消费”时可能发生的背压问题。 |
| W02-D4（第 11 天，周四） | 1h | 实现流式输出与请求级计时 | 1. 将模型输出接入 `StreamingResponse`。<br>2. 记录接收时间、首 chunk 时间、结束时间。<br>3. 编写流式客户端，逐 chunk 打印时间戳。 | 1. 客户端在完整响应结束前能收到至少 2 个 chunk。<br>2. 日志含 request ID、TTFT、总耗时、输入/输出 Token。<br>3. 自动测试验证最终拼接文本非空且结束事件仅一次。 |
| W02-D5（第 12 天，周五） | 1h | 取消与并发限制 | 1. 增加信号量限制活跃生成数。<br>2. 客户端主动断开后检测 disconnect，并停止/丢弃后续生成。<br>3. 定义并发满时排队或拒绝策略。 | 1. 并发测试中活跃生成数从不超过配置值。<br>2. 取消请求后，活跃计数在约定超时内回落为 0。<br>3. 代码测试覆盖正常、满载、取消三条路径，`pytest -q` 全通过。 |
| W02-D6（第 13 天，周六） | 5h | 主项目：完成服务、容器化与故障处理 | 1. 完成同步与流式接口，统一错误模型。<br>2. 增加超时、并发、Token 统计、结构化日志和健康检查。<br>3. 编写多阶段 Dockerfile 与 `.dockerignore`。<br>4. 构建镜像并以受限 CPU/内存运行；配置 volume/cache，避免每次重复下载权重。<br>5. 编写冒烟、流式、取消、超长输入、并发上限测试。 | 1. `docker build` 成功且镜像可启动；`curl http://localhost:8000/healthz` 返回 200。<br>2. 五类测试均有自动化脚本和退出码，失败时返回可诊断日志。<br>3. `docker inspect` 能看到配置的 CPU/内存限制。<br>4. 日志可用同一 request ID 串起接收、首 Token、完成或取消事件。<br>5. README 从构建到调用无需依赖未记录的手工步骤。 |
| W02-D7（第 14 天，周日） | 2h | 端到端验收与复盘 | 1. 删除运行中的测试容器后，从 README 重建并启动。<br>2. 顺序执行全部测试并保存输出。<br>3. 写 Week 2 Review 和已知限制。 | 1. 冷启动后 `/readyz` 状态按设计变化，最终可请求。<br>2. `pytest -q` 与 E2E 脚本均为 0 退出码。<br>3. `docs/w02-review.md` 写明取消的实现边界、CPU 推理限制和下一周迁移 vLLM 的接口复用点。 |

#### 第 3 周：vLLM 入门与基础压测

**本周目标：** 跑通 vLLM 的 OpenAI 兼容服务，理解其请求路径，并建立可重复的 TTFT、总延迟和吞吐基线。  
**本周交付物：** `serving/vllm/` 启动配置、`benchmark/basic_bench.py`、原始结果 CSV、Transformers/vLLM 初步对比。

| 天 | 时长 | 学习任务 | 具体操作步骤 | 检测与验收 |
|---|---:|---|---|---|
| W03-D1（第 15 天，周一） | 1h | vLLM 定位与架构概览 | 1. 阅读所用版本的 vLLM 官方入门和架构资料。<br>2. 画出 API Server、Engine、Scheduler、Worker、Model Runner 的关系。<br>3. 将 Transformers 单请求执行与 vLLM 批处理执行作对比。 | 1. `docs/vllm-request-path.md` 有一张调用链图。<br>2. 能脱稿说明 vLLM 主要解决的 3 个服务问题。<br>3. 所有版本相关结论都注明本地 `vllm --version` 或资料版本。 |
| W03-D2（第 16 天，周二） | 1h | 环境检查与可复现启动配置 | 1. 检查 Python、PyTorch、驱动、CUDA、GPU 显存和模型大小兼容性。<br>2. 查看 `vllm serve --help`，只使用当前版本存在的参数。<br>3. 写启动脚本和环境检查脚本。 | 1. `serving/vllm/check_env.sh` 能以 0 或带原因的非 0 退出。<br>2. 保存版本输出与 GPU/无 GPU 判定。<br>3. 无 GPU 时在 `docs/deferred-validations.md` 登记云 GPU 补验，并继续完成客户端与 Mock 接口验收。 |
| W03-D3（第 17 天，周三） | 1h | OpenAI 兼容接口与参数映射 | 1. 启动 vLLM 小模型服务。<br>2. 调用模型列表、Chat Completions 或 Completions 接口。<br>3. 测试 temperature、top_p、max_tokens、stop 的映射。 | 1. 模型列表接口能返回预期模型 ID。<br>2. 固定 seed 且使用确定性参数时，两次输出满足设定的可重复条件。<br>3. 无效模型名与超长输入得到可识别的 4xx，而不是服务崩溃。 |
| W03-D4（第 18 天，周四） | 1h | 流式、取消与并发观察 | 1. 用 OpenAI 兼容客户端发起流式请求。<br>2. 收到若干 Token 后主动关闭连接。<br>3. 并发发起 2～4 个不同长度请求，记录到达与完成顺序。 | 1. 流式客户端能记录首 chunk 时间和结束原因。<br>2. 取消后服务中的运行/等待请求数最终回落；保存指标或日志证据。<br>3. 数据能显示短请求不必等待所有长请求完全结束后才进入系统。 |
| W03-D5（第 19 天，周五） | 1h | 基础压测指标与测量方法 | 1. 定义 TTFT、E2E latency、输出 Token 吞吐、请求吞吐和错误率。<br>2. 规定计时点、预热次数、重复次数和超时。<br>3. 设计 CSV schema。 | 1. `benchmark/README.md` 给出每个指标的公式和单位。<br>2. CSV 至少包含 request ID、输入/输出 Token、并发、TTFT、E2E、状态。<br>3. 能说明客户端测量与服务端测量为何可能不同。 |
| W03-D6（第 20 天，周六） | 5h | 主实验：编写压测器并建立基线 | 1. 实现异步流式压测客户端，准确识别首有效 Token。<br>2. 支持并发 1、2、4 和固定输入/输出长度；增加预热和超时。<br>3. 对 Transformers 服务与 vLLM 各执行至少 3 轮。<br>4. 保存原始 CSV，计算中位数、P50/P95、吞吐和错误率。<br>5. 同步采集 CPU、内存；有 GPU 时采集利用率与显存。 | 1. `python benchmark/basic_bench.py --help` 显示模型地址、并发、输入/输出长度、轮数等参数。<br>2. 单元测试用伪流式响应验证 TTFT 不等于总耗时。<br>3. 每种后端和并发组合至少有 3 轮有效数据，原始结果不可手改。<br>4. 汇总表可由脚本从 CSV 重建，且明确失败请求是否进入分位数。<br>5. 没有 vLLM GPU 环境时先完成 Mock/Transformers 基线，并保留一条可直接在云 GPU 执行的命令。 |
| W03-D7（第 21 天，周日） | 2h | 数据校验与初步结论 | 1. 检查 Token 计数、异常值、预热数据和失败请求处理。<br>2. 画并发—TTFT、并发—吞吐两张图。<br>3. 写 Week 3 Review，区分事实、推断和待验证假设。 | 1. 图可由脚本重新生成，轴、单位、样本数齐全。<br>2. 随机抽查 3 个请求，原始时间戳与计算结果一致。<br>3. 报告至少写出一个“吞吐改善但尾延迟变差”的可能机制，且不把小样本结果夸大为普遍结论。 |

#### 第 4 周：可观测性与第一阶段里程碑

**本周目标：** 为两类模型服务建立请求、运行时和硬件三层指标，完成最小模型服务第一里程碑。  
**本周交付物：** Prometheus 配置、Grafana Dashboard、告警草案、Transformers/vLLM 对比报告、可一键复现的里程碑包。

| 天 | 时长 | 学习任务 | 具体操作步骤 | 检测与验收 |
|---|---:|---|---|---|
| W04-D1（第 22 天，周一） | 1h | 指标类型与标签设计 | 1. 复习 Counter、Gauge、Histogram、Summary。<br>2. 为请求数、错误数、活跃/排队请求、TTFT、E2E、Token 数设计指标。<br>3. 审查模型名、状态码等标签的基数风险。 | 1. `monitoring/metrics-spec.md` 包含名称、类型、单位、标签和采集点。<br>2. 指标名遵循统一命名与秒/字节等基础单位。<br>3. request ID、用户输入等高基数字段未被用作指标标签。 |
| W04-D2（第 23 天，周二） | 1h | 应用指标接入 Prometheus | 1. 在 Transformers 服务暴露 `/metrics`。<br>2. 配置 Prometheus scrape target。<br>3. 发正常、错误、取消请求并查询指标。 | 1. Prometheus Targets 页面显示服务为 UP。<br>2. 三类请求后计数器增量符合预期，Histogram bucket 有数据。<br>3. 重启服务后能说明进程内 Counter 重置带来的查询注意事项。 |
| W04-D3（第 24 天，周三） | 1h | GPU 与系统指标采集 | 1. 确认可用的 CPU/内存采集方式。<br>2. 有 NVIDIA GPU 时接入 DCGM Exporter 或采样 `nvidia-smi`；无 GPU 时准备配置并登记补验。<br>3. 定义 GPU 利用率、显存、功耗与温度面板。 | 1. 至少 CPU、进程内存能形成时间序列。<br>2. GPU 环境下查询能返回利用率和显存；无 GPU 时配置通过语法检查且 deferred 记录完整。<br>3. 能解释“显存占满但 GPU 利用率低”至少 3 种原因。 |
| W04-D4（第 25 天，周四） | 1h | Grafana Dashboard V1 | 1. 建请求量、错误率、TTFT、E2E、Token 吞吐、活跃/排队数面板。<br>2. 增加 backend/model 变量。<br>3. 配置时间范围和单位。 | 1. Dashboard JSON 已纳入 `monitoring/grafana/`。<br>2. 发起一次压测后每个核心面板有数据且单位正确。<br>3. 随机选一个点，Grafana 数值与 Prometheus 查询基本一致。 |
| W04-D5（第 26 天，周五） | 1h | SLO 与告警草案 | 1. 为可用性、P95 TTFT、错误率、队列长度定义初始目标。<br>2. 写告警表达式、持续时间、严重级别和处理提示。<br>3. 人工触发一个低风险告警。 | 1. 告警规则通过 `promtool check rules` 或等价校验。<br>2. 至少一个测试告警从 pending 进入 firing，再在恢复后消失。<br>3. 文档注明阈值只是基线，需用真实负载校准。 |
| W04-D6（第 27 天，周六） | 5h | 主项目：整合服务、监控、压测与对比 | 1. 用 Compose 或脚本启动服务、Prometheus、Grafana。<br>2. 执行短输入短输出、长输入短输出、短输入长输出三类负载。<br>3. 对 Transformers/vLLM 运行相同协议与并发；每组至少 3 次。<br>4. 保存原始 CSV、Dashboard 截图和服务日志。<br>5. 写 `reports/milestone-01.md`：环境、方法、结果、瓶颈、限制和下一步。 | 1. 一条 documented 命令可启动全栈，健康检查均通过。<br>2. 报告含 TTFT、P95 E2E、Token 吞吐、错误率和资源利用对比；数值均能追溯到 CSV。<br>3. 指标面板能观察到压测开始、峰值和结束。<br>4. 对至少一个异常点给出日志/指标证据，不凭感觉解释。<br>5. 若 vLLM 尚未补验，报告明确标为未完成，不能用 Mock 数据冒充真实性能。 |
| W04-D7（第 28 天，周日） | 2h | 第一阶段正式验收 | 1. 按全新使用者视角执行 README。<br>2. 跑单元、集成、E2E 和监控冒烟测试。<br>3. 检查秘密、模型缓存、生成文件是否正确忽略。<br>4. 写 Week 4 Review 与里程碑清单。 | 1. 服务、容器、流式、取消、并发和指标六项全部有通过证据。<br>2. `git status` 不出现密钥、大模型权重或不应提交的原始缓存。<br>3. 里程碑清单逐项链接到代码、测试、Dashboard 或报告；任何 deferred 项都有日期和补验条件。 |

### 阶段二：模型服务工程与 vLLM 深入（第 5～8 周）

#### 第 5 周：vLLM 核心配置与容量边界

**本周目标：** 掌握采样、上下文、显存预算、并发、前缀缓存和量化等核心配置，并用实验而非经验拍脑袋选择参数。  
**本周交付物：** `serving/vllm/configs/`、配置矩阵实验数据、`docs/vllm-capacity-guide.md`。

| 天 | 时长 | 学习任务 | 具体操作步骤 | 检测与验收 |
|---|---:|---|---|---|
| W05-D1（第 29 天，周一） | 1h | 采样参数与输出可重复性 | 1. 实验 temperature、top_p、top_k、seed、stop 和 max_tokens。<br>2. 为确定性与创造性场景各写一套配置。<br>3. 记录参数冲突或版本限制。 | 1. 测试脚本能批量发送至少 6 组采样配置。<br>2. 确定性配置重复 3 次满足预设一致性；随机配置能观察到差异。<br>3. 能解释 max_tokens 与上下文上限之间的约束。 |
| W05-D2（第 30 天，周二） | 1h | 权重、激活与 KV Cache 显存预算 | 1. 根据参数量和 dtype 粗估权重显存。<br>2. 记录加载后的实际显存。<br>3. 将剩余空间分成运行时开销与 KV Cache 预算。 | 1. `docs/vllm-capacity-guide.md` 有估算公式、假设与实测。<br>2. 估算与实测误差被量化并解释至少 2 个来源。<br>3. 能回答为什么提高 GPU memory utilization 可能增加容量也可能导致 OOM。 |
| W05-D3（第 31 天，周三） | 1h | 最大上下文与最大并发配置 | 1. 从本地 `--help` 找到上下文、并发序列和批 Token 相关参数。<br>2. 设计 2×2 小矩阵：短/长上下文与低/高并发。<br>3. 预测各组合对容量和延迟的影响。 | 1. 保存配置矩阵和事前预测，不允许实验后改预测。<br>2. 每个参数写明单位、默认值来源和所用版本。<br>3. 启动前即可通过预算解释至少一个不可行组合。 |
| W05-D4（第 32 天，周四） | 1h | Prefix Caching 实验设计 | 1. 构造共享长前缀与完全不同前缀两组请求。<br>2. 规定 cache 开/关、冷/热运行顺序。<br>3. 决定观察 TTFT、吞吐和缓存指标。 | 1. 数据集包含相同前缀组与对照组，长度分布一致。<br>2. 实验方案能隔离预热与缓存效果。<br>3. 能说明 prefix caching 主要减少哪一阶段的重复工作及不适用场景。 |
| W05-D5（第 33 天，周五） | 1h | 量化基础与可行性检查 | 1. 梳理 FP16/BF16、INT8/INT4 权重或 KV Cache 量化的目标与代价。<br>2. 检查模型、GPU和当前 vLLM 版本支持矩阵。<br>3. 选一项可执行量化方案或写延后补验。 | 1. `docs/quantization-decision.md` 明确选择、兼容证据、预期收益与质量风险。<br>2. 启动前检查能阻止明显不兼容组合。<br>3. 能解释显存下降不保证延迟一定下降。 |
| W05-D6（第 34 天，周六） | 5h | 主实验：配置矩阵与容量拐点 | 1. 固定模型、硬件和数据集，准备基础、长上下文、高并发、缓存开/关、量化等可行配置。<br>2. 每个配置启动后保存完整命令、日志和加载显存。<br>3. 对每组运行预热和至少 3 轮压测。<br>4. 逐级增加并发，直到错误率、排队或 P95 明显越过预设阈值。<br>5. 汇总 TTFT、TPOT、吞吐、显存、错误率与容量边界。 | 1. `serving/vllm/configs/` 中每个配置可独立启动，启动参数均由本地版本验证。<br>2. 原始数据包含成功及失败点，不能只保留最好结果。<br>3. 至少找出一个服务饱和或 OOM 临界点，并能从队列、显存或日志解释。<br>4. Prefix Cache 对照严格共享相同负载；量化若执行，需同时报告质量抽查。<br>5. 无真实 GPU 的配置项在 deferred 文档逐项登记，不计入真实性能结论。 |
| W05-D7（第 35 天，周日） | 2h | 形成配置选择指南 | 1. 校验所有汇总数值可追溯。<br>2. 写场景化建议：交互低延迟、批量吞吐、长上下文。<br>3. 复盘预测与实测差异。 | 1. `docs/vllm-capacity-guide.md` 对三个场景各给参数起点、监控项和回退条件。<br>2. 报告至少保留一个失败配置及原因。<br>3. 能在白板上根据模型大小、显存和上下文给出粗略容量判断，并明确假设。 |

#### 第 6 周：Continuous Batching、KV Cache 与 PagedAttention

**本周目标：** 从调度与显存管理角度理解 vLLM 的吞吐优势，能够估算 KV Cache、解释碎片，并通过实验观察批处理取舍。  
**本周交付物：** 调度模拟器、KV Cache 估算器、源码阅读笔记、Continuous Batching 实验报告。

| 天 | 时长 | 学习任务 | 具体操作步骤 | 检测与验收 |
|---|---:|---|---|---|
| W06-D1（第 36 天，周一） | 1h | Static 与 Continuous Batching | 1. 用时间线画出 3 个不同输出长度请求在静态批处理中的空槽。<br>2. 再画连续批处理中新请求插入过程。<br>3. 总结吞吐、单请求延迟和公平性的取舍。 | 1. `docs/continuous-batching.md` 有两张同一负载的时间线。<br>2. 可根据图计算空闲 step 数。<br>3. 能解释为何 continuous batching 不是“所有请求同时开始”。 |
| W06-D2（第 37 天，周二） | 1h | 编写最小调度模拟器 | 1. 用离散 step 模拟请求到达、prefill、decode 与完成。<br>2. 实现 static 和 continuous 两种策略。<br>3. 输出每请求 TTFT/E2E 与总体槽位利用率。 | 1. `python serving/simulate_scheduler.py` 对固定种子输出可复现。<br>2. 单元测试覆盖请求中途完成与新请求加入。<br>3. 同一负载下两策略的利用率差异可由时间线人工核对。 |
| W06-D3（第 38 天，周三） | 1h | KV Cache 组成与容量估算 | 1. 从层数、KV heads、head dimension、dtype bytes、Token 数和并发推导近似公式。<br>2. 实现命令行估算器。<br>3. 用两个模型配置交叉检查。 | 1. `python serving/estimate_kv_cache.py --help` 可用。<br>2. 至少 3 个手算样例与程序输出误差在舍入范围内。<br>3. 明确 GQA/MQA 与 MHA 对 KV Cache 的影响，公式假设写入 README。 |
| W06-D4（第 39 天，周四） | 1h | 碎片与 PagedAttention | 1. 用可变长度请求模拟连续内存预留造成的内部浪费。<br>2. 改用固定块按需分配并计算块尾浪费。<br>3. 说明 block table 如何把逻辑 Token 映射到物理块。 | 1. `serving/simulate_kv_blocks.py` 输出连续分配与分页分配的占用/浪费。<br>2. 两个手工案例可核对计算。<br>3. 能说明 PagedAttention 不能消除所有碎片，也不等同于操作系统换页。 |
| W06-D5（第 40 天，周五） | 1h | 按问题阅读 vLLM 源码 | 1. 锁定已安装版本/tag。<br>2. 从请求对象追到 Scheduler，再定位 KV Cache/Block 管理相关模块。<br>3. 只记录关键类、输入、输出和状态变化。 | 1. `docs/vllm-source-map.md` 标明版本/commit、文件路径和关键符号。<br>2. 画出 waiting、running、finished 等状态迁移或当前版本等价状态。<br>3. 至少一个理解由源码行或单元测试验证，而不是只靠博客。 |
| W06-D6（第 41 天，周六） | 5h | 主实验：混合长度负载下的调度与缓存 | 1. 生成短、中、长三档输入/输出的混合数据集，并固定随机种子。<br>2. 选择低、中、高三档并发，压测 vLLM。<br>3. 采集 TTFT、TPOT/ITL、E2E、吞吐、队列、KV Cache/显存指标。<br>4. 与等批等待的简化基线或模拟器结果对比。<br>5. 观察长请求占比变化对短请求 P95 的影响，记录异常与失败。 | 1. 数据集生成器能复现相同长度分布。<br>2. 每档至少 3 轮，报告中写中位数和样本量。<br>3. 能用队列/缓存数据解释至少一个吞吐—尾延迟取舍。<br>4. 估算器与运行时可用 KV 容量量级一致；差异来源有说明。<br>5. 实验报告不把模拟器结果冒充真实 vLLM 实测。 |
| W06-D7（第 42 天，周日） | 2h | 概念答辩与周复盘 | 1. 写 Week 6 Review。<br>2. 不看资料回答 8 个核心问题。<br>3. 将错误答案回链到源码或实验修正。 | 1. 自测覆盖 Prefill/Decode、KV Cache、分页、碎片、抢占/排队、吞吐/延迟、公平性和 OOM。<br>2. 8 题至少 6 题能完整回答，剩余题有补学链接。<br>3. 周报附调度模拟、KV 估算、源码图和实测报告四类产物。 |

#### 第 7 周：LLM Gateway 工程治理

**本周目标：** 在 vLLM 前建设可生产化演进的 Gateway V1，覆盖 OpenAI 兼容、鉴权、限流、超时/取消、重试、熔断和可追踪日志。  
**本周交付物：** `gateway/` 服务、策略文档、单元/集成/故障注入测试、Gateway V1 镜像。

| 天 | 时长 | 学习任务 | 具体操作步骤 | 检测与验收 |
|---|---:|---|---|---|
| W07-D1（第 43 天，周一） | 1h | Gateway 边界与 API 契约 | 1. 画 Client—Gateway—Backend 链路。<br>2. 定义 OpenAI 兼容字段、透传字段和禁止字段。<br>3. 统一错误码、request ID 和模型路由接口。 | 1. `gateway/docs/api-contract.md` 有请求、流式事件和错误示例。<br>2. Contract test 能对已有 vLLM 请求样例通过。<br>3. 能说明 Gateway 负责治理而不重复实现推理调度。 |
| W07-D2（第 44 天，周二） | 1h | API Key 鉴权与安全日志 | 1. 实现 Key 哈希存储/比较和启停状态。<br>2. 在中间件中鉴权并绑定租户上下文。<br>3. 对日志、异常和 metrics 做脱敏。 | 1. 缺失、错误、禁用、正确 Key 四类测试符合 401/403/成功契约。<br>2. 全库日志测试中不出现明文 Key。<br>3. Key 配置不硬编码进仓库，示例只含占位符。 |
| W07-D3（第 45 天，周三） | 1h | 限流、并发与配额语义 | 1. 区分请求速率、并发数和 Token 配额。<br>2. 选 Token Bucket/滑动窗口等算法并写理由。<br>3. 定义 per-key 与全局限制以及 429 响应头。 | 1. `gateway/docs/traffic-policy.md` 含算法、状态和边界。<br>2. 伪时钟单测验证突发、补充、耗尽和恢复。<br>3. 并发限制在取消、异常后不会泄漏计数。 |
| W07-D4（第 46 天，周四） | 1h | 超时、取消与流式转发 | 1. 分开连接、首 Token、闲置和总超时。<br>2. 实现客户端断连向后端传播取消。<br>3. 确保流式错误只按协议结束一次。 | 1. 故障后端分别模拟首 Token 慢与中途卡住，两种超时均被准确触发。<br>2. 取消后 Gateway 和 Backend 活跃请求都回落。<br>3. 自动测试验证无重复结束事件、无悬挂任务。 |
| W07-D5（第 47 天，周五） | 1h | 重试、熔断与幂等边界 | 1. 列出连接失败、429、5xx、已开始流式等场景是否可重试。<br>2. 实现带抖动的有界退避。<br>3. 定义熔断 closed/open/half-open 状态。 | 1. 策略表明确“已输出 Token 后默认不自动重试”。<br>2. 故障注入单测验证最大尝试次数和退避区间。<br>3. 熔断测试能经历 open、拒绝、half-open、恢复完整状态机。 |
| W07-D6（第 48 天，周六） | 5h | 主项目：集成 Gateway V1 | 1. 串联鉴权、路由、限流、并发、超时、取消、重试、熔断与流式代理。<br>2. 增加结构化日志、request ID 传播和 Token 用量记录。<br>3. 连接 vLLM 与可控故障 Mock 后端。<br>4. 编写正常、未授权、超限、超时、断连、后端故障、熔断恢复 E2E 测试。<br>5. 构建镜像并补部署/配置 README。 | 1. 七类 E2E 场景全部自动化且退出码为 0。<br>2. 同一 request ID 能在 Gateway 与 Backend 日志关联。<br>3. 20 次并发取消/异常后，任务数、连接数和信号量回到基线附近，无持续增长。<br>4. 429/5xx 不泄露内部堆栈或凭据。<br>5. `docker build gateway/` 成功，新终端可按 README 启动并请求。 |
| W07-D7（第 49 天，周日） | 2h | 代码审查与 Gateway 验收 | 1. 运行格式化、静态检查、单元和 E2E 测试。<br>2. 对策略默认值做一次威胁/故障复盘。<br>3. 写 Week 7 Review。 | 1. 测试报告按鉴权、流控、超时、重试、熔断、流式分类，关键路径均覆盖。<br>2. 复盘列出至少 3 个残余风险及缓解方式。<br>3. Gateway V1 有版本标签或明确 commit，README 标出能力与非目标。 |

#### 第 8 周：监控、压测整合与性能报告 V1

**本周目标：** 将 Gateway、vLLM、系统/GPU 指标和压测器整合成闭环，用同一实验回答负载、排队、缓存、GPU 与用户延迟之间的关系。  
**本周交付物：** LLM Benchmark Lab V1、Grafana Dashboard V2、自动汇总脚本、`reports/performance-report-v1.md`、第 8 周里程碑包。

| 天 | 时长 | 学习任务 | 具体操作步骤 | 检测与验收 |
|---|---:|---|---|---|
| W08-D1（第 50 天，周一） | 1h | 端到端指标语义统一 | 1. 统一 Client、Gateway、vLLM 三处时间戳与指标定义。<br>2. 定义 TTFT、TPOT、ITL、E2E、排队时间、Token 吞吐。<br>3. 明确超时/失败样本进入哪些统计。 | 1. `benchmark/metric-contract.md` 有公式、计时点、时钟和单位。<br>2. 同一测试请求在三层 request ID 可关联。<br>3. 误差分解能解释客户端 E2E 与后端执行时间差。 |
| W08-D2（第 51 天，周二） | 1h | 真实负载生成与数据集 | 1. 支持固定并发和开放到达率两种模式。<br>2. 生成短/长输入输出以及共享前缀数据集。<br>3. 固定 seed 并记录数据集摘要。 | 1. 同一 seed 生成文件哈希一致。<br>2. 数据摘要包含样本数、输入/输出长度 P50/P95 和共享前缀比例。<br>3. 压测器能从文件读取而非在测量期间临时 tokenize 全部数据。 |
| W08-D3（第 52 天，周三） | 1h | 分位数与吞吐计算校验 | 1. 实现 P50/P95/P99、成功率、请求/Token 吞吐汇总。<br>2. 用手工小数据验证插值/取值规则。<br>3. 区分 wall-clock 吞吐与单请求 tokens/s。 | 1. 至少 5 个确定输入的统计单测通过。<br>2. 失败、取消、零输出样本的处理有测试。<br>3. 报告模板自动写入样本数与统计口径，避免只报分位数不报 N。 |
| W08-D4（第 53 天，周四） | 1h | Dashboard V2 与关联视图 | 1. 添加 Gateway 限流/熔断、vLLM 队列/KV、GPU/系统资源面板。<br>2. 用 request rate、backend、model 变量过滤。<br>3. 在同一时间轴对齐 TTFT、队列和 GPU 利用率。 | 1. Dashboard JSON 可导入且核心查询无错误。<br>2. 压测时能同时看到负载上升、队列变化与延迟变化。<br>3. 至少保存一个带注释的饱和窗口截图。 |
| W08-D5（第 54 天，周五） | 1h | 实验矩阵与报告预注册 | 1. 选低/中/高并发和四种长度组合。<br>2. 加入 Prefix Cache 开/关或另一项配置对照。<br>3. 事先写主要假设、阈值、停止条件与硬件预算。 | 1. `reports/performance-report-v1-plan.md` 在正式实验前提交。<br>2. 实验矩阵控制变量清晰，每组至少 3 次。<br>3. 定义饱和点判据，例如 P95/错误率/队列超过具体阈值，而非事后挑选。 |
| W08-D6（第 55 天，周六） | 5h | 主实验：端到端性能与稳定性测试 | 1. 启动 Gateway、vLLM、Prometheus、Grafana 和硬件采集。<br>2. 预热后按随机化顺序运行完整矩阵，每组至少 3 轮。<br>3. 在高负载场景注入超时、取消和短暂后端故障。<br>4. 保存客户端 CSV、服务日志、Prometheus 快照/查询结果、配置和截图。<br>5. 自动生成汇总图表，分析 TTFT/TPOT、吞吐、P99、队列、KV/显存和 GPU 的关系。 | 1. 所有运行都有 run ID、配置哈希、开始结束时间和环境版本，可一一关联证据。<br>2. 数据完整性脚本能发现缺列、负耗时、重复 ID 和样本不足。<br>3. 找出饱和点，并用至少两类证据支持瓶颈判断。<br>4. 故障期间错误率、重试/熔断、恢复时间均可量化，恢复后指标回到基线范围。<br>5. 结果脚本可从原始数据完全重建表和图；若缺 GPU 实测，相关结论明确留空而非模拟填充。 |
| W08-D7（第 56 天，周日） | 2h | 报告 V1 与第 8 周里程碑验收 | 1. 完成摘要、环境、方法、指标、结果、瓶颈、失败尝试、限制和建议。<br>2. 逐项检查 Gateway V1、vLLM、Dashboard、Benchmark Lab。<br>3. 从空白终端执行复现流程。<br>4. 写 Week 8 Review 和后续技术债。 | 1. `reports/performance-report-v1.md` 的每个数字都链接/指向原始 run ID，至少有 4 张带单位图表。<br>2. 流式、取消、鉴权、限流、TTFT、TPOT/ITL、P99、吞吐、GPU/系统指标均有测试或数据证据。<br>3. 一条 documented 流程可启动、压测并生成报告，关键命令均成功。<br>4. 明确写出至少 2 个无效优化或负面结果、3 个下一阶段待验证假设；里程碑缺项不得标记完成。 |

### 阶段三：Kubernetes 与 Mini LLM Platform（第 9～14 周）

#### 第 9 周：Kubernetes 核心对象与模型服务入门

**本周目标**：掌握 Kubernetes 声明式资源、Pod 生命周期、Deployment 发布、Service 服务发现及常用工作负载的适用边界。

**本周交付物**：可在本地集群运行的模型服务基础清单、对象关系图、部署与回滚记录。

| 天 | 时长 | 学习任务 | 具体操作步骤 | 检测与验收 |
|---|---:|---|---|---|
| W09-D1（第 57 天，周一） | 1h | 建立本地 Kubernetes 实验环境 | 选择 kind 或 minikube；创建 <code>ai-infra</code> 集群与 <code>model-serving</code> Namespace；记录集群版本、节点、context 和常用 kubectl 命令。 | <code>kubectl cluster-info</code> 成功；<code>kubectl get nodes</code> 至少 1 个 Ready 节点；Namespace 存在；环境信息写入 <code>docs/k8s-environment.md</code>。 |
| W09-D2（第 58 天，周二） | 1h | 理解 Pod 生命周期与排障入口 | 编写一个包含资源声明和环境变量的 Pod YAML；观察 Pending、Running、Succeeded/Failed 状态；练习 <code>get</code>、<code>describe</code>、<code>logs</code>、<code>exec</code> 和删除重建。 | YAML 可重复创建；保存一次 <code>describe</code> 与日志输出；能口述 Pod 重建后 IP 和本地文件为何不可靠，并在笔记中列出至少 3 个排障入口。 |
| W09-D3（第 59 天，周三） | 1h | Deployment、ReplicaSet 与滚动发布 | 为已有 Gateway 或 Mock 模型服务编写 Deployment；设置 2 个副本；修改镜像标签触发 rollout；查看 ReplicaSet 变化并执行 rollback。 | <code>kubectl rollout status</code> 成功；2 个副本 Ready；升级前后存在不同 ReplicaSet；<code>rollout undo</code> 后镜像恢复，命令和输出保存到 <code>docs/w09-rollout.md</code>。 |
| W09-D4（第 60 天，周四） | 1h | Service 与集群内 DNS | 创建 ClusterIP Service；启动临时客户端 Pod，通过 Service 名称访问健康检查；分别记录 Pod IP、Service IP 与 DNS 名称。 | 临时 Pod 连续请求 10 次全部返回成功；删除一个后端 Pod 后请求仍可恢复；在 <code>docs/k8s-networking.md</code> 画出 Client→Service→Pod 流向并解释 Service 不等于进程代理。 |
| W09-D5（第 61 天，周五） | 1h | 选择合适的工作负载对象 | 分别运行一个 Job 和 CronJob；阅读 StatefulSet 的稳定身份语义；为 Gateway、vLLM、一次性权重转换、周期清理任务选择对象并写理由。 | Job 显示 Complete；CronJob 至少生成 1 个成功 Job；完成包含 4 个场景、对象选择和反例的决策表，且每项理由不少于 2 句。 |
| W09-D6（第 62 天，周六） | 5h | 搭建 Kubernetes 模型服务 V0 | 用现有 Gateway 加 Mock/轻量模型后端制作镜像或使用已构建镜像；编写 Namespace、Deployment、Service 清单；加入端口、资源请求、标签和版本注解；实现 <code>make deploy</code>、<code>make smoke-test</code>、<code>make undeploy</code> 或等价脚本；演练扩容和滚动更新。 | 从空 Namespace 执行 README 后 15 分钟内可访问接口；扩至 3 副本且全部 Ready；冒烟测试至少覆盖健康检查、普通请求、流式请求 3 项；回滚成功；所有清单通过客户端 dry-run。 |
| W09-D7（第 63 天，周日） | 2h | 系统验收与周复盘 | 清空后按 README 重建；故意设置错误镜像标签并用事件和日志定位；修复后归档命令、对象关系图和问题记录；完成周报。 | 全新重建与清理各成功 1 次；故障记录包含现象、证据、根因、修复 4 部分；对象关系图覆盖 Deployment、ReplicaSet、Pod、Service；本周 7 项证据均有文件路径或提交号。 |

#### 第 10 周：调度、配置、存储与健康探针

**本周目标**：让模型服务具备明确的资源边界、可控的节点放置、外部化配置、持久存储与正确的生命周期管理。

**本周交付物**：生产化 Kubernetes 清单 V1、调度验证记录、探针与故障演练报告。

| 天 | 时长 | 学习任务 | 具体操作步骤 | 检测与验收 |
|---|---:|---|---|---|
| W10-D1（第 64 天，周一） | 1h | requests、limits 与 QoS | 为服务设置 CPU/内存 requests 和 limits；创建 Burstable 与 Guaranteed 示例 Pod；观察调度结果和 QoSClass；模拟超过内存限制。 | <code>kubectl describe pod</code> 能看到资源配置；两个示例的 QoSClass 符合预期；至少捕获一次 OOMKilled 或用受控测试说明无法触发的原因；写出 requests 与 limits 各自影响。 |
| W10-D2（第 65 天，周二） | 1h | 节点选择、亲和性与污点容忍 | 给节点添加实验标签；先用 nodeSelector 放置 Pod，再改为 required/preferred nodeAffinity；添加 taint 并配置 toleration；验证不匹配时 Pending。 | 记录 4 次调度结果：nodeSelector 成功、affinity 成功、无 toleration 失败、有 toleration 成功；Pending 事件明确包含调度原因；实验后恢复节点标签与 taint。 |
| W10-D3（第 66 天，周三） | 1h | ConfigMap 与 Secret 外部化配置 | 把模型名、日志级别、并发上限移入 ConfigMap；把 API Key 移入 Secret；分别以环境变量和卷挂载使用；验证配置更新的生效方式。 | 清单中不含明文 API Key；Pod 内可读取配置但日志不打印密钥；更新 ConfigMap 后能说明哪些方式需要重启；提交一项自动检测密钥不入库的检查或命令。 |
| W10-D4（第 67 天，周四） | 1h | PV、PVC 与模型缓存 | 创建 StorageClass 可用的 PVC；挂载到服务的模型缓存目录；写入标记文件后删除 Pod 并重建；比较 emptyDir 与 PVC 的生命周期。 | 新 Pod 仍可读取标记文件；PVC 状态 Bound；<code>docs/k8s-storage.md</code> 记录访问模式、容量、回收策略以及模型权重缓存为何需要持久卷。 |
| W10-D5（第 68 天，周五） | 1h | startup、readiness、liveness 与优雅退出 | 为 Mock 服务加入可控启动延迟和失败开关；分别配置三类探针；设置 preStop 与 terminationGracePeriodSeconds；观察 Endpoints 变化。 | 慢启动阶段 Pod 不接流量且未被 liveness 误杀；readiness 失败时 Service Endpoint 移除；终止时在宽限期内完成在途请求；保存事件时间线。 |
| W10-D6（第 69 天，周六） | 5h | 整合生产化部署清单 V1 | 整合资源配额、affinity、toleration、ConfigMap、Secret、PVC、三类探针、preStop 与 PDB；添加 Namespace ResourceQuota/LimitRange；编写部署参数说明和验证脚本；逐项注入错误配置、存储不可用和探针失败。 | <code>kubectl apply</code> 后所有资源健康；验证脚本至少 10 项且全通过；PDB、Quota、PVC、探针均有实际状态证据；3 类故障都能被检测并恢复；清理脚本不删除 Namespace 外资源。 |
| W10-D7（第 70 天，周日） | 2h | 故障演练与设计复盘 | 执行 Pod 删除、错误配置、readiness 失败、节点放置不匹配 4 个演练；记录恢复时间和用户影响；更新部署决策说明及周报。 | 4 份演练均含预期、实际、证据、修复；服务在 Pod 删除后自动恢复；能解释 startup 与 liveness 参数的边界；生产化清单从空环境复现成功。 |

#### 第 11 周：Kubernetes GPU 资源与模型冷启动

**本周目标**：理解 NVIDIA GPU 在 Kubernetes 中的暴露和调度链路，并能测量、优化模型服务冷启动。

**本周交付物**：GPU/vLLM Deployment 清单、冷启动阶段时间线、至少 3 次实验的原始数据与优化结论。

| 天 | 时长 | 学习任务 | 具体操作步骤 | 检测与验收 |
|---|---:|---|---|---|
| W11-D1（第 71 天，周一） | 1h | 梳理 GPU 容器与 K8s 链路 | 画出 Host Driver→Container Runtime→NVIDIA Container Toolkit→Device Plugin→Pod 的链路；核对本机驱动、CUDA runtime 和容器运行时；记录 GPU 型号与显存。 | 产出 <code>docs/k8s-gpu-stack.md</code>；有 GPU 时保存 <code>nvidia-smi</code> 和容器内检测结果；无 GPU 时明确标注替代环境、缺失环节和补验日期。 |
| W11-D2（第 72 天，周二） | 1h | Device Plugin 与扩展资源 | 安装或审阅 NVIDIA Device Plugin 清单；查看节点 Capacity/Allocatable 中的 <code>nvidia.com/gpu</code>；理解 GPU 为整数扩展资源及不可超卖的默认语义。 | 有 GPU 时节点显示正确 GPU 数量并且插件 Pod Ready；无 GPU 时完成清单 dry-run 和字段追踪；笔记能回答谁上报资源、谁作调度决策、谁把设备注入容器。 |
| W11-D3（第 73 天，周三） | 1h | GPU Pod 调度与隔离验证 | 创建请求 1 GPU 的测试 Pod；结合节点标签、affinity 和 toleration 固定到 GPU 节点；在容器内运行设备查询；再提交超出容量的 Pod 观察 Pending。 | 有 GPU 时容器可见且只见分配设备，超额 Pod Pending；保存调度事件和设备输出；无 GPU 时用 <code>kubectl apply --dry-run=server</code> 或测试集群替代，并登记延后补验。 |
| W11-D4（第 74 天，周四） | 1h | 模型权重分发与缓存方案 | 对比镜像内置、initContainer 下载、PVC 预热、对象存储直读、节点本地缓存；为目标模型估算权重大小和传输时间；选定一套本阶段方案。 | 决策表至少比较启动速度、镜像体积、一致性、并发下载、成本 5 个维度；给出按实际或假设带宽计算的冷下载时间；明确缓存失效与版本校验方法。 |
| W11-D5（第 75 天，周五） | 1h | 定义冷启动分段指标 | 将冷启动拆成调度等待、镜像拉取、权重获取、进程初始化、权重载入、GPU warm-up、readiness 生效；设计事件采集脚本和 JSONL 数据格式。 | 用 Mock 服务跑通一次采集；每个阶段有起止时间或可解释的代理指标；总时长与阶段和误差小于 5%；数据格式包含运行编号、环境、模型、缓存状态。 |
| W11-D6（第 76 天，周六） | 5h | 部署 GPU vLLM 服务并优化冷启动 | 编写 vLLM Deployment/Service，设置 GPU limit、模型 PVC/initContainer、startup/readiness、共享内存与优雅退出；执行冷缓存和热缓存启动；加入预拉镜像或预热请求；无 GPU 时以 Mock 服务验证控制流程并保留 GPU 清单。 | 有 GPU 时 OpenAI 兼容接口完成 1 次推理且容器可见 GPU；冷/热各至少 1 次完整时间线；优化后某个阶段有量化变化；无 GPU 时清单通过 schema/dry-run、Mock 流程全通且补验项写入统一清单。 |
| W11-D7（第 77 天，周日） | 2h | 冷启动实验与容量复盘 | 冷缓存、热缓存、预热后三种配置各运行至少 3 次；计算中位数；分析最大阶段与波动来源；记录模型大小、PVC、网络和探针配置。 | 原始数据不少于 9 条；报告给出各阶段中位数和总冷启动对比；所有结论可追溯到数据；提出 2 个后续优化并写明收益假设与风险。 |

#### 第 12 周：Helm 与生产发布

**本周目标**：把模型服务清单封装为可校验、可升级、可回滚的 Helm Chart，并设计适合稀缺 GPU 的发布策略。

**本周交付物**：<code>model-serving</code> Helm Chart、dev/gpu 两套 values、发布验证脚本及回滚演练报告。

| 天 | 时长 | 学习任务 | 具体操作步骤 | 检测与验收 |
|---|---:|---|---|---|
| W12-D1（第 78 天，周一） | 1h | Helm 对象模型与 Chart 骨架 | 创建 Chart；理解 Chart.yaml、values.yaml、templates、release、revision；把第 11 周 Deployment 和 Service 迁入模板。 | <code>helm lint</code> 通过；<code>helm template</code> 可生成合法 Deployment/Service；模板中不再硬编码 release 名和 Namespace。 |
| W12-D2（第 79 天，周二） | 1h | values 分层与输入校验 | 参数化镜像、模型、GPU 数、资源、存储、探针和副本数；提供 values-dev.yaml 与 values-gpu.yaml；添加 values.schema.json 约束范围和必填项。 | 两套 values 均能渲染；非法 GPU 数或缺失镜像时 Helm 明确失败；默认值可运行 Mock 环境，GPU values 明确声明 GPU 资源。 |
| W12-D3（第 80 天，周三） | 1h | 模板复用与安全配置 | 编写 helpers 统一名称、labels、selector；使用 checksum annotation 触发配置变更滚动；设计 existingSecret 接口，避免在 values 中保存密钥。 | 所有对象标签一致且 selector 稳定；ConfigMap 修改后 Pod template hash 改变；渲染产物与 Git 搜索均无真实密钥；名称长度测试不超过 Kubernetes 限制。 |
| W12-D4（第 81 天，周四） | 1h | Chart 静态与运行测试 | 执行 lint、template、Kubernetes schema 校验；添加 Helm test Pod 调用健康接口；对关键 values 建立最小自动化测试矩阵。 | 自动脚本一次运行完成至少 2 套 values 渲染和校验；安装后 <code>helm test</code> 成功；故意删除必填值时测试必须失败。 |
| W12-D5（第 82 天，周五） | 1h | GPU 服务发布策略 | 比较 RollingUpdate、Recreate、蓝绿和金丝雀；计算 1 副本 GPU 服务在 maxSurge 下的额外 GPU 需求；设置 PDB、preStop 和 revisionHistoryLimit。 | 产出发布决策记录；至少给出 1、2、4 副本的峰值 GPU 需求；策略能解释容量不足时如何避免永远 Pending，以及何时接受短暂中断。 |
| W12-D6（第 83 天，周六） | 5h | 完成 Chart 的安装、升级与回滚闭环 | 补齐 ServiceAccount、ConfigMap、Secret 引用、PVC、PDB、探针、affinity/toleration 和 NOTES；打包 Chart；依次执行 install、upgrade、失败升级、rollback、uninstall；保存 release revision。 | 从空 Namespace 使用一条 Helm 命令安装；升级后接口版本变化且数据卷保留；失败发布能在约定时间内回滚；回滚后冒烟测试通过；Chart 包和 README 可复现。 |
| W12-D7（第 84 天，周日） | 2h | 生产发布演练与复盘 | 模拟错误镜像、错误探针和容量不足三类发布问题；使用 history/status/events 定位；验证 rollback；归档运行手册和周报。 | 三类问题均有检测信号和恢复步骤；至少一次 rollback 的 revision、耗时、接口可用性证据完整；运行手册包含发布前检查、发布中观察和回滚阈值。 |

#### 第 13 周：Mini LLM Platform 控制面

**本周目标**：设计并实现平台控制面的领域模型、API、持久化与幂等协调逻辑，使用户不再直接编写 Kubernetes 资源。

**本周交付物**：Platform API V0、数据库迁移、Kubernetes 适配层、创建与查询服务的纵向切片。

| 天 | 时长 | 学习任务 | 具体操作步骤 | 检测与验收 |
|---|---:|---|---|---|
| W13-D1（第 85 天，周一） | 1h | 定义需求、资源模型与状态机 | 定义 ModelTemplate、ModelService、Operation、Quota 四个核心对象；画出 Pending→Provisioning→Ready→Updating→Deleting→Failed 状态机；列 API 契约。 | OpenAPI 草案可被解析；每个状态写明进入条件、退出条件和失败路径；至少覆盖创建、查询、列表、更新、删除 5 个动作。 |
| W13-D2（第 86 天，周二） | 1h | 设计数据库与迁移 | 设计模型模板、服务实例、操作记录、事件表；添加 tenant、version、desired_state、observed_state、timestamps；编写首个迁移和回滚。 | 空数据库可向上迁移并回滚；唯一约束阻止同租户重名；状态字段和乐观锁版本有测试；ER 图与实际表一致。 |
| W13-D3（第 87 天，周三） | 1h | Kubernetes 适配层 | 定义 K8sClient 接口；实现 Deployment、Service、ConfigMap 的 build/apply/get/delete；为单元测试实现 Fake Client；统一 labels 和 owner 标识。 | 不连接集群的单元测试可验证生成对象；真实或 kind 集群可 apply 一组资源；重复 apply 不产生额外对象；labels 能从资源反查平台服务 ID。 |
| W13-D4（第 88 天，周四） | 1h | 幂等 API 与协调循环 | 实现创建请求落库、生成 desired spec、队列/轮询触发 reconcile；定义幂等键和重试策略；处理 API 进程在 apply 前后崩溃的情况。 | 相同幂等键提交 3 次只产生 1 个服务；reconcile 连续执行 3 次资源数量不变；故障注入后重启可收敛到 Ready 或明确 Failed。 |
| W13-D5（第 89 天，周五） | 1h | 模型模板、输入校验与配额 | 建立至少 2 个模板（Mock 和 vLLM）；校验模型名、GPU、上下文、显存比例；实现租户 GPU/副本配额预检查；返回结构化错误。 | 有效请求通过；至少 6 个非法边界用例被拒绝；超额请求不写入 K8s；错误响应包含稳定错误码、字段和可读原因。 |
| W13-D6（第 90 天，周六） | 5h | 实现控制面纵向切片 | 实现 create/get/list API、数据库事务、reconciler、K8s 资源生成与状态回写；接入第 12 周 Chart 或等价资源构建；提供启动依赖和初始化脚本。 | 一次 API 调用可从数据库记录推进到 Deployment/Service Ready；查询 API 返回 observed status 和 endpoint；重启平台后状态不丢；单元/集成测试总计不少于 12 项。 |
| W13-D7（第 91 天，周日） | 2h | 一致性与失败场景测试 | 测试数据库成功但 K8s 失败、K8s 已存在、重复请求、平台重启、手工删除 Pod 五种场景；补充 ADR 与周报。 | 五种场景均得到预期状态且可再次收敛；无孤立的重复 Deployment/Service；ADR 记录一致性边界、最终一致选择和已知限制。 |

#### 第 14 周：Mini LLM Platform MVP 闭环与阶段里程碑

**本周目标**：补齐更新、删除、状态、日志、鉴权与可观测性，完成“创建—观察—更新—删除”的平台 MVP。

**本周交付物**：Mini LLM Platform MVP、端到端验收脚本、架构与运行文档、第 14 周阶段报告。

| 天 | 时长 | 学习任务 | 具体操作步骤 | 检测与验收 |
|---|---:|---|---|---|
| W14-D1（第 92 天，周一） | 1h | Operation 与异步状态查询 | 为长操作返回 operation_id；实现 operation 查询、状态转换、错误详情和更新时间；区分平台状态、K8s 状态与模型健康状态。 | 创建请求无需等待 Pod Ready 才返回；客户端可轮询到终态；成功和失败各有集成测试；状态不出现无定义跳转。 |
| W14-D2（第 93 天，周二） | 1h | 平台可观测性 | 添加结构化日志、request_id/service_id、HTTP 指标、reconcile 次数/错误/耗时和平台事件查询；建立基础 Prometheus 面板或查询。 | 一次创建请求可用 request_id 串起 API、reconcile 与 K8s 对象；至少暴露请求量、错误率、协调耗时、状态数 4 类指标；错误日志不包含 Secret。 |
| W14-D3（第 94 天，周三） | 1h | 最小鉴权与租户隔离 | 实现 API Key 或 JWT 中间件；把 tenant 写入资源和查询条件；限制普通用户只能查看/操作自己的服务；定义管理员能力。 | 未认证返回 401；跨租户查询/删除返回 403 或 404；同名服务可在不同租户存在；至少 5 个鉴权自动测试通过。 |
| W14-D4（第 95 天，周四） | 1h | 更新、删除与垃圾回收 | 支持安全更新镜像/模型参数/副本；实现删除状态、K8s 资源清理和数据库保留策略；处理资源已被手工删除的情况。 | 更新后 generation/revision 增加且 endpoint 恢复 Ready；删除 API 可重复调用；相关 Deployment、Service、ConfigMap 全部清理；审计记录仍可查询。 |
| W14-D5（第 96 天，周五） | 1h | CLI 与用户路径打磨 | 编写简易 CLI 或脚本封装 login、template list、service create/get/list/update/delete、logs/endpoint；统一错误输出；准备演示数据。 | 新用户只看 README 可执行完整命令序列；CLI 退出码能区分成功和失败；所有输出均含 service_id 或 operation_id 便于追踪。 |
| W14-D6（第 97 天，周六） | 5h | 完成 MVP 端到端闭环 | 从空集群启动数据库、平台和监控；通过平台创建 Mock 模型服务，观察到 Ready，调用推理接口，更新版本，再删除；补全架构图、时序图、部署文档、验收脚本和演示。 | <code>make e2e</code> 或等价命令自动完成创建—观察—调用—更新—删除且退出码为 0；全过程不手写 kubectl apply；数据库、K8s 与 API 状态一致；全程日志和指标可追踪。 |
| W14-D7（第 98 天，周日） | 2h | 第 14 周里程碑验收 | 在清理后的环境由 README 重装；运行单测、集成测试和 E2E；按里程碑清单检查；记录耗时、失败点、限制与下一阶段性能基线需求。 | 新环境 30 分钟内完成 MVP 复现；测试报告无失败；必须满足“平台完成创建—观察—删除闭环”；阶段报告含架构、证据链接、3 个已知限制和下阶段待测指标。 |

### 阶段四：性能压测与系统优化（第 15～20 周）

#### 第 15 周：性能模型、指标与分析工具

**本周目标**：建立 LLM 服务性能指标体系和瓶颈分析方法，并得到一份可复现的端到端基线。

**本周交付物**：指标定义与计算测试、瓶颈决策树、采集工具配置、性能基线报告 V0。

| 天 | 时长 | 学习任务 | 具体操作步骤 | 检测与验收 |
|---|---:|---|---|---|
| W15-D1（第 99 天，周一） | 1h | 定义 TTFT、TPOT、ITL 与吞吐 | 用时间戳定义请求到达、首 token、逐 token、结束；明确 TTFT、ITL、TPOT、端到端延迟、tokens/s、requests/s；写一个固定样例手算。 | <code>docs/performance-metrics.md</code> 含公式、单位、边界；固定时间戳样例的程序结果与手算完全一致；明确首 token 是否计入输出 token。 |
| W15-D2（第 100 天，周二） | 1h | Prefill/Decode 性能模型 | 对目标模型估算参数权重、KV Cache 和单请求 token 计算；比较 prefill 的并行计算与 decode 的逐 token 访存；列出输入/输出长度影响。 | 至少完成 2 组上下文/并发的 KV Cache 估算；计算过程含维度、精度字节数和假设；能解释为何同一优化可能改善吞吐但恶化 TTFT。 |
| W15-D3（第 101 天，周三） | 1h | 建立瓶颈定位决策树 | 将症状映射到 CPU/tokenizer、队列、GPU 计算、显存容量、显存带宽、PCIe/网络、磁盘/权重加载；为每类选择指标和验证实验。 | 决策树至少含 7 类瓶颈，每类有“观测→假设→实验→判定阈值”；给 3 个虚拟案例能沿树得到可证伪的下一步，而不是直接调参数。 |
| W15-D4（第 102 天，周四） | 1h | GPU/主机指标采集 | 练习 <code>nvidia-smi</code>、dmon 或 DCGM Exporter，配合 pidstat/vmstat/iostat；建立统一采样脚本，记录时间、GPU 利用率、显存、功耗、CPU、内存和 IO。 | 采样脚本持续 60 秒生成带时间戳 CSV/JSONL；列完整且时间单调；有 GPU 时捕获空闲与负载差异，无 GPU 时用 Mock 指标验证管道并登记补验。 |
| W15-D5（第 103 天，周五） | 1h | Profiler 使用与实验设计 | 对轻量 PyTorch/服务路径运行 PyTorch Profiler；有 NVIDIA 环境时准备 Nsight Systems 命令；标记 warm-up、测量窗口和同步点；识别至少一个热点。 | 导出可打开的 trace 或表格；报告列出 top 5 CPU/CUDA 操作及时间占比；说明 profiler overhead 和同步对数字的影响；保存准确复现命令。 |
| W15-D6（第 104 天，周六） | 5h | 建立端到端性能基线 | 固定模型、硬件、输入 512 token、输出 128 token 与一组并发；预热后运行至少 3 轮；同时采集客户端逐 token 时间、vLLM/服务指标、GPU/主机指标；写脚本合并同一时间线。 | 3 轮原始请求和系统指标均保留；报告给出 TTFT、TPOT/ITL、E2E、输出吞吐、错误率的 P50/P95/P99 或适用统计；运行间差异有量化；环境和命令完整。 |
| W15-D7（第 105 天，周日） | 2h | 基线复盘与假设排序 | 检查时间戳、warm-up、失败请求、token 计数和样本量；用决策树判断当前首要瓶颈；提出 3 个可验证优化假设并按预期收益/成本排序。 | 基线报告所有图表可由原始数据重建；指出至少 2 个测量偏差来源；每个优化假设包含要改的单变量、成功阈值和回退条件；不使用无基线的“显著提升”。 |

#### 第 16 周：LLM Benchmark Lab 负载生成与统计

**本周目标**：实现可配置、可校验、可复现的 LLM 压测工具，覆盖流式时间戳、长度分布、并发/速率模型和分位数统计。

**本周交付物**：Benchmark Lab V1、负载矩阵原始数据、自动摘要与第 16 周性能报告。

| 天 | 时长 | 学习任务 | 具体操作步骤 | 检测与验收 |
|---|---:|---|---|---|
| W16-D1（第 106 天，周一） | 1h | 设计负载与结果数据模型 | 定义 request_id、到达时间、首/逐 token 时间、输入/输出 token、状态码、错误、配置；定义固定长度、随机长度、数据集回放三类请求源。 | JSON Schema 或类型定义可验证样例；成功、超时、取消、服务错误 4 类记录均能表达；同一 seed 生成的请求长度序列完全一致。 |
| W16-D2（第 107 天，周二） | 1h | 实现异步流式负载发生器 | 使用异步 HTTP 客户端实现 OpenAI 兼容流式请求；支持固定并发、超时、连接复用、最大请求数和随机 seed；正确解析 SSE 结束标记。 | Mock Server 测试下并发峰值等于配置且无连接泄漏；首 token 和结束时间均记录；至少 20 个请求全部结束；超时测试返回预期错误类型。 |
| W16-D3（第 108 天，周三） | 1h | 逐 token 指标与错误处理 | 从流式事件计算 TTFT、ITL、TPOT、E2E 和 token 吞吐；处理空响应、断流、非 200、取消和重试；区分服务端 token usage 与客户端 tokenizer。 | 使用人工构造事件序列的单测不少于 8 项且全通过；对固定样例计算误差为 0；失败请求不混入成功延迟分位数，但计入总错误率。 |
| W16-D4（第 109 天，周四） | 1h | 统计与分位数正确性 | 实现 P50/P95/P99、中位数、均值、标准差、成功率和置信样本数；定义 warm-up 丢弃规则；禁止用“分位数的平均值”替代全样本分位数。 | 与独立统计库或手算样例对比，所有结果在容差内；小样本、空样本、单样本均有测试；摘要明确样本数、过滤数和分位数算法。 |
| W16-D5（第 110 天，周五） | 1h | 设计压力模型与测试矩阵 | 支持 closed-loop 固定并发和 open-loop 固定到达率/阶梯加压；设计短短、短长、长短、长长及 1/4/16/32 并发矩阵；定义饱和判据。 | 配置文件能表达至少 8 个场景；到达率测试的实际速率误差不超过 10%；饱和判据同时使用错误率、队列/TTFT 和吞吐增益，不只看 GPU 利用率。 |
| W16-D6（第 111 天，周六） | 5h | 完成 Benchmark Lab V1 并执行矩阵 | 整合 CLI、请求源、异步引擎、采样器、统计器和报告器；运行 smoke、固定并发、阶梯加压和长度组合；每配置预热后重复 3 次；输出原始 JSONL、summary CSV 和 Markdown 报告。 | 一条命令可重跑指定场景；至少 8 个场景×3 轮有完整原始数据；汇总可自动生成且与抽样手算一致；任一失败都保留错误详情和非零退出策略；README 从空环境可执行。 |
| W16-D7（第 112 天，周日） | 2h | 数据审计、饱和点判断与周复盘 | 检查丢包、超时、实际长度和时间同步；绘制并发/到达率对吞吐、P95 TTFT、P99 E2E、错误率的关系；识别饱和点并说明因果证据与局限。 | 图表全部能由脚本重建；报告明确测试环境、变量、重复次数和饱和点；结论至少引用 3 项指标，不把相关性写成确定因果；列出第 17 周要验证的 2 个优化实验。 |

#### 第 17 周：建立 Workload 矩阵与性能基线

**本周目标：** 把“压测一下”变成可复现的实验协议，覆盖请求长度、并发、到达模式和流式响应等关键变量，并冻结后续优化使用的 V1 基线。

**本周交付物：** `benchmark/workloads.yaml`、指标定义、环境采集脚本、Benchmark Runner V1、原始基线数据及 `reports/week17-baseline.md`。

| 天 | 时长 | 学习任务 | 具体操作步骤 | 检测与验收 |
|---|---:|---|---|---|
| W17-D1（第 113 天，周一） | 1h | 定义 Workload 维度与矩阵 | 1. 列出输入长度、输出长度、并发数、到达模式、流式开关和请求内容重复率六个维度。<br>2. 在 `benchmark/workloads.yaml` 定义至少 8 个场景，必须含短入短出、短入长出、长入短出、长入长出及低/高并发。<br>3. 为每个场景写唯一 ID 和用途。 | YAML 能被解析；脚本打印出恰好 8 个以上唯一场景；四种长度组合和两档并发均被自动断言覆盖；保存解析输出。 |
| W17-D2（第 114 天，周二） | 1h | 固化指标口径和时间戳 | 1. 定义请求开始、首字节、首 Token、各 Token 和结束时间戳。<br>2. 写清 TTFT、TPOT、ITL、端到端延迟、请求吞吐和 Token 吞吐公式。<br>3. 用一组手工时间戳写指标计算单元测试。 | `docs/benchmark-metrics.md` 含 6 个指标的单位与公式；至少 5 个单元测试通过；测试能识别“首字节不等于首 Token”和时间戳倒序。 |
| W17-D3（第 115 天，周三） | 1h | 采集可复现实验环境 | 1. 编写 `benchmark/capture_env.sh`，采集 OS、CPU、内存、Python、依赖版本及可用时的 GPU、驱动、CUDA 信息。<br>2. 记录模型、服务启动参数和 Git 提交。<br>3. 将结果保存为机器可读文件。 | 执行脚本生成 `benchmark/results/env.json`；必填字段缺失时脚本非零退出；无 GPU 时明确写 `gpu_available: false`，而不是静默缺字段。 |
| W17-D4（第 116 天，周四） | 1h | Workload 冒烟与数据格式校验 | 1. 启动当前模型服务；若无 GPU，使用真实的小模型 CPU 服务，不以静态 Mock 代替延迟测量。<br>2. 对四种长度组合各发 2 个请求。<br>3. 校验流式事件顺序、Token 计数和 JSONL 字段。 | 原始 JSONL 至少有 8 条成功请求；每条均满足时间戳单调、输出 Token 大于 0、场景 ID 可追溯；校验器退出码为 0。 |
| W17-D5（第 117 天，周五） | 1h | 设计重复、预热与随机性控制 | 1. 规定每组预热次数、正式重复次数、随机种子和组间冷却方式。<br>2. 设定失败请求处理和异常值保留规则。<br>3. 为协议写配置校验器。 | `docs/benchmark-protocol-v1.md` 明确“至少 3 次正式重复并报告中位数”；删除种子或把重复数改为 1 时，配置校验测试必须失败。 |
| W17-D6（第 118 天，周六） | 5h | 实现 Benchmark Runner V1 并跑基线 | 1. 实现固定并发、流式解析、请求取消和原始事件落盘。<br>2. 计算单请求 TTFT、TPOT、ITL 与端到端延迟。<br>3. 实现汇总分位数和请求/Token 吞吐。<br>4. 对至少 8 个场景预热后各跑 3 次。<br>5. 采集同期 CPU、内存和可用时的 GPU 指标。 | CLI 能以一条命令运行矩阵；单元测试覆盖正常、超时、空输出和流中断；生成不少于 24 组正式运行记录；成功率、P50/P95/P99、吞吐均非空且原始数据可追溯。 |
| W17-D7（第 119 天，周日） | 2h | 分析并冻结 V1 基线 | 1. 自动从原始数据生成汇总表和至少 3 张图。<br>2. 写出 3 个有数据依据的瓶颈假设。<br>3. 在全新终端按 README 重跑一个代表场景。<br>4. 给基线打版本标签或记录提交号。 | `reports/week17-baseline.md` 包含环境、协议、原始数据路径、中位数及 P95；3 张图可从脚本再生；代表场景重跑成功且关键指标与原记录量级一致；提交号已记录。 |

#### 第 18 周：服务参数、量化与 Prefix Cache 对照实验

**本周目标：** 使用控制变量法量化调度参数、显存参数、量化和 Prefix Cache 对吞吐、延迟、容量及输出质量的影响。

**本周交付物：** 参数实验矩阵、显存估算表、量化与缓存正确性样本、完整对照数据及 `reports/week18-serving-tuning.md`。

| 天 | 时长 | 学习任务 | 具体操作步骤 | 检测与验收 |
|---|---:|---|---|---|
| W18-D1（第 120 天，周一） | 1h | 设计控制变量实验 | 1. 从 W17 选择 3 个代表 Workload。<br>2. 选定 `max_num_seqs`、`max_num_batched_tokens`、`gpu_memory_utilization`、量化方式和 Prefix Cache 五类变量。<br>3. 为每项写基线值、候选值、主指标与风险。 | `benchmark/experiments/week18.yaml` 可解析且每组只改变一个主变量；脚本检测到混杂变量时失败；至少包含 12 个实验单元。 |
| W18-D2（第 121 天，周二） | 1h | 实验调度批处理参数 | 1. 固定模型、请求集和显存参数。<br>2. 用 2～3 个 `max_num_seqs` 或等效并发批处理档位各做预热和 3 次短测。<br>3. 记录队列长度、TTFT、TPOT 和吞吐。 | 生成至少 6 条正式运行记录；每条配置快照不同且请求集哈希相同；能指出吞吐最高档与 P95 TTFT 最低档，结论附数值而非感受。 |
| W18-D3（第 122 天，周三） | 1h | 验证显存参数与容量估算 | 1. 估算权重、KV Cache 和运行时余量。<br>2. 逐步调整 `gpu_memory_utilization` 或等效缓存预算。<br>3. 记录最大可接纳并发和 OOM/拒绝边界；无 NVIDIA GPU 时用小模型真实测 RAM 容量，同时登记 GPU 补验。 | `benchmark/memory_estimator.py` 对固定样例有单测；实测至少 3 个预算点；预测和实测容量差异被计算；无 GPU 时 `docs/deferred-validations.md` 含 GPU 命令、计划日期和通过阈值。 |
| W18-D4（第 123 天，周四） | 1h | 实验批 Token 上限与长请求 | 1. 选择长输入场景。<br>2. 设置 3 档批 Token 上限或等效调度限制。<br>3. 观察抢占、排队、失败及 P99 变化。 | 三档配置各有原始日志；至少一个调度现象可从日志或指标定位；报告中给出容量与尾延迟的数值权衡；所有失败均被分类。 |
| W18-D5（第 124 天，周五） | 1h | 量化正确性与可运行性检查 | 1. 选择一种环境支持的 INT8/INT4 量化模型或动态量化小模型。<br>2. 用固定 20 条提示分别调用基线与量化版本。<br>3. 比较加载内存、成功率、输出长度和至少一种任务相关正确性指标。 | 20 对输出及配置哈希落盘；量化模型确实加载并完成推理；内存差值和指标差值可计算；若目标 GPU 量化后端不可用，CPU 实测只算预演，并登记具体 GPU 补验命令。 |
| W18-D6（第 125 天，周六） | 5h | 运行量化与 Prefix Cache 主实验 | 1. 构造共享前缀占 0%、50%、90% 的请求集并保存哈希。<br>2. 缓存开/关各预热并重复 3 次。<br>3. 基线与量化版本跑同一代表矩阵。<br>4. 同步采集缓存命中、TTFT、吞吐、显存/内存和错误率。<br>5. 自动生成对比图。 | 缓存实验至少 18 组、量化对比至少 6 组正式数据；开/关请求集完全相同；图表能展示命中率与 TTFT/吞吐关系；不得把加载失败或 OOM 当成性能结果。 |
| W18-D7（第 126 天，周日） | 2h | 形成调参建议与边界 | 1. 汇总各变量的收益、代价和适用 Workload。<br>2. 给出交互式、批处理、长上下文三套建议配置。<br>3. 复跑一项最大收益实验验证可复现性。 | `reports/week18-serving-tuning.md` 含至少 5 个数据化结论、3 套配置及反例；复跑改善方向与原实验一致；每个建议都链接到原始运行 ID。 |

#### 第 19 周：系统瓶颈定位与 Profiler 实战

**本周目标：** 建立从症状到证据的定位流程，能区分 CPU、GPU 计算、显存/内存带宽、队列、网络和存储瓶颈，并使用 Profile 证据验证。

**本周交付物：** 统一监控脚本、CPU/GPU Trace、故障注入数据、瓶颈决策树及 `reports/week19-bottleneck-analysis.md`。

| 天 | 时长 | 学习任务 | 具体操作步骤 | 检测与验收 |
|---|---:|---|---|---|
| W19-D1（第 127 天，周一） | 1h | 构建瓶颈决策树 | 1. 为高 TTFT、低 Token 吞吐、高 P99、GPU 利用率低和显存占满五种症状列候选原因。<br>2. 为每个原因指定需观察的指标和排除实验。<br>3. 用 W17 数据走一遍决策树。 | `docs/bottleneck-decision-tree.md` 至少含 5 种症状、每种 3 个候选原因；对一个历史场景输出明确的下一步实验，不能仅凭单一 GPU 利用率下结论。 |
| W19-D2（第 128 天，周二） | 1h | 实现时间对齐的系统采样 | 1. 扩展采样器记录进程 CPU、RSS、线程、磁盘和网络。<br>2. 有 NVIDIA GPU 时加入 `nvidia-smi dmon` 或 DCGM；无 GPU 时明确跳过。<br>3. 将采样时间与请求事件统一为单调时钟/UTC 映射。 | 运行 3 分钟负载后生成连续时序数据；采样间隔误差 P95 小于设定间隔的 20%；请求 ID 可映射到同一时间轴；GPU 缺失状态有显式字段。 |
| W19-D3（第 129 天，周三） | 1h | CPU Profile：网关与 Tokenizer | 1. 对固定请求集分别采集空闲和负载下的 `py-spy`、`perf` 或语言对应 Profile。<br>2. 生成火焰图。<br>3. 找出 Top 5 热点并做一次关闭 Token 计数或序列化的 A/B 小实验。 | 保存两份可打开的 Profile；热点函数累计占比可见；A/B 各至少 3 次且请求集相同；能用数据确认或否定一个 CPU 假设。 |
| W19-D4（第 130 天，周四） | 1h | GPU/PyTorch 执行 Trace | 1. 对一个短场景和一个长场景采集 PyTorch Profiler；有 GPU 时补 Nsight Systems Trace。<br>2. 标注 Prefill、Decode 和主机等待区间。<br>3. 比较两种场景的算子/等待占比。 | 两个 Trace 均可被工具打开且含 shape 或调用栈信息；报告列出 Top 10 算子/事件；无 GPU 时 CPU Trace 必须实际运行，并把 Nsight 补验命令、硬件和日期登记。 |
| W19-D5（第 131 天，周五） | 1h | 故障注入隔离队列、网络与存储 | 1. 分别注入固定网关延迟、模型加载/读取延迟或队列并发上限。<br>2. 用同一小负载跑基线和两个注入场景。<br>3. 检查指标能否正确指向注入点。 | 每种注入可通过配置开关启停；三组原始数据完整；决策树能在不知道标签的情况下至少正确识别 2 个注入瓶颈；恢复配置后指标回归基线量级。 |
| W19-D6（第 132 天，周六） | 5h | 三档负载的端到端 Profile | 1. 运行低负载、饱和前和过载后三个场景，各重复 3 次。<br>2. 同步采集请求指标、系统时序、CPU Profile 和可用时的 GPU Trace。<br>3. 标记饱和点并关联队列增长。<br>4. 对首要瓶颈做一个最小干预。<br>5. 复跑验证。 | 至少 9 组运行拥有统一 run ID；饱和点有请求率、队列和 P95/P99 三类证据；干预前后各 3 次；结论能由 Trace 和时序交叉支持。 |
| W19-D7（第 133 天，周日） | 2h | 输出因果链与排障手册 | 1. 按“症状—假设—实验—证据—结论—边界”写报告。<br>2. 制作 15 分钟排障清单。<br>3. 从原始数据重新生成一张关键关联图。 | `reports/week19-bottleneck-analysis.md` 至少完整论证 2 条因果链并否定 1 个假设；清单可在 15 分钟内走完；关键图可由一条命令再生。 |

#### 第 20 周：综合优化报告与性能阶段里程碑

**本周目标：** 选择有证据的优化项进行受控 A/B，兼顾性能、正确性与稳定性，交付可复现的 Benchmark Lab 和阶段性能报告。

**本周交付物：** 优化假设清单、回归测试、容量/成本模型、最终 A/B 数据、自动报告流水线及 `reports/performance-milestone-v1.md`。

| 天 | 时长 | 学习任务 | 具体操作步骤 | 检测与验收 |
|---|---:|---|---|---|
| W20-D1（第 134 天，周一） | 1h | 选择优化项并预注册标准 | 1. 从 W18、W19 选择 2～3 个首要优化。<br>2. 为每项写机制假设、唯一主指标、护栏指标和预期幅度。<br>3. 在运行前冻结接受/拒绝阈值。 | `docs/week20-experiment-preregistration.md` 有时间戳或提交号；每项只有一个主指标；至少包含错误率、输出正确性和 P99 护栏，事后不可无记录改阈值。 |
| W20-D2（第 135 天，周二） | 1h | 优化 CPU/请求路径 | 1. 根据 Profile 选择一个热点，例如 Token 计数、序列化、连接复用或日志。<br>2. 实现最小改动。<br>3. 跑微基准和端到端小负载。 | 代码含测试；微基准与端到端各至少 3 次；报告 CPU 时间、TTFT 或吞吐的中位数变化；功能输出和错误处理测试全通过。 |
| W20-D3（第 136 天，周三） | 1h | 优化调度/服务配置 | 1. 应用 W18 中一个最佳参数组合。<br>2. 固定模型、数据和请求顺序与 V1 基线比较。<br>3. 检查是否只是通过牺牲尾延迟换吞吐。 | A/B 配置 diff 只含预注册项；两侧各 3 次；主指标达到或未达到阈值均如实记录；P95/P99、失败率和资源峰值完整。 |
| W20-D4（第 137 天，周四） | 1h | 稳定性与取消回归 | 1. 构造超时、客户端取消、流中断和过载四类场景。<br>2. 验证资源释放、请求状态和重试边界。<br>3. 检查优化前后的行为一致性。 | 4 类场景自动测试通过；取消后活动请求和占用资源在规定时间内下降；无重复计费/幽灵请求；优化没有降低既有成功率护栏。 |
| W20-D5（第 138 天，周五） | 1h | 建立容量与成本模型 | 1. 根据吞吐、目标利用率和峰值流量估算所需副本。<br>2. 计算每百万 Token 的资源时成本；无云价格时用参数化单价。<br>3. 用 W17/W20 实测数据校准。 | `benchmark/capacity_model.py` 有至少 4 个边界单测；输入 QPS、长度分布和单价可输出副本数与成本；预测吞吐与实测误差被明确计算。 |
| W20-D6（第 139 天，周六） | 5h | 运行最终受控 A/B 与自动化报告 | 1. 清理并固定环境，不删除旧原始数据。<br>2. 按预注册协议对基线和优化版完成矩阵实验，各场景至少 3 次。<br>3. 运行正确性、稳定性和性能测试。<br>4. 自动生成表格、置信区间/离散度和图表。<br>5. 写入环境与提交哈希。 | 基线和优化数据可按配对 run ID 对齐；原始数据、配置、日志和图全部齐全；一条命令可重建报告；任何未达阈值项明确标为“无收益/回退”。 |
| W20-D7（第 140 天，周日） | 2h | 完成阶段报告与新环境复现 | 1. 写摘要、方法、结果、瓶颈、失败尝试、成本和限制。<br>2. 在新虚拟环境或容器按 README 重跑代表实验。<br>3. 对照第 20 周里程碑逐项勾选。 | `reports/performance-milestone-v1.md` 链接所有原始数据；包含中位数、P95/P99 及至少一个负结果；新环境重跑成功；Benchmark Lab 无原始数据时仍能从命令开始生成新报告。 |

### 阶段五：分布式推理与 GPU 通信（第 21～26 周）

#### 第 21 周：分布式并行基础与可执行最小模型

**本周目标：** 通过代码理解 Data、Tensor、Pipeline Parallel 的切分方式、内存与通信代价，并在无 GPU 的本机也完成多进程正确性预演。

**本周交付物：** 三类并行 Toy 实现、通信/内存估算器、两进程 `torchrun` 实验、对比报告和多卡延后补验记录（如适用）。

| 天 | 时长 | 学习任务 | 具体操作步骤 | 检测与验收 |
|---|---:|---|---|---|
| W21-D1（第 141 天，周一） | 1h | Data Parallel 最小实验 | 1. 用 PyTorch 写单进程参考计算。<br>2. 用两个本地进程切分同一 batch 并聚合结果。<br>3. 记录每进程输入、输出和内存。 | 单进程与两进程拼接结果在设定容差内一致；两个 rank 都有独立日志；改变 batch 不可整除时仍正确处理或明确拒绝。 |
| W21-D2（第 142 天，周二） | 1h | Tensor Parallel 矩阵切分推导与实现 | 1. 构造一个线性层。<br>2. 分别按输出维和输入维切成两片。<br>3. 用实际张量执行局部矩阵乘并重组。 | `distributed/toy_tp.py` 对至少 5 个 shape 与参考输出一致；测试包含非方阵和偏置；代码打印各 shard shape 与需要的通信操作。 |
| W21-D3（第 143 天，周三） | 1h | Pipeline Parallel 与气泡可视化 | 1. 把两层网络拆成两个进程/阶段。<br>2. 用队列传递 4 个 micro-batch。<br>3. 记录开始结束时间并绘制时间线。 | 所有 micro-batch 输出与串行参考一致且顺序可追踪；图中可计算气泡占比；把 micro-batch 从 1 改为 4 后时间线发生合理变化。 |
| W21-D4（第 144 天，周四） | 1h | 用本地进程执行集合通信 | 1. 使用 `torchrun --standalone --nproc-per-node=2` 和 Gloo 初始化进程组。<br>2. 实际执行 broadcast、all-reduce、all-gather。<br>3. 在日志中记录操作前后张量。 | 命令退出码为 0；两个 rank 的 all-reduce 结果符合手算值；all-gather 顺序与 rank 一致；不能以画图或阅读代替运行。 |
| W21-D5（第 145 天，周五） | 1h | 编写并行内存与通信估算器 | 1. 输入参数量、精度、batch、序列长度和并行度。<br>2. 估算 DP/TP/PP 每 rank 权重占用与通信字节。<br>3. 用 Toy shape 手算核对。 | `distributed/parallel_estimator.py` 有至少 6 个单测；TP=1 时通信为 0；并行度非法时拒绝；Toy 实际张量字节与估算一致。 |
| W21-D6（第 146 天，周六） | 5h | 构建 Parallelism Lab V1 | 1. 统一三种 Toy 实现的 CLI、随机种子和日志格式。<br>2. 对单进程、两进程 DP/TP/PP 各跑 3 次。<br>3. 检查正确性、峰值内存和耗时。<br>4. 故意制造一次 shape/rank 错误并改善报错。<br>5. 自动生成比较表。 | 至少 12 条运行记录；全部正确性误差低于预设容差；失败用例在超时前退出并给出可定位错误；表格能比较三种并行的计算、通信和内存特征。 |
| W21-D7（第 147 天，周日） | 2h | 形成选择框架与多卡补验计划 | 1. 写出模型放不下、吞吐不足、跨节点受限三种场景的并行选择。<br>2. 有多卡时将本周周四的集合通信实验换成 NCCL 复跑；无多卡时在 `docs/deferred-validations.md` 登记云资源规格、目标日期、原命令和 NCCL 通过阈值。<br>3. 从空目录按 README 重跑一个两进程实验。 | `reports/week21-parallelism-basics.md` 的每个选择都引用实测或估算；本地 Gloo 两进程复跑成功；无多卡记录含负责人、日期、预算/资源、命令、产物路径和客观通过条件。 |

#### 第 22 周：torch.distributed 与 NCCL 集合通信

**本周目标：** 掌握进程组、rank、同步和主要集合通信的真实行为，构建可切换 Gloo/NCCL 的通信基准并验证故障处理。

**本周交付物：** 集合通信实验套件、正确性测试、延迟/带宽曲线、故障注入记录、NCCL 运行手册及无多卡时的可直接执行补验包。

| 天 | 时长 | 学习任务 | 具体操作步骤 | 检测与验收 |
|---|---:|---|---|---|
| W22-D1（第 148 天，周一） | 1h | 初始化进程组与 rank 映射 | 1. 编写支持 `gloo`/`nccl` 参数的启动脚本。<br>2. 本地用两个 CPU 进程执行 Gloo。<br>3. 有两卡时再执行 NCCL 并记录 rank 到 GPU 映射。 | Gloo 两 rank 必须实际启动、Barrier 成功并干净退出；日志含 global/local rank、world size、主机和设备；有多卡时每 rank 绑定不同 GPU。 |
| W22-D2（第 149 天，周二） | 1h | Broadcast 与 AllGather 正确性 | 1. 为每个 rank 构造可识别张量。<br>2. 执行 broadcast、all-gather 或 `all_gather_into_tensor`。<br>3. 对 0 元素、非连续张量或不支持情况做边界处理。 | 自动测试手算期望值；两个 rank 日志一致；至少一个边界用例得到正确结果或明确错误；Gloo 必跑，NCCL 有资源时复跑。 |
| W22-D3（第 150 天，周三） | 1h | AllReduce 与 ReduceScatter | 1. 执行 sum/max all-reduce。<br>2. 执行 `reduce_scatter_tensor`；后端不支持时用 reduce + scatter 对照实现。<br>3. 记录输入输出字节数。 | 至少 4 组张量与手算一致；对照实现和原生实现结果一致（在支持环境）；通信字节估算测试通过；不能只保留 API 示例而无运行日志。 |
| W22-D4（第 151 天，周四） | 1h | 构建严谨的通信计时器 | 1. 加入 warm-up、Barrier、设备同步和多次迭代。<br>2. 测 1 KB、1 MB、16 MB 三档消息。<br>3. 计算延迟及算法/总线带宽，并记录原始每次耗时。 | 同一配置至少 20 次正式迭代；CSV 无负值/空值；P50/P95 可重算；同步开关测试能暴露错误计时或在文档中说明差异。 |
| W22-D5（第 152 天，周五） | 1h | 超时与 rank 故障注入 | 1. 让一个 rank 延迟超过超时或提前退出。<br>2. 观察另一 rank 的错误和清理行为。<br>3. 增加超时、异常传播与进程销毁逻辑。 | 故障实验在规定超时内退出且无残留子进程；日志指出失败 rank/阶段；修复后正常运行再次通过；保存失败与恢复两份日志。 |
| W22-D6（第 153 天，周六） | 5h | 运行通信带宽曲线实验 | 1. 本地 Gloo 用两进程完整跑消息大小曲线。<br>2. 有两卡时用 NCCL 对同一组大小跑 3 轮，并采集拓扑和 NCCL 调试日志。<br>3. 无多卡时完成 NCCL 环境预检脚本、云上启动脚本、结果模板和命令 dry-run，再登记补验。<br>4. 自动绘制延迟/带宽曲线。 | Gloo 曲线必须来自实际执行；有多卡时 NCCL 每档 3 轮且无校验错误；无多卡时预检测试、参数校验和本地 dry-run 全通过，补验记录含预定日期；图表可由 CSV 再生。 |
| W22-D7（第 154 天，周日） | 2h | 写 NCCL 集合通信报告与 Runbook | 1. 解释各集合通信映射到 TP 的位置。<br>2. 汇总小消息延迟与大消息带宽现象。<br>3. 写 NCCL 启动、诊断、超时和日志采集步骤。<br>4. 复跑一项随机通信正确性测试。 | `reports/week22-collectives.md` 引用实际 run ID；Runbook 中命令可复制执行且不依赖隐含变量；随机测试至少 20 组 shape 全通过；未完成的 NCCL 实测明确标“延后补验”，不冒充完成。 |

#### 第 23 周：Tensor Parallel 与硬件拓扑

**本周目标：** 从列并行、行并行组合出两 rank MLP，量化通信量，并让 TP 部署决策显式考虑 PCIe、NVLink 与跨 NUMA/跨节点拓扑。

**本周交付物：** 两进程 TP MLP、拓扑采集/解析器、通信预测器、TP 预演或实测数据及 `reports/week23-tp-topology.md`。

| 天 | 时长 | 学习任务 | 具体操作步骤 | 检测与验收 |
|---|---:|---|---|---|
| W23-D1（第 155 天，周一） | 1h | 两进程 Column Parallel Linear | 1. 将权重按输出维切到两个 rank。<br>2. 用 Gloo 本地真实执行局部 GEMM 和 all-gather。<br>3. 与未切分线性层比较。 | 5 组固定/随机 shape 的最大误差低于容差；两个 rank 都实际参与；输出拼接维度自动断言；错误分片能触发清晰失败。 |
| W23-D2（第 156 天，周二） | 1h | 两进程 Row Parallel Linear | 1. 将输入和权重按输入维切分。<br>2. 局部 GEMM 后 all-reduce。<br>3. 正确处理一次偏置添加。 | 与参考层在至少 5 个 shape 上一致；偏置没有被重复累加；日志记录 all-reduce 张量大小；测试能抓到错误的双重偏置。 |
| W23-D3（第 157 天，周三） | 1h | 组合 TP MLP Forward | 1. 串联 Column Parallel、激活和 Row Parallel。<br>2. 从相同 checkpoint/随机权重切片。<br>3. 对不同 batch、sequence 和 hidden size 验证。 | `distributed/tp_mlp.py` 在至少 12 个 shape 上通过正确性测试；参数切片可逆或可与原权重核对；两进程命令退出码为 0。 |
| W23-D4（第 158 天，周四） | 1h | 采集并解析硬件拓扑 | 1. Linux/NVIDIA 环境执行 `nvidia-smi topo -m`、`lspci` 和 NUMA 查询；其他环境使用仓库内固定拓扑样例。<br>2. 编写解析器输出 GPU 对之间的链路类别。<br>3. 为 NVLink、同 PCIe Switch、跨 NUMA 样例写测试。 | 当前环境原始拓扑或“无 GPU”探测结果落盘；解析器对 3 类 fixture 测试通过；无 GPU 时不是阅读样例，而是实际运行探测器和解析测试。 |
| W23-D5（第 159 天，周五） | 1h | 通信量与拓扑代价预测 | 1. 从 TP MLP shape 计算每层 collective 字节。<br>2. 输入链路带宽/延迟估算通信时间。<br>3. 与 W22 本地或 NCCL 实测点比较。 | 预测器有单位测试和单位标注；至少输出 6 组 shape；与可用实测的误差被计算；不能用理论峰值直接冒充实际带宽。 |
| W23-D6（第 160 天，周六） | 5h | TP 正确性、性能与服务部署预演 | 1. 用两进程 Gloo 对 TP MLP 跑 3 档 shape、各 3 次。<br>2. 有两卡时切 NCCL 复跑，并用 vLLM `tensor_parallel_size=2` 启动可放入两卡的小模型完成请求。<br>3. 无多卡时生成并校验 TP=2 启动配置、模型可分片检查、端口/挂载预检，并登记云上补验。<br>4. 对比预测与实测。 | Gloo TP 正确性与计时数据必须实际产生；有多卡时 OpenAI 兼容请求成功且两卡均有显存/利用率证据；无多卡时启动包测试和 dry-run 成功，补验含具体模型、卡型、日期、命令及成功判据。 |
| W23-D7（第 161 天，周日） | 2h | 输出拓扑感知的 TP 决策报告 | 1. 画出当前/目标机器拓扑。<br>2. 说明何时 TP 收益会被通信抵消。<br>3. 给同 NUMA、跨 NUMA、跨节点三种部署建议。<br>4. 从干净进程复跑 TP MLP 正确性。 | `reports/week23-tp-topology.md` 含拓扑图、每层通信量和至少 3 条数据/模型支持的建议；干净复跑全通过；多卡缺失项显式链接延后补验，不把 Gloo 速度当 GPU 结论。 |

#### 第 24 周：单机多卡基准与扩展效率

**本周目标：** 用同一模型与 Workload 严格比较单卡和 TP=2，计算加速比、扩展效率与通信占比；无多卡时完成可一键搬到云上的全流程预演并锁定补验机制。

**本周交付物：** 多卡预检工具、单卡/双卡原始数据或本地预演数据、扩展效率计算器、敏感性分析、补验包及 `reports/week24-multigpu-scaling.md`。

| 天 | 时长 | 学习任务 | 具体操作步骤 | 检测与验收 |
|---|---:|---|---|---|
| W24-D1（第 162 天，周一） | 1h | 冻结单卡/双卡对比协议 | 1. 固定模型 revision、精度、上下文、请求集和服务版本。<br>2. 定义单卡与 TP=2 的可比配置、预热、3 次重复和冷却。<br>3. 编写预检脚本检查 GPU 数、P2P、端口、磁盘、模型缓存和依赖。 | 协议 YAML 通过 schema；请求集有哈希；预检在当前机器运行并明确输出 READY 或缺失项；把 GPU 数伪造为不足时脚本必须拒绝正式多卡实验。 |
| W24-D2（第 163 天，周二） | 1h | 获取单设备参考基线 | 1. 有 NVIDIA GPU 时启动单卡服务并跑一个短、一个长场景，各 3 次。<br>2. 无 GPU 时用单进程 CPU/Gloo 跑 TP Toy 参考，验证数据流水线，同时登记单卡 GPU 补验。<br>3. 保存环境和配置。 | GPU 路径至少 6 组有效运行并有 GPU 时序；无 GPU 路径至少 6 组真实 CPU 运行且标记为 `rehearsal`；两种路径都要求原始数据可由汇总脚本读取，缺失 GPU 不得写成单卡结论。 |
| W24-D3（第 164 天，周三） | 1h | 获取 TP=2 对照或双进程预演 | 1. 有两卡时以完全相同请求集启动 TP=2 并跑短/长场景各 3 次。<br>2. 无两卡时用两个 Gloo 进程实际跑同一 Toy 请求集，验证分片、汇总、计时和日志链路。<br>3. 运行配置差异检查。 | 多卡路径两张 GPU 均有进程、显存和通信证据；无多卡路径两个 rank 均完成不少于 6 组运行且正确性通过；diff 除设备数/TP 必需项外无意外变量。 |
| W24-D4（第 165 天，周四） | 1h | 实现加速比与扩展效率计算 | 1. 实现 `speedup = throughput_tp2 / throughput_single`。<br>2. 实现 `efficiency = speedup / 2`，同时比较 TTFT、TPOT、P99 和每请求显存。<br>3. 为缺失、失败和预演数据加状态检查。 | 计算器至少 8 个单测通过；预演数据输出明确水印且不能生成 GPU 扩展结论；正式数据缺少任一重复时程序非零退出；结果保留小数和单位。 |
| W24-D5（第 166 天，周五） | 1h | 长度与并发敏感性小实验 | 1. 选择两档输入长度、两档输出长度和两档并发。<br>2. 有多卡时至少实测其中 4 个代表组合；无多卡时在两进程预演完成 8 格运行并验证自动编排。<br>3. 找出通信更易被摊薄/放大的场景。 | 运行矩阵无漏格或有明确跳过原因；每格配置和 run ID 唯一；至少生成一张效率/耗时随 Workload 变化的图；预演图标题必须标注非 GPU 性能结论。 |
| W24-D6（第 167 天，周六） | 5h | 完成单机多卡正式基准或云上补验包 | 1. 有两卡时按冻结协议跑完整单卡与 TP=2 矩阵，各 3 次，同步采集 GPU、NCCL 和请求指标。<br>2. 计算加速比、扩展效率、通信占比和容量变化。<br>3. 无两卡时从空环境执行“一键预演”：检查配置、启动两进程 Gloo、跑完整矩阵、汇总、绘图；随后将云命令、凭据占位、模型缓存、结果回传和关机清单写入补验包。<br>4. 设置明确补验日期。 | 正式路径数据完整、两侧请求哈希一致且每场景至少 3 次；预演路径所有流水线步骤实际成功且测试能阻止其被标为正式 GPU 数据；`docs/deferred-validations.md` 含不晚于计划里程碑的日期、卡型、预算、命令、通过阈值和证据路径。 |
| W24-D7（第 168 天，周日） | 2h | 扩展效率报告与复现审计 | 1. 写单卡/TP=2 对比、通信解释、拓扑限制和成本权衡。<br>2. 抽取一个场景从原始 JSONL 重算全部指标。<br>3. 对照补验清单审计缺失证据。<br>4. 更新第 26 周前的后续实验入口。 | `reports/week24-multigpu-scaling.md` 明确状态为“多卡已实测”或“本地预演完成、GPU 待补验”；正式结论含 speedup、efficiency、P95/P99 和通信证据；手工重算与脚本一致；待补验项有可执行入口而非纯阅读记录。 |

#### 第 25 周：跨节点通信、网络与分布式推理设计

**本周目标**：理解跨节点推理为何更难，能够用拓扑、带宽、时延和消息规模解释性能。  
**本周交付**：`docs/cross-node-inference.md`、网络基准脚本、两节点实验或可直接上云执行的预演包。

| 天 | 时长 | 学习任务 | 具体操作步骤 | 检测与验收 |
|---|---:|---|---|---|
| W25-D1（第 169 天，周一） | 1h | 建立跨节点性能模型 | ① 画出两节点、每节点多 GPU 的数据路径；② 标注 HBM、NVLink/PCIe、NIC、交换机；③ 写出 `总时延≈计算+节点内通信+节点间通信+排队`。 | 图中至少包含 8 个组件和 3 类链路；能口述为何相同 TP 数跨节点通常慢于单节点；笔记保存到 `docs/cross-node-inference.md`。 |
| W25-D2（第 170 天，周二） | 1h | 学习 RDMA、InfiniBand 与 RoCE | ① 对比 TCP、RDMA；② 整理 IB 与 RoCE 的差异；③ 记录 GPUDirect RDMA、零拷贝、拥塞控制的作用。 | 完成一张至少 6 行的对比表；能回答“RDMA 减少了哪些 CPU/内存路径”；列出 RoCE 正常运行依赖的 3 个网络条件。 |
| W25-D3（第 171 天，周三） | 1h | 理解 NCCL 网络选择与环境信息 | ① 阅读当前 NCCL 官方文档中与接口选择、调试、拓扑有关的条目；② 设计需要采集的环境清单；③ 写排障决策树。 | 环境清单含 NIC、MTU、路由、GPU 拓扑、NCCL/驱动版本；决策树至少有 8 个节点；不要求死记环境变量，要求说明每项观测目的。 |
| W25-D4（第 172 天，周四） | 1h | 运行主机网络基准 | ① 用 `ping` 测 RTT；② 用 `iperf3`（可用时）测单流/多流吞吐；③ 保存命令、原始输出和机器信息。 | 生成 `distributed/results/network-baseline.*`；至少 3 次测量并计算中位数；若只有一台机器，用两个容器/网络命名空间完成流程验证并明确其不能代表 RDMA。 |
| W25-D5（第 173 天，周五） | 1h | 准备两节点 NCCL 测试包 | ① 编写主机清单与启动脚本；② 加入 rendezvous 地址、rank、world size 参数；③ 添加超时、日志目录和 preflight 检查。 | `--dry-run` 可打印每个 rank 的完整启动配置；缺少 NIC/GPU/端口时脚本非零退出并提示原因；README 能让另一人照做。 |
| W25-D6（第 174 天，周六） | 5h | 执行跨节点通信实验或完整本地预演 | ① 1h 检查驱动、端口、时间同步和拓扑；② 2h 对至少 4 个消息规模跑 AllReduce；③ 1h 对比节点内/跨节点；④ 1h 采集日志并画带宽曲线。无两节点资源时，用本地多进程完成脚本、测试、结果解析和云上执行清单。 | 有资源：原始结果覆盖至少 4 个消息规模、各 3 次，报告节点内/跨节点带宽比；无资源：测试包通过本地 2-rank 演练、结果解析单测和错误注入，并在 `deferred-validations.md` 登记唯一待补项。 |
| W25-D7（第 175 天，周日） | 2h | 分析网络瓶颈并复盘 | ① 计算通信占比；② 写“小消息受时延、大消息受带宽”的数据证据；③ 记录一个异常和排查路径；④ 提交周报。 | 文档至少引用 2 组自己的原始数据；给出一项可证伪的优化假设；本周 12h 记录完整，代码、数据、图表均已提交。 |

#### 第 26 周：分布式推理可靠性与阶段报告

**本周目标**：完成分布式推理阶段闭环，覆盖启动、故障、恢复、扩展效率和成本判断。  
**本周交付**：分布式实验报告、故障演练记录、可复现运行包；第 26 周里程碑验收。

| 天 | 时长 | 学习任务 | 具体操作步骤 | 检测与验收 |
|---|---:|---|---|---|
| W26-D1（第 176 天，周一） | 1h | 建立分布式故障模型 | ① 枚举进程退出、GPU Xid、网络中断、超时、OOM、节点丢失；② 标注可检测信号；③ 定义快速失败与恢复策略。 | `docs/distributed-failure-model.md` 至少覆盖 6 类故障，每类含现象、检测、影响、动作；能解释为何部分 collective 会导致所有 rank 一起卡住。 |
| W26-D2（第 177 天，周二） | 1h | 设计超时、健康检查与日志关联 | ① 为 rank 日志加入 run ID/rank/host；② 设置合理超时；③ 设计健康状态与失败原因结构。 | 两个 rank 的日志可按 run ID 合并排序；故意配置错误地址时在限定时间内退出，而非无限等待；错误信息能定位到 rank。 |
| W26-D3（第 178 天，周三） | 1h | 设计模型权重分发与启动流程 | ① 比较共享存储、对象存储、本地缓存；② 画启动时序；③ 定义下载校验、并发抑制、预热与就绪门槛。 | 时序图覆盖下载、校验、加载、进程组、预热、ready；能解释如何避免所有副本同时拉权重；给出 checksum 验证方案。 |
| W26-D4（第 179 天，周四） | 1h | 故障注入：rank 退出 | ① 启动 2-rank 小程序；② 在 collective 前后分别终止一个 rank；③ 记录另一 rank 行为、超时时间和退出码。 | 保存两种注入的完整日志；程序在配置的超时范围内结束；报告准确区分故障发生位置造成的差异。 |
| W26-D5（第 180 天，周五） | 1h | 故障注入：配置与资源异常 | ① 分别制造 world size 不匹配、端口占用、显存不足或其可控模拟；② 检查 preflight；③ 改进错误提示。 | 至少 3 个失败用例自动执行且均被正确识别；错误输出含“原因+建议动作”；测试脚本最终返回成功表示故障均按预期发生。 |
| W26-D6（第 181 天，周六） | 5h | 完成分布式推理综合报告 | ① 1h 整理 W21～W25 数据；② 1h 计算吞吐、延迟、加速比和扩展效率；③ 1h 分析计算/通信/显存权衡；④ 1h 写故障与恢复；⑤ 1h 固化复现脚本和图表。 | `reports/distributed-inference-report.md` 含环境、方法、原始数据链接、至少 3 张图、限制与结论；所有数字可由脚本重算；无多卡时明确区分实测、模拟和待补验，禁止混写。 |
| W26-D7（第 182 天，周日） | 2h | 第 26 周里程碑答辩 | ① 从空环境按 README 复现核心实验；② 录制或模拟 10 分钟讲解；③ 回答 TP 通信、拓扑、扩展效率、故障四组问题；④ 修复最后缺口。 | 复现脚本无手工改代码即可运行；自测问题正确率≥80%；报告通过里程碑表要求；多卡缺失时 `deferred-validations.md` 有资源需求、命令和预计补验日期。 |

### 阶段六：CUDA、Triton 与综合收官（第 27～32 周）

#### 第 27 周：CUDA 执行模型与开发环境

**本周目标**：建立正确的 GPU 编程心智模型，能编译、调试和验证第一个 CUDA Kernel。  
**本周交付**：CUDA 环境自检、Vector Add、错误检查模板和执行模型笔记。

| 天 | 时长 | 学习任务 | 具体操作步骤 | 检测与验收 |
|---|---:|---|---|---|
| W27-D1（第 183 天，周一） | 1h | 理解 Grid/Block/Thread/Warp | ① 画 CUDA 层级；② 推导一维索引公式；③ 用 1000 个元素、256 threads/block 手算 block 数和边界线程。 | 手算结果为 4 blocks 且能说明越界保护；写出 thread、warp、block 的调度关系；完成 5 道索引自测且全对。 |
| W27-D2（第 184 天，周二） | 1h | 理解 GPU 内存层级 | ① 对比 register、shared、L1/L2、global、host pinned；② 记录容量/作用域/速度的相对关系；③ 分析一次 H2D→Kernel→D2H。 | 形成内存层级图；能指出一次不必要的数据拷贝；对给定访问模式判断是否可能合并访问并说明理由。 |
| W27-D3（第 185 天，周三） | 1h | 建立 CUDA 开发与诊断环境 | ① 记录 GPU compute capability、驱动、toolkit、编译器；② 创建最小编译命令/CMake；③ 加入 CUDA API 和 kernel 错误检查宏。 | `kernels/cuda/env-check` 可输出环境并编译运行；故意访问无效设备时能捕获并打印可读错误；README 列出兼容性。 |
| W27-D4（第 186 天，周四） | 1h | 编写 Vector Add | ① 分配 host/device 内存；② 拷贝输入；③ 启动 kernel；④ 拷回并释放；⑤ 用 CPU 参考值比较。 | 对长度 0、1、1000、非 block 整倍数均通过；最大误差在定义容差内；`compute-sanitizer`（可用时）无错误。无 GPU 时只可完成编译与代码审查，登记补验。 |
| W27-D5（第 187 天，周五） | 1h | 正确测量 Kernel 时间 | ① 对比主机计时与 CUDA Event；② 加 warm-up；③ 同步后重复测量；④ 区分端到端和 kernel-only。 | 输出 warm-up 次数、重复次数、中位数；能解释异步执行为何让朴素计时失真；结果文件同时包含 kernel-only 与端到端。 |
| W27-D6（第 188 天，周六） | 5h | 扩展 Vector Add 实验 | ① 1h 参数化数据类型和长度；② 1h 扫描 block size；③ 1h 测 pageable/pinned memory；④ 1h 计算有效带宽；⑤ 1h 写测试与结果图。 | 至少 5 个长度×3 个 block size；每组≥20 次并报告中位数；正确性测试全过；图中明确单位、环境和误差处理；不宣称微小抖动为提升。 |
| W27-D7（第 189 天，周日） | 2h | 复盘 CUDA 基础 | ① 闭卷画执行/内存模型；② 解释边界检查、同步、错误检查；③ 整理常用模板；④ 提交周报。 | 10 个自测问题正确≥8；从删除构建目录开始能一条命令重建并测试；列出当前实现的 3 个性能限制。 |

#### 第 28 周：Reduction、矩阵运算与同步

**本周目标**：掌握 shared memory、同步和并行归约，理解正确性与性能的共同约束。  
**本周交付**：Reduction、Matrix Add、朴素 MatMul 及完整正确性测试。

| 天 | 时长 | 学习任务 | 具体操作步骤 | 检测与验收 |
|---|---:|---|---|---|
| W28-D1（第 190 天，周一） | 1h | 学习 shared memory 与同步 | ① 画 block 内协作流程；② 解释 `__syncthreads()`；③ 分析条件分支中同步的死锁风险；④ 估算每 block shared memory。 | 能指出两个错误同步示例的 bug；完成 shared memory 容量计算；笔记含“可见性、屏障、作用域”三个概念。 |
| W28-D2（第 191 天，周二） | 1h | 推导并行 Reduction | ① 从串行求和改成树形归约；② 画每轮活跃线程；③ 比较相邻寻址与顺序寻址；④ 记录浮点加法顺序影响。 | 对 256 元素推导出 8 轮；能解释分支发散与 bank conflict 风险；定义相对/绝对误差验收规则。 |
| W28-D3（第 192 天，周三） | 1h | 设计通用正确性测试 | ① 建 CPU/PyTorch reference；② 覆盖随机、全零、极值、奇数长度；③ 固定随机种子；④ 输出差异摘要。 | 测试至少覆盖 8 个 shape；故意注入索引 bug 时测试能失败；恢复后全部通过；失败信息含 shape、最大误差和位置。 |
| W28-D4（第 193 天，周四） | 1h | 实现 Block Reduction | ① 每线程加载元素；② 在 shared memory 归约；③ 每 block 输出部分和；④ 用第二阶段或 CPU 合并。 | 对 1～1M 多种长度与 reference 对齐；sanitizer 无越界/竞态；代码处理非 2 次幂长度。 |
| W28-D5（第 194 天，周五） | 1h | 实现 Matrix Add 与二维索引 | ① 使用二维 grid/block；② 实现边界检查；③ 比较不同 block 形状；④ 加测试。 | 至少测试 7 个含非整除维度的 shape；输出与 reference 对齐；能解释 `(32,8)` 与 `(16,16)` 访问差异。 |
| W28-D6（第 195 天，周六） | 5h | 实现并分析朴素 MatMul | ① 1h 推导 M/N/K 索引；② 1.5h 实现朴素 kernel；③ 1h 建正确性测试；④ 1h 扫描 shape；⑤ 0.5h 对比高性能库并分析差距。 | 至少 6 个矩阵 shape 正确；报告 GFLOP/s、延迟、与库基线差距；明确此实现仅为学习基线；代码无越界且构建测试一键运行。 |
| W28-D7（第 196 天，周日） | 2h | 整理 Kernel 设计模式 | ① 对 Vector Add/Reduction/MatMul 画数据映射；② 总结同步点和内存访问；③ 记录一次 bug 的定位过程；④ 周复盘。 | 三张图能对应到代码行/函数；bug 记录含现象、最小复现、根因、修复和测试；本周交付全部提交。 |

#### 第 29 周：CUDA 性能分析与 Kernel 优化

**本周目标**：用 profiler 证据而非猜测优化 kernel，理解 coalescing、occupancy、register 与 shared memory 权衡。  
**本周交付**：一个优化过的 Reduction 或 MatMul、Nsight 分析记录、优化前后报告。

| 天 | 时长 | 学习任务 | 具体操作步骤 | 检测与验收 |
|---|---:|---|---|---|
| W29-D1（第 197 天，周一） | 1h | 理解合并访问与算术强度 | ① 分析 warp 的连续/跨步访问；② 计算 Vector Add 与 MatMul 的粗略算术强度；③ 判断 memory-bound/compute-bound。 | 完成两个字节/操作量推导；能预测哪个优化方向更可能有效；预测写入实验假设，后续不得事后修改。 |
| W29-D2（第 198 天，周二） | 1h | 理解 occupancy 与资源限制 | ① 整理 threads、register、shared memory 对驻留 block 的影响；② 使用 occupancy 工具/计算器；③ 对两个 block size 做预测。 | 保存资源报告；能解释高 occupancy 不等于高性能；给出一项可测试的 block size 假设。 |
| W29-D3（第 199 天，周三） | 1h | 学习 Nsight Systems/Compute 工作流 | ① 用 Systems 定位 CPU、拷贝、kernel 时间线；② 用 Compute 查看内存与执行指标；③ 建立 profiler 命令模板。 | 对 W27 程序保存一份时间线和 kernel 指标；能指出瓶颈证据所在指标；命令和工具版本已记录。 |
| W29-D4（第 200 天，周四） | 1h | Profile 基线 Kernel | ① 固定代表 shape；② warm-up；③ 收集基线；④ 记录 kernel time、带宽/吞吐、occupancy、主要 stall。 | `kernels/results/cuda-baseline/` 包含原始 profiler 文件和摘要；重复运行差异在解释范围内；没有只凭单次数据下结论。 |
| W29-D5（第 201 天，周五） | 1h | 实施单变量优化 | ① 只改一种因素，如访问模式、shared memory 或展开；② 重跑正确性；③ 同配置 benchmark；④ 比较 profiler。 | 正确性全部通过；表格明确唯一变量；无提升也算有效，但必须用指标解释并保留结果，禁止删掉失败实验。 |
| W29-D6（第 202 天，周六） | 5h | 完成两轮证据驱动优化 | ① 1h 复核基线；② 1.5h 优化 A；③ 1h profile；④ 1h 优化 B；⑤ 0.5h 汇总速度、资源与适用 shape。 | 至少两轮“假设→改动→正确性→测量→结论”；每轮≥30 次计时；报告速度比及误差；若退化，准确记录回滚理由，不删除代码历史。 |
| W29-D7（第 203 天，周日） | 2h | 写 CUDA 优化报告 | ① 整理图表；② 区分端到端与 kernel-only；③ 写适用范围与反例；④ 做 8 分钟讲解。 | `reports/cuda-kernel-optimization.md` 可由原始数据重生成；至少包含一次失败优化；讲解能回答“为何更快/为何没快/何时不该用”。 |

#### 第 30 周：Triton 编程模型与基础算子

**本周目标**：掌握 Triton 的 program instance、block、mask 与自动调优基本方式。  
**本周交付**：Triton Vector Add、Fused Activation、基准与测试框架。

| 天 | 时长 | 学习任务 | 具体操作步骤 | 检测与验收 |
|---|---:|---|---|---|
| W30-D1（第 204 天，周一） | 1h | 对比 CUDA 与 Triton 编程模型 | ① 对比 thread-centric 与 block/vector-centric；② 理解 program ID、offset、mask；③ 将 CUDA Vector Add 映射为 Triton 伪代码。 | 完成概念对照表；能手算给定长度下每个 program 的 offsets/mask；指出 Triton 仍需关心的 3 个硬件因素。 |
| W30-D2（第 205 天，周二） | 1h | 搭建 Triton 环境 | ① 固定 PyTorch/Triton/CUDA 版本；② 运行官方最小示例；③ 建 pytest 与 benchmark 入口；④ 记录缓存行为。 | `kernels/triton/env-check` 一键检查并运行；版本不兼容时给出明确提示；有 GPU 时最小 kernel 通过，无 GPU 时登记实际执行补验。 |
| W30-D3（第 206 天，周三） | 1h | 实现 Triton Vector Add | ① 生成 offsets；② mask load/store；③ 定义 grid；④ 与 PyTorch 对齐；⑤ 处理非整除长度。 | 长度 1、17、1024、非 2 次幂及大数组全部正确；dtype/shape 参数化测试通过；代码不依赖固定长度。 |
| W30-D4（第 207 天，周四） | 1h | 建立 Triton Benchmark | ① warm-up；② 使用框架 benchmark 工具；③ 扫描输入长度；④ 输出延迟和 GB/s；⑤ 对比 PyTorch。 | 至少 8 个长度、每点足够重复；CSV 与图自动生成；图中注明硬件、dtype、端到端或 kernel-only。 |
| W30-D5（第 208 天，周五） | 1h | 实现融合激活算子 | ① 选择 bias+ReLU 或其他简单逐元素融合；② 写 PyTorch reference；③ Triton 一次读写完成；④ 测正确性与速度。 | 多 shape/dtype 输出在容差内；证明 kernel launch/中间张量减少；性能结论覆盖小、中、大 3 类 shape。 |
| W30-D6（第 209 天，周六） | 5h | 完成 Triton 基础算子实验集 | ① 1h 扩充测试；② 1h 扫 BLOCK_SIZE；③ 1h 分析生成代码/资源；④ 1h 比较融合前后；⑤ 1h 写 README 和结果。 | `pytest` 全过；BLOCK_SIZE 至少 4 档；报告说明性能拐点；一条命令生成数据和图；保留未提升的 shape 与原因。 |
| W30-D7（第 210 天，周日） | 2h | 复盘 Triton 与选择项目算子 | ① 闭卷写 kernel；② 对比 CUDA/Triton 调试体验；③ 在 Softmax 与 RMSNorm 中选 W31 项目；④ 定义基线、shape 和成功标准。 | 30 分钟内从空文件写出正确 Vector Add；项目方案含至少 8 个 shape、2 个 dtype、正确性容差和“不退化”阈值；周报完成。 |

#### 第 31 周：Triton Softmax / RMSNorm 优化项目

**本周目标**：独立完成一个中等复杂度算子的设计、正确性、基准、分析和优化。  
**本周交付**：可选择 Softmax 或 RMSNorm 的 Triton 实现、测试、数据和正式报告。

| 天 | 时长 | 学习任务 | 具体操作步骤 | 检测与验收 |
|---|---:|---|---|---|
| W31-D1（第 211 天，周一） | 1h | 定义算子规格与数值标准 | ① 写输入/输出、轴、dtype、布局；② 定义 PyTorch reference；③ 设计数值容差；④ 列边界 shape。 | `spec.md` 含功能、非功能和排除项；测试矩阵≥8 shapes×2 dtypes（硬件支持时）；极端值/非整除列数在清单中。 |
| W31-D2（第 212 天，周二） | 1h | 推导数据映射与内存流量 | ① 选择一行/一个 program 或合理映射；② 推导 load、计算、store；③ 估算中间张量和理论字节量；④ 识别长行限制。 | 有清晰数据流图和字节量公式；能说明 mask、数值稳定与累积精度；列出至少 2 个预期瓶颈。 |
| W31-D3（第 213 天，周三） | 1h | 实现正确的第一版 | ① 编写 kernel；② 接 Python wrapper；③ 加 shape/dtype 检查；④ 跑最小正确性用例。 | 第一版对至少 4 个代表 shape 与 reference 对齐；非法输入给可读错误；代码无硬编码测试数据。 |
| W31-D4（第 214 天，周四） | 1h | 扩展正确性与数值测试 | ① 参数化完整矩阵；② 随机/极端输入；③ 重复性检查；④ 与高精度 reference 对比。 | 全部计划用例通过或明确记录硬件不支持项；测试能捕获故意去掉稳定化/epsilon 后的错误；最大误差被保存。 |
| W31-D5（第 215 天，周五） | 1h | 建立 PyTorch 基线与初测 | ① 固定环境；② warm-up；③ 测 PyTorch eager/可选 compile；④ 测 Triton；⑤ 输出第一版曲线。 | 原始 CSV 含 shape、dtype、实现、延迟、重复参数；至少每点 3 个统计量；发现异常点时先复测，不直接删除。 |
| W31-D6（第 216 天，周六） | 5h | 优化并完成项目报告 | ① 1h profile 第一版；② 1h 调 BLOCK_SIZE/warps；③ 1h 做融合或访存优化；④ 1h 全量回归与 benchmark；⑤ 1h 写报告和使用说明。 | 正确性 100% 通过；至少 2 个优化尝试；报告对每个 shape 给速度比，不要求全部更快；说明数值、资源、长行和硬件限制；API 示例可直接运行。 |
| W31-D7（第 217 天，周日） | 2h | 代码审查与可复现验收 | ① 清理接口与注释；② 从新环境按 README 安装/运行；③ 运行 lint/test/benchmark smoke；④ 录制或模拟技术讲解。 | 一条命令跑测试、一条命令跑 smoke benchmark；报告数字能由数据重建；讲解覆盖映射、数值稳定、性能证据及失败方案。 |

#### 第 32 周：系统整合、作品集与面试收官

**本周目标**：把 32 周成果整合为可运行、可解释、可展示的 AI Infra 作品集。  
**本周交付**：三项目 Release、综合架构文档、演示脚本、简历要点、面试题库和最终验收记录。

| 天 | 时长 | 学习任务 | 具体操作步骤 | 检测与验收 |
|---|---:|---|---|---|
| W32-D1（第 218 天，周一） | 1h | 盘点成果与缺口 | ① 对照最终清单检查 Gateway、Platform、Benchmark、Distributed、Kernels；② 标绿/黄/红；③ 只选影响演示的关键缺口。 | 形成 `reports/final-gap-list.md`；每个缺口有负责人（自己）、预计时间和验收；本周范围不超过可用 11 小时。 |
| W32-D2（第 219 天，周二） | 1h | 完成综合架构与请求链路 | ① 画组件架构图；② 画请求时序；③ 画 Kubernetes/GPU 部署图；④ 标注指标与故障边界。 | 三张图均能对应仓库真实组件；请求链路从 Agent/客户端贯穿 Gateway、vLLM、GPU、监控；不存在尚未实现却画成已实现的组件。 |
| W32-D3（第 220 天，周三） | 1h | 固化端到端演示脚本 | ① 编写启动/清理/健康检查；② 准备流式请求、并发压测、指标查看；③ 添加无 GPU 的 mock 演示模式。 | 从干净终端按脚本可完成“启动→请求→观测→停止”；失败会非零退出；演示总时长≤10 分钟，且有预期输出样例。 |
| W32-D4（第 221 天，周四） | 1h | 写量化简历条目 | ① 用“问题—行动—指标—结果”写 3～5 条；② 每个数字链接到报告；③ 删除无法证明的形容词。 | 每条都能在 2 分钟内找到代码或数据证据；不把本地实验包装成生产规模；至少覆盖服务治理、平台、性能三类能力。 |
| W32-D5（第 222 天，周五） | 1h | 建立面试题与项目叙事 | ① 整理 30 题：后端/K8s、vLLM、性能、分布式、CUDA/Triton 各 6 题；② 为每题写要点；③ 准备 2/5/10 分钟版本。 | 闭卷随机 10 题正确≥8；三种时长讲述均不超时；每个项目能说明一次失败实验、取舍和限制。 |
| W32-D6（第 223 天，周六） | 5h | 最终集成、回归与 Release | ① 1h 修复红色缺口；② 1h 跑 Gateway/Platform 测试；③ 1h 跑 benchmark/kernel smoke；④ 1h 执行端到端演示；⑤ 1h 整理 README、变更日志和 Release。 | 自动测试全部通过或有明确、非核心的已知问题；三项目 README 含架构、运行、测试、结果和限制；Release 有版本号、环境矩阵、演示截图/日志；无密钥或大模型权重误提交。 |
| W32-D7（第 224 天，周日） | 2h | 最终答辩与 32 周验收 | ① 30 分钟跑最终清单；② 30 分钟做 10 分钟演示并复盘；③ 30 分钟随机面试自测；④ 30 分钟写下一阶段 90 天计划。 | 最终作品集清单全部勾选；自测正确率≥80%；所有 deferred validation 已完成或有明确资源/日期/命令；`reports/final-review.md` 记录 384 小时完成情况、最强证据、薄弱点和后续计划。 |

## 7. 阶段里程碑总表

| 时间点 | 必须交付 | 一票否决条件 |
|---|---|---|
| 第 4 周末 | 最小模型 HTTP 服务、Docker 镜像、基础指标、Transformers/vLLM 初测 | 只能本地手工运行；无 README 或无请求级指标 |
| 第 8 周末 | LLM Gateway V1、vLLM 服务、Grafana 面板、性能报告 V1 | 不支持流式与取消；没有 TTFT/吞吐数据 |
| 第 14 周末 | Kubernetes 上的 Mini LLM Platform MVP | 只能手写资源；平台不能完成创建—观察—删除闭环 |
| 第 20 周末 | Benchmark Lab 与系统化性能报告 | 结论只看平均值；无原始数据或不可复现 |
| 第 26 周末 | 多 GPU/替代环境的分布式推理与 NCCL 报告 | 不解释通信开销；没有扩展效率计算 |
| 第 32 周末 | CUDA/Triton 算子、三项目整合、作品集与面试材料 | 正确性未验证；性能数字无基线或无环境说明 |

## 8. 每周复盘模板

```markdown
# Week XX Review

- 计划时间：12 h；实际时间：__ h
- 完成任务：__/7 天
- 本周提交：__ 个；测试：__ 项
- 本周最重要的三个结论：
  1. 
  2. 
  3. 
- 一个失败实验及原因：
- 一个仍无法解释的问题：
- 下周风险与调整：
- 里程碑状态：绿 / 黄 / 红
```

## 9. 最终作品集验收清单

- [ ] 仓库首页能在 3 分钟内说明问题、架构、运行方式和关键成果。
- [ ] LLM Gateway、Mini LLM Platform、Benchmark Lab 均有独立 README 和自动化测试。
- [ ] 至少有 1 张架构图、1 张请求时序图和 1 张部署图。
- [ ] 至少有 2 份使用原始数据、实验变量和分位数的性能报告。
- [ ] 至少记录 3 个真实故障：现象、假设、证据、根因、修复、预防。
- [ ] 至少 1 个 Triton Kernel 通过正确性测试并与 PyTorch 基线比较。
- [ ] 有多 GPU 条件时，至少 1 份 TP/NCCL 扩展效率报告；否则保留明确的补验记录。
- [ ] 能用 10 分钟完整讲述“请求进入网关—平台部署—vLLM 调度—GPU 执行—指标观测”的链路。
- [ ] 简历条目只写可复现的数据，不使用“显著提升”等无基线表述。

## 10. 学习完成的定义

32 周走完不等于完成。只有当最终清单全部通过、三个项目可复现、关键性能数字有原始数据支撑，并且你能解释一次成功优化和一次失败优化时，这套学习计划才算真正完成。
