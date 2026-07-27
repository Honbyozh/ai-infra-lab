# AI Infra 全景认知方案

## 目的

当前不预设最终走推理、训练、Kernel、GPU 集群或 AI Platform。本阶段只建立 AI Infra 全景，并通过少量代表性体验选出两个候选方向。

全景阶段不设逐日计划，不维护任务状态机，不为每个概念单独写学习日志。知识以脱稿解释验收，工程以代码和测试验收，性能以原始数据验收；只有路线决策和难以重新获得的结论才进入文档。

## 全景地图

```text
AI 应用
聊天、搜索、Agent、推荐、多模态
            |
            v
服务与产品层
API、Gateway、鉴权、限流、路由、流式输出
            |
            v
AI Platform / MLOps
训练任务、模型注册、部署、配额、权限、生命周期
            |
            v
训练与推理系统
PyTorch、vLLM、SGLang、TensorRT-LLM、训练 Runtime
            |
            v
分布式执行
DP、TP、PP、EP、NCCL、RDMA、通信计算重叠
            |
            v
编译器与算子
torch.compile、XLA、Triton、CUDA Kernel、算子库
            |
            v
加速器与节点
GPU、HBM、CPU、内存、PCIe、NVLink、网卡、SSD
            |
            v
集群基础设施与云
Kubernetes、Slurm、调度、网络、存储、故障恢复、容量
```

可观测性、可靠性、安全和成本横跨所有层。

## 执行方法

按下面六个模块建立认知。每个模块只需要回答三个共同问题：

1. 它解决什么问题？
2. 它处在系统的哪一层，与上下层如何交互？
3. 这个方向的工程师实际构建、测量或调试什么？

学习时由 Agent 逐题追问。能够脱稿解释并处理一个简单案例即可继续；回答不清楚的内容只追加到本文末尾的“未决问题”，不另建日志。

### 模块一：模型生命周期与工作负载

应该知道：

- 数据准备、训练、checkpoint、评估、模型转换、部署、推理、监控和回滚；
- 预训练、微调、后训练与推理的区别；
- tensor、shape、dtype、device、参数量、FLOPs 和显存；
- Transformer、Attention、MLP、token、sequence length 和 batch；
- 训练的 forward、backward、optimizer step；
- 推理的 prefill、decode 和 KV Cache；
- dense model 与 Mixture-of-Experts；
- compute-bound 与 memory-bound。

至少能解释：

- 为什么训练通常比推理保存更多状态；
- 为什么 prefill 与 decode 的性能特征不同；
- 为什么参数量不能直接代表运行速度；
- 长上下文为什么会增加 KV Cache 压力。

### 模块二：硬件、Runtime、编译器与 Kernel

应该知道：

- CPU、cache、NUMA、系统内存和内存带宽；
- GPU SM、warp、Tensor Core、HBM 容量和带宽；
- PCIe、NVLink、NVSwitch；
- arithmetic intensity 与 Roofline；
- PyTorch/JAX、eager/graph、Runtime、Driver；
- `torch.compile`、TorchInductor、XLA、Triton 和 CUDA；
- cuBLAS、cuDNN、FlashAttention 与自定义 Kernel；
- graph optimization、operator fusion 和 code generation。

至少能解释：

- GPU 理论算力很高，程序为什么仍可能很慢；
- 显存容量不足与显存带宽不足的区别；
- 一个 PyTorch 算子如何最终在 GPU 上执行；
- 单个 Kernel 更快为什么不一定改善端到端性能。

### 模块三：分布式训练系统

应该知道：

- Data、Tensor、Pipeline、Expert 和 Context Parallel；
- broadcast、all-reduce、all-gather、reduce-scatter 和 all-to-all；
- NCCL、NVLink、RDMA、InfiniBand/RoCE 与网络拓扑；
- DDP、FSDP、ZeRO、Megatron-LM 和 DeepSpeed；
- activation、gradient、optimizer state 和 mixed precision；
- 数据加载、checkpoint、容错、恢复和 elastic training；
- 通信计算重叠、straggler、吞吐和扩展效率。

至少能解释：

- 模型权重能放进 GPU，为什么仍可能无法训练；
- 不同并行方式分别切分和通信什么；
- 更多 GPU 为什么不一定更快；
- 数据、通信、计算和 checkpoint 如何分别拖慢训练。

### 模块四：推理系统

应该知道：

- online serving 与 offline inference；
- static batching 与 continuous batching；
- request scheduler、KV Cache 和 prefix cache；
- quantization、speculative decoding 和 tensor parallel；
- prefill/decode disaggregation、KV Cache 传输和 model routing；
- vLLM、SGLang、TensorRT-LLM、NVIDIA Dynamo 和 Triton Inference Server；
- TTFT、TPOT、ITL、吞吐、goodput、P50/P95/P99 和 cost per token。

至少能解释：

- 提高吞吐为什么可能损害延迟；
- continuous batching 实际在调度什么；
- KV Cache 为什么同时是容量问题和调度问题；
- 请求长度分布为什么会改变服务性能；
- 什么情况下值得分离 prefill 和 decode。

### 模块五：GPU 集群、数据与 AI Platform

应该知道：

- Kubernetes、Slurm、node、pod、container 和 job；
- GPU device plugin、GPU Operator、MIG 和 GPU sharing；
- gang scheduling、topology-aware scheduling、queue、quota 和 preemption；
- 节点健康、GPU Xid、异构 GPU 与容量规划；
- object storage、distributed filesystem、local SSD、cache 和 prefetch；
- dataset sharding、checkpoint、model loading 和 model distribution；
- experiment tracking、model registry、workflow orchestration 和 metadata lineage；
- control plane、data plane、multi-tenancy、RBAC、发布和回滚。

至少能解释：

- 普通 CPU 调度为什么不能完全解决 GPU 调度；
- 总空闲 GPU 足够，任务为什么仍可能无法调度；
- 模型冷启动和 checkpoint 为什么会成为存储问题；
- AI Platform、MLOps、训练 Runtime 与推理 Runtime 的边界。

### 模块六：可观测性、可靠性、安全与成本

应该知道：

- logs、metrics、traces、profiles 和 benchmark；
- GPU metrics、请求级指标、分布式 trace、PyTorch Profiler 和 Nsight；
- SLI、SLO、SLA、availability 和 failure domain；
- retry、timeout、backpressure、降级和恢复；
- secret、供应链、容器隔离、多租户隔离和权重保护；
- GPU-hour、利用率、容量规划、cost per token；
- latency、throughput、reliability 与 cost 的权衡。

至少能解释：

- benchmark 与生产观测的区别；
- 一次性能下降需要收集哪些不同层级的证据；
- GPU 利用率高为什么不一定代表业务效率高；
- 性能更快但成本更高的优化是否值得上线。

## 方向比较

完成六个模块后，对以下方向形成初步判断：

| 方向 | 主要问题 | 典型产出 | 环境门槛 |
|---|---|---|---|
| AI Platform / MLOps | 提升模型开发、部署和治理效率 | 工作流、控制面、平台服务 | 中 |
| GPU 集群基础设施 | 管理昂贵的大规模计算资源 | 调度、容量和可靠性系统 | 高 |
| 分布式训练系统 | 让大模型训练更快、更稳定 | 训练 Runtime、并行和容错 | 很高 |
| 推理系统与性能 | 降低延迟和单位推理成本 | 推理引擎、调度和性能优化 | 高 |
| Kernel / ML 编译器 | 提升底层执行效率 | Kernel、编译器和 Runtime 优化 | 高 |
| 网络与通信 | 减少跨 GPU 和跨节点通信瓶颈 | NCCL、RDMA 和拓扑优化 | 很高 |
| AI 数据与存储 | 持续向训练和推理供应数据 | 数据管线、checkpoint 和缓存 | 高 |
| 硬件—软件协同 | 共同设计模型执行与加速器 | Runtime、编译器和架构方案 | 最高 |

每个方向只比较：实际工作内容、核心能力、技术上限、岗位广度、硬件依赖和可迁移能力，不在此阶段制定专项课程。

## 候选方向体验

概念学习后选择两个候选方向，各完成一个小型代表性体验，例如：

- 阅读一段 vLLM 或 SGLang 调度源码；
- 分析一份 GPU Profiler 报告；
- 阅读一个短小的 Triton/CUDA Kernel；
- 推导一次集合通信成本；
- 分析一个 GPU 集群故障案例；
- 设计一个最小模型部署控制面；
- 分析训练任务的数据、通信和 checkpoint 路径。

体验的目的不是做项目，而是判断是否愿意长期处理这类问题、是否喜欢该抽象层级，以及是否能接受它的实验环境要求。

## 最小记录

全景阶段只维护以下三部分，不建立每日或每模块日志。

### 已经能解释

- 待学习后追加，每条只写一句。

### 未决问题

- 待学习后追加，只记录真正影响理解或方向选择的问题。

### 方向选择备忘录

全景和两个体验完成后，用一页回答：

- 第一候选方向及理由；
- 第二候选方向及理由；
- 只需保持工作知识的方向；
- 两次代表性体验带来的判断变化；
- 仍未验证的假设；
- 下一步只展开哪个方向的第一阶段。

## 完成条件

全景阶段只需要满足：

1. 能脱稿讲清 AI Infra 的主要层级和相互关系；
2. 能分别讲清一次训练任务和一次推理请求的端到端路径；
3. 能比较主要方向的工作内容和环境要求；
4. 完成两个候选方向的代表性体验；
5. 完成一页方向选择备忘录。

满足以上条件后再生成专项学习方案。此前的长期路线都只视为候选，不作为当前执行计划。
