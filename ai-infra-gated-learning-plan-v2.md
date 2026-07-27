# AI Infra 能力门禁学习方案 v2

> 本版在原方案基础上做了三处结构性改动，用于降低启动摩擦、提高早期验收可信度、并让状态机可执行：
> 1. **逃生阀机制**：区分硬验收与软验收，软验收失败可挂"补强任务"放行，避免整条依赖链锁死（见 §3.4、§6 各任务）。
> 2. **外部答案锚点**：将部分外部校准从 S6 前移到 S1，FND-08/FND-09 引入真实模型配置反推，杜绝纯自证式假门禁（见 §3.5、FND-08、FND-09）。
> 3. **可执行进度索引**：提供 `progress.yaml` schema 与校验脚本 `plan_check.py`，让状态机不依赖人肉记忆（见 §12）。

## 1. 定位

本方案的目标不是泛化的 AI Platform 或 MLOps，而是形成以下纵向能力：

```text
模型推理语义
  -> 推理 Runtime 与性能工程
  -> KV Cache、批处理、内存和调度
  -> GPU Kernel、Runtime 与编译器
  -> 多 GPU 并行、NCCL 与高速网络
  -> 集群级容量、可靠性和 cost-to-serve
```

目标角色是：**高性能分布式推理系统工程师，并具备向 GPU Runtime / Kernel 下钻的能力。**

贯穿全程的项目统一命名为 `Distributed Inference Lab`。它是用于复现、测量、分析和扩展真实推理引擎的实验工作台，不以重新实现一个生产级 vLLM 为目标。

建议的工作量分配原则是：

- 约 30% 用于最小自研实现，以建立机制理解；
- 约 50% 用于阅读、修改和扩展 vLLM、SGLang、PyTorch、Triton 等真实项目；
- 约 20% 用于基准、性能报告、故障分析和外部交流。

## 2. 不采用固定周期

本方案不规定周数、日期、每日时长或预计完成时间。进度只由任务验收决定：

- 阅读结束不代表完成；
- 投入时间不代表完成；
- 文档总结不代表实验通过；
- 计划运行结果、模拟输出和缺少原始数据的性能结论均不算证据；
- 任务的全部**硬验收**通过后，状态才可变为 `passed`；软验收未过时可通过挂补强任务放行（见 §3.4）；
- 前置任务全部 `passed` 后，后继任务才从 `locked` 变为 `ready`。

虽然时间不参与验收，但每次任务记录仍应包含**实际投入、会话次数，以及任务开始前预估的粗粒度量级（小时 / 天 / 周）**。当实际投入超过预估量级一档时，触发一次任务是否过大的复核。连续三次会话没有新增证据时，应停止继续堆时间，将任务标记为 `blocked` 并记录阻塞诊断。

同时进行的任务不得超过两个**实验任务**；知识任务可并行消化，不受此限，但处于 `in_progress` 的任务总数不超过四个。

## 3. 状态与证据

### 3.1 任务状态

| 状态 | 含义 |
|---|---|
| `locked` | 至少一个前置任务未通过 |
| `ready` | 前置依赖满足，可以开始 |
| `in_progress` | 已产生有效工作记录，但尚未通过全部验收 |
| `pending_experiment` | 知识或实验准备已完成，仍缺真实执行证据 |
| `blocked` | 多次尝试没有新增证据，或存在明确外部阻塞 |
| `passed` | 所有硬验收均有可核查证据 |
| `passed_with_reinforcement` | 硬验收通过、存在软验收缺口，已创建补强任务并登记（见 §3.4） |

状态变化由依赖规则确定，而不是由日期确定。所谓"解锁"是更新共享进度索引中的状态，不是假设 Markdown 会自动执行工作流。

### 3.2 证据等级

证据从强到弱依次为：

1. 自动化正确性测试和原始性能数据；
2. 可复现命令、Profiler 输出和故障日志；
3. 源代码、配置和 Git diff；
4. 公式推导、源码注释和架构图；
5. 文字解释和自测答案。

性能结论必须包含环境、变量、预热方法、重复运行结果、原始数据和统计方式。不得只保留最好结果。

### 3.3 累计复验

`passed` 表示任务在当时有充分证据，不表示能力永久有效。后续阶段的集成门禁必须再次使用早期能力，例如重新估算显存、解释 Roofline、拆解端到端延迟。复验失败时不抹去历史证据，但必须创建补强任务。

### 3.4 逃生阀：硬验收 / 软验收与补强任务（新增）

每个任务的验收项被显式标注为两类：

- **硬验收（H）**：涉及正确性、可复现性、机制理解底线的项。任一未过则任务不得 `passed`，也不得放行后继关键路径。
- **软验收（S）**：涉及深度、完备性、边界完善度的项。允许暂时缺口。

放行规则：

1. 当一个任务的全部 **H** 项通过、但存在未过的 **S** 项时，可将状态置为 `passed_with_reinforcement`，并**强制**创建一条 `reinforce-<原任务ID>` 补强任务登记到进度索引，写明缺口内容与触发复验的下游门禁。
2. `passed_with_reinforcement` 视同 `passed` 参与解锁，但对应补强任务必须在**引用该能力的下一个集成门禁之前**清偿，否则该集成门禁的前置检查判定失败。
3. 一个任务累计允许的未清偿补强任务不超过 2 条；超过则该任务回退为 `in_progress`。

该机制把 §3.3 的"复验失败可挂补强"理念，同样应用于**初次学习**，从而避免单点卡死锁死整条链。

### 3.5 外部答案锚点（新增）

从 S1 起，凡涉及可被客观核对的推导（shape、显存、KV Cache、FLOPs 等），验收必须包含**至少一个外部锚点对照**，而非纯自证：

- 使用真实开源模型的公开配置（`config.json` 中的 `num_hidden_layers`、`num_attention_heads`、`num_key_value_heads`、`hidden_size`、`head_dim`、`torch_dtype` 等）反推数值；
- 将自己的手算结果与"从权重文件实际大小 / 框架实际显存占用 / 官方模型卡"中可得到的数字对齐，误差需落在可解释范围内并说明来源。

外部校准不再等到 S6 才引入；S1 的知识门禁即要求最低限度的客观锚点。

## 4. 双机和云实验模式

每台机器通过被 Git 忽略的 `.machine.local.yaml` 声明 `machine_id`、`role` 和能力。

支持两种运行模式：

- 模式 A：一台学习机和一台实验机；
- 模式 B：两台本地机器都是学习机，按需租用 CPU 或 GPU 云服务器作为临时实验机。

学习机负责阅读、推导、源码分析、架构设计、自测和文档维护。实验机负责代码实现、编译、服务运行、Profiler、基准以及硬件实验。学习机不得用文字总结代替真实实验验收。

除角色外，实验任务还应声明所需能力，避免把所有实验机视为同一种资源：

| 能力标签 | 含义 |
|---|---|
| `linux_exec` | 可运行 Linux 命令和保存原始输出 |
| `pytorch_cpu` | 可运行和修改 PyTorch CPU 程序 |
| `cpp_build` | 可编译、测试和调试 C++ |
| `single_gpu` | 单张可用 GPU |
| `cuda_build` | 可构建 CUDA/Triton 实验 |
| `gpu_profiler` | 可运行相应 GPU Profiler |
| `multi_gpu` | 单机多 GPU |
| `nvlink` | 可验证 NVLink/NVSwitch 相关行为 |
| `multi_node` | 至少两个可通信计算节点 |
| `rdma` | 可进行 RDMA/InfiniBand 实验 |

实验记录必须包含 machine ID、Git commit、软硬件环境、完整命令、原始结果和失败记录。全仓库只维护一份共享进度索引，不按机器复制学习路线或任务日志。

知识任务可以在等待实验资源时继续，但第一阶段最终门禁以及后续每个工程阶段的门禁都必须包含真实实验。不得跨越阶段持续积累实验债务。

## 5. 总体阶段路线

后续阶段只在前一阶段通过后展开原子任务。展开下一阶段本身也是一项任务，用于根据已有证据、当时的硬件和上游项目状态制定任务图。

### S1：最小必要基础
建立进入真实推理系统的最低通行能力。详细任务见第 6 节。退出后解锁：`PLAN-S2`。

### S2：单 GPU 推理系统
核心方向：真实推理引擎的执行路径、continuous batching、KV Cache、prefix cache、内存管理、量化、调度和端到端性能分析。
阶段退出标准：能够修改一个真实推理引擎的非平凡执行路径，用正确性测试和原始性能数据说明影响，并完成一次端到端瓶颈分析。主要环境：`single_gpu`、`gpu_profiler`。

### S3：GPU 性能工程与 CUDA/Triton
核心方向：GPU 执行与内存模型、Roofline、访存合并、共享内存、Tensor Core、算子融合、CUDA/Triton、编译器和 Runtime。
阶段退出标准：至少一个推理关键算子通过正确性测试、Profiler 分析和可靠基线对比；能够解释成功优化和失败优化各一次。主要环境：`single_gpu`、`cuda_build`、`gpu_profiler`。

### S4：分布式推理与高速通信
核心方向：TP、PP、EP、MoE、NCCL 集合通信、NVLink、RDMA、拓扑、通信计算重叠、Prefill/Decode 分离和 KV Cache 传输。
阶段退出标准：完成真实多 GPU 或多节点实验，给出扩展效率、通信占比、瓶颈证据和失败边界，不能只展示吞吐增长。主要环境：`multi_gpu`，按任务增加 `nvlink`、`multi_node` 或 `rdma`。

### S5：集群级调度、可靠性与成本
核心方向：拓扑感知调度、异构 GPU、路由、容量规划、弹性、故障恢复、可观测性以及 cost-to-serve。
阶段退出标准：从请求、模型、Runtime、节点和 fleet 多层解释一个真实瓶颈或故障，并通过数据给出容量或成本决策。主要环境：`multi_gpu` 或 `multi_node`。

### S6：外部验证和行业级作品集
核心方向：上游贡献、公开复现、性能报告、故障复盘、跨硬件原理迁移和技术表达。
阶段退出标准至少包括：一个被上游接受的实质贡献、两份可复现性能报告、一个公开故障分析、一次跨后端或跨硬件迁移分析，以及对整条推理链路的完整讲解。

## 6. S1 最小必要基础：任务图

S1 只设置 11 个原子任务。知识任务与实验任务分离，使学习机可以推进理论，同时保证阶段最终状态不会在没有真实执行证据时通过。每个任务验收项标注 (H) 硬验收 / (S) 软验收。

```text
FND-01 推理系统全景 [ready]
  +-> FND-02 Linux 执行与内存模型
  |     +-> FND-03 Linux 观察实验
  |     +-> FND-06 最小 C++ 系统素养
  |             +-> FND-07 C++ 构建与调试实验
  |
  +-> FND-04 Tensor 与 PyTorch 语义
        +-> FND-05 Tensor 行为实验
        +-> FND-08 Transformer 推理与 KV Cache
                +-> FND-09 容量与性能模型 <- FND-02
                        +-> FND-10 基准方法与诊断设计

FND-03 + FND-05 + FND-07 + FND-08 + FND-09 + FND-10
  -> FND-11 S1 集成门禁
  -> PLAN-S2
```

### FND-01：推理系统全景
- 类型：`knowledge`；前置：无；允许角色：学习机、实验机
- 目标：建立从请求进入到 token 输出的最小端到端心智模型。
- 必须产出：一张端到端链路图和一份术语边界说明。
- 验收：
  - (H) 从请求、tokenization、batching、model forward、sampling 到 streaming，按顺序解释主要步骤；
  - (H) 在链路图上标出 CPU、GPU、内存、网络和存储可能参与的位置；
  - (S) 区分训练、离线推理和在线推理；
  - (S) 写出至少三个当前尚不能证明的性能假设。
- 证据：链路图、脱稿自测记录、假设列表。通过后解锁：`FND-02`、`FND-04`。

### FND-02：Linux 执行与内存模型
- 类型：`knowledge`；前置：`FND-01`；允许角色：学习机、实验机
- 目标：建立定位 CPU、内存和 I/O 问题所需的系统模型。
- 必须产出：进程、线程、调度、虚拟内存、page cache、I/O、CPU cache、NUMA 和内存带宽的关系图。
- 验收：
  - (H) 解释进程、线程、用户态和内核态的边界；
  - (H) 解释 runnable、running、sleeping、上下文切换和 load average；
  - (H) 区分虚拟内存、RSS、page cache、swap 和文件映射；
  - (S) 解释 cache locality、NUMA locality 和内存带宽为何会影响推理；
  - (H) 针对高 load 低 CPU、高 CPU 低吞吐、高 I/O wait 三种现象分别提出证据收集顺序。
- 证据：关系图、诊断决策表、自测答案。通过后解锁：`FND-03`、`FND-06`，并满足 `FND-09` 一个前置。

### FND-03：Linux 观察实验
- 类型：`experiment`；前置：`FND-02`；所需能力：`linux_exec`
- 目标：把系统概念映射到真实进程和原始指标。
- 必须产出：一个目标进程的可复现实验记录。
- 验收：
  - (H) 采集目标进程的 PID、线程、状态、CPU、RSS、上下文切换、文件描述符和 I/O 证据；
  - (H) 制造至少一种 CPU、内存或 I/O 行为并观察指标变化；
  - (S) 对观察结果提出解释，并指出指标不能证明的内容；
  - (H) 从新终端按记录能够复现主要结果。
- 证据：完整命令、原始输出、machine ID、Git commit、环境说明。通过后满足 `FND-11` 一个前置。

### FND-04：Tensor 与 PyTorch 语义
- 类型：`knowledge`；前置：`FND-01`；允许角色：学习机、实验机
- 目标：能沿推理程序跟踪 tensor 的语义和潜在内存行为。
- 必须产出：tensor shape、dtype、device、stride、storage、view、copy 和 contiguous 的关系说明。
- 验收：
  - (H) 对给定 tensor 操作逐步推导 shape、dtype 和 device；
  - (H) 根据 stride 判断一个简单 tensor 是否连续；
  - (H) 区分 view、reshape、transpose、contiguous 和 clone 可能产生的行为；
  - (S) 解释 dtype 转换和 device 传输可能引入的成本；
  - (S) 阅读一个最小 PyTorch forward，标注主要中间 tensor。
- 证据：手工推导、带注释的最小程序阅读记录、自测答案。通过后解锁：`FND-05`、`FND-08`。

### FND-05：Tensor 行为实验
- 类型：`experiment`；前置：`FND-04`；所需能力：`pytorch_cpu`
- 目标：用运行结果验证 tensor shape、stride、view/copy 和 dtype 判断。
- 必须产出：一组最小可复现 tensor 实验和正确性断言。
- 验收：
  - (H) 实际打印并核对 shape、stride、dtype、device、data pointer 或 storage 关系；
  - (H) 构造至少一个非连续 tensor，并验证哪些操作共享存储、复制或失败；
  - (S) 比较至少两种 dtype 的内存占用；
  - (H) 使用断言把预期行为固化为可重复测试。
- 证据：源代码、测试输出、machine ID、Git commit、环境说明。通过后满足 `FND-11` 一个前置。

### FND-06：最小 C++ 系统素养
- 类型：`knowledge`；前置：`FND-02`；允许角色：学习机、实验机
- 目标：具备阅读推理引擎底层代码所需的最低 C++ 语言和构建心智模型。
- 必须产出：一份小型 C++ 代码阅读注释和构建链路图。
- 验收：
  - (H) 区分栈、堆、值语义、引用、指针和对象生命周期；
  - (H) 解释 RAII、`unique_ptr`、`shared_ptr` 和移动语义解决的问题；
  - (H) 解释头文件、翻译单元、编译、链接和动态库之间的关系；
  - (S) 阅读一段包含类、模板容器和资源管理的小程序并描述执行路径；
  - (S) 明确高级模板元编程不属于本阶段目标。
- 证据：代码注释、构建链路图、自测答案。通过后解锁：`FND-07`。

### FND-07：C++ 构建与调试实验
- 类型：`experiment`；前置：`FND-06`；所需能力：`cpp_build`
- 目标：验证最低限度的 C++ 修改、构建、测试和调试能力。
- 必须产出：一个小型 C++ 程序、测试和调试记录。
- 验收：
  - (H) 修改一个包含资源生命周期或内存访问的小程序；
  - (H) 从干净构建目录完成编译和链接；
  - (H) 运行自动化测试；
  - (H) 使用调试器或 sanitizer 定位并修复一个预先构造的问题；
  - (S) 解释修复发生在哪个抽象层，而不只给出修改结果。
- 证据：源码、构建命令、测试输出、调试证据、machine ID、Git commit。通过后满足 `FND-11` 一个前置。

### FND-08：Transformer 推理与 KV Cache
- 类型：`knowledge`；前置：`FND-04`；允许角色：学习机、实验机
- 目标：理解自回归推理中真正影响执行和内存的模型语义。
- 必须产出：简化 Attention 与 KV Cache 推导、prefill/decode 时序图，以及一次外部锚点对照。
- 验收：
  - (H) 推导 Q、K、V、attention score 和输出的 shape；
  - (H) 解释 causal mask 和自回归生成；
  - (H) 比较无 KV Cache 与使用 KV Cache 时每步重复计算的差异；
  - (H) 解释 prefill 与 decode 的输入形态、并行度和资源特征；
  - (H) 用伪代码或公式表达简化 Attention 和 KV Cache 更新；
  - (H) **外部锚点**：取一个真实开源模型的 `config.json`，用其真实层数 / head 数 / KV head 数 / hidden size 推导单层 Q/K/V 的 shape，并核对与实现或模型卡是否一致（见 §3.5）；
  - (S) 列出该简化模型没有覆盖的真实模型特性。
- 证据：公式、shape 推导、伪代码、时序图、外部锚点核对记录、自测记录。通过后解锁：`FND-09`，并满足 `FND-11` 一个前置。

### FND-09：容量与性能模型
- 类型：`knowledge`；前置：`FND-02`、`FND-08`；允许角色：学习机、实验机
- 目标：在运行前能够对显存、计算量、访存量和瓶颈做数量级预测。
- 必须产出：一份带公式的容量计算表、性能假设表，以及外部锚点对照。
- 验收：
  - (H) 根据参数量和 dtype 估算权重显存；
  - (H) 根据层数、KV head、head dimension、序列长度、batch 和 dtype 估算 KV Cache；
  - (H) 对简化 Attention 或矩阵乘给出 FLOPs 与主要访存量估算；
  - (H) 计算 arithmetic intensity，并用 Roofline 思想形成可验证的瓶颈假设；
  - (H) **外部锚点**：对一个真实开源模型，用公开配置反推权重显存并与实际权重文件大小（或官方模型卡显存要求）对齐，误差落在可解释范围并说明来源（见 §3.5）；
  - (S) 解释为什么 prefill 更可能偏计算受限、decode 更可能偏内存带宽受限，并说明例外；
  - (S) 对估算误差来源作出边界说明。
- 证据：公式、计算表、至少两个手算样例、外部锚点核对记录、假设与边界。通过后解锁：`FND-10`，并满足 `FND-11` 一个前置。

### FND-10：基准方法与诊断设计
- 类型：`knowledge`；前置：`FND-09`；允许角色：学习机、实验机
- 目标：能够设计不会轻易产生误导结论的最小推理实验。
- 必须产出：一份可交给实验机执行的基准协议和诊断决策树。
- 验收：
  - (H) 定义 TTFT、TPOT、ITL、端到端延迟、吞吐、错误率和 P50/P95/P99；
  - (H) 说明平均值、中位数和高分位数各自可能隐藏的问题；
  - (H) 明确自变量、控制变量、预热、重复运行和原始数据格式；
  - (H) 设计用于比较 batch、序列长度或 dtype 的单变量实验；
  - (S) 针对 CPU、内存、I/O 和 GPU 异常列出工具、指标及其证据边界；
  - (S) 明确 Linux 工具、PyTorch Profiler 和 GPU Profiler 分别能回答什么问题。
- 证据：实验协议、数据 schema、诊断决策树、预期失败场景。通过后满足 `FND-11` 一个前置。

### FND-11：S1 集成门禁
- 类型：`integration`；前置：`FND-03`、`FND-05`、`FND-07`、`FND-08`、`FND-09`、`FND-10`
- 所需能力：`linux_exec`、`pytorch_cpu`、`cpp_build`；`single_gpu` 为可选增强，不是 S1 强制条件
- **前置检查**：所有前置任务必须为 `passed` 或 `passed_with_reinforcement`；且引用到的能力对应的**未清偿补强任务必须为 0**（见 §3.4）。
- 目标：用一次真实执行把系统、tensor、模型和性能方法连接起来。
- 必须产出：最小 Attention/KV Cache 实现、基准原始数据、分析报告和复现说明。
- 验收：
  - (H) 在实验机实现并测试简化 Attention 和增量 KV Cache；
  - (H) 使用自动化断言验证 shape、dtype 和数值正确性；
  - (H) 对至少两个输入长度和两个 batch 条件运行基准；
  - (H) 保留每次原始结果，而不是只保留汇总；
  - (H) 将实测内存或耗时趋势与 `FND-09` 的预测比较；
  - (H) 解释至少一个预测一致点和一个偏差；
  - (S) 给出从请求到 token 的完整链路图，并标注当前能够观测和不能观测的位置；
  - (H) 从新终端按文档复现测试和基准；
  - (H) 记录 machine ID、Git commit、环境和失败尝试；
  - (H) 完成一次不依赖笔记的综合口头或书面自测。
- 证据：源码、测试、原始数据、分析报告、复现输出、综合自测。通过后解锁：`PLAN-S2`。

## 7. 第一阶段初始状态

| 任务 | 初始状态 | 原因 |
|---|---|---|
| `FND-01` | `ready` | 无前置依赖 |
| `FND-02` | `locked` | 等待 `FND-01` |
| `FND-03` | `locked` | 等待 `FND-02` |
| `FND-04` | `locked` | 等待 `FND-01` |
| `FND-05` | `locked` | 等待 `FND-04` |
| `FND-06` | `locked` | 等待 `FND-02` |
| `FND-07` | `locked` | 等待 `FND-06` |
| `FND-08` | `locked` | 等待 `FND-04` |
| `FND-09` | `locked` | 等待 `FND-02`、`FND-08` |
| `FND-10` | `locked` | 等待 `FND-09` |
| `FND-11` | `locked` | 等待六项集成前置任务 |
| `PLAN-S2` | `locked` | 等待 `FND-11` |

任何任务在没有实际证据前都不得初始化为 `passed`。

## 8. 阶段滚动展开规则

`PLAN-S2` 以及后续 `PLAN-SN` 任务应执行以下流程：

1. 汇总上一阶段通过证据、失败实验、仍不稳定的能力，以及**所有未清偿补强任务**；
2. 确认当前可用机器的能力标签；
3. 选择真实上游项目、模型、硬件和基线；
4. 将下一阶段拆成不超过 12 个原子任务；
5. 同时设计知识任务、真实实验任务和阶段集成门禁；
6. 将上一阶段能力嵌入新的集成验收，形成累计复验；
7. 检查依赖图不存在循环、不可达任务或无证据验收；
8. 只展开紧邻的下一阶段，不提前细化更远阶段。

## 9. 外部验证原则

从 S2 开始，单纯自我验收不足以证明行业能力（S1 已通过 §3.5 引入最低限度外部锚点）。任务设计应逐步加入：

- 复现官方 benchmark 或论文结果；
- 与至少一个真实推理引擎对照；
- 阅读并修改上游源码；
- 提交可审查的 issue、复现、文档、测试或 PR；
- 保存 reviewer 反馈和修改过程；
- 解释一次有效优化和一次无效或退化的优化。

最终作品集中的性能数字必须可复现、有基线、有环境说明，并能够回答"为什么有效"和"在什么边界下失效"。

## 10. 非目标

本路线不以以下成果作为核心成功标准：

- 功能繁多的管理后台；
- 通用 CRUD 型 Mini AI Platform；
- 只完成 Kubernetes 部署；
- 没有性能证据的 Gateway；
- 与真实引擎没有联系的长期玩具框架；
- 只会孤立编写 Kernel，却不能解释端到端影响；
- 只会调用 Profiler，却不能形成和验证性能假设。

Kubernetes、Gateway、可观测性和平台治理会在需要支撑推理性能、可靠性或成本问题时学习，而不是与主线平行扩张。

## 11. 完成定义

整条路线的完成不是"所有文档已阅读"，而是能够：

1. 从模型结构推导推理计算和内存行为；
2. 从请求级指标定位到 Runtime、通信或 Kernel；
3. 在单 GPU 和分布式环境中设计可复现实验；
4. 修改真实系统并证明正确性、收益和边界；
5. 对延迟、吞吐、容量、可靠性和成本作出有数据支持的权衡；
6. 通过上游贡献和公开报告获得外部验证；
7. 完整解释一次成功优化、一次失败优化和一个真实故障。

达到这些条件后，`Distributed Inference Lab` 才是能力证据，而不仅是一个学习仓库。

## 12. 可执行进度索引（新增）

状态机不依赖人肉记忆。全仓库维护唯一一份 `progress.yaml`，配套校验脚本 `plan_check.py` 在每次提交前运行。

### 12.1 `progress.yaml` schema

```yaml
# progress.yaml —— 全仓库唯一进度索引
tasks:
  FND-01:
    type: knowledge          # knowledge | experiment | integration
    status: ready            # locked|ready|in_progress|pending_experiment|blocked|passed|passed_with_reinforcement
    deps: []                 # 前置任务 ID 列表
    caps: []                 # 所需能力标签
    estimate: hours          # hours|days|weeks 粗粒度预估
    sessions: 0              # 累计会话次数
    evidence:                # 证据文件路径（相对仓库根）
      - path: null
        kind: null           # test|rawdata|repro|code|derivation|selftest
    reinforcements: []       # 未清偿补强任务 ID，如 reinforce-FND-08
  FND-02:
    type: knowledge
    status: locked
    deps: [FND-01]
    caps: []
    estimate: days
    sessions: 0
    evidence: []
    reinforcements: []
  # ... 其余任务同构
```

### 12.2 校验脚本 `plan_check.py`

用法：`uv run plan_check.py progress.yaml`

```python
#!/usr/bin/env python3
"""Distributed Inference Lab 进度索引校验器。
校验项：
  1. 依赖图无环、无指向不存在任务的边；
  2. 无 evidence 的任务不得为 passed / passed_with_reinforcement；
  3. status=ready 的任务其全部 deps 必须已 passed/passed_with_reinforcement；
  4. deps 未全部通过的任务必须为 locked；
  5. passed_with_reinforcement 必须至少登记一条 reinforcements；
  6. 单任务未清偿补强不得超过 2 条；
  7. integration 任务的前置若存在未清偿补强，则该门禁不可置为 ready 以上；
  8. 打印当前真正可执行（ready）的任务集合。
"""
import sys, yaml

PASSED = {"passed", "passed_with_reinforcement"}

def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["tasks"]

def has_cycle(tasks):
    color = {}  # 0=visiting,1=done
    def dfs(n, stack):
        if color.get(n) == 1: return False
        if n in stack: return True
        stack.add(n)
        for d in tasks[n].get("deps", []):
            if d in tasks and dfs(d, stack): return True
        stack.discard(n); color[n] = 1
        return False
    return any(dfs(n, set()) for n in tasks)

def check(tasks):
    errs, warns = [], []
    ids = set(tasks)

    # 1. 边有效性 + 环
    for tid, t in tasks.items():
        for d in t.get("deps", []):
            if d not in ids:
                errs.append(f"{tid}: 依赖不存在的任务 {d}")
    if has_cycle(tasks):
        errs.append("依赖图存在环")

    for tid, t in tasks.items():
        st = t.get("status")
        deps = t.get("deps", [])
        deps_ok = all(tasks[d]["status"] in PASSED for d in deps if d in ids)
        ev = [e for e in (t.get("evidence") or []) if e and e.get("path")]
        reinf = t.get("reinforcements") or []

        # 2. passed 必须有证据
        if st in PASSED and not ev:
            errs.append(f"{tid}: 状态 {st} 但无 evidence")
        # 3./4. ready 与 locked 的一致性
        if st == "ready" and not deps_ok:
            errs.append(f"{tid}: 标为 ready 但前置未全部通过")
        if st == "locked" and deps_ok and deps:
            warns.append(f"{tid}: 前置已全通过，可从 locked 提升为 ready")
        # 5. pwr 必须有补强登记
        if st == "passed_with_reinforcement" and not reinf:
            errs.append(f"{tid}: passed_with_reinforcement 但未登记补强任务")
        # 6. 补强上限
        if len(reinf) > 2:
            errs.append(f"{tid}: 未清偿补强 {len(reinf)} 条，超过上限 2")
        # 7. integration 门禁：前置补强需清偿
        if t.get("type") == "integration" and st not in {"locked"}:
            pending = [d for d in deps if d in ids and tasks[d].get("reinforcements")]
            if pending:
                errs.append(f"{tid}: 集成门禁前置存在未清偿补强 {pending}")

    ready = sorted(
        tid for tid, t in tasks.items()
        if t.get("status") == "ready"
    )
    return errs, warns, ready

def main():
    if len(sys.argv) < 2:
        print("usage: uv run plan_check.py progress.yaml"); sys.exit(2)
    tasks = load(sys.argv[1])
    errs, warns, ready = check(tasks)
    for w in warns: print(f"[WARN] {w}")
    for e in errs:  print(f"[FAIL] {e}")
    print(f"\n可执行(ready)任务: {ready or '无'}")
    sys.exit(1 if errs else 0)

if __name__ == "__main__":
    main()
```

脚本以非零退出码阻止"无证据 passed""依赖成环""带未清偿补强闯集成门禁"等违规状态进入仓库，把 §3 的规则从纸面变成可强制执行的约束。
