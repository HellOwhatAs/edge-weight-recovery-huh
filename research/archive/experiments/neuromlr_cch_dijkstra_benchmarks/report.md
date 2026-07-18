# 最终 NeuroMLR-Greedy 质量与 CCH/Dijkstra 效率实验

## 结论摘要

本实验完成了两项互相分离的比较：

1. 在给定真实首边和真实末边的同一 edge-to-edge 任务上，以完全相同的
   原始道路轨迹比较本项目与 NeuroMLR-Greedy；
2. 在本项目内部固定图表示、数据、初始化、优化器、整数权重和线程数，
   仅替换 CCH 与普通 Dijkstra shortest-path oracle。

500 条共同 test 轨迹上的主质量结果为：本项目 Edge F1 `0.766015`，
NeuroMLR-Greedy `0.768496`。本项目低 `0.002481`，即约 `0.25` 个百分点，
因此结论是“达到接近的路线质量”，不是质量提升。

在严格保证 21 个训练状态的最短距离、解码路径、次梯度、objective 和最终
权重全部一致的 4,971-OD、20-update workload 上，Dijkstra/CCH 的
setup-plus-core 时间比为 `1.68×`。在同一 checkpoint 和 500 条单线程
node-to-node 查询上，query-only 推理时间比为 `8.05×`。

外部基线按用户在实验开始后的最终范围决定改为 NeuroMLR-Greedy；
NeuroMLR-Dijkstra 未运行，不能从本报告推导其质量或时间。

## 仓库、协议与防止 test 泄漏

- 起始提交：`d5c8dd59f56b0fd783ef00a3d1cfe2428e4e3863`
- 实验分支：`neuromlr-cch-dijkstra-benchmarks`
- 随机种子：`20260716`
- 冻结质量协议：`protocol.json`，SHA-256 `c386dab9…`
- NeuroMLR 官方仓库：`https://github.com/idea-iitd/NeuroMLR`
- NeuroMLR commit：`c45e3b5811e5a59b36e4682307d2196c02dac360`

checkpoint、数据过滤、validation 结果和 oracle pilot 全部冻结后才生成
hash-bound test unlock。`test_access_receipt.json` 记录 test 源 pickle 只被
解码一次；再次调用清单生成器会因 receipt 已存在而失败。

原协议中的 50,000-OD 训练效率 workload 在 test 之后暴露出动态 tie
导致的模型分叉。质量数据、checkpoint 和 test 结果均未修改；效率部分的
修正单独记录在 `efficiency_protocol_amendment.json`，且只使用共同 train
清单。

## 公平质量协议

对完整原始道路序列 `(e1, e2, ..., eL)`：

- 两个方法均已知真实首边 `e1` 和真实末边 `eL`；
- NeuroMLR 训练仍包含正确的 `L-1` 个下一道路预测目标；
- `L-1` 不表示评价时删除一条道路，预测和评价均使用包含首尾边的完整序列；
- 本项目把 `e1`、`eL` 固定为 line graph 的 source/target state，使用已学习的
  `edge_transition_arcs` 权重，并解码完整原始道路 ID；
- 主指标为逐轨迹、非长度加权的 raw-edge set macro Precision、Recall、F1、
  Jaccard，以及完整序列 Exact Match。

NeuroMLR 原生按物理道路长度加权的 Precision/Recall 仅作 supplementary。
本项目原生 node-to-node 推荐也只单列报告，未与 NeuroMLR 的 edge-to-edge
结果混为一列。

## 共同数据

统一过滤要求道路 ID 可一对一映射、相邻道路连续、至少 5 条道路，并在任一
原始图节点重复时删除整条轨迹。两个方法使用相同 manifest ID 和相同 raw
edge 序列。

| Split | 源记录 | 合格 | 最终使用 | 合格覆盖率 | 环路删除 | 过短删除 |
|---|---:|---:|---:|---:|---:|---:|
| Train | 785,709 | 605,935 | 605,935 | 77.12% | 160,877 | 18,897 |
| Validation | 20,000 | 15,399 | 500 | 77.00% | 4,143 | 458 |
| Test | 10,000 | 7,678 | 500 | 76.78% | 2,092 | 230 |

三个 split 的不连续、越界/不可表示和空路径计数均为 0。manifest SHA-256：

- train：`4cd26d89fe7baf6c49155aa9860ee4eaa4ceb98791af4468a2eb05f871252a4b`
- validation：`2b631a725e41cee3b5ce7fd4b072f4138ba22ef3a2b38dedbd2804333bb9663a`
- test：`d340e0715853f3245538f00525f4edeed6edca19c5e5326253f160baace1c5a9`

完整过滤审计见 `train_audit.json`、`validation_audit.json` 和
`test_audit.json`。

## 模型和 checkpoint 选择

本项目保持推荐配置不变：完整共同北京训练集、`edge_transition_arcs`、长度
初始化、全局静态权重、`relative_projected_subgradient`、`eta0=0.0002`、
`lambda=100000`。500 次更新中每 25 次保存 checkpoint，只按共同 validation
edge-to-edge F1、Exact Match、最早 update 排名。16 Rayon 线程来自预冻结
8-vs-16 全 workload 试跑：查询中位数从 `5289.132 ms` 降至
`3514.573 ms`，最终权重完全相同。

NeuroMLR 保留上游 GCN/MLP 结构和默认维度：128 维 Lipschitz 特征、1 个配置
GNN layer、3 个宽度 256 的 MLP hidden layer、Adam AMSGrad、学习率
`0.001`、batch size 32、50 epochs、无 traffic feature。未做大范围超参搜索。

| 方法 | 选中 checkpoint | Validation F1 | Validation Exact |
|---|---:|---:|---:|
| 本项目 edge-to-edge | update 200 | 0.777764 | 0.486 |
| NeuroMLR-Greedy | epoch 45 | 0.775495 | 0.520 |

NeuroMLR validation 有 6/500 条 Greedy rollout 未同时保持真实首尾边；这些
预测未被修补，仍参与指标。

## NeuroMLR 复现适配

官方 checkout 保持 clean。适配器只处理执行和数据兼容性，主要包括：

- 从上游实际存在的 `model_all.Model` 导入模型；
- 训练模式只接受共同 train/validation，禁止提前加载 test；
- 禁用上游会合并平行原始道路 ID 的 `condense_edges()`；
- 固定 Python、NumPy、Torch 和 CUDA 随机种子；
- 用 state dict、模型配置、上游 commit 和图 hash 保存 checkpoint；
- 使用 Python 3.12、Torch 2.7.1+cu128、PyG 2.6.1 适配 RTX 5060 Ti；
- Lipschitz 初始化对平行反向边取最小成本，避免 SciPy COO 重复项求和；
- Greedy 保留上游 300-step 上限和 closest-point truncation，不补齐目的边。

全部修改与原因见 `upstream_adaptations.md`，依赖锁定见
`environment/requirements.lock`。

## 最终质量结果

| 方法 | Edge Precision | Edge Recall | Edge F1 | Exact Match | Edge Jaccard |
|---|---:|---:|---:|---:|---:|
| 本项目 edge-to-edge | 0.778879 | 0.759719 | 0.766015 | 0.476 | 0.698639 |
| NeuroMLR-Greedy | 0.774549 | 0.768722 | 0.768496 | 0.508 | 0.708265 |
| 本项目 − NeuroMLR | +0.004330 | −0.009003 | −0.002481 | −0.032 | −0.009626 |

NeuroMLR-Greedy 在 test 中有 4/500 条 endpoint mismatch。本项目
edge-to-edge 查询按定义固定两个 endpoint，因此为 0。NeuroMLR 原生长度加权
supplementary Precision/Recall 为 `0.783657 / 0.778932`，不作为主结果。

本项目原生 node-to-node supplementary 结果为 F1 `0.694946`、Exact Match
`0.412`。该值对应不同任务，不能与上表作公平优劣比较。

两种主方法的完整 raw-road 预测分别位于：

- `generated/predictions/project-edge-to-edge-test.jsonl`
- `generated/predictions/neuromlr-greedy-test.jsonl`

其 SHA-256 和共同评价结果已写入 `summary.json`。

## NeuroMLR 时间（supplementary）

在同一机器上，NeuroMLR 50 epochs 的纯训练时间为 `20,270.69 s`，总时间
`20,281.93 s`，峰值 RSS `3,066,064 KiB`。Greedy 对 500 条 test 路径做
1 次 warm-up 和 5 次重复，平均核心预测时间 `1.6738 s`，即
`3.348 ms/query`、`298.72 query/s`；峰值 CUDA allocation 为
`404,156,416 bytes`。

本项目共同数据 500-update 质量训练耗时 `1,871.58 s`。两者的 epoch/update
语义、GPU/CPU 利用、预处理和计时边界不同，因此不表述为算法级加速倍数。
按最终范围决定，未运行 NeuroMLR-Dijkstra 推理。

## 普通 Dijkstra oracle 与一致性

新增 Dijkstra 使用与 CCH 相同的 routing topology 和同一个正整数 `u32`
量化权重向量。查询缓冲区复用，并只重置实际访问节点；这避免把朴素的全图
数组清零成本错误计入 Dijkstra，从而人为放大 CCH 优势。

选中 checkpoint 的 500 条 validation edge-to-edge 查询中，两种 oracle 的
距离和完整路线 500/500 全部相同。最终 node-to-node test 中距离错误为 0；
499/500 路线相同，`test:000000281` 是唯一等距 tie-breaking 路线差异。

## 训练效率

### 动态 tie 与 workload 修正

最初 50,000-OD workload 只检查初始路径相同。正式试跑在 state 2 出现新的
等距路线差异，state 9 后因模型已经分叉而出现距离和差异，最终权重最大差
`296,572.74`。这是 tie-breaking 诱发的优化轨迹分叉，不是同一权重下的距离
错误，该 run 已排除。

随后实现固定点筛选：在共享 CCH 训练轨迹的 21 个状态逐 OD 比较 CCH 与
Dijkstra，删除任一状态出现等距路径差异的 OD，再从相同初始化重放。50k
筛选约 20 分钟未完成，因此不报告其时间。可完整收敛的共同 workload 从
5,000 个初始 OD 经 8 轮删除 29 个动态 tie OD，冻结为 4,971 OD / 4,979 条
观测。报告不把该结果外推为全量训练实测。

### 正式测量

4 次独立测量交替使用 CCH-first / Dijkstra-first，均为 16 线程、20 updates、
21 个 oracle state。每次的距离和、predicted-count checksum、objective 和最终
权重均完全相同；最终权重 SHA-256 四次也一致。

| 项目（4 次均值） | CCH | Dijkstra |
|---|---:|---:|
| 一次性 topology/adjacency setup | 1.895930 s | 0.000777 s |
| 21 次 customization/metric setup | 0.934938 s | 0.022595 s |
| 21 批最短路查询 | 0.879651 s | 6.243287 s |
| 20 次 optimizer update | 0.026736 s | 0.026910 s |
| Core end-to-end | 1.882789 s | 6.334542 s |
| Setup + core | 3.778719 s | 6.335320 s |
| 独立进程峰值 RSS | 112,380 KiB | 152,572 KiB |

- 主训练加速口径：`6.335320 / 3.778719 = 1.68×`，CCH 时间减少约 40.4%。
- 不含一次性 topology setup 的 core 比值：`3.36×`。
- 只看最短路查询阶段：`7.10×`。

简历中的训练倍数只能使用主口径，并必须同时注明 4,971-OD、20-update、
16-thread workload。

## 固定权重推理效率

推理固定 update-200 checkpoint、相同的 500 条 test node-to-node 查询、1 个
线程、1 次 warm-up 和 5 次测量。文件读取和 checkpoint 解析不进入核心查询
时间；CCH topology preprocessing 和 customization 单列。

| 项目 | CCH | Dijkstra |
|---|---:|---:|
| 一次性 topology/adjacency setup | 1.808506 s | 0.000825 s |
| customization/metric setup | 0.043216 s | 0.001074 s |
| 500 条查询均值 | 0.029860 s | 0.240334 s |
| 单查询均值 | 0.0597 ms | 0.4807 ms |
| 吞吐量 | 16,744.66 query/s | 2,080.44 query/s |
| 峰值 RSS | 62,068 KiB | 66,280 KiB |

query-only 比值为 `0.240334 / 0.029860 = 8.05×`，CCH 核心查询时间减少约
87.6%。500 条查询的短 workload 不把 CCH setup 摊入 query-only 倍数；setup
已按要求单独报告。

## 可复现性与原始记录

- 机器和线程：`environment/hardware.json`
- Python 环境：`environment/requirements.lock`
- 上游文件身份：`upstream.json`
- 完整命令：`commands.md`
- 冻结协议和修正：`protocol.json`、`efficiency_protocol_amendment.json`
- test 门禁：`validation_evidence.json`、`test_unlock.json`、
  `test_access_receipt.json`
- 排除运行：`excluded_runs.json`
- 机器可读最终结果：`summary.json`
- 大型原始 benchmark、预测和 manifest：`generated/`（Git ignore，身份 hash
  保留在审计和 summary 中）

## 局限

- 外部主基线是用户最终指定的 NeuroMLR-Greedy，不包含 NeuroMLR-Dijkstra。
- 只有一个固定随机种子，不提供多 seed 方差。
- 训练效率只实测严格一致的 4,971-OD workload；不能声称是全量 Dijkstra
  训练时间。
- CCH 的优势取决于图规模、路径解码、customization 频率和查询数量；当前
  完整路径任务的实测不支持“几十倍”结论。

## 质量检查

`cargo fmt --check`、release locked build、`cargo test --all-targets`（50 个库
测试及 14 个 contract 测试）、严格 clippy `-D warnings`、Python compile、
4 个 Python unittest 和 `git diff --check` 均通过。

## 可直接用于简历的候选表述

1. 在相同的完整北京共同训练集和首尾道路固定协议下，本方法与
   NeuroMLR-Greedy 达到接近的路线复现质量（Edge F1 `0.7660` vs
   `0.7685`，相差 `0.25` 个百分点）。
2. 在严格保持 21 个训练状态、次梯度与最终模型完全一致的 4,971-OD、
   20-update workload 上，CCH 相较普通 Dijkstra 将包含拓扑预处理的训练
   setup-plus-core 时间加速 `1.68×`。
3. 在同一 checkpoint、500 条单线程 node-to-node 查询上，CCH 核心最短路
   推理相较普通 Dijkstra 加速 `8.05×`。

不要把第 1 条改写为质量超过 NeuroMLR，也不要省略第 2、3 条的 workload 和
计时边界。
