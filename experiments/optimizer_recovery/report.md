# 北京 10% 相对权重优化恢复报告

## 结论

Review 指出的优化退化已经复现、定位并解决。问题不在数据、图构造或初始最短路：旧 edge-only 实现与当前统一实现的 `original_edges` update 0 validation mean regret 均为 `650449.9359347331`，逐位相同。退化来自把无量纲乘子优化改成了直接权重空间的统一欧氏步长，同时改变了坐标尺度和正则化几何。

统一训练器现在支持图表示无关的相对权重优化：

```text
q_i = w_i / w_i^0
w_i = w_i^0 q_i

g_q_i = w_i^0 (n_i^obs - n_i^pred) / N
        + lambda / m (q_i - 1)

q <- project(q - eta_k g_q)
eta_k = eta0 / sqrt(k + 1)
```

checkpoint 仍只保存一个直接权重向量 `w`；没有恢复旧的 q/r 双块模型，没有表示专用状态，也没有修改原图或 line graph 定义。等价的直接权重表述是对相对正则化目标使用 `diag((w_i^0)^2)` 对角预条件。

北京 10% 回归验证通过：

- `original_edges` 的 Edge F1 从 update 0 的 0.589902 提高到 0.685404，Exact Match 从 0.336643 提高到 0.373640；
- 旧 edge-only 最佳结果为 F1 0.682145、Exact Match 0.368454，新通用实现已复现到同一水平并略高；
- `edge_transition_arcs` 的 F1 从 0.603467 提高到 0.694125，Exact Match 从 0.346952 提高到 0.377245；
- 在共同解码指标上，训练后的 line graph 比 original edges 高 0.008720 F1 和 0.003605 Exact Match。

因此，“训练几乎没有效果”的阻断问题已经解除。后续 NeuroMLR 对比应使用 `edge_transition_arcs + relative_projected_subgradient`，不得沿用直接权重欧氏优化结果。不过 line graph 的最低 validation objective 位于本次注册预算的最后 update 299，学习有效已经确认，收敛仍未确认。

## 根因排查

### 排除数据、初始化和图查询变化

旧结果来自代码提交 `62b23eb9471ce490f05512af3c942235a5962410`，其机器摘要保存在提交 `fa26aa84f3c5e528f662835917d288e4f4368ebb` 的 `experiments/summaries/beijing_edge_only_10pct.json`。旧实现与当前回归使用相同的 train/validation 文件、过滤后样本数、unique OD 数、原图初始权重和 CCH 查询协议。

最强对照信号如下：

| 对照 | 旧 edge-only | 当前统一 `original_edges` | 差值 |
|---|---:|---:|---:|
| Update 0 validation mean regret | 650449.9359347331 | 650449.9359347331 | 0，逐位相同 |
| 最后状态 299 mean regret | 311137.484885 | 311137.504282 | +0.019397 |
| 所选 checkpoint mean regret | 310343.733747（state 289） | 310213.247090（update 290） | -130.486657 |
| 所选 Edge F1 | 0.682145 | 0.685404 | +0.003259 |
| 所选 Exact Match | 0.368454 | 0.373640 | +0.005186 |

最后状态的 mean regret 只差约 0.019，说明相对坐标更新轨迹已在新的表示无关训练器中恢复。所选 checkpoint 的小差异来自当前保存 cadence 使用 update 290，而旧摘要选择 state 289；本次目标是回归到旧结果附近，不声称 checkpoint 逐字节相同。

### 确认几何变化

直接权重优化使用：

```text
g_w_i = (n_i^obs - n_i^pred) / N
        + lambda / m (w_i - w_i^0)
```

所有道路权重坐标共享同一绝对步长。由于 `w_i^0` 的量级跨度很大，这与旧乘子优化不是变量改名，而是改变了梯度尺度、条件数和正则化含义。

另有一个 review 公式细节已经核查：仓库中的直接权重正则梯度符号实际是正号 `+ lambda/m * (w-w0)`，不是 review 文本中写出的负号。这个笔误不改变主要诊断；真正的退化来自尺度和正则化几何。

## 实现

- `relative_projected_subgradient` 显式选择相对坐标；历史 `projected_subgradient` 仍保留为直接权重语义，避免旧配置被静默重解释。
- 相对模式使用 `lambda/(2m) ||w/w0 - 1||^2`，更新、objective 日志和 checkpoint 选择使用同一个正则化定义。
- 两种图表示通过同一个 `OptimizerGeometry::RelativeWeights` 路径，不检查坐标代表原始边还是转移弧。
- 日志新增 optimizer kind、参数化名称和 `max_abs_parameter_delta`；矩阵运行器验证配置与完成事件的 optimizer kind 一致。
- 单元与合同测试覆盖直接/相对公式、投影、正则化配对、两个图表示以及 checkpoint resume 的精确一致性。

## 数据与范围

| Split | Variant | SHA-256 | 原始样本 | 有效轨迹 | 丢弃环路 | Unique OD |
|---|---|---|---:|---:|---:|---:|
| Train | `scale_10pct_seed42` | `8943d8958f3b4fadd7d3eb2f351b97268543961e441436e0ad68408cee45cc0a` | 78,570 | 62,348 | 16,222 | 61,253 |
| Validation | `scale_fixed_seed20260715` | `c855d1ebc396576463c363cf2b94480569938de77908aac560df2573d75a1ade` | 20,000 | 15,812 | 4,188 | 15,730 |

两种表示使用完全相同的最少两条边、丢弃环路结果。没有读取 test，没有运行 NeuroMLR，没有运行全量训练，也没有修改 start cost、第一边参数、目标路线定义或图拓扑。

上一轮单边审计仍适用，因为图定义和初始化没有变化：train 有 28 条、validation 有 10 条有效轨迹受 line graph 零成本单边路线影响；validation 暴露比例为 0.0632%。它不是本次优化退化的主因。

## 实际配置与命令

两次恢复运行只改变 `graph.representation`：

| 配置 | Representation | Optimizer | eta0 | lambda | Bounds | Updates | Validation cadence | Threads |
|---|---|---|---:|---:|---|---:|---:|---:|
| `original_edges_relative_10pct_u299.json` | `original_edges` | `relative_projected_subgradient` | 0.0002 | 100000 | `[0.1, 10] * w0` | 299 | 10 | 4 |
| `edge_transition_arcs_relative_10pct_u299.json` | `edge_transition_arcs` | `relative_projected_subgradient` | 0.0002 | 100000 | `[0.1, 10] * w0` | 299 | 10 | 4 |

参数直接沿用旧 edge-only 的相对坐标设置，没有根据本次结果增加学习率或 lambda 网格。299 次 optimizer update 对齐旧 300-state 运行的实际更新数。

运行命令：

```bash
python3 scripts/run_experiment_matrix.py \
  --config experiments/optimizer_recovery/configs/original_edges_relative_10pct_u299.json \
  --output-root artifacts/optimizer_recovery \
  --binary target/release/train \
  --evaluate-binary target/release/evaluate \
  --timeout-seconds 3600 \
  --rayon-threads 4

python3 scripts/run_experiment_matrix.py \
  --config experiments/optimizer_recovery/configs/edge_transition_arcs_relative_10pct_u299.json \
  --output-root artifacts/optimizer_recovery \
  --binary target/release/train \
  --evaluate-binary target/release/evaluate \
  --timeout-seconds 3600 \
  --rayon-threads 4
```

每个 run 从保存的 update 0、10、…、290、299 checkpoint 中选择 validation objective 最低者，再用统一 evaluator 在固定 validation 上解码。raw objective 仅用于各自 run 内 checkpoint 选择，不跨图表示比较。

## 路线质量结果

### Update 0 到所选 checkpoint

| Representation | Checkpoint | Precision | Recall | Edge F1 | Exact Match | Jaccard | Mean regret |
|---|---:|---:|---:|---:|---:|---:|---:|
| `original_edges` update 0 | 0 | 0.596529 | 0.590161 | 0.589902 | 0.336643 | 0.519793 | 650449.936 |
| `original_edges` selected | 290 | 0.699076 | 0.679209 | **0.685404** | **0.373640** | 0.611801 | 310213.247 |
| `edge_transition_arcs` update 0 | 0 | 0.610961 | 0.602860 | 0.603467 | 0.346952 | 0.533143 | 636965.750 |
| `edge_transition_arcs` selected | 299 | 0.707199 | 0.688856 | **0.694125** | **0.377245** | 0.620388 | 321414.632 |

| Representation | Δ Precision | Δ Recall | Δ Edge F1 | Δ Exact | Δ Jaccard | Mean regret 降幅 |
|---|---:|---:|---:|---:|---:|---:|
| `original_edges` | +0.102548 | +0.089048 | **+0.095503** | **+0.036997** | +0.092007 | -340236.689（-52.31%） |
| `edge_transition_arcs` | +0.096237 | +0.085996 | **+0.090658** | **+0.030293** | +0.087245 | -315551.117（-49.54%） |

这与直接权重运行的 F1 仅约 `2e-5` 改善形成明确对照，足以证明历史轨迹学习现在带来了有意义的路线复现增益。

### 两种表示在相对优化下的共同指标对比

| 指标 | `original_edges` | `edge_transition_arcs` | Line graph - original |
|---|---:|---:|---:|
| Edge Precision | 0.699076 | **0.707199** | +0.008122 |
| Edge Recall | 0.679209 | **0.688856** | +0.009647 |
| Edge F1 | 0.685404 | **0.694125** | **+0.008720** |
| Exact Match | 0.373640 | **0.377245** | **+0.003605** |
| Edge Jaccard | 0.611801 | **0.620388** | +0.008588 |

line graph 的初始优势在训练后缩小，但在五个共同路线指标上仍一致更高。两种表示的 raw mean regret 和 regularized objective 定义在不同坐标集合上，不用于跨表示优劣判断。

## 运行成本与量化

| Representation | Training wall | Total wall | Peak RSS | 连续变化坐标 | 整数量化后变化坐标 |
|---|---:|---:|---:|---:|---:|
| `original_edges` | 125.95 s | 128.17 s | 158.25 MiB | 45,279 / 72,156（62.75%） | 45,139 / 72,156（62.56%） |
| `edge_transition_arcs` | 304.26 s | 309.70 s | 215.41 MiB | 69,615 / 188,249（36.98%） | 69,306 / 188,249（36.82%） |

同为 299-update 相对优化时，line graph 的训练时间是 2.42 倍，peak RSS 是 1.36 倍。其整数量化变化比例从上一轮直接权重运行的 9.88% 提高到 36.82%，表明 CCH 整数 metric 上的大量坐标已真正移动；但整数舍入仍是 oracle 风险，不等于严格连续权重最短路。

## 推荐与剩余风险

推荐后续 NeuroMLR 正式对比使用：

```text
graph representation = edge_transition_arcs
optimizer kind       = relative_projected_subgradient
```

依据是两种表示都已证明能从 update 0 获得显著学习增益，而 line graph 在相同相对优化协议下仍具有更高的共同解码 Edge F1 和 Exact Match。直接权重欧氏优化只应作为历史退化对照，不应进入正式比较。

剩余风险：

- **收敛：** original 最佳 update 290，update 299 略回退；line graph 最佳 update 299 位于预算边界，因此 line graph 尚未确认收敛。本轮不临时追加训练。
- **量化：** CCH 仍使用四舍五入后的 `u32` metric；相对优化显著降低了量化停滞，但没有消除连续/整数 oracle 差异。
- **单边路线：** validation 中 10/15,812 条受零成本单边语义影响，规模很小但不可由正转移权重修复。
- **开发集边界：** 只有一个 train seed 和固定 validation；test 从未读取，结果不是测试集或泛化结论。
- **超参数边界：** line graph 暂时复用旧 edge-only 的 `eta0=0.0002, lambda=100000`，本次没有进行表示专用调参。

完整机器摘要见 [`summary.json`](summary.json)，原始日志、checkpoint、外部评估和运行命令保存在本地忽略目录 `artifacts/optimizer_recovery/`。
