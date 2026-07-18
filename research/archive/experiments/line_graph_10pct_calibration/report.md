# 北京 10% 图表示路线复现校准

> **状态：优化退化诊断基线。** 本报告冻结的直接权重欧氏优化后来被确认存在尺度退化，因此下述“进入 NeuroMLR”建议不能单独作为执行依据。相对权重优化已在 `original_edges` 上复现旧结果，并重新训练两种表示；当前结论和正式推荐见[相对权重优化恢复报告](../optimizer_recovery/report.md)。本报告及原始数据不被覆盖，用于保留问题发现过程和直接优化对照。

## 结论

推荐 `edge_transition_arcs` 进入后续 NeuroMLR 正式对比。按各自筛选出的学习率、最低 validation objective checkpoint 和完全相同的 15,812 条 validation 轨迹，line graph 的 Edge F1 为 0.603495，`original_edges` 为 0.589923，绝对提高 0.013572（1.357 个百分点）；Exact Match 分别为 0.346762 和 0.335947，提高 0.010815（1.081 个百分点）。此结论只使用解码后的共同原图边路线指标，不跨表示比较 raw objective。

这个优势主要来自图表示及其初始度量，而不是本轮 200 次更新：update 0 时 line graph 相对 original edges 的 F1 优势已经是 0.013565。两种表示从 update 0 到所选 checkpoint 的 F1 改善都只有约 2–3e-5，且 Exact Match 均略降。因此本轮回答的是“当前冻结模型下哪种表示复现路线更好”，不能解读为优化器已经带来显著学习增益。

代价是 line graph 的训练 wall time 为 207.69 秒，是 original edges 的 2.42 倍；peak RSS 为 215.28 MiB，是 1.36 倍。两种表示的最低 validation objective 都位于 update 200，均标记为“尚未确认收敛”，按预注册约束没有追加训练。

## 范围与 Git 起点

起始工作树干净，分支为 `decouple-graph-representation`，HEAD 为：

```text
a587e9ed2b239a448f4eeaaebf274b313a596359
Model transitions as direct line-graph arc weights
```

实验分支从该提交创建为 `line-graph-10pct-calibration`。本轮没有修改模型架构、目标函数、optimizer 或图定义；没有加入 start cost、第一边参数或其他模型结构；没有调 lambda；没有读取 test、运行 NeuroMLR、比较 CCH/Dijkstra 或运行全量训练。代码变更仅增加审计、checkpoint 选择/评估、汇总工具和本轮配置。

最终本地提交与提交后的 Git 状态记录在交付消息中；commit SHA 无法写入产生该 SHA 的同一提交内容。

## 数据身份与过滤

| Split | Variant | 文件 bytes | SHA-256 | 原始样本 | 有效轨迹 | 丢弃环路 | Too short | Unique OD |
|---|---|---:|---|---:|---:|---:|---:|---:|
| Train | `scale_10pct_seed42` | 16,680,638 | `8943d8958f3b4fadd7d3eb2f351b97268543961e441436e0ad68408cee45cc0a` | 78,570 | 62,348 | 16,222 (20.65%) | 0 | 61,253 |
| Validation | `scale_fixed_seed20260715` | 4,160,448 | `c855d1ebc396576463c363cf2b94480569938de77908aac560df2573d75a1ade` | 20,000 | 15,812 | 4,188 (20.94%) | 0 | 15,730 |

Train 的 `source_sha256` 为 `d7fdfb5870c54df79d1044ecb12a076e0244dbd5d3bc74fd67d1bdcc2b7c0fce`，validation 的 `source_sha256` 为 `97d9e9231fc6599084e6af9eaa081e08c09f6108f27f304a26264fff1ee0ec6d`。两种表示都沿用最少两条原图边且丢弃环路的同一加载结果；无空轨迹、越界或不连续轨迹被接受。

## 单边零转移成本审计

审计在 `edge_transition_arcs` 初始权重上实际执行 unique OD 最短路查询，而不是只按拓扑推断。

| Split | 有原图直接边的样本 | 占有效轨迹 | 直接边存在且观测长度 > 1 | 初始零成本单边预测 OD | 占 unique OD | 受影响样本 |
|---|---:|---:|---:|---:|---:|---:|
| Train | 28 | 0.0449% | 28（条件比例 100%） | 28 | 0.0457% | 28 |
| Validation | 10 | 0.0632% | 10（条件比例 100%） | 10 | 0.0636% | 10 |

正转移权重无法击败零成本直接边，所以这 10 条 validation 观测在冻结语义下始终不能 Exact Match；其规模相当于 validation 的 0.0632 个百分点，远小于 line graph 相对 original edges 的 1.081 个百分点 Exact Match 优势。该问题真实存在但覆盖面很小，本轮只报告影响，没有改变模型语义。

## 实际配置与命令

所有 run 固定 `lambda=0.001`、权重边界 `[0.1*w0, 10*w0]`、`validation_every=10`、`RAYON_NUM_THREADS=4`、CCH full customization、unique-OD grouping 和 `test_policy=never_read`。

| 阶段 | Representation | eta0 | Updates | 配置 |
|---|---|---:|---:|---|
| 初始筛选 | `original_edges` | 300, 1000, 3000 | 50 | `experiments/configs/original_edges_eta{300,1000,3000}_10pct_u50.json` |
| 边界补充 | `original_edges` | 100 | 50 | `experiments/configs/original_edges_eta100_10pct_u50.json` |
| 初始筛选 | `edge_transition_arcs` | 300, 1000, 3000 | 50 | `experiments/configs/edge_transition_arcs_eta{300,1000,3000}_10pct_u50.json` |
| 边界补充 | `edge_transition_arcs` | 100 | 50 | `experiments/configs/edge_transition_arcs_eta100_10pct_u50.json` |
| Development | `original_edges` | 300 | 200 | `experiments/configs/original_edges_eta300_10pct_u200.json` |
| Development | `edge_transition_arcs` | 100 | 200 | `experiments/configs/edge_transition_arcs_eta100_10pct_u200.json` |

两个初始网格都由 eta 300 获胜，因其位于下边界，各自只补充了相邻的 eta 100。没有运行 eta 10000，也没有增加其他候选。

实际命令组如下；每个矩阵 run 又自动用 `target/release/evaluate --split validation --variant scale_fixed_seed20260715` 评估最低-objective checkpoint 和 update 0，完整展开后的命令数组保存在机器 summary 中。

```bash
RAYON_NUM_THREADS=4 target/release/audit_single_edge \
  --config experiments/configs/edge_transition_arcs_eta300_10pct_u50.json \
  --output artifacts/line_graph_10pct_calibration/single_edge_audit.json

python3 scripts/run_experiment_matrix.py \
  --config experiments/configs/original_edges_eta300_10pct_u50.json \
  --config experiments/configs/original_edges_eta1000_10pct_u50.json \
  --config experiments/configs/original_edges_eta3000_10pct_u50.json \
  --output-root artifacts/line_graph_10pct_calibration/screening \
  --timeout-seconds 3600 --rayon-threads 4

python3 scripts/run_experiment_matrix.py \
  --config experiments/configs/edge_transition_arcs_eta300_10pct_u50.json \
  --config experiments/configs/edge_transition_arcs_eta1000_10pct_u50.json \
  --config experiments/configs/edge_transition_arcs_eta3000_10pct_u50.json \
  --output-root artifacts/line_graph_10pct_calibration/screening \
  --timeout-seconds 3600 --rayon-threads 4

python3 scripts/run_experiment_matrix.py \
  --config experiments/configs/original_edges_eta100_10pct_u50.json \
  --config experiments/configs/edge_transition_arcs_eta100_10pct_u50.json \
  --output-root artifacts/line_graph_10pct_calibration/screening \
  --timeout-seconds 3600 --rayon-threads 4

python3 scripts/run_experiment_matrix.py \
  --config experiments/configs/original_edges_eta300_10pct_u200.json \
  --config experiments/configs/edge_transition_arcs_eta100_10pct_u200.json \
  --output-root artifacts/line_graph_10pct_calibration/development \
  --timeout-seconds 7200 --rayon-threads 4
```

## 学习率筛选

每个 run 先在 update 0/10/20/30/40/50 中按最低 validation objective 选 checkpoint，再计算共同路线指标。下表所有 run 都选中了 update 50；“时间”是训练进程 wall time，不含随后两次 validation 解码评估。

| Representation | eta0 | Precision | Recall | Edge F1 | Exact | Jaccard | Validation objective | 最佳 update | 时间 (s) | 变化坐标 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `original_edges` | 100 | 0.596346 | 0.590130 | 0.589785 | 0.336074 | 0.519602 | 650399.768 | 50 | 23.26 | 41,019 |
| `original_edges` | **300** | **0.596377** | 0.590174 | **0.589821** | **0.336137** | **0.519629** | 650299.504 | 50 | 23.16 | 41,021 |
| `original_edges` | 1000 | 0.595933 | 0.590334 | 0.589646 | 0.335505 | 0.519369 | 649950.545 | 50 | 22.88 | 41,030 |
| `original_edges` | 3000 | 0.595586 | 0.590203 | 0.589359 | 0.333797 | 0.518748 | 648970.476 | 50 | 23.03 | 41,057 |
| `edge_transition_arcs` | **100** | **0.610809** | 0.602858 | **0.603380** | **0.346382** | **0.532977** | 636920.707 | 50 | 55.51 | 59,950 |
| `edge_transition_arcs` | 300 | 0.610668 | 0.602762 | 0.603260 | 0.345181 | 0.532818 | 636830.691 | 50 | 55.45 | 59,957 |
| `edge_transition_arcs` | 1000 | 0.610344 | 0.602873 | 0.603134 | 0.346130 | 0.532723 | 636517.241 | 50 | 55.05 | 59,989 |
| `edge_transition_arcs` | 3000 | 0.609931 | 0.602703 | 0.602808 | 0.344675 | 0.532172 | 635635.897 | 50 | 55.35 | 60,041 |

按 Edge F1、再 Exact Match、再更低同表示 validation objective 的预注册规则，`original_edges` 选择 eta 300，`edge_transition_arcs` 选择 eta 100。没有发生指标并列。筛选差异很小，因此更稳妥的表述是 original edges 的合适量级约为 100–300、选 300；line graph 约为 100–300、选 100，而不是声称存在尖锐最优点。

## 200-update development 结果

| Representation | eta0 | Precision | Recall | Edge F1 | Exact | Jaccard | Validation objective | 最佳 update | 训练时间 (s) | 总时间 (s) | 连续/量化变化坐标 | Peak RSS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `original_edges` | 300 | 0.596459 | 0.590318 | 0.589923 | 0.335947 | 0.519683 | 650133.583 | 200 | 85.96 | 88.05 | 41,026 / 25,682 | 158.28 MiB |
| `edge_transition_arcs` | 100 | 0.610980 | 0.602904 | 0.603495 | 0.346762 | 0.533138 | 636870.900 | 200 | 207.69 | 212.81 | 59,955 / 18,598 | 215.28 MiB |

两行 raw objective 只用于各自行内 checkpoint 选择，不能因 line graph 数值更低就判定其更优。两种表示的最佳 checkpoint 都是 update 200，故均“尚未确认收敛”；没有自动追加更长训练。

相对各自 update 0 的共同路线质量变化如下：

| Representation | ΔPrecision | ΔRecall | ΔEdge F1 | ΔExact | ΔJaccard |
|---|---:|---:|---:|---:|---:|
| `original_edges` | -0.000069 | +0.000157 | +0.000022 | -0.000696 | -0.000111 |
| `edge_transition_arcs` | +0.000018 | +0.000044 | +0.000028 | -0.000190 | -0.000005 |

这说明 objective 虽继续下降，共同路线复现质量几乎没有变化；学习率结论应视为本冻结优化设置下的有限校准，而不是已证明收敛或显著优于初始化。

## 表示间对比

| 指标 | `original_edges` | `edge_transition_arcs` | Line graph 差值/倍率 |
|---|---:|---:|---:|
| Edge F1 | 0.589923 | 0.603495 | +0.013572 |
| Exact Match | 0.335947 | 0.346762 | +0.010815 |
| 训练 wall time | 85.96 s | 207.69 s | 2.42x |
| Peak RSS | 158.28 MiB | 215.28 MiB | 1.36x |
| 坐标数 | 72,156 | 188,249 | 2.61x |

line graph 在 Precision、Recall、F1、Exact Match 和 Jaccard 上都高于 original edges，路线质量优势口径一致；其运行成本和内存也明确更高。在后续 NeuroMLR 正式对比以路线质量为主目标的前提下，选择 `edge_transition_arcs`。若后续场景把训练吞吐或内存设为硬约束，`original_edges` 仍是更便宜的基线。

## 风险与解释边界

- **尚未确认收敛：** 两种表示的最低 validation objective 都在 update 200；按约定没有延长训练。
- **量化风险：** CCH 路由使用四舍五入后的 `u32` 权重。所选点最大量化误差小于 0.5，但 original edges 仅 25,682/72,156（35.59%）、line graph 仅 18,598/188,249（9.88%）坐标在整数度量上发生变化。连续 objective 改善不一定改变解码路线，本轮极小的质量增益与此一致。
- **单边路线风险：** validation 有 10 条样本、10 个 OD 固定受零成本单边语义影响，覆盖约 0.063%；问题真实但不足以解释两表示间超过 1 个百分点的路线质量差异。
- **统计风险：** 只使用一个 10% seed 和一个固定 validation，没有重复种子或置信区间；细小的 eta 排名和 update-0 增量不应过度解释。
- **目标口径：** raw objective 的坐标空间和路径成本定义不同，禁止跨表示比较；推荐只基于共同解码路线指标。
- **测试边界：** 本轮从未读取 test，因此结论仍是 development 结论，不能宣称 test 泛化性能。

## 产物与 checkpoint 依据

机器可读结果为 [`summary.json`](summary.json)。它包含全部 8 个筛选 run、2 个 development run、每个 validation checkpoint 的 objective/变化坐标、所选 checkpoint、update-0 与所选点的完整指标、配置 SHA-256、实际命令、训练/评估时间、peak RSS、健康检查和 `test_read=false` 证据。

两个 development 最终配置仍纳入版本控制；screening 配置的内容、SHA-256 和实际命令保留在 `summary.json`，独立旧配置文件不再保留。完整日志、checkpoint 和 runner result 已在薄档案清理中删除。

## 验证

最终提交前执行以下命令，全部通过：

```bash
cargo fmt --check
cargo build --release --locked
cargo test --locked --all-targets
cargo clippy --locked --all-targets -- -D warnings
git diff --check
```

`cargo test --locked --all-targets` 共通过 40 个测试：library 32 个、单边审计 2 个、evaluate 1 个、配置/CLI/checkpoint contracts 5 个；其余 bin target 无单测。`cargo clippy` 在 `-D warnings` 下无警告。契约测试明确锁定了两份既有 smoke 和本轮允许的 8 份筛选、2 份 development 配置矩阵。
