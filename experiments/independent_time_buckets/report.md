# 北京独立时间桶静态模型实验

## 研究问题与实现边界

本轮只比较两个对象：

1. 一套模型使用全部有效 train，在全部固定 validation 上选择 checkpoint；
2. 五套互相独立的静态模型，每套只使用对应出发时间桶的 train，并只在同桶 validation 上选择 checkpoint。

两者均固定为：

```text
graph representation = edge_transition_arcs
optimizer            = relative_projected_subgradient
baseline             = road length
eta0                 = 0.0002
lambda               = 100000
relative bounds      = [0.1, 10]
updates              = 500
checkpoint cadence   = 25
test                  = never_read
```

时间只参与数据选择。配置中的可选 `data.departure_time_filter` 在共同的路径合法性过滤之后、图映射和优化之前保留一个桶。进入训练循环后，每个 run 仍求解原来的静态凸问题：

```text
J_b(q) = (1 / |D_b|) sum_(trip in D_b)
         [cost_(w0*q)(observed) - dist_(w0*q)(source,target)]
         + lambda / (2m) * ||q - 1||^2.
```

没有全局加桶残差、双正则、residual learning-rate multiplier、旅行时间基线、专用 optimizer、专用 checkpoint 或专用 evaluator。每个模型保存普通 schema-2 direct-weight checkpoint；推理只是 `start_time -> bucket -> static checkpoint -> CCH query`。

此前完成的共享残差与整段平均速度实验没有抹除，已整体归档到 [`../archive/full_data_shared_temporal_residual`](../archive/full_data_shared_temporal_residual)，但不再是活跃实现或推荐模型。

## 时间与数据审计

原始 pickle 时间字段保留为完整轨迹的 Unix 秒。full train 的 `MMDD` key 中：

- `785,709 / 785,709` 与 UTC+8 日期一致；
- `695,129 / 785,709` 与 UTC 日期一致。

因此使用 `Asia/Shanghai`、UTC+8，并按 `start_time` 选择预先由 train 冻结的五个粗桶：

| 出发时间 | 有效 train | 固定 validation |
|---|---:|---:|
| 00:00-06:00 | 35,201 | 1,009 |
| 06:00-10:00 | 106,282 | 2,399 |
| 10:00-16:00 | 220,543 | 5,456 |
| 16:00-20:00 | 151,360 | 4,066 |
| 20:00-24:00 | 109,889 | 2,882 |
| **合计** | **623,275** | **15,812** |

train 原始 785,709 条中共同路径过滤丢弃 162,434 条环路；validation 原始 20,000 条中丢弃 4,188 条环路。其他拒绝类型均为 0。精确日期、小时、duration、文件哈希和过滤记录见 [`time_audit.json`](time_audit.json)，桶定义见 [`time_buckets.json`](time_buckets.json)。

## 公平比较协议

五个桶不做各自的超参数搜索，直接复用全量静态基线已经冻结的学习率、正则和训练长度。这样模型之间唯一的实验变量是是否拆分训练数据。每桶在自己的 21 个注册 checkpoint 中按 validation Edge F1 最大选择，Exact Match 和较早 update 只用于精确 tie-break。

总体指标按固定 validation 样本数加权聚合；relative regret 使用各桶的 additive regret 与 observed-cost totals 精确合并。全量静态 checkpoint 也使用当前 evaluator 和同一桶定义重新评价。所有 regret 均是相同长度基线下的毫米单位，可以直接比较；不比较不同训练集上的 raw objective。

独立模型拥有五次 checkpoint 选择机会，而全量静态模型只有一次总体选择机会，这一协议偏向独立模型。若独立模型仍未改善，则结论不会由选择自由度不足造成。

## 实际命令

审计：

```bash
RAYON_NUM_THREADS=4 target/release/audit_time \
  --config experiments/independent_time_buckets/configs/static_night_00_06_u500.json \
  --output experiments/independent_time_buckets/time_audit.json
```

五个普通静态 run 的薄调度层：

```bash
python3 scripts/run_bucketed_static_experiment.py \
  --config experiments/independent_time_buckets/configs/static_night_00_06_u500.json \
  --config experiments/independent_time_buckets/configs/static_morning_06_10_u500.json \
  --config experiments/independent_time_buckets/configs/static_day_10_16_u500.json \
  --config experiments/independent_time_buckets/configs/static_evening_16_20_u500.json \
  --config experiments/independent_time_buckets/configs/static_late_20_24_u500.json \
  --output-root artifacts/independent_time_buckets/formal \
  --binary target/release/train \
  --evaluate-binary target/release/evaluate \
  --timeout-seconds 10800 \
  --rayon-threads 4
```

该脚本只调用现有 `run_experiment_matrix.py`、`train`、`evaluate` 和通用 checkpoint selector，然后按样本数聚合五个静态评价；它不实现训练目标或 optimizer。

全量静态参照使用同一 evaluator 和当前桶定义重新评价全部已注册 checkpoint：

```bash
RAYON_NUM_THREADS=4 python3 scripts/select_route_checkpoint.py \
  --run-dir artifacts/full_data_time_conditioning/static_final/static_full_eta0002_u500 \
  --evaluate-binary target/release/evaluate \
  --time-buckets experiments/independent_time_buckets/time_buckets.json \
  --rayon-threads 4 \
  --timeout-seconds 900
```

最终机器摘要由以下命令生成：

```bash
python3 scripts/summarize_independent_time_buckets.py
```

## 正式结果

### 总体比较

下表中的三个阶段使用同一个 15,812 条轨迹的固定 validation 和同一评价实现。10% 结果保留作历史参照；本轮的直接对照是后两行。

| 阶段 | 有效 train | checkpoint | Edge Precision | Edge Recall | Edge F1 | Exact Match | Edge Jaccard | mean regret (mm) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 已有 10% 静态 line graph | 62,348 | 299 | 0.707199 | 0.688856 | 0.694125 | 0.377245 | 0.620388 | 321,414.6 |
| 全量静态 line graph | 623,275 | 400 | **0.713366** | **0.695423** | **0.700554** | **0.388186** | **0.627908** | **303,899.5** |
| 五个独立分桶静态模型 | 623,275（分桶） | 475/275/200/275/150 | 0.711458 | 0.694377 | 0.699198 | 0.384202 | 0.626007 | 320,424.2 |

完整训练集相对已有 10% 静态模型带来 `+0.006429` Edge F1、`+0.010941` Exact Match、`+0.007520` Edge Jaccard，并将 mean regret 降低约 `17,515.2 mm`。这仍是当前最明确的正向结果。

独立分桶相对全量静态模型则为：

| 指标 | 差值（独立分桶 - 全量静态） |
|---|---:|
| Edge Precision | -0.001908 |
| Edge Recall | -0.001046 |
| Edge F1 | **-0.001356** |
| Exact Match | **-0.003984** |
| Edge Jaccard | -0.001900 |
| mean regret | +16,524.8 mm |
| relative regret | +0.000831 |

因此，简单时间分桶没有显著提高路线复现质量；在这一个固定 validation split 上，它的所有总体路线指标都变差，regret 也上升。这里没有执行正式显著性检验，所以“没有显著提高”指没有观察到方向一致、量级明确的 validation 改善，而不是统计检验结论。

### 逐时间桶结果

每个桶的“全量静态”行都使用同一个 update-400 checkpoint；“独立静态”行使用该桶自己选择的普通静态 checkpoint。

| 出发时间 | train / validation | 模型 | update | Precision | Recall | F1 | Exact | Jaccard | mean regret (mm) |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 00:00-06:00 | 35,201 / 1,009 | 全量静态 | 400 | 0.710416 | 0.689286 | 0.695822 | 0.382557 | 0.623204 | 367,418.1 |
| 00:00-06:00 | 35,201 / 1,009 | 独立静态 | 475 | 0.696627 | 0.681513 | 0.684981 | 0.360753 | 0.610160 | 354,028.8 |
| 06:00-10:00 | 106,282 / 2,399 | 全量静态 | 400 | 0.708264 | 0.690202 | 0.695201 | 0.373489 | 0.621250 | 369,851.7 |
| 06:00-10:00 | 106,282 / 2,399 | 独立静态 | 275 | 0.701254 | 0.684750 | 0.689142 | 0.362234 | 0.613956 | 378,773.1 |
| 10:00-16:00 | 220,543 / 5,456 | 全量静态 | 400 | 0.714652 | 0.697487 | 0.702208 | 0.392229 | 0.629705 | 282,479.1 |
| 10:00-16:00 | 220,543 / 5,456 | 独立静态 | 200 | 0.714949 | 0.700113 | 0.703794 | 0.390213 | 0.630827 | 302,822.6 |
| 16:00-20:00 | 151,360 / 4,066 | 全量静态 | 400 | 0.710223 | 0.693665 | 0.698344 | 0.387113 | 0.625379 | 271,560.1 |
| 16:00-20:00 | 151,360 / 4,066 | 独立静态 | 275 | 0.709309 | 0.690619 | 0.696364 | 0.384899 | 0.622646 | 282,964.3 |
| 20:00-24:00 | 109,889 / 2,882 | 全量静态 | 400 | 0.720646 | 0.700490 | 0.706651 | 0.396253 | 0.635261 | 312,938.9 |
| 20:00-24:00 | 109,889 / 2,882 | 独立静态 | 150 | 0.721568 | 0.701338 | 0.707844 | 0.398334 | 0.637205 | 346,260.6 |

独立模型只在 `10:00-16:00`（`+0.001586`）和 `20:00-24:00`（`+0.001192`）提高 F1；Exact Match 只在 `20:00-24:00` 提高（`+0.002082`）。最稀疏的 `00:00-06:00` 桶损失最大：F1 `-0.010841`、Exact `-0.021804`；`06:00-10:00` 也分别下降 `-0.006058` 和 `-0.011255`。分桶没有产生跨桶一致的收益，且负向结果并非总体样本权重变化造成。

夜间桶的 mean regret 虽下降 `13,389.3 mm`，但其路线复现指标明显变差；白天和晚间两个 F1 略升的桶，mean regret 反而分别增加 `20,343.5 mm` 与 `33,321.6 mm`。这说明当前长度代价下的 regret 与轨迹边集合复现不是同一个目标，不能用单个 regret 改善替代主要路线指标。

## Checkpoint、收敛与资源

全量静态模型选择 update 400；五个独立模型依次选择 475、275、200、275、150。六个选择都早于 update 500，因此 validation 路线质量不再直接停留在预算边界。五个独立模型在 update 500 的 F1 均不高于各自所选值。不过夜间选择距离边界只有一个 25-update cadence，且各训练目标在有限预算内不保证达到数值最优，所以这里只确认 development checkpoint 的路线质量已出现内部峰值，不宣称凸目标已经数值收敛。

五桶调度总 wall time 为 5,311.6 秒；各桶纯训练时间约为 329.5、903.1、1,716.8、1,206.6、879.2 秒，峰值 RSS 为 1,483,556--1,502,368 KiB。由于总体结果为负且协议预先固定，本轮没有为单桶临时扩大超参数搜索或训练预算。

## 风险与结论

所有选中 checkpoint 的 CCH `u32` 量化均无零权重；五桶最大相对量化误差不超过 `2.29e-4`，全量静态参照为 `3.08e-4`。量化仍可能改变近似并列路径的排序，故保留为 oracle 风险，但没有观察到零权重故障。

本实验只使用一个固定日期 validation split，也没有估计置信区间。独立分桶尤其损失低频转移的跨时段共享，夜间 35,201 条 train 对约 18.8 万个 line-graph 转移坐标最为稀疏。时间戳只有整段起止信息，但本轮不再估计旅行时间或逐边速度，因此不会把整段平均速度误解释为逐边观测。

最终判断是：**保留全量静态 line graph 为推荐配置；时间只作为可选的数据切分与 checkpoint 调度机制。** 当前五桶独立静态模型没有抵消训练样本拆分带来的统计效率损失，不提供总体正向收益。共享残差和旅行时间代理实验仅作为历史证据归档，不再扩大其实现。完整数值、文件身份、逐桶 additive totals、量化诊断和 `test_read=false` 证明见 [`summary.json`](summary.json)。
