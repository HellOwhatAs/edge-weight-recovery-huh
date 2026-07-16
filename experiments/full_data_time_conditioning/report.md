# 北京全量时间条件化路线学习实验

## 范围与开发协议

本轮固定使用：

```text
graph representation = edge_transition_arcs
optimizer            = relative_projected_subgradient
train                = all
validation           = scale_fixed_seed20260715
test                  = never_read
```

所有模型继续使用 directed line graph：坐标只表示合法转移 `(e,f)`，代价取进入道路 `f` 的基线；source/target edge state 的 offset 均为 0，不增加第一条边代价、start 参数或更高阶状态。静态与时间模型采用相同的完整路径过滤和 decoded original-edge evaluator。checkpoint 在各自 run 内按 validation Edge F1 最大选择，Exact Match 和较早 update 只用于精确 tie-break；不跨不同基线比较 regularized objective。

## 数据与时间审计

原始 pickle 的时间字段为 Unix 秒。full train 的 key 保留 `MMDD`：

- `785,709 / 785,709` 与 start timestamp 的 UTC+8 日期一致；
- `695,129 / 785,709` 与 UTC 日期一致。

因此时间解释固定为 `Asia/Shanghai`；2009 年这些日期使用 UTC+8。分桶只看出发时间。由 train 小时分布预先冻结 5 个粗桶：

| 本地出发时间 | 有效 train | 固定 validation |
|---|---:|---:|
| 00:00-06:00 | 35,201 | 1,009 |
| 06:00-10:00 | 106,282 | 2,399 |
| 10:00-16:00 | 220,543 | 5,456 |
| 16:00-20:00 | 151,360 | 4,066 |
| 20:00-24:00 | 109,889 | 2,882 |
| **合计** | **623,275** | **15,812** |

过滤后 train 本地日期为 `2009-05-01, 05-02, 05-04, 05-08, 05-09`；validation 为 `2009-05-13, 05-16`。按 0–23 时的样本数为：

```text
train      = [12123, 6119, 4746, 3178, 3094, 5941, 11319, 23919,
              34212, 36832, 34963, 33474, 34729, 38688, 39434, 39255,
              38646, 39669, 38126, 34919, 32486, 31090, 27853, 18460]
validation = [380, 200, 123, 81, 81, 144, 336, 717,
              767, 579, 820, 937, 611, 1016, 1091, 981,
              1060, 1016, 1060, 930, 873, 830, 671, 508]
```

有效 train 全程 duration 中位数 633 秒、p90 1,597 秒、p99 3,012 秒；validation 分别为 643、1,626、3,111 秒。

full train 原始 785,709 条中丢弃 162,434 条环路；validation 原始 20,000 条中丢弃 4,188 条环路。其余错误类型均为 0。时间字段没有非正 duration。完整审计见 [`time_audit.json`](time_audit.json)，桶定义见 [`time_buckets.json`](time_buckets.json)。

## 凸的共享时间模型

第 `b` 个时间桶使用：

```text
q_b = q_global + residual_b
w_b = w0_b * q_b
```

目标为：

```text
(1/N) sum_b sum_(t in b) [cost_observed(w_b) - dist_shortest(w_b)]
+ lambda_global / (2m) * ||q_global - 1||^2
+ lambda_residual / (2mB) * sum_b ||residual_b||^2
```

它对 `q_global` 和所有 residual 联合凸。全局范围 `[0.6, 9.5]`、residual 范围 `[-0.5, 0.5]`，保证有效相对权重仍在 `[0.1, 10]`。所有桶共享同一全局道路/转移偏好；稀疏桶 residual 向 0 收缩且不能自由学习完整无约束模型。训练器只接收 graph problem 给出的映射、计数与 CCH 查询，不按坐标语义分支。checkpoint 明确保存全局相对参数、每桶 residual、桶定义、每桶道路基线和基线诊断。

## Train-only 旅行时间基线

每条有效 train 轨迹只提供全程时长，因此先计算：

```text
trip_average_speed = sum(full road sequence lengths) / (end_time - start_time)
```

这只是整条轨迹的平均速度 proxy，不是真实逐边速度。方法采用以下保守处理：

1. 速度截断为 `[1, 33.333...]` m/s；401 条触发下截断，307 条触发上截断；
2. 截断后 train 全网均值为 `8.7681816625` m/s；
3. 道路全局速度用 42 个伪观测向全网均值收缩；
4. 道路—时间桶速度用 74 个伪观测向道路全局速度收缩；
5. 两个伪计数分别来自 train-only 支持度分位数，validation 不参与；
6. `w0` 与 `length / smoothed_speed` 成正比。

进入 CCH 前再乘以所有道路/桶共同的 train-only 常数 `8.7681816625`。除以该数可恢复毫秒；正的全局缩放不改变路径排序，却把初始最大相对 `u32` 量化误差降到约 `8.8e-4`，且没有坐标量化为 0。line graph 仍只把“进入道路”的该值放在转移弧上，第一条边语义不变。

## 有限校准

### 静态学习率

只筛选 `eta0 in {0.0002, 0.0004, 0.0008}`，每个上限 60 updates：

| eta0 | 状态 | update 60 F1 | update 60 Exact |
|---:|---|---:|---:|
| 0.0002 | 完成 | 0.694756 | 0.383569 |
| 0.0004 | 完成 | 0.689208 | 0.371174 |
| 0.0008 | update 10 明显发散后终止 | — | — |

`0.0008` 的 validation objective 从 636,965.750 升到 1,135,726.472。虽然 `0.0004` 在 update 60 的 mean regret 略低，但 decoded F1/Exact 明显更差且轨迹抖动，因此正式 run 冻结 `0.0002`。

### 时间 residual 步长

使用长度基线对 residual step multiplier 做 `{1,2}` 的 40-update 对照；另保留 multiplier 5 的 update-25 不稳定诊断：

| multiplier | checkpoint | F1 | Exact | 结论 |
|---:|---:|---:|---:|---|
| 1 | 40 | 0.692307 | 0.383569 | 采用 |
| 2 | 40 | 0.692333 | 0.381229 | F1 实质持平，Exact 更差 |
| 5 | 25 | 0.675727 | 0.363015 | 早期振荡明显，终止 |

不为 `2.6e-5` 的 F1 差异继续扩展搜索；正式时间模型固定 multiplier 1。train-scaled 旅行时间基线随后只做一次 40-update 健康检查：F1 从 0.580927 提高到 0.688144，证明优化和 CCH 路径可正常恢复，但同预算仍落后长度时间模型。

## 实际命令

时间审计：

```bash
RAYON_NUM_THREADS=4 target/release/audit_time \
  --config experiments/full_data_time_conditioning/configs/travel_time_baseline_audit_full.json \
  --output experiments/full_data_time_conditioning/time_audit.json
```

静态学习率 screen 使用 `scripts/run_experiment_matrix.py`，正式静态命令为：

```bash
python3 scripts/run_experiment_matrix.py \
  --config experiments/full_data_time_conditioning/configs/static_full_eta0002_u500.json \
  --output-root artifacts/full_data_time_conditioning/static_final \
  --binary target/release/train \
  --evaluate-binary target/release/evaluate \
  --timeout-seconds 7200 \
  --rayon-threads 4
```

两个正式时间模型分别使用：

```bash
python3 scripts/run_temporal_experiment.py \
  --config experiments/full_data_time_conditioning/configs/temporal_length_full_eta0002_u500.json \
  --output-root artifacts/full_data_time_conditioning/temporal_final \
  --binary target/release/train_temporal \
  --evaluate-binary target/release/evaluate_temporal \
  --timeout-seconds 10800 \
  --rayon-threads 4

python3 scripts/run_temporal_experiment.py \
  --config experiments/full_data_time_conditioning/configs/temporal_travel_time_full_eta0002_u500.json \
  --output-root artifacts/full_data_time_conditioning/temporal_final \
  --binary target/release/train_temporal \
  --evaluate-binary target/release/evaluate_temporal \
  --timeout-seconds 10800 \
  --rayon-threads 4
```

每个正式 run 保存 update 0、25、…、500。`scripts/select_route_checkpoint.py` 对所有已注册 checkpoint 使用同一个 validation evaluator，并以最大 Edge F1 选择 development checkpoint。日志、checkpoint、逐 checkpoint 评价和 runner result 保存在忽略目录 `artifacts/full_data_time_conditioning/`。

## 正式结果

### 总体指标

四个阶段都评价同一组 15,812 条 validation 轨迹：

| 阶段 | checkpoint | Precision | Recall | F1 | Exact | Jaccard | mean regret | regret 单位 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 已有 10% 静态 line graph | 299 | 0.707199 | 0.688856 | 0.694125 | 0.377245 | 0.620388 | 321,414.6 | mm |
| 全量静态 line graph | 400 | 0.713366 | 0.695423 | 0.700554 | 0.388186 | 0.627908 | 303,899.5 | mm |
| 全量时间条件化，长度基线 | 400 | 0.715654 | 0.696363 | 0.702280 | 0.388629 | 0.629917 | 295,214.9 | mm |
| 全量时间条件化，旅行时间基线 | 425 | 0.716090 | 0.697769 | 0.703176 | 0.389704 | 0.631127 | 290,476.5 | scaled-ms |

最后一行的 CCH direct cost 是 `8.7681816625 * milliseconds`；因此其 mean regret 对应 33,128.5 ms。它不能与前三行以毫米表示的 regret 直接相减。路线重现指标不受这个正的全局缩放影响。

逐阶段增量为：

| 对比 | ΔPrecision | ΔRecall | ΔF1 | ΔExact | ΔJaccard | Δmean regret |
|---|---:|---:|---:|---:|---:|---:|
| 全量静态 − 已有 10% | +0.006167 | +0.006567 | +0.006429 | +0.010941 | +0.007520 | −17,515.2 mm |
| 时间条件化长度 − 全量静态 | +0.002288 | +0.000939 | +0.001726 | +0.000443 | +0.002009 | −8,684.6 mm |
| 旅行时间基线 − 时间条件化长度 | +0.000436 | +0.001407 | +0.000896 | +0.001075 | +0.001210 | 不同单位，不相减 |
| 旅行时间模型 − 全量静态 | +0.002724 | +0.002346 | +0.002623 | +0.001518 | +0.003219 | 不同单位，不相减 |

### 分时间桶指标

| 时间桶 | n | 阶段 | Precision | Recall | F1 | Exact | Jaccard | mean regret | 单位 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| 00:00-06:00 | 1,009 | 10% 静态 | 0.704852 | 0.683336 | 0.689670 | 0.376611 | 0.617577 | 383,763.4 | mm |
|  | 1,009 | 全量静态 | 0.710416 | 0.689286 | 0.695822 | 0.382557 | 0.623204 | 367,418.1 | mm |
|  | 1,009 | 时间条件化长度 | 0.706529 | 0.687783 | 0.693389 | 0.397423 | 0.623508 | 370,014.2 | mm |
|  | 1,009 | 时间条件化旅行时间 | 0.706858 | 0.687418 | 0.692828 | 0.381566 | 0.620643 | 334,794.2 | scaled-ms |
| 06:00-10:00 | 2,399 | 10% 静态 | 0.697932 | 0.679658 | 0.684872 | 0.356398 | 0.608590 | 393,443.8 | mm |
|  | 2,399 | 全量静态 | 0.708264 | 0.690202 | 0.695201 | 0.373489 | 0.621250 | 369,851.7 | mm |
|  | 2,399 | 时间条件化长度 | 0.707737 | 0.687316 | 0.693608 | 0.368487 | 0.618794 | 363,116.2 | mm |
|  | 2,399 | 时间条件化旅行时间 | 0.709939 | 0.689983 | 0.696105 | 0.371822 | 0.622117 | 361,373.3 | scaled-ms |
| 10:00-16:00 | 5,456 | 10% 静态 | 0.710768 | 0.691137 | 0.696938 | 0.384714 | 0.623641 | 298,513.3 | mm |
|  | 5,456 | 全量静态 | 0.714652 | 0.697487 | 0.702208 | 0.392229 | 0.629705 | 282,479.1 | mm |
|  | 5,456 | 时间条件化长度 | 0.719318 | 0.699871 | 0.705730 | 0.396078 | 0.634011 | 267,855.8 | mm |
|  | 5,456 | 时间条件化旅行时间 | 0.721348 | 0.702901 | 0.708389 | 0.391312 | 0.636339 | 269,680.1 | scaled-ms |
| 16:00-20:00 | 4,066 | 10% 静态 | 0.703193 | 0.686732 | 0.691319 | 0.372110 | 0.616771 | 287,869.1 | mm |
|  | 4,066 | 全量静态 | 0.710223 | 0.693665 | 0.698344 | 0.387113 | 0.625379 | 271,560.1 | mm |
|  | 4,066 | 时间条件化长度 | 0.709670 | 0.690695 | 0.696668 | 0.377275 | 0.622395 | 262,570.5 | mm |
|  | 4,066 | 时间条件化旅行时间 | 0.707378 | 0.691150 | 0.695618 | 0.387605 | 0.622610 | 269,239.3 | scaled-ms |
| 20:00-24:00 | 2,882 | 10% 静态 | 0.714627 | 0.697124 | 0.702019 | 0.387925 | 0.630140 | 330,310.5 | mm |
|  | 2,882 | 全量静态 | 0.720646 | 0.700490 | 0.706651 | 0.396253 | 0.635261 | 312,938.9 | mm |
|  | 2,882 | 时间条件化长度 | 0.726946 | 0.708251 | 0.713996 | 0.404233 | 0.644281 | 310,355.5 | mm |
|  | 2,882 | 时间条件化旅行时间 | 0.726782 | 0.707498 | 0.713480 | 0.407356 | 0.644445 | 285,277.7 | scaled-ms |

旅行时间模型各桶的 physical mean regret 依次为 38,182.9、41,214.2、30,756.7、30,706.4、32,535.6 ms。

全量静态相对 10% 静态在五桶的 F1 均提升，范围为 `+0.004633` 到 `+0.010329`，说明总体增益不是样本构成造成的。时间条件化长度相对全量静态只在 10:00-16:00 和 20:00-24:00 提升 F1，另外三桶下降。旅行时间模型相对全量静态在 06:00-10:00、10:00-16:00 和 20:00-24:00 提升，夜间和晚高峰下降；它相对时间条件化长度则只有早高峰和白天两桶提升 F1。

### Checkpoint、收敛和资源

| 正式 run | 选择 update | 预算 | 是否边界 | 选择 F1 | update 500 F1 | 选择 objective | 最终 objective |
|---|---:|---:|---|---:|---:|---:|---:|
| 全量静态 | 400 | 500 | 否 | 0.700554 | 0.697173 | 304,143.7 | 295,869.3 |
| 时间条件化长度 | 400 | 500 | 否 | 0.702280 | 0.700468 | 295,402.9 | 288,492.7 |
| 时间条件化旅行时间 | 425 | 500 | 否 | 0.703176 | 0.700894 | 290,676.3 | 286,135.0 |

三个 development checkpoint 都位于预注册预算内部，后续 75–100 updates 没有刷新最佳 decoded F1，因而不再是“最佳点停留在预算边界”的结果。另一方面，三个 run 的 regularized validation objective 到 update 500 仍下降；因此确认的是 decoded 路线质量已进入带抖动的平台区，而不是凸目标已经达到数值收敛。继续无限延长训练不在本轮协议内。

| 正式 run | 训练 wall time | peak RSS | 选择 checkpoint 最大相对量化误差 | 零 `u32` 权重 |
|---|---:|---:|---:|---:|
| 全量静态 | 5,418.3 s | 1,549,424 KiB | 3.08e-4 | 0 |
| 时间条件化长度 | 6,041.7 s | 1,654,420 KiB | 3.92e-4 | 0 |
| 时间条件化旅行时间 | 5,056.6 s | 1,665,172 KiB | 8.76e-4 | 0 |

所有 formal runner、训练日志、checkpoint、逐 checkpoint 评价和选择依据仍在 `artifacts/full_data_time_conditioning/`。可提交、机器可读的汇总为 [`summary.json`](summary.json)，其中保留配置哈希、实际 argv、完整 checkpoint trace、分桶 delta、资源和量化诊断。

## 结论

全量数据本身是本轮唯一清晰且跨桶一致的改进：相对已有 10% 模型，F1 提高 0.006429、Exact Match 提高 0.010941，五个时间桶 F1 全部提高，mean regret 同时下降 17,515.2 mm。

共享时间残差在总体上再提高 F1 0.001726，但 Exact Match 只提高 0.000443，且三个桶的 F1 下降。train-only 旅行时间基线又比长度时间模型提高 F1 0.000896；最终模型相对全量静态总计提高 F1 0.002623、Exact Match 0.001518，但分桶仍有两桶 F1 下降，相对长度时间模型也只有两桶 F1 提升。

因此，在这个单一固定 validation 划分上，粗粒度时间条件化和保守旅行时间 proxy 产生了小幅总体增益，但没有证据支持“相对当前全量静态 line graph 显著且稳定提高路线复现质量”的结论。收益主要集中在白天和深夜，量级远小于全量数据本身的收益；本轮也没有做独立 test 或跨日期重复划分，不能把小差异解释为统计显著性。

## 质量检查

以下检查在最终源代码和记录上通过：

```text
cargo fmt --check                                      pass
cargo build --release --locked                         pass
cargo test --locked --all-targets                      pass (52 tests)
cargo clippy --locked --all-targets -- -D warnings     pass
Python source compilation                              pass (7 scripts)
full_data_time_conditioning JSON parse                 pass (13 files)
summary/bucket/interior-checkpoint/test-read assertions pass
git diff --check                                       pass
```

另逐一核对三份 formal 配置的当前 JSON 与训练日志中的执行配置语义一致；summary 同时保留执行时和当前文件的 SHA-256（静态配置只规范化了末尾空行，两个时间配置 byte-identical）。formal training 日志中没有 `split=test` 事件，所有 finished event 和 evaluator 输出均记录 `test_read=false`。

## 预先保留的解释边界

- 这是一个固定 validation、单次数据划分的 development 实验，没有读取 test，也不提供跨数据集泛化结论或正式显著性检验。
- train 只有 5 个离散日期、validation 只有 2 个日期；hour effect 可能混合日期、供需和天气等未观测因素。
- 全程平均速度只能提供弱的道路旅行时间 proxy；没有逐边时间戳，不能声称恢复了真实逐边速度。
- full train 中 16,244 条道路没有速度支持；各时间桶有 22,108–34,104 个道路—桶单元为零支持。两级收缩给出有限基线，但不能创造局部交通观测。
- CCH 仍按 `u32` 选择路线，再用直接 `f64` 权重计算返回路径代价；固定点缩放降低但没有消除连续目标与整数 oracle 的差异。
- validation 中既有的 10 条零成本单边 line-graph 查询仍受无第一边代价语义影响，本轮没有修改它。
- 不跨长度基线与 scaled-time 基线比较 raw objective 或 mean regret 数值；只有 decoded 路线指标可直接比较。
