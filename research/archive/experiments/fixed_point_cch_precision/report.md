# Fixed-point CCH decimal-precision study

Status: **superseded**. This study was run against the retired monolithic CLI
and is retained only to document why frozen v1 uses integer CCH weights.

## Question

`routingkit-cch` accepts `u32` metrics. The established runs rounded direct
weights to integer millimetres for route selection. This study tested whether
scaling by 10 or 100 before rounding improved validation route quality while
leaving the relative optimizer and 10% Beijing data fixed.

Both representations used `eta0=0.0002`, `lambda=100000`, multiplier bounds
`[0.1, 10]`, 299 updates, validation cadence 10, and four Rayon threads. The
validation split contained 15,812 accepted paths; the test split was not read.

## One decimal place

Scaling by 10 completed normally. Checkpoints were selected by minimum
validation objective, with the earlier update breaking exact ties.

| Representation | Selected update | Baseline F1 | Selected F1 | Selected Exact | Mean regret |
| --- | ---: | ---: | ---: | ---: | ---: |
| original edges | 299 | 0.589902 | 0.685614 | 0.372945 | 310,231.4 |
| transition arcs | 290 | 0.603467 | 0.696468 | 0.378257 | 322,016.8 |

For comparison, evaluating the older integer-quantized checkpoints under their
own integer oracle produced F1 `0.684231` for original edges at update 299 and
`0.697511` for transition arcs at update 290. One decimal place therefore moved
the two representations in opposite directions and did not provide a consistent
quality improvement. It also increased training wall time to 140.5 seconds and
338.4 seconds respectively in these runs.

## Two decimal places

The original-edge run failed before training. A valid coordinate upper bound
scaled to `2,322,804,000`, reaching the CCH infinity-sentinel range. This is a
representation-independent safety failure of the proposed scale, not an
optimizer result. No two-decimal line-graph run was warranted.

## Decision

Do not add decimal-place configuration to the production API. Frozen v1 keeps
the simple integer quantization contract. A future precision change should use
an oracle with an explicitly larger or floating-point metric domain and should
be evaluated as a new research adapter, not as a production configuration
switch.

The ignored 104 MB checkpoint tree was removed after the values above and the
machine-readable summary were preserved. Original run files remain recoverable
from the Git-era artifact backup only if one exists; the tracked evidence is the
durable record.
