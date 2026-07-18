# Archived shared-temporal experiment

This directory preserves the completed `q_bucket = q_global + residual_bucket`
and train-only trip-average travel-time study as historical evidence. It is no
longer an active model or recommendation. Paths embedded in the report and
machine summary are the execution-time paths from commit `807e86c`. The large
ignored artifacts were pruned after this thin archive was created.

The active experiment uses ordinary independent static checkpoints selected by
departure-time data partition. No residual optimizer, temporal checkpoint, or
travel-time baseline remains in the active implementation.

The retired input configs are embedded in `summary.json` and remain available
at commit `807e86c`; separate stale config files are intentionally omitted.
The identical departure bucket definition is retained once in
`../../independent_time_buckets/time_buckets.json`.
