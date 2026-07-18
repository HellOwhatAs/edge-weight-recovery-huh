# Archived shared-temporal experiment

This directory preserves the completed `q_bucket = q_global + residual_bucket`
and train-only trip-average travel-time study as historical evidence. It is no
longer an active model or recommendation. Paths embedded in the report and
machine summary are the execution-time paths from commit `807e86c`; the ignored
local artifacts remain under `artifacts/full_data_time_conditioning/`.

The active experiment uses ordinary independent static checkpoints selected by
departure-time data partition. No residual optimizer, temporal checkpoint, or
travel-time baseline remains in the active implementation.
