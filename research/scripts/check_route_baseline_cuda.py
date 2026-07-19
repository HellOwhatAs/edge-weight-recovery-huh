#!/usr/bin/env python3
"""Fail-fast CUDA preflight using only a synthetic tensor operation."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cuda"], default="cuda")
    parser.parse_args()
    try:
        import torch
    except Exception as error:
        print(f"CUDA preflight could not import torch: {error}", file=sys.stderr)
        return 2
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        print("CUDA_VISIBLE_DEVICES must be exactly 0", file=sys.stderr)
        return 2
    if not torch.cuda.is_available():
        print("torch.cuda.is_available() is false", file=sys.stderr)
        return 2
    try:
        device = torch.device("cuda")
        left = torch.arange(64, dtype=torch.float32, device=device).reshape(8, 8)
        result = left @ left.T
        torch.cuda.synchronize(device)
        checksum = float(result.sum().cpu())
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        report = {
            "schema": "ewr.route-baseline-cuda-preflight/v1",
            "status": "ok",
            "resolved_device": str(device),
            "visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "device_name": torch.cuda.get_device_name(device),
            "device_capability": list(torch.cuda.get_device_capability(device)),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "python": platform.python_version(),
            "free_cuda_bytes": free_bytes,
            "total_cuda_bytes": total_bytes,
            "synthetic_matmul_checksum": checksum,
        }
    except Exception as error:
        print(f"CUDA synthetic operation failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
