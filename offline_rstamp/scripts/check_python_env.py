#!/usr/bin/env python3
"""Check common Python dependencies for offline STAMP/R-STAMP runs."""

from __future__ import annotations

import importlib
import json
import os
import sys


PACKAGES = [
    "torch",
    "torchvision",
    "transformers",
    "accelerate",
    "deepspeed",
    "peft",
    "bitsandbytes",
    "numpy",
    "PIL",
    "cv2",
    "tqdm",
]


def check_package(name: str) -> dict:
    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "unknown")
        return {"ok": True, "version": version}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def main() -> int:
    report = {
        "python": sys.version,
        "executable": sys.executable,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "transformers_offline": os.environ.get("TRANSFORMERS_OFFLINE", ""),
        "hf_datasets_offline": os.environ.get("HF_DATASETS_OFFLINE", ""),
        "packages": {name: check_package(name) for name in PACKAGES},
    }

    torch_info = report["packages"].get("torch", {})
    if torch_info.get("ok"):
        import torch

        report["torch_cuda_available"] = torch.cuda.is_available()
        report["torch_cuda_device_count"] = torch.cuda.device_count()
        report["torch_cuda_devices"] = [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ]

    print(json.dumps(report, indent=2, ensure_ascii=False))

    required = ["torch", "transformers", "accelerate", "deepspeed", "peft", "numpy", "PIL", "tqdm"]
    missing = [name for name in required if not report["packages"][name]["ok"]]
    if missing:
        print("ERROR: missing required packages: " + ", ".join(missing))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

