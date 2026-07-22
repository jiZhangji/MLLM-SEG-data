from __future__ import annotations

import csv
import json
import os
import random
import statistics
import threading
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch


def select_indices(length: int, warmup: int, samples: int, seed: int) -> tuple[list[int], list[int]]:
    required = warmup + samples
    if required > length:
        raise ValueError(f"Benchmark needs {required} items, but the dataset has only {length}.")
    selected = random.Random(seed).sample(range(length), required)
    return selected[:warmup], selected[warmup:]


def cuda_elapsed(call: Callable[[], Any]) -> tuple[Any, float]:
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = call()
    torch.cuda.synchronize()
    return result, time.perf_counter() - started


class ProcessGpuMemoryMonitor:
    """Poll total GPU memory owned by this process, including Torch and CuPy."""

    def __init__(self, device_index: int = 0, interval_seconds: float = 0.01) -> None:
        self.device_index = device_index
        self.interval_seconds = interval_seconds
        self.peak_bytes = 0
        self.backend = "torch"
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(self.device_index)
        try:
            import pynvml

            pynvml.nvmlInit()
            visible = [
                value.strip()
                for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
                if value.strip()
            ]
            nvml_index = self.device_index
            if self.device_index < len(visible) and visible[self.device_index].isdigit():
                nvml_index = int(visible[self.device_index])
            handle = pynvml.nvmlDeviceGetHandleByIndex(nvml_index)
            pid = os.getpid()

            def poll() -> None:
                while not self._stop.is_set():
                    used = 0
                    try:
                        processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                        used = sum(
                            int(process.usedGpuMemory)
                            for process in processes
                            if process.pid == pid and process.usedGpuMemory is not None
                        )
                    except pynvml.NVMLError:
                        pass
                    self.peak_bytes = max(self.peak_bytes, used)
                    self._stop.wait(self.interval_seconds)

            self.backend = "nvml_process_poll_10ms"
            self._thread = threading.Thread(target=poll, daemon=True)
            self._thread.start()
        except Exception:
            self.backend = "torch_max_memory_allocated"

    def finish(self) -> tuple[float, str]:
        torch.cuda.synchronize()
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=2.0)
        self.peak_bytes = max(
            self.peak_bytes, int(torch.cuda.max_memory_allocated(self.device_index))
        )
        return self.peak_bytes / (1024**3), self.backend


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def write_report(
    output_dir: Path,
    metadata: dict[str, Any],
    rows: list[dict[str, Any]],
    peak_gpu_gib: float,
    memory_backend: str,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("No measured benchmark rows were produced.")
    output_dir.mkdir(parents=True, exist_ok=True)
    row_path = output_dir / "timings.csv"
    with row_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    totals = [float(row["total_seconds"]) for row in rows]
    summary = {
        **metadata,
        "samples": len(rows),
        "e2e_mean_seconds": statistics.fmean(totals),
        "e2e_median_seconds": statistics.median(totals),
        "e2e_p95_seconds": percentile(totals, 95),
        "fps": len(totals) / sum(totals),
        "peak_gpu_gib": peak_gpu_gib,
        "peak_gpu_measurement": memory_backend,
        "timings_csv": str(row_path.resolve()),
    }
    component_keys = sorted(
        key for key in rows[0] if key.endswith("_seconds") and key != "total_seconds"
    )
    summary["component_mean_seconds"] = {
        key: statistics.fmean(float(row[key]) for row in rows) for key in component_keys
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def assert_rtx_4090(allow_other_gpu: bool) -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("This end-to-end benchmark requires CUDA.")
    name = torch.cuda.get_device_name(0)
    if "4090" not in name and not allow_other_gpu:
        raise RuntimeError(
            f"Expected a unified RTX 4090 benchmark, but CUDA device 0 is {name!r}. "
            "Use --allow-other-gpu only for debugging, never for the paper table."
        )
    return name
