"""Parameter, MAC/FLOP, and peak-memory measurement helpers."""

from __future__ import annotations

import os
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional


def _parameters(module: Any) -> Iterable[Any]:
    return () if module is None else module.parameters()


def count_parameters(module: Any, *, trainable_only: bool = False) -> int:
    seen = set()
    total = 0
    for parameter in _parameters(module):
        identifier = id(parameter)
        if identifier in seen:
            continue
        seen.add(identifier)
        if trainable_only and not parameter.requires_grad:
            continue
        total += int(parameter.numel())
    return total


def _first_attribute(model: Any, names: Iterable[str]) -> Any:
    for name in names:
        value = getattr(model, name, None)
        if value is not None:
            return value
    return None


def parameter_partitions(model: Any) -> Dict[str, int]:
    """Count the deployed student, training heads, and frozen teacher separately."""

    student = getattr(model, "student", model)
    teacher = getattr(model, "teacher", None)
    heads = _first_attribute(
        model,
        (
            "prediction_heads",
            "distill_prediction_heads",
            "distillation_head",
            "distill_heads",
            "distill_linear_projs",
            "target_adapters",
        ),
    )
    deployed = count_parameters(student)
    prediction_heads = count_parameters(heads)
    return {
        "deployed_backbone_parameters": deployed,
        "training_only_prediction_head_parameters": prediction_heads,
        "total_training_student_head_parameters": deployed + prediction_heads,
        "frozen_teacher_parameters": count_parameters(teacher),
        "trainable_parameters": count_parameters(model, trainable_only=True),
    }


def audiovisual_input_constructor(_: Any) -> Dict[str, Any]:
    """Construct the fixed Chapter 3 profiling input."""

    import torch

    return {
        "source": {
            "audio": torch.ones((1, 104, 100)),
            "video": torch.ones((1, 1, 100, 88, 88)),
        },
        "features_only": True,
        "mask": False,
    }


def profile_macs(
    model: Any,
    *,
    input_constructor: Any = audiovisual_input_constructor,
) -> Dict[str, Any]:
    """Measure numeric MACs with ptflops and report FLOPs as exactly ``2*MACs``."""

    try:
        from ptflops import get_model_complexity_info
    except ImportError as exc:
        raise RuntimeError("ptflops is required for numeric Chapter 3 profiling") from exc

    was_training = bool(getattr(model, "training", False))
    model.eval()
    try:
        macs, profiler_parameters = get_model_complexity_info(
            model,
            (1, 768),
            input_constructor=input_constructor,
            as_strings=False,
            print_per_layer_stat=False,
            verbose=False,
            backend="aten",
        )
    finally:
        if was_training:
            model.train()
    numeric_macs = int(macs)
    return {
        "input_convention": {
            "audio": [1, 104, 100],
            "video": [1, 1, 100, 88, 88],
        },
        "macs": numeric_macs,
        "flops": 2 * numeric_macs,
        "flop_definition": "2 * MACs",
        "profiler_parameters": int(profiler_parameters),
        "profiler": "ptflops/aten",
    }


def profile_distillation_model(model: Any) -> Dict[str, Any]:
    student = getattr(model, "student", model)
    result = {**parameter_partitions(model), **profile_macs(student)}
    for attribute in ("initialization_report", "initialization_copy_report"):
        report = getattr(model, attribute, None)
        if report is not None:
            result["initialization_copy_report"] = report
            break
    return result


def select_conformer_ffn_match(
    target: Mapping[str, Any],
    candidates: Iterable[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Select the requested parameter/FLOP match with a deterministic tie-break."""

    target_parameters = int(target["deployed_backbone_parameters"])
    target_flops = int(target["flops"])
    if target_parameters <= 0 or target_flops <= 0:
        raise ValueError("target parameters and FLOPs must be positive")
    scored = []
    for candidate in candidates:
        ffn_dim = int(candidate["student_ffn_dim"])
        if ffn_dim <= 0 or ffn_dim % 128:
            raise ValueError(f"Conformer FFN dimension must be a positive multiple of 128: {ffn_dim}")
        parameters = int(candidate["deployed_backbone_parameters"])
        flops = int(candidate["flops"])
        parameter_mismatch = abs(parameters - target_parameters) / target_parameters
        flop_mismatch = abs(flops - target_flops) / target_flops
        scored.append(
            {
                **dict(candidate),
                "student_ffn_dim": ffn_dim,
                "relative_parameter_mismatch": parameter_mismatch,
                "relative_flop_mismatch": flop_mismatch,
                "max_relative_mismatch": max(parameter_mismatch, flop_mismatch),
            }
        )
    if not scored:
        raise ValueError("at least one Conformer profiling candidate is required")
    scored.sort(
        key=lambda item: (
            item["max_relative_mismatch"],
            item["relative_parameter_mismatch"],
            item["student_ffn_dim"],
        )
    )
    return {
        "criterion": (
            "minimize max(relative parameter mismatch, relative FLOP mismatch); "
            "tie-break by relative parameter mismatch then FFN dimension"
        ),
        "target": dict(target),
        "candidates": scored,
        "selected": scored[0],
        "hydra_override": f"model.student_ffn_dim={scored[0]['student_ffn_dim']}",
    }


def _descendant_pids(root_pid: int) -> set[int]:
    """Return *root_pid* and descendants without requiring psutil."""

    descendants = {int(root_pid)}
    try:
        output = subprocess.check_output(
            ["ps", "-e", "-o", "pid=,ppid="],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        return descendants
    pairs = []
    for line in output.splitlines():
        fields = line.split()
        if len(fields) == 2:
            try:
                pairs.append((int(fields[0]), int(fields[1])))
            except ValueError:
                continue
    changed = True
    while changed:
        changed = False
        for child, parent in pairs:
            if parent in descendants and child not in descendants:
                descendants.add(child)
                changed = True
    return descendants


@dataclass
class PeakMemoryResult:
    peak_bytes: int
    samples: int
    source: str
    error: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "peak_gpu_memory_bytes": int(self.peak_bytes),
            "peak_gpu_memory_mib": self.peak_bytes / (1024 * 1024),
            "memory_samples": int(self.samples),
            "memory_source": self.source,
            "memory_monitor_error": self.error,
        }


class PeakMemoryMonitor:
    """Sample process-specific GPU memory once per second.

    NVML is preferred. ``nvidia-smi`` is used as a portable fallback. Child
    processes are followed so the wrapper can monitor Fairseq workers.
    """

    def __init__(self, pid: int, *, device_index: int = 0, interval: float = 1.0):
        self.pid = int(pid)
        self.device_index = int(device_index)
        self.interval = float(interval)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._peak = 0
        self._samples = 0
        self._source = "unavailable"
        self._error: Optional[str] = None

    def start(self) -> "PeakMemoryMonitor":
        if self._thread is not None:
            raise RuntimeError("memory monitor already started")
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> PeakMemoryResult:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.interval * 2))
        return PeakMemoryResult(
            peak_bytes=self._peak,
            samples=self._samples,
            source=self._source,
            error=self._error,
        )

    def _run(self) -> None:
        try:
            import pynvml

            pynvml.nvmlInit()
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
                self._source = "nvml-process"
                while not self._stop.is_set():
                    pids = _descendant_pids(self.pid)
                    processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                    current = sum(
                        int(process.usedGpuMemory)
                        for process in processes
                        if int(process.pid) in pids
                        and process.usedGpuMemory not in (None, getattr(pynvml, "NVML_VALUE_NOT_AVAILABLE", -1))
                    )
                    self._peak = max(self._peak, current)
                    self._samples += 1
                    self._stop.wait(self.interval)
            finally:
                pynvml.nvmlShutdown()
            return
        except BaseException as exc:
            self._error = f"NVML unavailable: {type(exc).__name__}: {exc}"
        self._source = "nvidia-smi-process"
        while not self._stop.is_set():
            try:
                pids = _descendant_pids(self.pid)
                output = subprocess.check_output(
                    [
                        "nvidia-smi",
                        f"--id={self.device_index}",
                        "--query-compute-apps=pid,used_memory",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    stderr=subprocess.DEVNULL,
                )
                current_mib = 0
                for line in output.splitlines():
                    fields = [field.strip() for field in line.split(",")]
                    if len(fields) == 2 and int(fields[0]) in pids:
                        current_mib += int(fields[1])
                self._peak = max(self._peak, current_mib * 1024 * 1024)
                self._samples += 1
            except BaseException as exc:
                if self._error:
                    self._error += f"; nvidia-smi unavailable: {type(exc).__name__}: {exc}"
                else:
                    self._error = f"nvidia-smi unavailable: {type(exc).__name__}: {exc}"
                self._source = "unavailable"
                return
            self._stop.wait(self.interval)
