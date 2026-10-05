"""Workload precision policy and PyTorch dynamic quantization helpers."""

from __future__ import annotations

from typing import Any

import torch


class WorkloadTask:
    def __init__(self, task_id, is_latency_critical=True, accuracy_floor=0.85,
                 base_duration_seconds=10, is_high_power=False):
        self.task_id = task_id
        self.is_latency_critical = is_latency_critical
        self.accuracy_floor = float(accuracy_floor)
        self.base_duration_seconds = base_duration_seconds
        self.is_high_power = bool(is_high_power)
        self.power_class = 'high' if self.is_high_power else 'low'
        self.precision = 'FP32'
        self.expected_accuracy = 0.94

    def scale_precision_to_int8(self):
        """Select INT8 only when its measured/declared accuracy meets the SLA."""
        int8_accuracy = 0.90
        if int8_accuracy < self.accuracy_floor:
            return False
        self.precision = 'INT8'
        self.expected_accuracy = int8_accuracy
        return True

    def select_precision(self, thermal_limit_exceeded):
        """Apply the workload policy without violating the accuracy floor."""
        if thermal_limit_exceeded and self.scale_precision_to_int8():
            return self.precision
        if thermal_limit_exceeded and self.is_latency_critical:
            return 'REJECTED_ACCURACY_FLOOR_BREACH'
        if thermal_limit_exceeded:
            return 'DEFERRED_WAITING_COOL_CLEAN_WINDOW'
        return self.precision


def quantize_model(model: torch.nn.Module) -> torch.nn.Module:
    """Return a CPU dynamic-INT8 copy of a model with supported layers quantized.

    Dynamic quantization is appropriate for inference-time CPU workloads:
    weights are quantized ahead of time and activations are quantized at runtime.
    The original model is not mutated.
    """
    if next(model.parameters(), torch.empty(0)).is_cuda:
        raise ValueError('dynamic INT8 quantization requires a CPU model')
    quantized = torch.ao.quantization.quantize_dynamic(
        model.cpu().eval(),
        {torch.nn.Linear, torch.nn.LSTM},
        dtype=torch.qint8,
    )
    return quantized


def model_for_precision(model: torch.nn.Module, precision: str) -> torch.nn.Module:
    """Return an inference model for FP32 or INT8 precision."""
    if precision == 'FP32':
        return model.eval()
    if precision == 'INT8':
        return quantize_model(model)
    raise ValueError(f'unsupported precision: {precision}')