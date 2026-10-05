"""Workload precision policy and PyTorch dynamic quantization helpers.

INT8 accuracy assumption (0.90) is derived from PyTorch dynamic quantization
benchmarks on LSTM-based models — typical accuracy drop is 1–3%.
This value MUST be validated against the actual deployed model before
production use.  See: https://pytorch.org/docs/stable/quantization.html
"""

from __future__ import annotations

import torch

# Empirical INT8 accuracy for dynamic-quantized LSTM inference.
# Override via WorkloadTask(accuracy_floor=...) per workload SLA.
_INT8_EXPECTED_ACCURACY = 0.90


class WorkloadTask:
    """Describes a submitted AI workload and its scheduling constraints."""

    def __init__(self, task_id: str,
                 is_latency_critical: bool = True,
                 accuracy_floor: float = 0.85,
                 base_duration_seconds: int = 10,
                 is_high_power: bool = False):
        self.task_id = task_id
        self.is_latency_critical = is_latency_critical
        self.accuracy_floor = float(accuracy_floor)
        self.base_duration_seconds = base_duration_seconds
        self.is_high_power = bool(is_high_power)
        self.power_class = 'high' if self.is_high_power else 'low'
        self.precision = 'FP32'
        self.expected_accuracy = 0.94  # typical FP32 baseline

    def scale_precision_to_int8(self) -> bool:
        """Downgrade to INT8 only when its accuracy meets the workload SLA.

        Returns:
            True  — precision downgraded; task may proceed at INT8.
            False — INT8 would breach accuracy_floor; caller must defer or reject.
        """
        if _INT8_EXPECTED_ACCURACY < self.accuracy_floor:
            return False
        self.precision = 'INT8'
        self.expected_accuracy = _INT8_EXPECTED_ACCURACY
        return True

    def select_precision(self, thermal_limit_exceeded: bool) -> str:
        """Return the selected precision string, applying accuracy-floor guard."""
        if not thermal_limit_exceeded:
            return self.precision  # no pressure — stay at current precision

        if self.scale_precision_to_int8():
            return self.precision  # INT8 is acceptable

        # INT8 would breach the floor; apply fallback policy
        return (
            'REJECTED_ACCURACY_FLOOR_BREACH'
            if self.is_latency_critical
            else 'DEFERRED_WAITING_COOL_CLEAN_WINDOW'
        )


# ---------------------------------------------------------------------------
# PyTorch helpers
# ---------------------------------------------------------------------------

def quantize_model(model: torch.nn.Module) -> torch.nn.Module:
    """Return a CPU dynamic-INT8 copy of *model* with Linear/LSTM layers quantized.

    Dynamic quantization weights are quantized offline; activations at runtime.
    The original model is NOT mutated.

    Raises:
        ValueError: if the model is on a CUDA device (quantization requires CPU).
    """
    if next(model.parameters(), torch.empty(0)).is_cuda:
        raise ValueError('dynamic INT8 quantization requires a CPU model')
    return torch.ao.quantization.quantize_dynamic(
        model.cpu().eval(),
        {torch.nn.Linear, torch.nn.LSTM},
        dtype=torch.qint8,
    )


def model_for_precision(model: torch.nn.Module,
                        precision: str) -> torch.nn.Module:
    """Return an inference-ready model at the requested precision.

    Args:
        model: The trained FP32 model.
        precision: 'FP32' or 'INT8'.

    Raises:
        ValueError: for unsupported precision strings.
    """
    if precision == 'FP32':
        return model.eval()
    if precision == 'INT8':
        return quantize_model(model)
    raise ValueError(f'unsupported precision: {precision!r}')