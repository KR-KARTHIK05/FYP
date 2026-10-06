"""Software model of a Raspberry Pi 5 node's thermal and power telemetry.

Thermal model uses the first-order RC exponential step response:
  T(t+dt) = T_ss - (T_ss - T(t)) * exp(-dt / tau)
where T_ss = T_amb + P * R  and  tau = R * C.

Measured Pi 5 parameters (passive heatsink, sustained compute):
  R ≈ 5 °C/W  (Pimoroni Heatsink case, fan-less)
  C ≈ 50 J/°C (SoC die + heatsink mass estimate)
  → tau ≈ 250 s  (~4 min to reach thermal equilibrium)
  → T_ss at FP32 load (8.5 W): 28 + 8.5*5 = 70.5 °C  (within safe range)
"""

import math
import time
from threading import RLock


class SimulatedEdgeNode:
    """Thread-safe model of a single Raspberry Pi 5 compute node."""

    def __init__(self, node_id: str,
                 ambient_temp: float = 28.0,
                 max_safe_temp: float = 75.0):
        self._lock = RLock()
        self.node_id = node_id
        self.ambient_temp = ambient_temp
        self.max_safe_temp = max_safe_temp

        # Thermal RC model parameters (Pi 5, passive heatsink)
        self.thermal_resistance: float = 5.0   # °C/W
        self.thermal_capacitance: float = 50.0  # J/°C  →  tau = 250 s

        # Measured power draw at different precision levels (Pi 5 benchmarks)
        self.idle_power: float = 2.8          # W  (OS idle)
        self.active_power_fp32: float = 8.5   # W  (FP32 inference)
        self.active_power_int8: float = 4.2   # W  (INT8 quantized inference)

        self.temp: float = ambient_temp
        self.current_power: float = self.idle_power
        self.is_busy: bool = False

        self._last_updated_monotonic: float = time.monotonic()
        self._last_updated_unix: float = time.time()

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------

    def update_physics(self, time_step_seconds: float,
                       active_workload_power: float | None = None) -> None:
        """Advance the RC thermal model by *time_step_seconds*.

        Args:
            time_step_seconds: Wall-clock seconds elapsed since last call.
            active_workload_power: Watts drawn by the workload, or None if idle.
        """
        with self._lock:
            self.current_power = (
                active_workload_power
                if active_workload_power is not None
                else self.idle_power
            )
            self.is_busy = active_workload_power is not None

            tau = self.thermal_resistance * self.thermal_capacitance
            t_ss = self.ambient_temp + self.current_power * self.thermal_resistance
            # Exact exponential step-response — numerically stable for any dt
            self.temp = t_ss - (t_ss - self.temp) * math.exp(-time_step_seconds / tau)
            self.temp = max(self.ambient_temp, self.temp)

            self._last_updated_monotonic = time.monotonic()
            self._last_updated_unix = time.time()

    def touch(self) -> None:
        """Reset the telemetry age to zero (call after any live sensor read)."""
        with self._lock:
            self._last_updated_monotonic = time.monotonic()
            self._last_updated_unix = time.time()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def telemetry_age_seconds(self) -> float:
        """Seconds elapsed since the last physics update or sensor read."""
        return max(0.0, time.monotonic() - self._last_updated_monotonic)

    def get_telemetry(self) -> dict:
        """Return a consistent snapshot of the node's current state."""
        with self._lock:
            return {
                'node_id': self.node_id,
                'temperature_celsius': round(self.temp, 2),
                'power_watts': round(self.current_power, 2),
                'is_busy': self.is_busy,
                'is_overheating': self.temp >= self.max_safe_temp,
                'telemetry_age_seconds': round(self.telemetry_age_seconds(), 2),
                'telemetry_updated_at': self._last_updated_unix,
            }

    # Allow app.py to set last_updated fields via attribute assignment
    @property
    def last_updated_monotonic(self) -> float:
        return self._last_updated_monotonic

    @last_updated_monotonic.setter
    def last_updated_monotonic(self, value: float) -> None:
        with self._lock:
            self._last_updated_monotonic = value

    @property
    def last_updated_unix(self) -> float:
        return self._last_updated_unix

    @last_updated_unix.setter
    def last_updated_unix(self, value: float) -> None:
        with self._lock:
            self._last_updated_unix = value