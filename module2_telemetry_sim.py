"""Software model of a Raspberry Pi node's thermal and power telemetry."""

import time


class SimulatedEdgeNode:
    def __init__(self, node_id, ambient_temp=28.0, max_safe_temp=75.0):
        self.node_id = node_id
        self.ambient_temp = ambient_temp
        self.max_safe_temp = max_safe_temp
        self.temp = ambient_temp
        self.idle_power = 2.8
        self.active_power_fp32 = 8.5
        self.active_power_int8 = 4.2
        self.thermal_resistance = 4.8
        self.thermal_capacitance = 18.0
        self.current_power = self.idle_power
        self.is_busy = False
        self.last_updated_monotonic = time.monotonic()
        self.last_updated_unix = time.time()

    def update_physics(self, time_step_seconds, active_workload_power=None):
        self.current_power = active_workload_power if active_workload_power is not None else self.idle_power
        self.is_busy = active_workload_power is not None
        heating = self.current_power / self.thermal_capacitance * time_step_seconds
        cooling = ((self.temp - self.ambient_temp) /
                   (self.thermal_resistance * self.thermal_capacitance) * time_step_seconds)
        self.temp = max(self.ambient_temp, self.temp + heating - cooling)
        self.last_updated_monotonic = time.monotonic()
        self.last_updated_unix = time.time()

    def telemetry_age_seconds(self):
        return max(0.0, time.monotonic() - self.last_updated_monotonic)

    def get_telemetry(self):
        return {
            'node_id': self.node_id,
            'temperature_celsius': round(self.temp, 2),
            'power_watts': round(self.current_power, 2),
            'is_busy': self.is_busy,
            'is_overheating': self.temp >= self.max_safe_temp,
            'telemetry_age_seconds': round(self.telemetry_age_seconds(), 2),
            'telemetry_updated_at': self.last_updated_unix,
        }