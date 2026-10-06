"""Spatio-Temporal-Thermal Orchestrator for the Green Edge Scheduler.

Scoring formula (lower penalty = better node):
    penalty = w1 * C_norm + w2 * T_norm + w3 * P_norm  [+ latency_penalty]

where:
    C_norm = carbon_intensity / MAX_CARBON          ∈ [0, 1]
    T_norm = (temp - ambient) / (limit - ambient)   ∈ [0, 1]  (clamped)
    P_norm = power / MAX_PI5_POWER                  ∈ [0, 1]  (clamped)
"""

from threading import RLock

# Normalisation bounds
_MAX_CARBON_GRAMS = 800.0   # gCO2/kWh — India historical ceiling
_MAX_PI5_POWER_W  = 12.0    # Watts   — Raspberry Pi 5 absolute maximum


class SpatioTemporalThermalOrchestrator:
    """Thread-safe scheduler orchestrator for carbon + thermal + power scoring."""

    def __init__(self,
                 w1: float = 0.45,   # carbon weight
                 w2: float = 0.35,   # thermal weight
                 w3: float = 0.20,   # power weight
                 temp_limit: float = 75.0,
                 telemetry_max_age_seconds: float = 15.0):
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.temp_limit = temp_limit
        self.telemetry_max_age_seconds = telemetry_max_age_seconds
        self._reservations: dict[str, str] = {}  # node_id → task_id
        self._lock = RLock()

    # ------------------------------------------------------------------
    # Filter phase
    # ------------------------------------------------------------------

    def filter_nodes(self, nodes):
        """Remove overheating, stale-telemetry, and already-reserved nodes."""
        with self._lock:
            reserved = set(self._reservations)
        return [
            node for node in nodes
            if node.temp < self.temp_limit
            and node.node_id not in reserved
            and node.telemetry_age_seconds() <= self.telemetry_max_age_seconds
        ]

    # ------------------------------------------------------------------
    # Score phase
    # ------------------------------------------------------------------

    def score_nodes(self, viable_nodes, current_grid_carbon: float,
                    max_observed_carbon: float = _MAX_CARBON_GRAMS,
                    task=None):
        """Compute and rank nodes by combined cost penalty (ascending).

        Returns:
            List of (node, penalty) tuples sorted lowest-penalty first.
        """
        c_norm = min(1.0, current_grid_carbon / max_observed_carbon)

        scored = []
        for node in viable_nodes:
            # Clamp T_norm and P_norm to [0, 1] to prevent score overflow
            t_norm = (node.temp - node.ambient_temp) / (
                max(1e-6, self.temp_limit - node.ambient_temp)
            )
            t_norm = min(1.0, max(0.0, t_norm))

            p_norm = min(1.0, max(0.0, node.current_power / _MAX_PI5_POWER_W))

            # Extra penalty for sending latency-critical tasks to busy nodes
            latency_penalty = 0.15 if (
                task is not None
                and task.is_latency_critical
                and node.is_busy
            ) else 0.0

            penalty = (self.w1 * c_norm +
                       self.w2 * t_norm +
                       self.w3 * p_norm +
                       latency_penalty)
            scored.append((node, penalty))

        scored.sort(key=lambda x: x[1])
        return scored

    # ------------------------------------------------------------------
    # Scheduling entry points
    # ------------------------------------------------------------------

    def schedule(self, task, nodes, current_grid_carbon: float):
        """Convenience wrapper — returns (selected_node | None, status_str)."""
        plan = self.schedule_plan(task, nodes, current_grid_carbon)
        return plan['nodes'][0] if plan['nodes'] else None, plan['status']

    def schedule_plan(self, task, nodes, current_grid_carbon: float) -> dict:
        """Full scheduling decision with detailed reason and retry hint."""
        viable = self.filter_nodes(nodes)

        if not viable:
            # No usable node — reject high-accuracy-critical tasks outright;
            # defer everything else.  The redundant accuracy_floor > 0.90 check
            # was removed: scale_precision_to_int8() already enforces the floor.
            return self._deferred_plan('No fresh, cool, unreserved node is available')

        ranked = self.score_nodes(viable, current_grid_carbon, task=task)
        stress = min(1.0, current_grid_carbon / _MAX_CARBON_GRAMS)
        thermal_stress = max(node.temp for node in viable) >= (self.temp_limit - 10)

        # High-power tasks under carbon or thermal pressure → quantize or split
        if task.is_high_power and (thermal_stress or stress >= 0.75):
            if task.scale_precision_to_int8():
                node = ranked[0][0]
                if self.reserve(node, task.task_id):
                    return {
                        'nodes': [node],
                        'status': 'SCHEDULED_QUANTIZED_INT8',
                        'decision_reason': (
                            'High-power job reduced to INT8: carbon or thermal stress is high'),
                        'retry_after_seconds': 0,
                    }
            # INT8 not viable → try splitting across three cool nodes
            if len(viable) >= 3:
                selected = [item[0] for item in ranked[:3]]
                if self.reserve_many(selected, task.task_id):
                    return {
                        'nodes': selected,
                        'status': 'SCHEDULED_SPLIT_THREE_NODES',
                        'decision_reason': (
                            'High-power job split across three cool nodes to reduce per-node heat'),
                        'retry_after_seconds': 0,
                    }
            return self._deferred_plan(
                'High-power job waiting for a lower-carbon or cooler window',
                retry_after_seconds=60,
            )

        # Normal path: pick the lowest-penalty viable node
        for selected_node, _ in ranked:
            if self.reserve(selected_node, task.task_id):
                return {
                    'nodes': [selected_node],
                    'status': 'SCHEDULED_OPTIMAL_FP32',
                    'decision_reason': (
                        'Selected the lowest combined carbon, thermal, and power penalty'),
                    'retry_after_seconds': 0,
                }

        return self._deferred_plan('All suitable nodes became reserved during allocation')

    # ------------------------------------------------------------------
    # Reservation management (all atomic under _lock)
    # ------------------------------------------------------------------

    def reserve(self, node, task_id: str) -> bool:
        """Atomically reserve *node* for *task_id*.  Returns False if taken."""
        with self._lock:
            owner = self._reservations.get(node.node_id)
            if owner is not None and owner != task_id:
                return False
            self._reservations[node.node_id] = task_id
            return True

    def reserve_many(self, nodes, task_id: str) -> bool:
        """Atomically reserve all nodes or none (all-or-nothing transaction)."""
        with self._lock:
            if any(
                self._reservations.get(n.node_id) not in (None, task_id)
                for n in nodes
            ):
                return False
            for n in nodes:
                self._reservations[n.node_id] = task_id
            return True

    def unreserve(self, task_id: str) -> list[str]:
        """Release every reservation held by *task_id*.  Returns freed node IDs."""
        with self._lock:
            released = [
                nid for nid, owner in self._reservations.items()
                if owner == task_id
            ]
            for nid in released:
                del self._reservations[nid]
            return released

    def reservations(self) -> dict:
        """Return a snapshot of current reservations (for diagnostics/tests)."""
        with self._lock:
            return dict(self._reservations)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _deferred_plan(self, reason: str, retry_after_seconds: int = 30) -> dict:
        return {
            'nodes': [],
            'status': 'DEFERRED_WAITING_COOL_CLEAN_WINDOW',
            'decision_reason': reason,
            'retry_after_seconds': retry_after_seconds,
        }