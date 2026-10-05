from threading import RLock


class SpatioTemporalThermalOrchestrator:
    def __init__(self, w1=0.45, w2=0.35, w3=0.20, temp_limit=75.0,
                 telemetry_max_age_seconds=15.0):
        # Configurable weights balancing carbon, thermal state, and power.
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.temp_limit = temp_limit
        self.telemetry_max_age_seconds = telemetry_max_age_seconds
        self._reservations = {}
        self._lock = RLock()

    def filter_nodes(self, nodes):
        """Filter hot or already-reserved nodes before scoring."""
        with self._lock:
            reserved = set(self._reservations)
        viable = [
            node for node in nodes
            if node.temp < self.temp_limit
            and node.node_id not in reserved
            and node.telemetry_age_seconds() <= self.telemetry_max_age_seconds
        ]
        return viable

    def score_nodes(self, viable_nodes, current_grid_carbon,
                    max_observed_carbon=800.0, task=None):
        """
        Score Phase: Lower score indicates lower environmental and hardware impact.
        Normalized cost penalty = w1*C_norm + w2*T_norm + w3*P_norm
        """
        scored_nodes = []
        c_norm = min(1.0, current_grid_carbon / max_observed_carbon)

        for node in viable_nodes:
            t_norm = (node.temp - node.ambient_temp) / (self.temp_limit - node.ambient_temp)
            p_norm = node.current_power / 12.0  # Normalized to max Pi power
            
            logical_penalty = 0.0
            if task is not None and task.is_latency_critical and node.is_busy:
                logical_penalty = 0.15
            penalty = ((self.w1 * c_norm) + (self.w2 * t_norm) +
                       (self.w3 * p_norm) + logical_penalty)
            scored_nodes.append((node, penalty))

        # Sort ascending (lowest penalty score wins)
        scored_nodes.sort(key=lambda x: x[1])
        return scored_nodes

    def schedule(self, task, nodes, current_grid_carbon):
        plan = self.schedule_plan(task, nodes, current_grid_carbon)
        selected = plan['nodes'][0] if plan['nodes'] else None
        return selected, plan['status']

    def schedule_plan(self, task, nodes, current_grid_carbon):
        viable = self.filter_nodes(nodes)
        if not viable:
            if task.is_latency_critical and task.accuracy_floor > 0.90:
                return {
                    'nodes': [], 'status': 'REJECTED_ACCURACY_FLOOR_BREACH',
                    'decision_reason': 'INT8 fallback would violate the accuracy floor',
                    'retry_after_seconds': 0,
                }
            return self._deferred_plan('No fresh, cool, unreserved node is available')

        ranked = self.score_nodes(viable, current_grid_carbon, task=task)
        stress = min(1.0, current_grid_carbon / 800.0)
        hottest = max(node.temp for node in viable)
        thermal_stress = hottest >= self.temp_limit - 10

        if task.is_high_power and (thermal_stress or stress >= 0.75):
            if task.scale_precision_to_int8():
                node = ranked[0][0]
                if self.reserve(node, task.task_id):
                    return {
                        'nodes': [node], 'status': 'SCHEDULED_QUANTIZED_INT8',
                        'decision_reason': 'High-power job reduced to INT8 because carbon or thermal stress is high',
                        'retry_after_seconds': 0,
                    }
            if len(viable) >= 3:
                selected = [item[0] for item in ranked[:3]]
                if self.reserve_many(selected, task.task_id):
                    return {
                        'nodes': selected, 'status': 'SCHEDULED_SPLIT_THREE_NODES',
                        'decision_reason': 'High-power job split across three cool nodes to reduce per-node heat',
                        'retry_after_seconds': 0,
                    }
            return self._deferred_plan(
                'High-power job is waiting for a lower-carbon or cooler window',
                retry_after_seconds=60,
            )

        for selected_node, _ in ranked:
            if self.reserve(selected_node, task.task_id):
                return {
                    'nodes': [selected_node], 'status': 'SCHEDULED_OPTIMAL_FP32',
                    'decision_reason': 'Selected the lowest combined carbon, thermal, and power penalty',
                    'retry_after_seconds': 0,
                }
        return self._deferred_plan('All suitable nodes became reserved during allocation')

    def _deferred_plan(self, reason, retry_after_seconds=30):
        return {
            'nodes': [], 'status': 'DEFERRED_WAITING_COOL_CLEAN_WINDOW',
            'decision_reason': reason, 'retry_after_seconds': retry_after_seconds,
        }

    def reserve_many(self, nodes, task_id):
        with self._lock:
            if any(self._reservations.get(node.node_id) not in (None, task_id)
                   for node in nodes):
                return False
            for node in nodes:
                self._reservations[node.node_id] = task_id
            return True

    def reserve(self, node, task_id):
        """Reserve a node atomically for a task until completion or cancellation."""
        with self._lock:
            owner = self._reservations.get(node.node_id)
            if owner is not None and owner != task_id:
                return False
            self._reservations[node.node_id] = task_id
            return True

    def unreserve(self, task_id):
        """Release every reservation held by a task."""
        with self._lock:
            released = [
                node_id for node_id, owner in self._reservations.items()
                if owner == task_id
            ]
            for node_id in released:
                del self._reservations[node_id]
            return released

    def reservations(self):
        """Return a snapshot for diagnostics and tests."""
        with self._lock:
            return dict(self._reservations)