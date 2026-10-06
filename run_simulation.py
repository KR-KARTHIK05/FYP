"""Run a reproducible 24-hour comparison of baseline and proposed scheduling."""

import numpy as np
import matplotlib.pyplot as plt

from module1 import run_forecast
from module2_telemetry_sim import SimulatedEdgeNode
from module3_orchestrator import SpatioTemporalThermalOrchestrator
from module4_quantization_sim import WorkloadTask


def run_benchmark(carbon_forecast, scheduler_type):
    nodes = [SimulatedEdgeNode(f'pi-node-{index}') for index in range(1, 4)]
    safety_limit = 65.0 if scheduler_type == 'proposed' else 75.0
    orchestrator = SpatioTemporalThermalOrchestrator(temp_limit=safety_limit)
    temperatures = {node.node_id: [] for node in nodes}
    total_carbon = 0.0
    quantized = deferred = 0
    random = np.random.default_rng(42)
    for hour, carbon in enumerate(carbon_forecast):
        for minute in range(60):
            assignments = {}
            for task_index in range(3):
                if scheduler_type == 'proposed':
                    task = WorkloadTask(f't-{hour}-{minute}-{task_index}', random.random() > 0.3)
                    # Use node.node_id for comparison since assignments stores nodes
                    available_nodes = [node for node in nodes if node not in assignments]
                    if available_nodes:
                        selected, status = orchestrator.schedule(task, available_nodes, carbon)
                    else:
                        selected, status = None, 'DEFERRED_CAPACITY_WINDOW'
                    quantized += status == 'SCHEDULED_QUANTIZED_INT8'
                    deferred += status.startswith('DEFERRED')
                else:
                    selected, status = nodes[task_index % len(nodes)], 'SCHEDULED_OPTIMAL_FP32'
                if selected is not None and status.startswith('SCHEDULED'):
                    assignments[selected] = status
            
            for node in nodes:
                status = assignments.get(node)
                active = status is not None
                power = node.active_power_int8 if active and 'INT8' in status else node.active_power_fp32
                node.update_physics(60, power if active else None)
                node.touch() # Reset telemetry age so they aren't filtered out
                watts = power if active else node.idle_power
                total_carbon += watts * 60 / 3600000 * carbon
                temperatures[node.node_id].append(node.temp)
            
            # Clear reservations at the end of the simulated minute since tasks only last 1 minute in the simulation
            if scheduler_type == 'proposed':
                for task_index in range(3):
                    orchestrator.unreserve(f't-{hour}-{minute}-{task_index}')
                    
    return total_carbon, temperatures, quantized, deferred


def execute_simulation():
    forecast = run_forecast()['forecast']
    baseline = run_benchmark(forecast, 'baseline')
    proposed = run_benchmark(forecast, 'proposed')
    reduction = (baseline[0] - proposed[0]) / baseline[0] * 100
    print(f'Baseline carbon: {baseline[0]:.2f} gCO2')
    print(f'Proposed carbon: {proposed[0]:.2f} gCO2')
    print(f'Carbon reduction: {reduction:.2f}%')
    print(f'INT8 events: {proposed[2]} | Deferred tasks: {proposed[3]}')
    timeline = np.linspace(0, 24, len(baseline[1]['pi-node-1']))
    figure, axes = plt.subplots(2, 1, figsize=(12, 7))
    axes[0].plot(timeline, baseline[1]['pi-node-1'], label='Baseline')
    axes[0].plot(timeline, proposed[1]['pi-node-1'], label='Proposed')
    axes[0].axhline(75, color='black', linestyle=':', label='75 C limit')
    axes[0].set_ylabel('Temperature (C)')
    axes[0].legend()
    axes[0].grid(True)
    axes[1].plot(range(1, 25), forecast, 'k-o')
    axes[1].set_xlabel('Hour')
    axes[1].set_ylabel('Grid carbon (gCO2/kWh)')
    axes[1].grid(True)
    figure.tight_layout()
    figure.savefig('simulation_benchmark_results.png', dpi=200)


if __name__ == '__main__':
    execute_simulation()