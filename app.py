"""Flask API for the forecast, telemetry, and scheduler webhook."""

from datetime import datetime
import os
from pathlib import Path
from threading import Thread
import time
import uuid
from flask import Flask, jsonify, render_template, request, send_from_directory
from PIL import Image

from module1 import (
    get_current_carbon_intensity,
    record_live_observation,
    record_live_sample,
    run_forecast,
)
from module2_telemetry_sim import SimulatedEdgeNode
from module3_orchestrator import SpatioTemporalThermalOrchestrator
from module4_quantization_sim import WorkloadTask

app = Flask(__name__)
orchestrator = SpatioTemporalThermalOrchestrator()
scheduler_mode = os.environ.get('SCHEDULER_MODE', 'python').lower()
configured_nodes = [name.strip() for name in os.environ.get(
    'SIMULATED_NODE_IDS', 'pi-node-1,pi-node-2,pi-node-3').split(',') if name.strip()]
_LOCAL_NODE_PROFILES = (
    {'temperature_celsius': 34.0, 'power_watts': 3.4, 'is_busy': False},
    {'temperature_celsius': 48.0, 'power_watts': 7.2, 'is_busy': True},
    {'temperature_celsius': 41.0, 'power_watts': 4.6, 'is_busy': False},
)
from collections import deque
from threading import RLock

nodes = {}
for index, name in enumerate(configured_nodes):
    node = SimulatedEdgeNode(name)
    profile = _LOCAL_NODE_PROFILES[index % len(_LOCAL_NODE_PROFILES)]
    node.temp = profile['temperature_celsius']
    node.current_power = profile['power_watts']
    node.is_busy = profile['is_busy']
    nodes[name] = node

workload_history = deque(maxlen=50)
workload_lock = RLock()

JOB_OUTPUT_DIR = Path(__file__).with_name('job_outputs')
JOB_OUTPUT_DIR.mkdir(exist_ok=True)


def _run_local_telemetry_simulation():
    """Keep local dashboard telemetry different and fresh without Kubernetes."""
    interval = max(1.0, float(os.environ.get('LOCAL_TELEMETRY_INTERVAL_SECONDS', '5')))
    elapsed = 0
    next_update = {index: index * interval * 0.3 for index in range(len(nodes))}
    while True:
        for index, node in enumerate(nodes.values()):
            if elapsed < next_update[index]:
                continue
            cycle = (elapsed + index * 4) % 20
            is_busy = cycle < (7 + index * 2)
            power = (
                node.active_power_int8 if index == 2
                else node.active_power_fp32
            ) if is_busy else node.idle_power + index * 0.35
            node_interval = interval + index * 0.75
            node.update_physics(node_interval, power if is_busy else None)
            next_update[index] += node_interval
        tick = min(0.5, interval / 2)
        time.sleep(tick)
        elapsed += tick


if scheduler_mode != 'kubernetes' and os.environ.get(
        'LOCAL_TELEMETRY_SIMULATION', 'true').lower() == 'true':
    Thread(target=_run_local_telemetry_simulation, daemon=True).start()


def _node_from_payload(payload):
    name = payload.get('node_id', payload.get('name'))
    node = nodes.setdefault(name, SimulatedEdgeNode(name))
    if 'temperature_celsius' in payload:
        node.temp = float(payload['temperature_celsius'])
    if 'power_watts' in payload:
        node.current_power = float(payload['power_watts'])
    node.is_busy = bool(payload.get('is_busy', node.is_busy))
    node.last_updated_monotonic = __import__('time').monotonic()
    node.last_updated_unix = __import__('time').time()
    return node


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/forecast')
def api_forecast():
    try:
        result = run_forecast()
        result.update(status='ok', updated_at=datetime.now().isoformat(timespec='seconds'))
        return jsonify(result)
    except Exception as error:
        return jsonify(status='error', message=str(error)), 500


@app.route('/api/carbon')
def api_carbon():
    """Return the current carbon value for the Kubernetes telemetry bridge."""
    try:
        carbon = get_current_carbon_intensity(
            request.args.get('fallback', type=float))
        aggregation = record_live_sample(datetime.now(), carbon)
        return jsonify(carbon_intensity=carbon, aggregation=aggregation)
    except (OSError, TypeError, ValueError, KeyError) as error:
        return jsonify(status='error', message=str(error)), 502


@app.route('/api/live')
def api_live():
    """Return the live carbon and node values without requesting a forecast."""
    try:
        carbon = get_current_carbon_intensity(
            request.args.get('fallback', type=float))
        aggregation = record_live_sample(datetime.now(), carbon)
        return jsonify(
            carbon_intensity=carbon,
            telemetry=[node.get_telemetry() for node in nodes.values()],
            aggregation=aggregation,
        )
    except (OSError, TypeError, ValueError, KeyError) as error:
        return jsonify(status='error', message=str(error)), 502


@app.route('/api/observation', methods=['POST'])
def api_observation():
    payload = request.get_json(silent=True) or {}
    if 'timestamp' not in payload or 'value' not in payload:
        return jsonify(status='error', message='timestamp and value are required'), 400
    try:
        record_live_observation(payload['timestamp'], payload['value'])
        return jsonify(status='ok', message='Observation added to the training dataset')
    except (TypeError, ValueError, KeyError) as error:
        return jsonify(status='error', message=str(error)), 400


@app.route('/api/telemetry', methods=['GET', 'POST'])
def api_telemetry():
    if request.method == 'POST':
        payload = request.get_json(silent=True) or {}
        if not payload.get('node_id') and not payload.get('name'):
            return jsonify(status='error', message='node_id is required'), 400
        _node_from_payload(payload)
    return jsonify([node.get_telemetry() for node in nodes.values()])


@app.route('/api/workloads')
def api_workloads():
    workloads = []
    with workload_lock:
        history_snapshot = list(workload_history)
    for workload in history_snapshot:
        node = nodes.get(workload['node'])
        state = (workload.get('reported_state') or workload.get('state')
                 or workload['status'])
        if not workload.get('reported_state') and state.startswith('SCHEDULED') and node is not None:
            state = 'RUNNING' if node.is_busy else 'ALLOCATED'
        workloads.append({**workload, 'state': state})
    return jsonify(workloads)


@app.route('/api/jobs/<job_id>')
def job_status(job_id):
    with workload_lock:
        for job in reversed(workload_history):
            if job['task_id'] == job_id:
                return jsonify(job)
    return jsonify(status='error', message='unknown job_id'), 404


@app.route('/job-outputs/<path:filename>')
def job_output(filename):
    return send_from_directory(JOB_OUTPUT_DIR, filename)


def _run_demo_job(job, source_path=None):
    allocated_nodes = [nodes[name] for name in job.get('allocated_nodes', [])
                       if name in nodes]
    if not allocated_nodes:
        job.update(state='FAILED', stage='node_missing',
                   error='Selected node disappeared before execution')
        orchestrator.unreserve(job['task_id'])
        return
    try:
        job.update(state='RUNNING', stage='loading', progress=15)
        for node in allocated_nodes:
            node.is_busy = True
        if job['job_type'] == 'grayscale':
            with Image.open(source_path) as image:
                job.update(stage='transforming', progress=45,
                           transformation='RGB/RGBA image -> grayscale (L)')
                output_name = f'{job["task_id"]}-grayscale.png'
                output_path = JOB_OUTPUT_DIR / output_name
                image.convert('L').save(output_path)
                job.update(stage='output_saved', progress=75,
                           output_url=f'/job-outputs/{output_name}')
            time.sleep(1)
        else:
            job.update(stage='generating', progress=35,
                       transformation=(
                           'Prompt -> simulated image-generation inference'
                           + (' split into three node shards' if len(allocated_nodes) == 3 else '')
                       ))
            for progress in (50, 65, 80):
                time.sleep(1)
                job.update(progress=progress)
            output_name = f'{job["task_id"]}-generated.txt'
            (JOB_OUTPUT_DIR / output_name).write_text(
                f'Simulated generated image for prompt: {job["prompt"]}\n'
                f'Inference precision: {job["precision"]}\n',
                encoding='utf-8',
            )
            job.update(stage='output_saved', progress=90,
                       output_url=f'/job-outputs/{output_name}')
        job.update(state='COMPLETED', stage='completed', progress=100,
                   completed_at=datetime.now().isoformat(timespec='seconds'))
    except (OSError, ValueError) as error:
        job.update(state='FAILED', stage='failed', error=str(error))
    finally:
        for node in allocated_nodes:
            node.is_busy = False
        if source_path is not None:
            source_path.unlink(missing_ok=True)
        if job['state'] in {'COMPLETED', 'FAILED'}:
            orchestrator.unreserve(job['task_id'])


@app.route('/api/jobs', methods=['POST'])
def submit_job():
    """Submit a visible dashboard demo job and run it through the scheduler."""
    if scheduler_mode == 'kubernetes':
        return jsonify(
            status='delegated',
            authority='green-edge-kube-scheduler',
            message='Submit a Kubernetes Job with schedulerName=green-scheduler in Kubernetes mode',
        ), 409
    job_type = request.form.get('job_type', 'grayscale')
    if job_type not in {'grayscale', 'image_generation'}:
        return jsonify(status='error', message='unsupported job_type'), 400
    upload = request.files.get('image')
    if job_type == 'grayscale' and (upload is None or not upload.filename):
        return jsonify(status='error', message='an image upload is required'), 400
    task_id = f'job-{uuid.uuid4().hex[:10]}'
    try:
        accuracy_floor = float(request.form.get('accuracy_floor', '0.85'))
    except ValueError:
        return jsonify(status='error', message='accuracy_floor must be a number'), 400
    if not 0 <= accuracy_floor <= 1:
        return jsonify(status='error', message='accuracy_floor must be between 0 and 1'), 400
    task_payload = {
        'task': {
            'task_id': task_id,
            'is_latency_critical': job_type == 'image_generation',
            'accuracy_floor': accuracy_floor,
            'is_high_power': job_type == 'image_generation',
        },
        'current_grid_carbon': request.form.get('current_grid_carbon', type=float),
    }
    selected, status, task = _schedule_payload(task_payload)
    with workload_lock:
        job = next(item for item in reversed(workload_history)
                   if item['task_id'] == task_id)
    if selected is not None:
        orchestrator.unreserve(task.task_id)
    plan = orchestrator.schedule_plan(task, list(nodes.values()),
                                      job['carbon'])
    job.update(
        node=plan['nodes'][0].node_id if plan['nodes'] else None,
        allocated_nodes=[node.node_id for node in plan['nodes']],
        status=plan['status'],
        decision_reason=plan['decision_reason'],
        retry_after_seconds=plan['retry_after_seconds'],
        precision=task.precision,
    )
    job.update(job_type=job_type, state='QUEUED', stage='scheduled',
               progress=5, prompt=request.form.get('prompt', '').strip())
    if not plan['nodes']:
        job.update(state='DEFERRED' if plan['status'].startswith('DEFERRED') else 'REJECTED',
                   stage='not_allocated', progress=0)
        return jsonify(job), 202
    source_path = None
    if upload is not None and upload.filename:
        source_path = JOB_OUTPUT_DIR / f'{task_id}-input'
        upload.save(source_path)
    Thread(target=_run_demo_job, args=(job, source_path), daemon=True).start()
    return jsonify(job), 202


@app.route('/api/workloads/<task_id>/status', methods=['POST'])
def workload_status(task_id):
    payload = request.get_json(silent=True) or {}
    state = payload.get('state')
    if state not in {'RUNNING', 'COMPLETED', 'FAILED'}:
        return jsonify(status='error', message='state must be RUNNING, COMPLETED, or FAILED'), 400
    with workload_lock:
        for workload in reversed(workload_history):
            if workload['task_id'] == task_id:
                workload['reported_state'] = state
                if state in {'COMPLETED', 'FAILED'}:
                    orchestrator.unreserve(task_id)
                return jsonify(status='ok', task_id=task_id, state=state)
            if state in {'COMPLETED', 'FAILED'}:
                orchestrator.unreserve(task_id)
            return jsonify(status='ok', task_id=task_id, state=state)
    return jsonify(status='error', message='unknown task_id'), 404


@app.route('/metrics')
def metrics():
    """Expose node data in Prometheus text format for scraping."""
    lines = [
        '# HELP edge_node_temperature_celsius Current simulated node temperature.',
        '# TYPE edge_node_temperature_celsius gauge',
        '# HELP edge_node_power_watts Current simulated node power.',
        '# TYPE edge_node_power_watts gauge',
        '# HELP edge_node_busy Whether a node is running a workload.',
        '# TYPE edge_node_busy gauge',
    ]
    for node in nodes.values():
        telemetry = node.get_telemetry()
        label = f'node="{node.node_id}"'
        lines.extend([
            f'edge_node_temperature_celsius{{{label}}} {telemetry["temperature_celsius"]}',
            f'edge_node_power_watts{{{label}}} {telemetry["power_watts"]}',
            f'edge_node_busy{{{label}}} {int(telemetry["is_busy"])}',
        ])
    return '\n'.join(lines) + '\n', 200, {'Content-Type': 'text/plain; version=0.0.4'}


def _schedule_payload(payload):
    task_data = payload.get('task', {})
    task = WorkloadTask(task_data.get('task_id', 'webhook-task'),
                        task_data.get('is_latency_critical', True),
                        task_data.get('accuracy_floor', 0.85),
                        is_high_power=task_data.get('is_high_power', False))
                        
    # Temporal Carbon Shifting: Use node annotation if provided, else fallback to API
    carbon = payload.get('current_grid_carbon')
    if not carbon and 'nodes' in payload and payload['nodes']:
        annotations = payload['nodes'][0].get('metadata', {}).get('annotations', {})
        carbon = annotations.get('green-edge.io/carbon-intensity')
        
    carbon = get_current_carbon_intensity(carbon)
    
    selected, status = orchestrator.schedule(task, list(nodes.values()), carbon)
    with workload_lock:
        workload_history.append({
            'task_id': task.task_id,
            'node': selected.node_id if selected else None,
            'precision': task.precision,
            'carbon': round(carbon, 2),
            'status': status,
        })
    return selected, status, task


@app.route('/api/scheduler/filter', methods=['POST'])
def scheduler_filter():
    payload = request.get_json(silent=True) or {}
    requested = {item.get('metadata', {}).get('name') for item in payload.get('nodes', [])}
    allowed = [node.node_id for node in nodes.values()
               if node.node_id in requested and node.temp < orchestrator.temp_limit]
    return jsonify({'nodes': [{'metadata': {'name': name}} for name in allowed]})


@app.route('/api/scheduler/prioritize', methods=['POST'])
def scheduler_prioritize():
    payload = request.get_json(silent=True) or {}
    carbon = float(payload.get('current_grid_carbon', 0))
    requested = [item.get('metadata', {}).get('name') for item in payload.get('nodes', [])]
    ranked = orchestrator.score_nodes([nodes[name] for name in requested if name in nodes], carbon)
    scores = {node.node_id: max(0, round((1 - penalty) * 10)) for node, penalty in ranked}
    return jsonify([{'host': name, 'score': scores.get(name, 0)} for name in requested])


@app.route('/api/scheduler/schedule', methods=['POST'])
def scheduler_schedule():
    if scheduler_mode == 'kubernetes':
        return jsonify(
            status='delegated',
            authority='green-edge-kube-scheduler',
            message='Kubernetes scheduling is authoritative; submit a Pod with schedulerName=green-scheduler',
        ), 409
    selected, status, task = _schedule_payload(request.get_json(silent=True) or {})
    return jsonify({'node': selected.node_id if selected else None,
                    'status': status, 'precision': task.precision})


@app.route('/api/scheduler/reservations')
def scheduler_reservations():
    return jsonify(orchestrator.reservations())


@app.route('/api/scheduler/status')
def scheduler_status():
    return jsonify({
        'mode': scheduler_mode,
        'authority': 'green-edge-kube-scheduler'
        if scheduler_mode == 'kubernetes' else 'flask-orchestrator',
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)