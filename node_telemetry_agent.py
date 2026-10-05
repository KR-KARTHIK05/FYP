"""Publish simulated node telemetry from a Kubernetes DaemonSet."""

import json
import os
import random
import ssl
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from module2_telemetry_sim import SimulatedEdgeNode


def publish(node, telemetry_url):
    payload = json.dumps(node.get_telemetry()).encode('utf-8')
    request = Request(telemetry_url, data=payload,
                      headers={'Content-Type': 'application/json'}, method='POST')
    with urlopen(request, timeout=5):
        pass


def publish_node_annotations(node, carbon_url):
    """Publish Flask-backed telemetry as annotations consumed by the Go plugin."""
    token_path = '/var/run/secrets/kubernetes.io/serviceaccount/token'
    host = os.environ.get('KUBERNETES_SERVICE_HOST')
    port = os.environ.get('KUBERNETES_SERVICE_PORT', '443')
    node_name = os.environ['NODE_NAME']
    if not host or not os.path.exists(token_path):
        return
    with open(token_path, encoding='utf-8') as token_file:
        token = token_file.read().strip()
    with urlopen(Request(carbon_url, timeout=5)) as response:
        carbon = json.loads(response.read().decode('utf-8'))['carbon_intensity']
    telemetry = node.get_telemetry()
    annotations = {
        'green-edge.io/temperature-celsius': str(telemetry['temperature_celsius']),
        'green-edge.io/power-watts': str(telemetry['power_watts']),
        'green-edge.io/carbon-intensity': str(carbon),
        'green-edge.io/busy': str(telemetry['is_busy']).lower(),
        'green-edge.io/telemetry-unix': str(int(telemetry['telemetry_updated_at'])),
    }
    patch = json.dumps({'metadata': {'annotations': annotations}}).encode('utf-8')
    request = Request(
        f'https://{host}:{port}/api/v1/nodes/{node_name}',
        data=patch,
        headers={
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/merge-patch+json',
        },
        method='PATCH',
    )
    ca_path = '/var/run/secrets/kubernetes.io/serviceaccount/ca.crt'
    context = ssl.create_default_context(cafile=ca_path) if os.path.exists(ca_path) else None
    with urlopen(request, timeout=5, context=context):
        pass


def main():
    node_id = os.environ['NODE_NAME']
    telemetry_url = os.environ.get(
        'TELEMETRY_URL',
        'http://green-scheduler-webhook:5000/api/telemetry',
    )
    carbon_url = os.environ.get(
        'CARBON_URL',
        'http://green-scheduler-webhook:5000/api/carbon',
    )
    interval = float(os.environ.get('TELEMETRY_INTERVAL_SECONDS', '5'))
    node = SimulatedEdgeNode(node_id)
    randomizer = random.Random()
    randomizer.seed(f'{node_id}-{time.time_ns()}')
    busy_remaining = randomizer.randint(0, 6)

    while True:
        if busy_remaining <= 0:
            if node.is_busy:
                busy_remaining = randomizer.randint(2, 8)
            else:
                busy_remaining = randomizer.randint(2, 10)
            node.is_busy = not node.is_busy

        if node.is_busy:
            base_power = (node.active_power_int8 if randomizer.random() < 0.25
                          else node.active_power_fp32)
            power = max(3.5, base_power + randomizer.uniform(-0.6, 0.6))
        else:
            power = max(2.4, node.idle_power + randomizer.uniform(-0.2, 0.2))
        node.update_physics(interval, power if node.is_busy else None)
        try:
            publish(node, telemetry_url)
            publish_node_annotations(node, carbon_url)
        except (HTTPError, OSError, URLError, KeyError, ValueError) as error:
            print(f'Unable to publish telemetry: {error}', flush=True)
        busy_remaining -= 1
        time.sleep(interval)


if __name__ == '__main__':
    main()