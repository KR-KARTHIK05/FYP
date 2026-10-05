"""Run a visible demo workload and report its lifecycle to the webhook."""

import json
import os
import time
from urllib.request import Request, urlopen


def report(status_url, task_id, state):
    payload = json.dumps({'state': state}).encode('utf-8')
    request = Request(status_url, data=payload,
                      headers={'Content-Type': 'application/json'}, method='POST')
    with urlopen(request, timeout=5):
        pass


def main():
    task_id = os.environ['TASK_ID']
    status_url = os.environ['STATUS_URL']
    duration = int(os.environ.get('WORKLOAD_DURATION_SECONDS', '30'))
    report(status_url, task_id, 'RUNNING')
    for step in range(1, duration + 1):
        print(f'{task_id}: inference_step={step}', flush=True)
        time.sleep(1)
    report(status_url, task_id, 'COMPLETED')
    print(f'{task_id}: inference_complete', flush=True)


if __name__ == '__main__':
    main()