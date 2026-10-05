# Green Edge Scheduler UI Guide

This document explains the terms shown in the dashboard, what each control is
used for, and how the value is produced by the application.

## API Status

### API Status

Shows whether the Flask backend is reachable. `API Online` means the dashboard
can communicate with the application APIs.

### Live carbon

Shows the current grid carbon-intensity value in `gCO2/kWh` (grams of carbon
dioxide per kilowatt-hour). The scheduler uses this value when scoring nodes and
deciding whether a high-power job should use INT8, split, or wait.

### Refresh

Reloads the API status, live carbon, telemetry, workloads, and forecast data.

## Submit a Demonstration Job

This section submits a workload to the Python scheduler in local mode. In
Kubernetes mode, Kubernetes and the Go scheduler plugin are authoritative.

### Workload type

Selects the type and expected power class of the job.

- **Low power - convert image to grayscale**: Uploads an image and converts it
  to grayscale using Pillow. This represents a small CPU workload.
- **High power - generate image from prompt (simulated)**: Uses a text prompt to
  simulate an image-generation workload. It is marked as high-power and
  latency-critical so the scheduler applies stricter carbon and thermal logic.

### Input image

The source image required by the grayscale workload. It is temporarily stored,
converted to grayscale, and made available as a job output.

### Image prompt

The text description used by the simulated image-generation workload. The
current demo writes a text output; it does not run a real image-generation
model.

### Minimum accuracy floor

The minimum acceptable accuracy for the workload, expressed from `0` to `1`.
The scheduler compares this value with the declared precision accuracies:

- FP32: approximately `0.94`
- INT8: approximately `0.90`

INT8 is allowed only when its expected accuracy is at least the requested
floor. For example, `0.85` allows INT8, while `0.95` does not.

### Grid carbon override for demo

An optional manual carbon-intensity value for this job, in `gCO2/kWh`.
Leaving it empty uses the live carbon/forecast value. Entering a value such as
`800` creates a reproducible high-carbon demonstration without changing the
actual forecast or future jobs.

### Upload and Schedule Job

Sends the job to the scheduler. The scheduler checks telemetry freshness,
temperature, reservations, carbon stress, power, workload type, and accuracy
requirements before selecting a node or scheduling plan.

## Observability

### Prometheus

Opens the Prometheus monitoring interface. Prometheus scrapes the `/metrics`
endpoint and stores node metrics for querying.

### Grafana

Opens Grafana dashboards for visualizing metrics collected by Prometheus.
Grafana and Prometheus must be running separately.

## API Endpoints

### `GET /api/forecast`

Returns the cached or newly trained 24-hour carbon forecast. The response
includes:

- `forecast`: predicted carbon values for the next 24 hours;
- `forecast_source`: `cache` or `retrained`;
- `updated_at`: response time;
- `last_trained_at`: time of the last LSTM training.

The dashboard polls this endpoint every 10 seconds, but the LSTM is not
retrained on every poll.

### `GET /api/live`

Returns the current carbon value and telemetry for every node. It also records
a live carbon sample in the current aggregation window.

### `POST /api/observation`

Adds a timestamped carbon observation to the training dataset and clears the
forecast cache. The next forecast request can retrain the model.

### `GET /api/telemetry`

Returns the latest telemetry for all configured nodes.

### `POST /api/telemetry`

Updates one node's telemetry. A payload normally includes `node_id`,
temperature, power, and busy/idle state.

### `GET /metrics`

Returns Prometheus-format metrics for node temperature, power, and busy state.

### `POST /api/scheduler/filter`

Filters out nodes that are too hot, reserved, or have stale telemetry.

### `POST /api/scheduler/prioritize`

Ranks requested nodes using carbon, temperature, power, and logical workload
factors.

### `POST /api/scheduler/schedule`

Runs the complete Python scheduling decision in local mode. In Kubernetes mode,
the request is delegated to the Go scheduler plugin.

## Live Node Telemetry

### Node

The simulated or Kubernetes node identifier, such as `pi-node-1`.

### Temp (°C)

The current simulated node temperature. It increases during active workloads
and cools toward the ambient temperature while idle.

### Power (W)

The current power draw in watts. Idle, FP32, and INT8 workloads use different
power levels.

### State

Shows whether the node is currently `Idle` or `Busy`.

### Thermal

Shows whether the node is within the configured thermal limit. `OK` means it is
safe for scheduling; `OVERHEATING` means it should not receive new work.

### Telemetry age

Shows how many seconds have passed since this specific node last published
telemetry:

```text
telemetry age = current monotonic time - node update monotonic time
```

Low age means fresh telemetry. The Python scheduler normally rejects telemetry
older than 15 seconds. Local simulation nodes update independently, so their
ages may be different.

## Live Workload Allocation

### Task

The unique identifier assigned to the submitted workload.

### Type

The workload category, such as `grayscale`, `image_generation`, or a scheduler
API task.

### Node

The node selected for the workload. A split job can list multiple allocated
nodes in the job details.

### Precision

The selected numeric precision:

- `FP32`: normal full-precision execution;
- `INT8`: reduced-precision execution intended to reduce CPU power and heat.

### Stage

The current execution phase, such as `scheduled`, `loading`,
`transforming`, `generating`, `output_saved`, or `completed`.

### Progress

An approximate completion percentage for the demonstration workload.

### Grid Carbon

The carbon-intensity value used by the scheduler for this workload. It may be
the live value or the optional demo override.

### State

The workload lifecycle state:

- `QUEUED`: accepted and waiting to execute;
- `RUNNING`: currently executing;
- `COMPLETED`: finished successfully;
- `DEFERRED`: waiting for a fresher, cooler, or cleaner scheduling window;
- `REJECTED`: could not satisfy the requested constraints;
- `FAILED`: execution encountered an error.

## 24-Hour Carbon Forecast

### Forecast chips

Each green value represents the predicted carbon intensity for one future hour.
There are 24 values because the LSTM produces a 24-hour forecast.

### Source: cache

The forecast was returned from memory without retraining the LSTM.

### Source: retrained

The input data changed or the cache was cleared, so the LSTM trained again.

### Response

The time when the latest `/api/forecast` response was generated. This can
change even when the forecast itself is cached.

### Last trained

The time when the LSTM model most recently trained. This is the timestamp to
watch when checking whether the forecast model actually ran again.

## Forecast update workflow

The dashboard and backend use two different update rates:

```text
Every 10 seconds:
    Refresh live carbon, telemetry, workloads, and cached forecast display.

Every 15 minutes by default:
    Average collected live carbon samples.
    Add one aggregate observation to the training dataset.
    Clear the forecast cache.

Next forecast request:
    Retrain the LSTM and generate a replacement 24-hour forecast.
```

The aggregation interval can be changed with:

```text
FORECAST_AGGREGATION_MINUTES=60
```

## Scheduling decision workflow

When a new job arrives:

1. The job type and accuracy floor are read.
2. Current carbon intensity is selected from the live source or demo override.
3. Nodes with stale telemetry, excessive temperature, or reservations are
   filtered out.
4. Remaining nodes are scored using carbon, thermal, power, and logical
   workload conditions.
5. A high-power job may select INT8 if the accuracy floor allows it.
6. If INT8 is not acceptable, the job may be split across three nodes.
7. If no safe plan is available, the job is deferred for a later window.
8. Reserved nodes are released when the workload completes or fails.
