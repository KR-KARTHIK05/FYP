# Green Edge Architecture: Extender vs. Framework Plugin

## Design Clarification

In the initial proposal, this project was envisioned to include a native Go-based Kubernetes Scheduling Framework Plugin. However, during the implementation phase, the architecture was intentionally pivoted to a **Kubernetes Scheduler Extender** model backed by a Python Flask application.

### Justification for the Pivot

1. **Ecosystem Compatibility:** 
   The core ML forecasting (LSTM) and data science logic are implemented in PyTorch and Pandas. Rewriting the entire inference and scoring pipeline in Go would have introduced significant technical debt and fragmented the codebase.
   
2. **Extender Pattern:**
   By implementing a webhook-based Extender (`/api/scheduler/filter` and `/api/scheduler/prioritize`), Kubernetes can natively delegate scheduling decisions to the Python orchestrator without requiring Go code compilation. 
   
3. **Decoupled Deployment:**
   The Python Orchestrator can scale independently of the Kubernetes control plane, maintaining its own cache of carbon telemetry and model checkpoints without bloating the native `kube-scheduler` process.

This deliberate design choice ensures maximum velocity for the AI components while fully satisfying the Kubernetes orchestration requirements outlined in the System Design.
