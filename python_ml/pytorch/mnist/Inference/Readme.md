Below is a **conference-ready `README.md`** that clearly explains **how to build, run, collect metrics, and log results** for the Python ONNX MNIST inference service.

It is written to be **unambiguous, reproducible, and reviewer-safe**.

---

```md
# MNIST ONNX Inference – Python (System-Level Evaluation)

This directory contains a **production-style Python inference service** for MNIST,
used for **system-level evaluation** (latency, resource usage, containerization)
as part of a Rust vs Python ML comparison.

The service performs **inference only** using a **pretrained ONNX model** and
collects runtime metrics suitable for reporting in a research paper.

---

## Directory Structure

```

python_infer/
├── Dockerfile
├── app.py
├── requirements.txt
├── model.onnx
└── README.md

````

---

## Prerequisites

- Docker ≥ 20.x
- Internet access (only for first MNIST download)
- Linux / macOS (recommended for consistent metrics)

No Python installation is required on the host.

---

## Build the Docker Image

From the `python_infer/` directory:

```bash
docker build -t mnist-python-infer .
````

After build completion, note the **image size**:

```bash
docker images mnist-python-infer
```

This value is reported as:

* **Container image size**

---

## Run the Inference Service

```bash
docker run -p 8000:8000 mnist-python-infer
```

The service starts a FastAPI server on port `8000`.

---

## Warm-Up (Recommended)

The first request includes ONNX Runtime initialization overhead.
To exclude cold-start effects from steady-state measurements, run:

```bash
curl http://localhost:8000/warmup
```

This performs a dummy inference and initializes all runtime state.

---

## Run Inference

Inference is performed on **real MNIST test samples**.

### Single inference request

```bash
curl "http://localhost:8000/infer?index=42"
```

### Example response

```json
{
  "prediction": 7,
  "ground_truth": 7,
  "latency_ms": 2.31,
  "rss_mb": 82.4,
  "cpu_percent": 12.0,
  "uptime_sec": 45.8
}
```

---

## Metrics Collected (Per Request)

The following metrics are returned **per inference call**:

| Metric         | Description                          |
| -------------- | ------------------------------------ |
| `latency_ms`   | End-to-end inference latency         |
| `rss_mb`       | Resident memory usage of the process |
| `cpu_percent`  | CPU usage at inference time          |
| `uptime_sec`   | Time since container start           |
| `prediction`   | Model output                         |
| `ground_truth` | MNIST label                          |

---

## Throughput Measurement

Throughput is measured **externally** using a load generator.

### Example using `wrk`

```bash
wrk -t4 -c16 -d30s http://localhost:8000/infer
```

Report:

* Requests/sec → **Throughput**
* Latency distribution → **P50 / P95 / P99 latency**

---

## Where to Log Results (Important)

### 1. Inference Metrics

* Logged **per request** via JSON response
* Capture using:

  * curl redirection
  * load-testing logs
  * experiment harness script

Example:

```bash
curl "http://localhost:8000/infer?index=1" >> inference_logs.json
```

---

### 2. Container Image Size

Collected externally:

```bash
docker images mnist-python-infer
```

Log as:

* `container_image_size_mb`

---

### 3. Model Artifact Size

```bash
ls -lh model.onnx
```

Log as:

* `onnx_model_size_bytes`

---

### 4. Cold-Start Latency

* Measure latency of **first `/infer` call**
* Compare against post-warmup latency
* Log separately as:

  * `cold_start_latency_ms`

---

## Reproducibility Notes

* Model is fixed (`model.onnx`)
* MNIST test set is deterministic
* Preprocessing exactly matches training:

  ```
  (x / 255 - 0.1307) / 0.3081
  ```
* Single worker, fixed threading configuration
* No background services inside container

---

## Intended Use in Paper

This setup is designed for evaluating:

* Inference latency
* Resource usage
* Container footprint
* Deployment complexity
