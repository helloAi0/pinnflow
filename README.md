<div align="center"> <img src="assets/pinnflow-banner.gif" alt="PinnFlow" width="100%"> </div> <div align="center">

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?logo=pytorch)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-served-009688.svg?logo=fastapi)](src/api/main.py)
[![React](https://img.shields.io/badge/React-Vite%20%2B%20Tailwind%20v4-61DAFB.svg?logo=react)](frontend)
[![Build Status](https://img.shields.io/github/actions/workflow/status/helloAi0/pinnflow/ci.yml?branch=main)](../../actions)

**Physics-informed reconstruction of unsteady incompressible flow from sparse and noisy observations.**

[Research Question](#research-question) • [Problem](#problem) • [Architecture](#system-architecture) • [Quickstart](#local-development) • [API](#api) • [Claims Registry](#scientific-validation) • [Limitations](#limitations)

</div>

---

PinnFlow is a reproducible scientific-computing framework for investigating Physics-Informed Neural Networks (PINNs) for spatiotemporal reconstruction of 2D incompressible Navier–Stokes flows.

The project combines a physics-based neural solver with a controlled experimental framework for studying the effects of observation sparsity, measurement noise, loss balancing, and sampling strategy.

> **Research status:** research prototype / experimental platform. Benchmark results are generated from the reproducible experiment pipeline and are not hard-coded into this repository — the [Results](#scientific-metrics) table ships empty until real multi-seed runs populate it. See [`CLAIMS.md`](CLAIMS.md) for exactly which claims below are `IMPLEMENTED`, `TESTED`, `EXPERIMENTALLY VERIFIED`, or `NOT YET DEMONSTRATED`, and [Test status](#testing) for the current, honest state of the automated suite.

---

## Research Question

The central research question is:

> **Under sparse and noisy observations of an unsteady cylinder wake, which combination of physics constraints, adaptive loss balancing, and sampling strategy provides the best accuracy–data-efficiency–robustness trade-off?**

PinnFlow evaluates this question using controlled observation budgets, reproducible random seeds, explicit evaluation splits, and multiple scientific error metrics.

---

## Problem

For an unsteady incompressible flow,

$$
u_t + uu_x + vu_y + p_x - \nu(u_{xx}+u_{yy}) = 0
$$

$$
v_t + uv_x + vv_y + p_y - \nu(v_{xx}+v_{yy}) = 0
$$

$$
u_x + v_y = 0.
$$

PinnFlow represents the velocity and pressure fields with a neural function

$$
(u,v,p)=f_\theta(x,y,t)
$$

and uses automatic differentiation to evaluate the governing-equation residuals directly during training.

The primary benchmark is the two-dimensional cylinder wake at Reynolds number $Re=100$ — the same regime the banner above animates: alternating vortex shedding behind a bluff body, the flow structure this whole framework exists to reconstruct from a handful of sparse sensors.

---

## Contributions

PinnFlow is designed as a **reproducible research and deployment framework**, rather than a single training script.

### Scientific contributions

* Controlled sparse-observation experiments.
* Explicit observation-noise injection.
* Comparison of data-only and physics-informed models.
* Adaptive loss-balancing experiments.
* Spatial, temporal, and physics-consistency evaluation.
* Failure-mode and error-field analysis.
* Reproducible multi-seed benchmarking.

### Engineering contributions

* Modular PyTorch implementation.
* Automatic-differentiation physics engine.
* FastAPI inference service, versioned (`/api/v1/...`) with unversioned aliases kept for backward compatibility.
* React + Tailwind scientific visualization console.
* Docker-based deployment (API, React console, and a legacy Streamlit comparison dashboard as separate services — see `render.yaml`).
* ONNX/TorchScript export.
* Automated unit, scientific, integration, and API tests.

---

## System Architecture

```text
                    Reference DNS
                         │
                 Observation Builder
                         │
              ┌──────────┴──────────┐
              │                     │
         Sparse/noisy            Clean
         observations           evaluation
              │                     │
              └──────────┬──────────┘
                         │
                    PINN Training
                         │
              ┌──────────┴──────────┐
              │                     │
          Data loss            Physics losses
                                │
                     continuity / momentum
                                │
                         Boundary constraints
                                │
                         Pressure gauge
              └──────────┬──────────┘
                         │
                     Checkpoint
                         │
             ┌───────────┴───────────┐
             │                       │
        Scientific Evaluation      Export
             │                       │
       Metrics / statistics       ONNX / Torch
             │                       │
             └──────────┬────────────┘
                        │
                  FastAPI Service
                        │
                  React Console
```

---

## Repository Structure

```text
src/
├── api/             FastAPI service (versioned + legacy routes)
├── config/          Central scientific configuration (single source of truth: Re, domain bounds, cylinder geometry)
├── data/            Dataset ingestion and observation generation
├── evaluation/      Metrics, statistics, and failure analysis
├── experiments/     Benchmark and training orchestration
├── export/          ONNX / TorchScript model export
├── geometry/        Domain and cylinder geometry
├── inference/       Canonical inference engine
├── losses/          Data, physics, and boundary losses
├── models/          Neural architectures + model factory
├── physics/         Navier–Stokes residual engine
├── sampling/        Collocation and adaptive sampling
├── training/        Training loops
└── visualization/   Scientific plotting

frontend/            React + Vite + Tailwind v4 scientific console

configs/
├── baseline_mlp.yaml
├── static_pinn.yaml
├── adaptive_pinn.yaml
├── adaptive_sampling.yaml
├── contemporary_baseline.yaml
├── benchmark.yaml
└── smoke.yaml

results/            Generated experiment artifacts (not committed — see Reproducibility)

tests/
├── unit/
├── integration/
├── scientific/
├── api/
└── reproducibility/

paper/
├── manuscript.md
└── figures/, tables/    (populate from results/ before submission — see Research Outputs)

CLAIMS.md            Scientific claims registry — every claim classified by evidence level
CITATION.cff
```

---

## Experimental Design

PinnFlow evaluates the effect of observation availability and noise.

### Observation budgets

```text
500
1,000
2,500
5,000
10,000
```

### Noise levels

```text
0%
1%
5%
10%
20%
```

### Random seeds

Main experiments use multiple independent seeds to quantify training variability — see `configs/benchmark.yaml` and `scripts/benchmark_multi_seed.py`.

### Evaluation splits

PinnFlow distinguishes:

1. Random point holdout.
2. Unseen sensor-location holdout.
3. Unseen temporal-window holdout.

The latter two are emphasized when measuring true reconstruction generalization, rather than interpolation within an already-seen sensor layout.

---

## Models

### Data-only baseline
A standard neural surrogate trained only from observations.

### Static PINN
A PINN with fixed weighting between data and physics objectives.

### Adaptive PINN
A PINN with learnable (homoscedastic-uncertainty) loss weighting.

### Fourier-constrained PINN
Fourier-feature input embedding to mitigate spectral bias against the wake's shedding frequency, plus a hard boundary-distance constraint enforcing exact no-slip on the cylinder wall.

### Additional research baselines
The benchmark harness (`configs/contemporary_baseline.yaml`) can include contemporary gradient-based, NTK-inspired, or adaptive-sampling methods to determine whether improvements are specific to one weighting strategy or general.

---

## Scientific Metrics

The evaluation suite reports:

* Relative $L_2$ error.
* $L_\infty$ error.
* Velocity error.
* Pressure error.
* Vorticity error.
* Divergence / mass-balance error.
* Individual PDE residuals.
* Error versus time.
* Error versus downstream distance.
* Training time.
* Inference latency.
* Parameter count.
* Memory usage.

All aggregate benchmark results are generated from saved experiment artifacts rather than manually entered values — see `src/evaluation/`.

---

## Reproducibility

Each experiment stores:

```text
Experiment ID
Git commit
Model configuration
Physics configuration
Dataset version
Observation budget
Noise level
Random seed
Python version
PyTorch version
Hardware
Training configuration
Checkpoint identifier
```

To reproduce an experiment:

```bash
python -m src.experiments.benchmark \
    --config configs/benchmark.yaml \
    --seed 0
```

Generated outputs are written under `results/raw/`, `results/processed/`, and `results/figures/` — this directory is intentionally not committed (it's regenerated output, not source), so a fresh clone won't contain it until you run an experiment.

---

## Scientific Validation

The physics engine uses automatic differentiation for first- and second-order derivatives. Analytical derivative tests validate the differentiation engine against closed-form functions (Taylor–Green vortex).

Scientific tests (`tests/scientific/`) verify PDE residual construction, boundary constraints, pressure gauge behavior, divergence behavior, sampling geometry, reproducibility, and evaluation metrics.

See [`CLAIMS.md`](CLAIMS.md) for the full registry — it's the most honest single document in this repo, and worth reading before citing anything from it in a paper or application. It explicitly marks what's `STATISTICALLY SUPPORTED` (multi-seed, $p<0.05$) versus merely `IMPLEMENTED`, and documents at least one known limitation plainly (long-horizon temporal extrapolation beyond the training window is `NOT YET DEMONSTRATED`).

---

## API

The FastAPI service exposes model inference and scientific diagnostics, with versioned routes (`/api/v1/...`) and unversioned aliases kept for backward compatibility.

### Health

```http
GET /api/v1/health
```

Returns model readiness, checkpoint metadata, configuration, and dataset availability — the console's status indicator reads this directly rather than assuming the model loaded.

### Point prediction

```http
POST /api/v1/predict/point
```

```json
{ "x": 2.0, "y": 0.3, "t": 10.0, "compute_vorticity": true }
```

### Field prediction

```http
POST /api/v1/predict/field
```

```json
{ "x_min": -1, "x_max": 8, "y_min": -2, "y_max": 2, "t": 10, "nx": 100, "ny": 100, "compute_vorticity": true }
```

### Reference field (real DNS data, for honest comparison)

```http
POST /api/v1/reference/field
```

Same shape as above. Interpolates the actual measurement points onto the requested grid; returns `503` (not a fabricated substitute) if the reference dataset hasn't been fetched on the server.

The API never performs inference using an unverified or uninitialized model without saying so — `weights_loaded` and `checkpoint` in `/health` reflect what's actually running.

---

## Web Console

The React application (`frontend/`) provides a scientific evaluation console with field visualization, reference comparison, absolute-error maps, PDE residual maps, vorticity, benchmark metrics, inference latency, model/checkpoint metadata, and experiment reproducibility information.

---

## Local Development

### Python

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Dataset preparation

```bash
python -m src.data.reference_pipeline
```

### Training

```bash
python -m src.experiments.train_final
```

### Benchmarking

```bash
python -m src.experiments.benchmark \
    --config configs/benchmark.yaml \
    --seed 0
```

### API

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### Frontend

```bash
cd frontend
npm ci
npm run dev
```

---

## Testing

```bash
pytest -q
```

**Current status on a clean clone: 35 of 38 tests pass.** The three failures are known and specific, not mysterious flakiness:

| Test | Cause | Fix needed |
|---|---|---|
| `tests/integration/test_export_parity.py` | `onnxscript` isn't in `requirements.txt` | Add it as a dependency |
| `tests/integration/test_training_pipeline.py` | Needs the real CFD dataset, no fallback/skip when absent | Add `@pytest.mark.skipif` or fetch the dataset in CI before running tests |
| `tests/test_pinnflow.py::test_data_pipeline_dimensions` | Same — needs the dataset, no graceful skip | Same fix |

None of these are correctness bugs in the physics or API — they're missing-dependency and missing-fixture gaps. Worth fixing before claiming a fully green suite in a paper or application, but not worth hiding.

Frontend:

```bash
cd frontend
npm ci
npm run lint
npm run build
```

CI (`.github/workflows/ci.yml`) runs the Python test suite, builds both Docker images, and builds + lints the frontend on every push — not just container builds, so a regression like the ones above shows up immediately rather than sitting on `main` unnoticed.

---

## Deployment

The production architecture is:

```text
React  →  FastAPI  →  Canonical model
```

`render.yaml` deploys this as a Blueprint: the API (Docker), the React console (static site build), and the legacy Streamlit comparison dashboard (Docker), with the console's API URL resolved automatically from the API service's own deployed URL rather than hardcoded.

---

## Limitations

The primary benchmark is a 2D cylinder wake at $Re=100$.

The results should therefore not be interpreted as evidence that one PINN strategy is universally superior for every PDE or flow regime.

Future work includes:

* multiple Reynolds numbers,
* higher-dimensional geometries,
* uncertainty quantification,
* adaptive residual sampling,
* experimental sensor placement,
* real CFD/experimental datasets,
* larger-scale GPU benchmarking,
* consolidating the dataset ingestion path (`src/data/` vs. the API's direct `.mat` reader) into one pipeline.

---

## Research Outputs

A paper-ready release should contain:

```text
paper/manuscript.md
paper/figures/
paper/tables/
results/
CITATION.cff
```

Every figure and table must be regenerable from repository scripts — none should be hand-edited after generation. `CLAIMS.md` is the mechanism that keeps the manuscript honest: every empirical sentence in `paper/manuscript.md` should trace to a row in that registry.

---

## Citation

```bibtex
@software{pinnflow,
  author  = {Taha},
  title   = {PinnFlow: Physics-Informed Reconstruction of Unsteady Flow from Sparse and Noisy Observations},
  year    = {2026},
  url     = {https://github.com/helloAi0/pinnflow}
}
```

Also see [`CITATION.cff`](CITATION.cff) for machine-readable citation metadata.

---

## License

MIT