<div align="center">

<a href="https://github.com/helloAi0/pinnflow">
  <img src="assets/pinnflow-banner.gif" alt="PINNFlow — Physics-Informed Flow Reconstruction" width="100%" />
</a>

PINNFlow

Physics-Informed Reconstruction of Unsteady 2D Incompressible Flow from Sparse and Noisy Observations

<p>
  <a href="https://github.com/helloAi0/pinnflow/actions"><img src="https://img.shields.io/github/actions/workflow/status/helloAi0/pinnflow/ci.yml?branch=main&label=CI&logo=github" alt="CI"/></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python"/></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch"/></a>
  <a href="https://fastapi.tiangolo.com/"><img src="https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white" alt="FastAPI"/></a>
  <a href="https://react.dev/"><img src="https://img.shields.io/badge/React-Scientific%20Console-61DAFB?logo=react&logoColor=111827" alt="React"/></a>
  <a href="https://www.docker.com/"><img src="https://img.shields.io/badge/Docker-Reproducible-2496ED?logo=docker&logoColor=white" alt="Docker"/></a>
  <img src="https://img.shields.io/badge/Domain-Scientific%20ML-7C3AED" alt="Scientific ML"/>
  <img src="https://img.shields.io/badge/License-MIT-green" alt="MIT License"/>
</p>

PINNFlow is a research and deployment framework for studying how physics constraints, adaptive loss balancing, and observation design affect neural reconstruction of unsteady incompressible flow.

</div>

Research Question

Under sparse and noisy observations of an unsteady cylinder wake, which combination of physics constraints, adaptive loss balancing, and sampling strategy provides the best accuracy–data-efficiency–robustness trade-off?

PINNFlow investigates this question through controlled observation budgets, reproducible random seeds, explicit evaluation splits, multi-model baselines, and scientific error metrics.

Problem

For a dimensionless 2D incompressible flow,

$$
u_t + uu_x + vu_y + p_x - \nu(u_{xx}+u_{yy}) = 0
$$

$$
v_t + uv_x + vv_y + p_y - \nu(v_{xx}+v_{yy}) = 0
$$

$$
u_x + v_y = 0.
$$

PINNFlow represents the continuous flow field as

$$
(u,v,p)=f_\theta(x,y,t)
$$

and uses automatic differentiation to evaluate first- and second-order derivatives directly inside the physics-informed objective.

The primary benchmark is the two-dimensional cylinder wake at Re = 100.

Why PINNFlow?

PINNFlow is intentionally structured as a reproducible research platform, not a single training notebook.

Scientific layer

Sparse-observation reconstruction

Explicit observation-noise injection

Navier–Stokes residuals through automatic differentiation

Boundary and pressure constraints

Controlled data-efficiency studies

Sensor-location and temporal holdout evaluation

Multi-seed statistical analysis

Failure-mode and error-field analysis

Engineering layer

Canonical PyTorch model factory

Modular training and evaluation pipeline

FastAPI inference service

React + TypeScript scientific console

TorchScript / ONNX export

Dockerized services

Automated unit, scientific, API, integration, and build checks

Reproducible experiment artifacts

System Architecture

                         ┌──────────────────────┐
                         │   Reference DNS/CFD  │
                         └──────────┬───────────┘
                                    │
                           Observation Builder
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
             Sparse / noisy                    Clean reference
             observations                       evaluation
                    │                               │
                    └───────────────┬───────────────┘
                                    │
                             ┌──────▼──────┐
                             │ PINN Engine │
                             └──────┬──────┘
                                    │
               ┌────────────────────┼────────────────────┐
               │                    │                    │
          Data objective      PDE objectives       BC / Gauge
               │                    │                    │
               └────────────────────┼────────────────────┘
                                    │
                              Verified Checkpoint
                                    │
                 ┌──────────────────┴──────────────────┐
                 │                                     │
        Scientific Evaluation                      Model Export
                 │                                     │
      ┌──────────┼───────────┐                  ONNX / TorchScript
      │          │           │                        │
   Metrics    Statistics   Failure Analysis            │
      │          │           │                        │
      └──────────┴───────────┘                        │
                 │                                     │
                 └──────────────────┬──────────────────┘
                                    │
                              FastAPI /api/v1
                                    │
                              React Console
                                    │
                        Research + Production UI

Repository Structure

pinnflow/
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── configs/                         # Reproducible experiment configurations
│   ├── baseline_mlp.yaml
│   ├── static_pinn.yaml
│   ├── adaptive_pinn.yaml
│   ├── adaptive_sampling.yaml
│   ├── contemporary_baseline.yaml
│   ├── benchmark.yaml
│   └── smoke.yaml
│
├── datasets/                        # Prepared/reference datasets (not ad-hoc copies)
│
├── docs/
│   ├── architecture.md
│   ├── api.md
│   ├── reproducibility.md
│   └── scientific-methodology.md
│
├── frontend/                        # React + TypeScript scientific console
│   ├── public/
│   └── src/
│       ├── components/
│       ├── services/
│       ├── types/
│       └── ...
│
├── paper/
│   ├── manuscript.md
│   ├── figures/
│   └── tables/
│
├── results/
│   ├── raw/
│   ├── processed/
│   └── figures/
│
├── scripts/                         # Reproducibility / benchmark utilities
│
├── src/
│   ├── api/                         # Versioned FastAPI service
│   ├── config/                      # Authoritative scientific configuration
│   ├── data/                        # Data loading / observation generation
│   ├── evaluation/                  # Metrics, statistics, failure analysis
│   ├── experiments/                 # Research experiment orchestration
│   ├── export/                      # ONNX / TorchScript export
│   ├── geometry/                    # Domain and cylinder geometry
│   ├── inference/                   # Canonical inference path
│   ├── losses/                      # Data / PDE / BC / gauge losses
│   ├── models/                      # Canonical model + factory
│   ├── physics/                     # Autograd derivatives / residuals
│   ├── sampling/                    # Collocation / adaptive sampling
│   ├── training/                    # Training loops
│   ├── utils/                       # Reproducibility / utilities
│   └── visualization/               # Scientific visualization
│
├── tests/
│   ├── unit/
│   ├── scientific/
│   ├── integration/
│   ├── api/
│   └── reproducibility/
│
├── CLAIMS.md
├── CITATION.cff
├── CHANGELOG.md
├── Dockerfile.api
├── Dockerfile.ui
├── Makefile
├── README.md
├── docker-compose.yml
├── pyproject.toml
└── requirements.txt

Experimental Design

PINNFlow evaluates the effect of observation availability and measurement noise.

Observation budgets

500
1,000
2,500
5,000
10,000

Noise levels

0%
1%
5%
10%
20%

Random seeds

Main experiments use multiple independent seeds to quantify training variability rather than relying on a single favorable run.

Evaluation splits

PINNFlow distinguishes:

Random point holdout

Unseen sensor-location holdout

Unseen temporal-window holdout

The latter two are emphasized when evaluating true reconstruction generalization.

Models

Data-only baseline

A neural surrogate trained only on observations.

Static PINN

A physics-informed model using fixed weights between data and physics objectives.

Adaptive PINN

A physics-informed model with learnable loss weighting.

Additional research baselines

The benchmark framework can incorporate contemporary:

gradient-based weighting,

NTK-inspired weighting,

residual-aware sampling,

adaptive sampling strategies.

These are treated as experimental baselines rather than assumed improvements.

Scientific Metrics

The evaluation suite reports:

Relative $L_2$ velocity error

Relative $L_2$ component errors

$L_\infty$ error

Pressure error

Vorticity error

Divergence / mass-balance error

Individual PDE residuals

Error versus time

Error versus downstream distance

Training wall time

Inference latency

Parameter count

Memory usage where available

Aggregate benchmark results are generated from saved experiment artifacts rather than manually entered values.

Reproducibility

Each experiment records:

Experiment ID
Git commit
Model configuration
Physics configuration
Dataset identifier/version
Observation budget
Noise level
Evaluation split
Random seed
Python version
PyTorch version
Hardware
Training configuration
Checkpoint identifier

Reproduce an experiment

python -m src.experiments.benchmark \
    --config configs/benchmark.yaml \
    --seed 0

Generated outputs are stored under:

results/raw/
results/processed/
results/figures/

Scientific Validation

The physics engine uses automatic differentiation for first- and second-order derivatives.

Scientific tests verify:

analytical derivative accuracy,

PDE residual construction,

boundary constraints,

pressure-gauge behavior,

divergence behavior,

sampling geometry,

evaluation metrics,

reproducibility,

API behavior.

No scientific result is considered established merely because the corresponding code exists; it must be supported by executed experiments.

API

The production inference service is exposed through a versioned FastAPI API.

Health

GET /api/v1/health

Returns model readiness, checkpoint metadata, configuration, and dataset availability.

Point prediction

POST /api/v1/predict/point

Example:

{
  "x": 2.0,
  "y": 0.3,
  "t": 10.0,
  "compute_vorticity": true
}

Field prediction

POST /api/v1/predict/field

Example:

{
  "x_min": -1,
  "x_max": 8,
  "y_min": -2,
  "y_max": 2,
  "t": 10,
  "nx": 100,
  "ny": 100,
  "compute_vorticity": true
}

The production API must never perform inference using an unverified or uninitialized model.

Web Console

The React scientific console provides:

field visualization,

reference comparison,

absolute-error maps,

PDE residual maps,

vorticity,

benchmark metrics,

inference latency,

model/checkpoint metadata,

experiment reproducibility information,

exportable field and metric data.

Local Development

Environment

python -m venv .venv

Linux/macOS:

source .venv/bin/activate

Windows:

.venv\Scripts\activate

Install:

pip install -r requirements.txt
pip install -e .

Dataset preparation

python -m src.data.reference_pipeline

Training

python -m src.experiments.train_final

Benchmarking

python -m src.experiments.benchmark \
    --config configs/benchmark.yaml \
    --seed 0

API

uvicorn src.api.main:app --host 0.0.0.0 --port 8000

Frontend

cd frontend
npm ci
npm run dev

Testing

Python

pytest -q

Frontend

cd frontend
npm ci
npm run lint
npm run build

CI is responsible for validating the real test suite, frontend build, API smoke tests, and container builds.

Deployment

Production architecture:

React / Vercel
       │
       │ HTTPS
       ▼
FastAPI / Render
       │
       ▼
Canonical inference engine
       │
       ▼
Verified checkpoint

Docker images are built from reproducible runtime artifacts and do not rely on host bind mounts for production execution.

Research Outputs

A paper-ready release should contain:

paper/manuscript.md
paper/figures/
paper/tables/
results/
CITATION.cff

Every paper figure and table should be regenerable from repository scripts and saved experiment artifacts.

Limitations

The primary benchmark is a two-dimensional cylinder wake at Re = 100.

Therefore, results should not be interpreted as evidence that a particular PINN strategy is universally superior across all PDEs, Reynolds-number regimes, geometries, or observation models.

Current research directions include:

multiple Reynolds numbers,

higher-dimensional geometries,

uncertainty quantification,

adaptive residual sampling,

experimental sensor placement,

real CFD/experimental datasets,

larger-scale GPU benchmarking.

Research Status

PINNFlow distinguishes between:

IMPLEMENTED
VALIDATED
EXPERIMENTALLY VERIFIED
STATISTICALLY SUPPORTED
NOT YET DEMONSTRATED

Scientific claims are documented separately in CLAIMS.md.

Results must not be presented as established findings until they are supported by reproducible experiment artifacts.

Citation

@software{pinnflow2026,
  title   = {PINNFlow: Physics-Informed Reconstruction of Unsteady Flow from Sparse and Noisy Observations},
  author  = {Taha},
  year    = {2026},
  url     = {https://github.com/helloAi0/pinnflow}
}

License

MIT License.

See LICENSE.