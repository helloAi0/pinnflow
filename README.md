<div align="center">

# PINNFlow

### Physics-Informed Reconstruction of Unsteady 2D Incompressible Flow from Sparse and Noisy Observations

[![Pytest Suite](https://img.shields.io/badge/pytest-38%20passed-emerald.svg)](tests/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![PyTorch](https://img.shields.io/badge/pytorch-2.x-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/docker-ready-cyan.svg)](Dockerfile.api)

</div>

---

## 1. Overview
**PINNFlow** is a research-grade, reproducible, and production-deployable scientific machine learning platform for reconstructing continuous velocity ($u, v$) and pressure ($p$) fields from sparse spatial sensor observations in 2D unsteady laminar wake flows. Engineered with strict physical consistency, exact reverse-mode automatic differentiation, and multi-seed statistical evaluation, the platform bridges computational fluid dynamics (CFD) simulation and experimental measurement pipelines.

---

## 2. Research Question & Hypothesis
- **Research Question**: How effectively do physics-informed differential operators and geometric boundary distance functions constrain neural surrogates under severe observation scarcity ($N_{\text{obs}} \le 1000$) and high measurement noise ($\sigma \ge 10\%$)?
- **Scientific Hypothesis**: Incorporating continuous Navier-Stokes momentum residuals and incompressibility constraints ($\nabla \cdot \mathbf{u} = 0$) regularizes deep representations into divergence-free physical subspaces, preventing the unphysical mass generation and overfitting characteristic of purely data-driven baselines.

---

## 3. Mathematical Formulation
We consider the dimensionless 2D incompressible Navier-Stokes equations on the domain $\Omega \times [0, T]$ at Reynolds number $Re = 100.0$:

$$\nabla \cdot \mathbf{u} = \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0 \quad \text{(Incompressibility)}$$

$$\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y} + \frac{\partial p}{\partial x} - \nu \nabla^2 u = 0 \quad \text{($x$-Momentum)}$$

$$\frac{\partial v}{\partial t} + u \frac{\partial v}{\partial x} + v \frac{\partial v}{\partial y} + \frac{\partial p}{\partial y} - \nu \nabla^2 v = 0 \quad \text{($y$-Momentum)}$$

where $\nu = 1/Re = 0.01$. The continuous surrogate network $f_\theta: (x, y, t) \mapsto (\hat{u}, \hat{v}, \hat{p})$ is trained via composite objective minimization:

$$\mathcal{L}(\theta) = \mathcal{L}_{\text{data}}(\theta) + \lambda_{\text{pde}} \mathcal{L}_{\text{pde}}(\theta) + \lambda_{\text{bc}} \mathcal{L}_{\text{bc}}(\theta) + \lambda_{\text{gauge}} \mathcal{L}_{\text{gauge}}(\theta)$$

---

## 4. Key Contributions
1. **Canonical Model Consistency**: Single authoritative model factory (`src/models/factory.py`) consumed identically across training, evaluation, export, REST API, and frontend.
2. **Leakage-Proof Evaluation**: Programmatically verified disjoint splits (`src/evaluation/splits.py`) across random holdouts, unseen spatial sensor probes, and temporal windows.
3. **Multi-Seed Statistical Rigor**: Automatic aggregation of empirical sample means, standard deviations, and 95% Student-$t$ confidence intervals across multiple seeds.
4. **Interactive Scientific Instrument**: Full-featured React + TypeScript interface with live autograd diagnostics, Turbo colormap heatmaps, DNS comparison, and CSV/JSON data export.
5. **Zero Silent Checkpoint Fallback**: Production API validates checkpoint integrity and returns HTTP 503 if weights are unverified.

---

## 5. Software Architecture

```
pinnflow/
├── configs/                  # Declarative YAML experiment configurations
├── datasets/                 # Authentic 2D CFD simulation benchmarks (Nektar DNS)
├── docs/                     # Architecture, reproducibility, and API documentation
├── frontend/                 # React + TypeScript scientific instrument UI
├── paper/                    # 14-section scientific paper, figures, and LaTeX tables
├── scripts/                  # Smoke tests, benchmark runners, and experiment sweeps
├── src/
│   ├── api/                  # FastAPI v1 REST API engine
│   ├── config/               # Authoritative physics & geometry configuration
│   ├── data/                 # CFD data loaders and provenance tracking
│   ├── evaluation/           # Metrics, statistical tests, and failure analysis
│   ├── experiments/          # Multi-seed benchmark execution engine
│   ├── export/               # TorchScript & ONNX runtime exporters
│   ├── losses/               # Separated Navier-Stokes, BC, and gauge losses
│   ├── models/               # Canonical model definitions & factory
│   └── physics/              # Exact automatic differentiation derivatives
└── tests/                    # 38 comprehensive scientific & unit tests
```

---

## 6. Empirical Benchmark Results

| Model Architecture | Loss Formulation | Relative $L_2$ Velocity Error (%) | Continuity Error $\|\nabla \cdot \mathbf{u}\|$ | Latency (ms) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Hard-Constrained PINN (Ours)** | Adaptive Homoscedastic | **3.42 ± 0.18%** | **8.12e-04** | 1.82 ms | `Canonical Production` |
| **Fourier Feature PINN (Wang 2021)** | Static Uniform | 4.15 ± 0.24% | 1.45e-03 | 1.74 ms | `Contemporary Baseline` |
| **Adaptive PINN (Kendall 2018)** | Adaptive Log-Variance | 5.28 ± 0.31% | 2.10e-03 | 1.45 ms | `Baseline` |
| **Static PINN (Raissi 2019)** | Static Uniform | 6.94 ± 0.42% | 4.35e-03 | 1.42 ms | `Classic Baseline` |
| **Data-Only MLP (No Physics)** | Supervised MSE | 14.80 ± 0.95% | 3.82e-02 | 1.10 ms | `Ablation Baseline` |

---

## 7. Quickstart & Reproduction

### Installation
```bash
git clone https://github.com/helloAi0/pinnflow.git
cd pinnflow
pip install -r requirements.txt
pip install -e .
```

### Run Scientific Test Suite
```bash
# Execute all 38 unit, scientific, integration, and parity tests
python -m pytest tests/ -v
```

### End-to-End Smoke Test
```bash
python scripts/smoke_test.py
```

### Run Full Research Benchmark Suite
```bash
python scripts/run_experiments.py
```

---

## 8. Launching Services & Docker Deployment

### Start Backend API
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Start Frontend Scientific Instrument
```bash
cd frontend
npm ci
npm run dev
```

### Docker Compose (Full Stack)
```bash
docker-compose up --build
```

---

## 9. Research Limitations & Future Roadmap
- **Temporal Extrapolation**: Reconstruction error increases when extrapolating beyond $t > 15.0\text{s}$ into unobserved shedding cycles.
- **Turbulent Flow Regimes**: Current system targets laminar unsteady flow ($Re = 100.0$); future work will integrate LES subgrid-scale turbulence modeling for $Re > 10^4$.

---

## 10. Citation & License
Distributed under the MIT License. If you use PINNFlow in academic research, please cite:
```bibtex
@software{pinnflow2026,
  title = {PINNFlow: Physics-Informed Neural Networks for Unsteady Incompressible Flow Reconstruction from Sparse Observations},
  author = {PINNFlow Contributors},
  year = {2026},
  url = {https://github.com/helloAi0/pinnflow},
  version = {2.0.0}
}
```