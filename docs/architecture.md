# PINNFlow Software Architecture & System Design

## 1. Architectural Philosophy
PINNFlow is engineered as a **unified, reproducible, and verifiable scientific machine learning platform** for fluid mechanics. Unlike typical prototype repositories with fragmented models across training scripts, API servers, and exports, PINNFlow strictly enforces a **single canonical model source of truth**.

```
                           +-------------------------------------+
                           |      Authoritative Physics &        |
                           |       Domain Configuration          |
                           |   (src/config/physics_config.py)    |
                           +------------------+------------------+
                                              |
                     +------------------------+------------------------+
                     |                                                 |
                     v                                                 v
        +-------------------------+                       +-------------------------+
        |  Canonical Model Factory|                       |  Continuous Physics &   |
        |  (src/models/factory.py)|                       |    Autograd Loss Engine |
        +------------+------------+                       |      (src/losses/)      |
                     |                                    +------------+------------+
                     |                                                 |
                     +------------------------+------------------------+
                                              |
                                              v
                              +-------------------------------+
                              |    Scientific Experiment &    |
                              |       Benchmark Runner        |
                              | (src/experiments/benchmark.py)|
                              +---------------+---------------+
                                              |
                     +------------------------+------------------------+
                     |                                                 |
                     v                                                 v
        +-------------------------+                       +-------------------------+
        |   Verified Checkpoint   |                       |  Statistical Evaluation |
        |  with Complete Metadata |                       |   & Figure Generation   |
        |  (src/models/checkpoint)|                       |  (src/evaluation/)      |
        +------------+------------+                       +-------------------------+
                     |
         +-----------+-----------+
         |                       |
         v                       v
+-----------------+     +-----------------+
|   FastAPI v1    |     |  TorchScript /  |
|   REST Server   |     |   ONNX Export   |
| (src/api/main)  |     | (src/export/)   |
+--------+--------+     +-----------------+
         |
         v
+-----------------+
| React Scientific|
| Instrument UI   |
|   (frontend/)   |
+-----------------+
```

---

## 2. Core Subsystems

### A. Physics & Configuration Layer (`src/config/physics_config.py`)
- Centralizes non-dimensional Navier-Stokes fluid properties:
  - Reynolds number $Re = 100.0$
  - Kinematic viscosity $\nu = 1/Re = 0.01$
  - Density $\rho = 1.0$
- Defines the canonical spatial-temporal observation window $[1.0, 8.0] \times [-2.0, 2.0] \times [0.0, 20.0]$ and cylinder obstacle geometry $(x_c=0, y_c=0, r=0.5)$.

### B. Loss Engine (`src/losses/`)
- **Data Loss (`data_loss.py`)**: Supervised $L_2$ regression on velocity and pressure components.
- **PDE Loss (`pde_loss.py`)**: Automatic differentiation residuals of $x$-momentum, $y$-momentum, and incompressibility continuity ($\nabla \cdot \mathbf{u} = 0$).
- **Boundary Loss (`boundary_condition_loss.py`)**: Penalties for cylinder solid-wall no-slip ($u=0, v=0$) and domain far-field conditions.
- **Pressure Gauge Loss (`pressure_gauge_loss.py`)**: Anchors reference pressure at $(x_0, y_0, t)$ to eliminate the arbitrary pressure gradient constant.
- **Weighting Strategies (`weighting.py`)**: Supports Static weighting, Multi-Task Homoscedastic uncertainty weighting (Kendall et al.), and Dynamic Gradient Pathology balancing (Wang et al., 2021).

### C. Canonical Model Factory (`src/models/factory.py`)
- Exposes `create_model(config)` supporting:
  - `standard_mlp`: Smooth $C^\infty$ Tanh/Sine coordinate network.
  - `fourier_pinn`: Random Fourier Feature mapping embedding high-frequency modes.
  - `hard_constrained_pinn`: Fourier features with exact distance masking $D(x,y) = 1 - \exp(-\text{relu}(r^2 - R^2))$ forcing $u=0, v=0$ on solid walls.
  - `adaptive_activation_pinn`: Layer-wise locally adaptive activation slopes (L-LAAF).

### D. Checkpoint Integrity Engine (`src/models/checkpoint.py`)
- Stores complete provenance bundles: weights, architecture hyperparameters, Re, training seed, Git commit SHA, and training loss metrics.
- Production inference verifies file integrity, architecture match, and tensor shape alignment; returns HTTP 503 if verification fails.

### E. REST API & React Instrument (`src/api/` & `frontend/`)
- REST v1 endpoints (`/api/v1/health`, `/api/v1/predict/point`, `/api/v1/predict/field`, `/api/v1/reference/field`, `/api/v1/metadata`).
- Interactive React interface providing real-time field rendering, absolute error maps against DNS data, autograd PDE residual diagnostics, multi-seed benchmark tables, and one-click data export tools (CSV/JSON).
