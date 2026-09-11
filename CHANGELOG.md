# Changelog

All notable changes to the PINNFlow research platform are documented in this file.

## [2.0.0] - 2026-09-11
### Scientific & Architecture Transformation
- **Canonical Model Architecture**: Consolidated all model variants (`standard_mlp`, `fourier_pinn`, `hard_constrained_pinn`, `adaptive_activation_pinn`) into `src/models/factory.py`.
- **Authoritative Physics Configuration**: Unified all physical parameters ($Re=100.0, \nu=0.01, \rho=1.0$) and geometric coordinates in `src/config/physics_config.py`.
- **Modular Physics Loss Hierarchy**: Separated losses into dedicated, mathematically documented modules (`src/losses/data_loss.py`, `pde_loss.py`, `boundary_condition_loss.py`, `pressure_gauge_loss.py`, `weighting.py`).
- **Analytical Physics Validation**: Created rigorous Taylor-Green vortex analytical validation tests with sub-$10^{-5}$ tolerance assertions.
- **Leakage-Resistant Data Splits**: Implemented disjoint random point splits, unseen spatial sensor location splits, and temporal forecasting splits with programmatic overlap verification.
- **Multi-Seed Statistical Aggregation**: Added empirical sample mean, standard deviation, and 95% Student-$t$ confidence interval calculation (`src/evaluation/statistics.py`).
- **Production REST API v1**: Built structured FastAPI v1 endpoints with request tracking, strict Pydantic validation, and mandatory checkpoint verification.
- **Scientific Instrument UI**: Rebuilt React frontend with interactive 5-tab workspace, multi-colormap heatmaps, live autograd diagnostics, benchmark comparisons, and data export tools (CSV/JSON).
- **Deployment & Export**: Added standalone Dockerfiles and TorchScript/ONNX export with numerical parity verification ($< 10^{-5}$).
- **Research Artifacts**: Populated 14-section paper manuscript, empirical benchmark tables, and `CLAIMS.md`.
