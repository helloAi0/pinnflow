# Scientific Reproducibility Guide

PINNFlow is engineered for deterministic, bitwise-verifiable scientific reproduction.

---

## 1. Quick Reproduction Commands

### A. End-to-End Smoke Test
Executes data verification, fast model training, checkpoint validation, and FastAPI querying:
```bash
python scripts/smoke_test.py
```

### B. Run Pytest Verification Suite
Runs 38 automated scientific, unit, integration, API, and reproducibility tests:
```bash
python -m pytest tests/ -v
```

### C. Run Full Scientific Benchmark
Executes multi-seed benchmarks, data scarcity sweeps, noise sweeps, sensor probe holdouts, and temporal holdouts:
```bash
python scripts/run_experiments.py
```

---

## 2. Dataset Provenance & Integrity

| Property | Description |
| :--- | :--- |
| **Dataset Source** | 2D Incompressible Cylinder Wake Simulation (Karniadakis / Raissi et al., 2019) |
| **Simulation Code** | Nektar Spectral/hp Element Navier-Stokes Solver |
| **Reynolds Number** | $Re = 100.0$ |
| **Total Points** | 1,000,000 ($N=5,000$ spatial probes $\times$ $T=200$ temporal snapshots) |
| **Spatial Range** | $x \in [1.0, 8.0], y \in [-2.0, 2.0]$ |
| **Temporal Range** | $t \in [0.0, 19.9]\text{s}, \Delta t = 0.1\text{s}$ |
| **Dataset SHA-256** | `99d9e1f9da7d2dff6f9fc53a06045952f4ea6c4d7e7c99276d4352125bb7ea76` |

---

## 3. Leakage-Resistant Evaluation Protocols

PINNFlow implements 3 distinct evaluation protocols (`src/evaluation/splits.py`):
1. **Random Spatio-Temporal Holdout (`random_point_split`)**: Uniform sampling across space-time.
2. **Unseen Sensor Probe Holdout (`sensor_location_split`)**: Completely holds out $20\%$ of spatial probes across all time steps to evaluate spatial interpolation.
3. **Temporal Window Forecasting (`temporal_holdout_split`)**: Trains on $t \le 14.0\text{s}$ and evaluates on future window $t > 14.0\text{s}$ to test temporal extrapolation.

Programmatic assertions (`verify_disjointness`) guarantee zero coordinate overlap ($\text{train} \cap \text{test} = \emptyset$).

---

## 4. Multi-Seed Statistical Reporting

To prevent reporting biased cherry-picked seeds, all paper benchmarks compute:
- Empirical Sample Mean ($\bar{x}$)
- Sample Standard Deviation ($s$)
- Standard Error of the Mean ($\text{SEM}$)
- Two-Sided $95\%$ Student-$t$ Confidence Intervals ($\bar{x} \pm t_{0.025, n-1} \cdot \text{SEM}$)
- Paired Student's $t$-test and Wilcoxon signed-rank test for method comparisons.
