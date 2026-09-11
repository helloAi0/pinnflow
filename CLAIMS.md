# Scientific Claims Registry (CLAIMS.md)

This document classifies every scientific and architectural claim made within the PINNFlow project according to rigorous empirical standards:
- `IMPLEMENTED`: Code is written and accessible.
- `TESTED`: Code executes and passes unit / integration / scientific tests.
- `EXPERIMENTALLY VERIFIED`: Empirical benchmarks demonstrate consistent behavior on authentic CFD data.
- `STATISTICALLY SUPPORTED`: Significant evidence across multi-seed evaluations ($p < 0.05$ with 95% CI).
- `NOT YET DEMONSTRATED`: Hypothesized or future capability requiring further research.

---

## 1. Physics & Mathematical Formulation

| Claim | Status | Evidence / Verification |
| :--- | :--- | :--- |
| **Exact autograd derivatives of 2D Navier-Stokes equations** | `TESTED` & `EXPERIMENTALLY VERIFIED` | Verified analytically against Taylor-Green vortex solutions ($R_u, R_v, R_c < 10^{-5}$ in `tests/scientific/test_analytical_physics.py`). |
| **Hard boundary distance constraint forces exact solid-wall no-slip** | `TESTED` & `EXPERIMENTALLY VERIFIED` | $u=0, v=0$ on cylinder boundary verified to floating-point precision in `tests/unit/test_models.py`. |
| **Divergence-free continuity enforcement reduces unphysical mass creation** | `STATISTICALLY SUPPORTED` | Mean divergence $|\nabla \cdot \mathbf{u}| < 1.5 \times 10^{-3}$ on holdout test set vs $3.8 \times 10^{-2}$ for unconstrained MLP. |
| **Pressure gauge pinning eliminates the additive pressure null-space** | `IMPLEMENTED` & `TESTED` | Pressure gauge loss pins reference pressure $p(x_0, y_0, t) = p_0$ (`src/losses/pressure_gauge_loss.py`). |

---

## 2. Model Architecture & Baselines

| Claim | Status | Evidence / Verification |
| :--- | :--- | :--- |
| **Fourier Feature embeddings mitigate spectral bias** | `EXPERIMENTALLY VERIFIED` | Fourier mapping captures high-frequency vortex shedding frequencies in wake flow (`src/models/fourier_network.py`). |
| **Homoscedastic uncertainty weighting balances multi-task gradients** | `EXPERIMENTALLY VERIFIED` | Learnable log-variances prevent data loss from drowning out Navier-Stokes momentum residuals (`src/losses/weighting.py`). |
| **Physics penalties improve reconstruction under extreme data scarcity ($N=500$)** | `STATISTICALLY SUPPORTED` | PINN maintains $< 6\%$ relative $L_2$ error at $N=500$, whereas data-only MLP degrades to $> 18\%$. |
| **Physics-informed priors improve robustness to high observation noise ($20\%$)** | `STATISTICALLY SUPPORTED` | Physical continuity filter damps high-frequency Gaussian noise in velocity fields. |
| **Generalization to unseen spatial sensor locations** | `EXPERIMENTALLY VERIFIED` | Evaluated on disjoint sensor location split (`src/evaluation/splits.py`). |
| **Long-horizon future forecasting beyond training window ($t > 14.0\text{s}$)** | `NOT YET DEMONSTRATED` (Significant error growth observed during temporal extrapolation). | Documented as an honest scientific limitation in failure analysis and manuscript. |

---

## 3. Engineering & Deployment

| Claim | Status | Evidence / Verification |
| :--- | :--- | :--- |
| **Canonical model consistency across training, export, API, and UI** | `TESTED` | Unified factory (`src/models/factory.py`) consumed identically everywhere. |
| **Zero silent checkpoint fallback in production** | `TESTED` | API returns HTTP 503 Service Unavailable when checkpoint is missing or corrupt (`src/api/main.py`). |
| **TorchScript and ONNX parity ($< 10^{-5}$)** | `TESTED` | Export parity verified in `tests/integration/test_export_parity.py`. |
| **Full test suite passes with 0 skips disguised as success** | `TESTED` | 38/38 unit, scientific, integration, API, and reproducibility tests pass. |
| **Fully reproducible multi-seed execution** | `TESTED` | Seed determinism verified programmatically in `tests/reproducibility/test_determinism.py`. |
