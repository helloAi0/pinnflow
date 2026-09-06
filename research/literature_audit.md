# Literature & Novelty Audit

## Baseline Literature
1. **Raissi et al. (2019):** *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations.*
   * **Contribution:** Established the baseline formulation of embedding Navier-Stokes residuals into the loss function.
   * **Limitation:** Proof-of-concept scripts; lacks production deployment architecture.
2. **Cai et al. (2021):** *Physics-Informed Neural Networks for Heat Transfer and Fluid Flows.*
   * **Contribution:** Extensive validation of PINNs across various fluid domains.
   * **Limitation:** Offline inference only; highly coupled, monolithic codebases.

## PINNFlow Novelty Matrix

| Feature | Traditional CFD (OpenFOAM) | Vanilla PINNs (Raissi et al.) | PINNFlow (Our Work) |
| :--- | :--- | :--- | :--- |
| **Resolution Method** | Mesh-based numerical integration | Mesh-free autograd optimization | Mesh-free autograd optimization |
| **Inference Speed** | Slow (hours/days) | Fast (milliseconds), offline | Fast (milliseconds), via REST API |
| **System Architecture** | Monolithic C++ binaries | Unstructured research scripts | Modular, containerized microservices |
| **Interactivity** | Post-processing via ParaView | Static matplotlib generation | Real-time Streamlit dashboard |
| **Reproducibility** | High (standardized solvers) | Low (seed dependent) | High (ONNX exports, CI/CD tested) |

## Core Thesis
While the underlying mathematical formulation aligns with Raissi et al., PINNFlow's primary novelty lies in bridging the gap between theoretical physics-informed machine learning and deployable scientific software. By decoupling the inference engine into a scalable FastAPI backend and an interactive Streamlit UI, PINNFlow demonstrates how surrogate neural solvers can be operationalized for real-time engineering diagnostics.