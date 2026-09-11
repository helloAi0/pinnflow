# Scientific Methodology & Mathematical Foundations

## 1. Governing Equations
We consider the two-dimensional, unsteady, incompressible Navier-Stokes equations governing laminar viscous flow past a circular cylinder at $Re = 100.0$:

$$\nabla \cdot \mathbf{u} = \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0 \quad \text{(Continuity)}$$

$$\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y} + \frac{1}{\rho}\frac{\partial p}{\partial x} - \nu \left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}\right) = 0 \quad \text{($x$-Momentum)}$$

$$\frac{\partial v}{\partial t} + u \frac{\partial v}{\partial x} + v \frac{\partial v}{\partial y} + \frac{1}{\rho}\frac{\partial p}{\partial y} - \nu \left(\frac{\partial^2 v}{\partial x^2} + \frac{\partial^2 v}{\partial y^2}\right) = 0 \quad \text{($y$-Momentum)}$$

where $\mathbf{u} = (u(x,y,t), v(x,y,t))$ denotes the Eulerian velocity field, $p(x,y,t)$ is the kinematic pressure, and $\nu = 1/Re = 0.01$ is the kinematic viscosity.

---

## 2. Physics-Informed Neural Surrogate Formulation
A parameterized neural network $f_\theta: (x, y, t) \mapsto (\hat{u}, \hat{v}, \hat{p})$ is trained by minimizing the composite loss functional:

$$\mathcal{L}(\theta) = \mathcal{L}_{\text{data}}(\theta) + \mathcal{L}_{\text{pde}}(\theta) + \mathcal{L}_{\text{bc}}(\theta) + \mathcal{L}_{\text{gauge}}(\theta)$$

### A. Supervised Observation Loss
$$\mathcal{L}_{\text{data}}(\theta) = \frac{1}{N_{\text{data}}} \sum_{i=1}^{N_{\text{data}}} \left[ (\hat{u}_i - u_i^*)^2 + (\hat{v}_i - v_i^*)^2 \right]$$

### B. Autograd Navier-Stokes Residuals
At $N_{\text{coll}}$ spatio-temporal collocation points $\mathbf{x}_c \in \Omega \times [0, T]$, exact derivatives are evaluated via reverse-mode automatic differentiation:
$$\mathcal{L}_{\text{pde}}(\theta) = \frac{1}{N_{\text{coll}}} \sum_{j=1}^{N_{\text{coll}}} \left[ \|\mathcal{R}_c(\mathbf{x}_c^{(j)})\|^2 + \|\mathcal{R}_u(\mathbf{x}_c^{(j)})\|^2 + \|\mathcal{R}_v(\mathbf{x}_c^{(j)})\|^2 \right]$$

### C. Hard Cylinder Boundary Constraints
To eliminate the penalty weight hyperparameter on the obstacle wall, solid-wall no-slip boundary conditions are enforced directly in the forward architecture:
$$D(x, y) = 1.0 - \exp\left(-\text{relu}\left((x - x_c)^2 + (y - y_c)^2 - R^2\right)\right)$$
$$\hat{u}(x, y, t) = D(x, y) \cdot \tilde{u}(x, y, t), \quad \hat{v}(x, y, t) = D(x, y) \cdot \tilde{v}(x, y, t)$$
which guarantees $\hat{u} = 0, \hat{v} = 0$ on $\partial \Omega_{\text{cyl}}$ by construction.
