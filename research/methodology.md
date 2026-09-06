# Methodology: Physics-Informed Neural Network (PINN)

## 1. Governing Equations
The fluid domain is governed by the incompressible Navier-Stokes equations. The physical residuals are defined as:

$$f_u = u_t + u u_x + v u_y + p_x - \nu (u_{xx} + u_{yy})$$
$$f_v = v_t + u v_x + v v_y + p_y - \nu (v_{xx} + v_{yy})$$
$$f_c = u_x + v_y$$

Where $u$ and $v$ are the stream-wise and transverse velocity components, $p$ is the pressure field, and $\nu$ is the kinematic viscosity. 

## 2. Loss Function Formulation
The network is optimized using a composite loss function that balances data fidelity (boundary/initial conditions) with physical adherence:

$$L_{total} = w_{data}L_{data} + w_{PDE}L_{PDE}$$

Where the physics-informed constraint is the Mean Squared Error (MSE) of the residuals:

$$L_{PDE} = \frac{1}{N_{c}} \sum_{i=1}^{N_{c}} \left( |f_u(x_i, y_i, t_i)|^2 + |f_v(x_i, y_i, t_i)|^2 + |f_c(x_i, y_i, t_i)|^2 \right)$$

## 3. Network Architecture & Hyperparameters
* **Input Layer:** 3 spatial-temporal coordinates $(x, y, t)$
* **Hidden Layers:** 8 layers, 20 neurons per layer
* **Activation Function:** Tanh
* **Output Layer:** 3 predicted fields $(u, v, p)$
* **Optimizer:** Adam (Learning Rate = $1e^{-3}$, decaying) followed by L-BFGS
* **Collocation Points ($N_c$):** 10,000 internal domain points