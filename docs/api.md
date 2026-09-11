# PINNFlow REST API Documentation (v1)

Base URL: `http://localhost:8000` (or configured deployment URL)

All endpoints return JSON responses. Errors follow standard HTTP status codes (400, 422, 500, 503) with descriptive error details.

---

## 1. Health & Service Readiness

### `GET /api/v1/health`
Returns the operational readiness and provenance of the inference model.

**Response (200 OK):**
```json
{
  "status": "ok",
  "trained": true,
  "checkpoint_name": "model_checkpoint.pth",
  "checkpoint_sha": "0c29b4d666d6...",
  "model_version": "2.0.0",
  "training_seed": 42,
  "git_commit": "e8a1f2b",
  "architecture": "hard_constrained_pinn",
  "reynolds_number": 100.0,
  "reference_dataset_available": true
}
```

---

## 2. Physics & System Metadata

### `GET /api/v1/metadata`
Returns canonical non-dimensional fluid properties and spatial-temporal domain extents.

**Response (200 OK):**
```json
{
  "physics": {
    "reynolds_number": 100.0,
    "density": 1.0,
    "kinematic_viscosity": 0.01,
    "cylinder_center": [0.0, 0.0],
    "cylinder_radius": 0.5,
    "x_min": 1.0,
    "x_max": 8.0,
    "y_min": -2.0,
    "y_max": 2.0,
    "t_min": 0.0,
    "t_max": 20.0,
    "pressure_gauge_coord": [8.0, 2.0]
  },
  "device": "cpu",
  "checkpoint_status": "verified"
}
```

---

## 3. Inference Endpoints

### `POST /api/v1/predict/point`
Evaluates predicted flow velocity, pressure, and autograd vorticity at a single coordinate $(x, y, t)$.

**Request Body:**
```json
{
  "x": 2.5,
  "y": 0.2,
  "t": 10.0,
  "compute_vorticity": true
}
```

**Response (200 OK):**
```json
{
  "x": 2.5,
  "y": 0.2,
  "t": 10.0,
  "u": 0.8523,
  "v": -0.0412,
  "p": 0.0128,
  "velocity_magnitude": 0.8533,
  "vorticity": -0.3214,
  "request_id": "a9b8c7d6"
}
```

---

### `POST /api/v1/predict/field`
Generates a 2D structured meshgrid evaluation and computes exact pointwise autograd Navier-Stokes residuals.

**Request Body:**
```json
{
  "x_min": 1.0,
  "x_max": 8.0,
  "y_min": -2.0,
  "y_max": 2.0,
  "nx": 100,
  "ny": 50,
  "t": 10.0,
  "compute_vorticity": true
}
```

**Response (200 OK):**
```json
{
  "x": [[1.0, ...]],
  "y": [[-2.0, ...]],
  "u": [[...]],
  "v": [[...]],
  "p": [[...]],
  "velocity_magnitude": [[...]],
  "vorticity": [[...]],
  "metrics": {
    "pde_loss": 0.000245,
    "continuity_residual_mean": 0.000812,
    "momentum_x_residual_mean": 0.000341,
    "momentum_y_residual_mean": 0.000219,
    "time_snapshot": 10.0,
    "reynolds_number": 100.0,
    "viscosity": 0.01,
    "nx": 100,
    "ny": 50,
    "trained_weights": true
  },
  "status": "success",
  "request_id": "e4f5g6h7"
}
```

---

### `POST /api/v1/reference/field`
Interpolates genuine high-fidelity DNS simulation reference data across the requested meshgrid.
