import os
import time
import uuid
import logging
from typing import Optional, List, Dict, Any
import numpy as np
import scipy.io
from scipy.interpolate import griddata
import torch
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from src.config.physics_config import PhysicsConfig, DEFAULT_PHYSICS_CONFIG
from src.models.checkpoint import load_and_validate_checkpoint, get_git_commit_sha
from src.losses.pde_loss import PDELoss

# Setup structured logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [req_id=%(request_id)s] %(message)s")
logger = logging.getLogger("PINNFlowAPI")

class RequestIdFilter(logging.Filter):
    def filter(self, record):
        if not hasattr(record, "request_id"):
            record.request_id = "system"
        return True

logger.addFilter(RequestIdFilter())

# ==========================================
# 1. Authoritative Configuration & State
# ==========================================
PHYSICS_CONFIG = DEFAULT_PHYSICS_CONFIG
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Checkpoint candidate resolution
ENV_CKPT = os.getenv("PINNFLOW_CHECKPOINT")
CANDIDATE_PATHS = [
    ENV_CKPT,
    "artifacts/fourier_constrained_seed8008.pth",
    "final_pinn_model.pth",
    "results/smoke/model_checkpoint.pth"
]
CKPT_PATH = next((p for p in CANDIDATE_PATHS if p and os.path.exists(p)), None)

# Global model state
MODEL: Optional[torch.nn.Module] = None
MODEL_METADATA: Dict[str, Any] = {}
CHECKPOINT_VALID = False
CHECKPOINT_ERROR: Optional[str] = None

if CKPT_PATH:
    is_valid, loaded_model, meta, err = load_and_validate_checkpoint(CKPT_PATH, device=DEVICE, physics_config=PHYSICS_CONFIG)
    if is_valid and loaded_model is not None:
        MODEL = loaded_model
        MODEL_METADATA = meta
        CHECKPOINT_VALID = True
        logger.info(f"[+] Loaded and verified canonical checkpoint: {CKPT_PATH} (SHA: {meta.get('checkpoint_sha', '')[:12]})", extra={"request_id": "startup"})
    else:
        CHECKPOINT_ERROR = err
        logger.warning(f"[-] Checkpoint validation failed: {err}", extra={"request_id": "startup"})
else:
    CHECKPOINT_ERROR = "No checkpoint file found at configured paths."
    logger.warning("[-] No checkpoint file found at startup.", extra={"request_id": "startup"})

# Reference CFD data caching
DATA_CANDIDATES = [
    "datasets/raissi_cylinder/cylinder_nektar_wake.mat",
    "data/cylinder_nektar_wake.mat"
]
DATA_PATH = next((p for p in DATA_CANDIDATES if os.path.exists(p) and os.path.getsize(p) > 1000000), None)

CACHED_DNS: Optional[Dict[str, Any]] = None
if DATA_PATH:
    try:
        mat = scipy.io.loadmat(DATA_PATH)
        CACHED_DNS = {
            "X_star": mat["X_star"].astype(np.float32),
            "t_star": mat["t"].astype(np.float32).flatten(),
            "U_star": mat["U_star"].astype(np.float32),
            "p_star": mat["p_star"].astype(np.float32)
        }
        logger.info(f"[+] Cached DNS reference dataset ({CACHED_DNS['X_star'].shape[0]} spatial points)", extra={"request_id": "startup"})
    except Exception as e:
        logger.warning(f"[-] Failed to cache reference data: {e}", extra={"request_id": "startup"})

# ==========================================
# 2. Strict Pydantic Contracts
# ==========================================
class PointRequest(BaseModel):
    x: float = Field(..., description="Spatial X coordinate")
    y: float = Field(..., description="Spatial Y coordinate")
    t: float = Field(..., description="Time coordinate (0.0 to 20.0)")
    compute_vorticity: bool = Field(default=False, description="Compute vorticity via autograd")

    @field_validator("t")
    @classmethod
    def validate_time(cls, v: float) -> float:
        if v < 0.0 or v > 50.0:
            raise ValueError("Time coordinate must be between 0.0 and 50.0")
        return v

class PointResponse(BaseModel):
    x: float
    y: float
    t: float
    u: float
    v: float
    p: float
    velocity_magnitude: float
    vorticity: Optional[float] = None
    request_id: str

class FieldRequest(BaseModel):
    x_min: float = Field(default=1.0, ge=-5.0, le=15.0)
    x_max: float = Field(default=8.0, ge=-5.0, le=15.0)
    y_min: float = Field(default=-2.0, ge=-5.0, le=5.0)
    y_max: float = Field(default=2.0, ge=-5.0, le=5.0)
    nx: int = Field(default=100, ge=2, le=200)
    ny: int = Field(default=50, ge=2, le=200)
    t: float = Field(default=10.0, ge=0.0, le=30.0)
    compute_vorticity: bool = Field(default=False)

class FieldMetrics(BaseModel):
    pde_loss: float
    continuity_residual_mean: float
    momentum_x_residual_mean: float
    momentum_y_residual_mean: float
    time_snapshot: float
    reynolds_number: float
    viscosity: float
    nx: int
    ny: int
    trained_weights: bool

class FieldResponse(BaseModel):
    x: List[List[float]]
    y: List[List[float]]
    u: List[List[float]]
    v: List[List[float]]
    p: List[List[float]]
    velocity_magnitude: List[List[float]]
    vorticity: Optional[List[List[float]]] = None
    metrics: FieldMetrics
    status: str
    request_id: str

class HealthResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    status: str
    trained: bool
    checkpoint_name: Optional[str] = None
    checkpoint_sha: Optional[str] = None
    model_version: str
    training_seed: Optional[int] = None
    git_commit: str
    architecture: Optional[str] = None
    reynolds_number: float
    reference_dataset_available: bool

# ==========================================
# 3. Application Setup & Middleware
# ==========================================
app = FastAPI(
    title="PINNFlow Scientific API",
    description="Production Scientific REST API for Navier-Stokes physics-informed flow reconstruction.",
    version="2.0.0"
)

# Configurable CORS
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS if CORS_ORIGINS != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_request_metadata(request: Request, call_next):
    req_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:8])
    request.state.request_id = req_id
    start_time = time.perf_counter()
    
    response = await call_next(request)
    
    duration = (time.perf_counter() - start_time) * 1000.0
    response.headers["X-Request-ID"] = req_id
    response.headers["X-Response-Time-Ms"] = f"{duration:.2f}"
    return response

def verify_service_ready():
    """Throws HTTP 503 if no valid trained checkpoint is loaded."""
    if not CHECKPOINT_VALID or MODEL is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"PINNFlow Inference Service NOT READY: {CHECKPOINT_ERROR or 'Trained checkpoint missing or invalid.'}"
        )

# ==========================================
# 4. API Endpoints (v1 & aliases)
# ==========================================
@app.get("/api/v1/health", response_model=HealthResponse)
@app.get("/health", response_model=HealthResponse)
def get_health():
    return HealthResponse(
        status="ok" if CHECKPOINT_VALID else "service_not_ready",
        trained=CHECKPOINT_VALID,
        checkpoint_name=MODEL_METADATA.get("checkpoint_filename"),
        checkpoint_sha=MODEL_METADATA.get("checkpoint_sha"),
        model_version=MODEL_METADATA.get("version", "2.0.0"),
        training_seed=MODEL_METADATA.get("training_seed"),
        git_commit=MODEL_METADATA.get("git_commit", get_git_commit_sha()),
        architecture=MODEL_METADATA.get("architecture"),
        reynolds_number=PHYSICS_CONFIG.reynolds_number,
        reference_dataset_available=CACHED_DNS is not None
    )

@app.get("/")
def get_root():
    return {
        "service": "PINNFlow Scientific Research Platform",
        "status": "healthy",
        "version": "2.0.0",
        "checkpoint_ready": CHECKPOINT_VALID,
        "docs_url": "/docs"
    }

@app.get("/api/v1/metadata")
def get_metadata():
    """Returns canonical physics constants and domain geometry."""
    return {
        "physics": PHYSICS_CONFIG.to_dict(),
        "model_metadata": MODEL_METADATA,
        "device": str(DEVICE),
        "git_commit": get_git_commit_sha(),
        "checkpoint_status": "verified" if CHECKPOINT_VALID else "unverified"
    }

@app.post("/api/v1/predict/point", response_model=PointResponse)
@app.post("/predict/point", response_model=PointResponse)
def predict_point(req: PointRequest, request: Request):
    verify_service_ready()
    req_id = getattr(request.state, "request_id", "default")

    coords = torch.tensor([[req.x, req.y, req.t]], dtype=torch.float32, device=DEVICE)
    vorticity_val = None

    if req.compute_vorticity:
        coords.requires_grad_(True)
        preds = MODEL(coords)
        u, v = preds[:, 0:1], preds[:, 1:2]
        grads_u = torch.autograd.grad(u, coords, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        grads_v = torch.autograd.grad(v, coords, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        vorticity_val = float((grads_v[:, 0:1] - grads_u[:, 1:2]).item())
    else:
        with torch.no_grad():
            preds = MODEL(coords)

    u_val = float(preds[0, 0].item())
    v_val = float(preds[0, 1].item())
    p_val = float(preds[0, 2].item())

    return PointResponse(
        x=req.x,
        y=req.y,
        t=req.t,
        u=u_val,
        v=v_val,
        p=p_val,
        velocity_magnitude=float(np.sqrt(u_val**2 + v_val**2)),
        vorticity=vorticity_val,
        request_id=req_id
    )

@app.post("/api/v1/predict/field", response_model=FieldResponse)
@app.post("/predict/field", response_model=FieldResponse)
def predict_field(req: FieldRequest, request: Request):
    verify_service_ready()
    req_id = getattr(request.state, "request_id", "default")

    nx = int(req.nx)
    ny = int(req.ny)

    x_line = np.linspace(req.x_min, req.x_max, nx)
    y_line = np.linspace(req.y_min, req.y_max, ny)
    X, Y = np.meshgrid(x_line, y_line)
    T = np.full_like(X, req.t)

    flat_coords = np.stack([X.flatten(), Y.flatten(), T.flatten()], axis=-1)
    coords_tensor = torch.tensor(flat_coords, dtype=torch.float32, device=DEVICE).requires_grad_(True)

    # Compute autograd residuals and predictions
    pde_module = PDELoss(physics_config=PHYSICS_CONFIG)
    preds = MODEL(coords_tensor)
    u_t, v_t, p_t = preds[:, 0:1], preds[:, 1:2], preds[:, 2:3]

    residuals = pde_module.compute_residuals(coords_tensor, preds)
    r_c = residuals["continuity"].detach().cpu().numpy()
    r_u = residuals["momentum_x"].detach().cpu().numpy()
    r_v = residuals["momentum_y"].detach().cpu().numpy()

    pde_loss = float(np.mean(r_c**2 + r_u**2 + r_v**2))
    cont_mean = float(np.mean(np.abs(r_c)))
    mom_x_mean = float(np.mean(np.abs(r_u)))
    mom_y_mean = float(np.mean(np.abs(r_v)))

    u_grid = u_t.detach().cpu().numpy().reshape(ny, nx)
    v_grid = v_t.detach().cpu().numpy().reshape(ny, nx)
    p_grid = p_t.detach().cpu().numpy().reshape(ny, nx)
    vel_mag_grid = np.sqrt(u_grid**2 + v_grid**2)

    vorticity_grid = None
    if req.compute_vorticity:
        w = residuals["vorticity"].detach().cpu().numpy().reshape(ny, nx)
        vorticity_grid = w.tolist()

    return FieldResponse(
        x=X.tolist(),
        y=Y.tolist(),
        u=u_grid.tolist(),
        v=v_grid.tolist(),
        p=p_grid.tolist(),
        velocity_magnitude=vel_mag_grid.tolist(),
        vorticity=vorticity_grid,
        metrics=FieldMetrics(
            pde_loss=pde_loss,
            continuity_residual_mean=cont_mean,
            momentum_x_residual_mean=mom_x_mean,
            momentum_y_residual_mean=mom_y_mean,
            time_snapshot=req.t,
            reynolds_number=PHYSICS_CONFIG.reynolds_number,
            viscosity=PHYSICS_CONFIG.kinematic_viscosity,
            nx=nx,
            ny=ny,
            trained_weights=CHECKPOINT_VALID
        ),
        status="success",
        request_id=req_id
    )

@app.post("/api/v1/reference/field")
@app.post("/reference/field")
def reference_field(req: FieldRequest, request: Request):
    if CACHED_DNS is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Reference CFD dataset is not available."
        )
    req_id = getattr(request.state, "request_id", "default")

    X_star = CACHED_DNS["X_star"]
    t_star = CACHED_DNS["t_star"]
    U_star = CACHED_DNS["U_star"]
    p_star = CACHED_DNS["p_star"]

    t_idx = int(np.argmin(np.abs(t_star - req.t)))
    u_exact = U_star[:, 0, t_idx]
    v_exact = U_star[:, 1, t_idx]
    p_exact = p_star[:, t_idx]

    x_line = np.linspace(req.x_min, req.x_max, req.nx)
    y_line = np.linspace(req.y_min, req.y_max, req.ny)
    X_grid, Y_grid = np.meshgrid(x_line, y_line)

    u_interp = griddata(X_star, u_exact, (X_grid, Y_grid), method="cubic", fill_value=0.0)
    v_interp = griddata(X_star, v_exact, (X_grid, Y_grid), method="cubic", fill_value=0.0)
    p_interp = griddata(X_star, p_exact, (X_grid, Y_grid), method="cubic", fill_value=0.0)

    vel_mag = np.sqrt(u_interp**2 + v_interp**2)

    return {
        "x": X_grid.tolist(),
        "y": Y_grid.tolist(),
        "u": u_interp.tolist(),
        "v": v_interp.tolist(),
        "p": p_interp.tolist(),
        "velocity_magnitude": vel_mag.tolist(),
        "metrics": {
            "time_snapshot_requested": req.t,
            "time_snapshot_matched": float(t_star[t_idx]),
            "time_index": t_idx
        },
        "status": "success",
        "request_id": req_id
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
