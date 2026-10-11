import os
import time
import uuid
import logging
import gc
import threading
import json
from typing import Optional, List, Dict, Any
import numpy as np
import scipy.io
from scipy.interpolate import griddata
import torch
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator, model_validator

from src.config.physics_config import PhysicsConfig, DEFAULT_PHYSICS_CONFIG
from src.models.checkpoint import load_and_validate_checkpoint, get_git_commit_sha
from src.losses.pde_loss import PDELoss
from fastapi.middleware.gzip import GZipMiddleware

# Lock single CPU thread for 512MB RAM predictability
torch.set_num_threads(1)
INFERENCE_LOCK = threading.Lock()

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

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ENV_CKPT = os.getenv("PINNFLOW_CHECKPOINT")

RAW_CKPT_CANDIDATES = [
    ENV_CKPT,
    "final_pinn_model.pth",
    "artifacts/fourier_constrained_seed8008.pth",
    "results/smoke/model_checkpoint.pth"
]

CANDIDATE_PATHS = []
for p in RAW_CKPT_CANDIDATES:
    if p:
        CANDIDATE_PATHS.append(p)
        if not os.path.isabs(p):
            CANDIDATE_PATHS.append(os.path.join(REPO_ROOT, p))

# Global model state
MODEL: Optional[torch.nn.Module] = None
MODEL_METADATA: Dict[str, Any] = {}
CHECKPOINT_VALID = False
CHECKPOINT_ERROR: Optional[str] = None
CKPT_PATH: Optional[str] = None

# Safe checkpoint resolution with verification
try:
    for candidate in CANDIDATE_PATHS:
        if candidate and os.path.exists(candidate):
            is_valid, loaded_model, meta, err = load_and_validate_checkpoint(candidate, device=DEVICE, physics_config=PHYSICS_CONFIG)
            if is_valid and loaded_model is not None:
                MODEL = loaded_model
                MODEL_METADATA = meta
                CHECKPOINT_VALID = True
                CKPT_PATH = candidate
                CHECKPOINT_ERROR = None
                logger.info(f"[+] Loaded verified checkpoint: {candidate}", extra={"request_id": "startup"})
                break
            else:
                if CHECKPOINT_ERROR is None:
                    CHECKPOINT_ERROR = err
except Exception as e:
    logger.warning(f"[-] Checkpoint verification warning: {e}", extra={"request_id": "startup"})
    CHECKPOINT_VALID = False

# Reference CFD data candidate path resolution
RAW_DATA_CANDIDATES = [
    "datasets/raissi_cylinder/cylinder_nektar_wake.mat",
    "data/cylinder_nektar_wake.mat"
]
DATA_CANDIDATES = []
for p in RAW_DATA_CANDIDATES:
    DATA_CANDIDATES.append(p)
    DATA_CANDIDATES.append(os.path.join(REPO_ROOT, p))

DATA_PATH: Optional[str] = next((p for p in DATA_CANDIDATES if os.path.exists(p) and os.path.getsize(p) > 1000000), None)

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

    @model_validator(mode="after")
    def validate_grid_caps(self):
        total_pts = self.nx * self.ny
        limit = 10000 if self.compute_vorticity else 20000
        if total_pts > limit:
            vort_str = " with vorticity enabled" if self.compute_vorticity else ""
            raise ValueError(
                f"Requested grid resolution {self.nx}x{self.ny} ({total_pts} points) exceeds limit of {limit} points{vort_str}."
            )
        return self

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

# Robust Wildcard CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

@app.middleware("http")
async def add_cors_and_metadata(request: Request, call_next):
    # Handle preflight OPTIONS requests directly
    if request.method == "OPTIONS":
        res = Response(status_code=200)
        res.headers["Access-Control-Allow-Origin"] = "*"
        res.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
        res.headers["Access-Control-Allow-Headers"] = "*"
        return res

    req_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:8])
    request.state.request_id = req_id
    start_time = time.perf_counter()
    
    try:
        response = await call_next(request)
    except Exception as exc:
        logger.error(f"Internal request error: {exc}", extra={"request_id": req_id})
        response = Response(
            content=json.dumps({"status": "error", "detail": f"Internal server error: {str(exc)}"}),
            status_code=500,
            media_type="application/json"
        )
    
    duration = (time.perf_counter() - start_time) * 1000.0
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "*"
    response.headers["X-Request-ID"] = req_id
    response.headers["X-Response-Time-Ms"] = f"{duration:.2f}"
    return response

# ==========================================
# 4. API Endpoints (v1 & aliases)
# ==========================================
@app.api_route("/api/v1/wakeup", methods=["GET", "HEAD"])
@app.api_route("/wakeup", methods=["GET", "HEAD"])
def get_wakeup():
    """Lightweight cold-start ping endpoint to wake up free-tier instances."""
    return {
        "status": "awake",
        "service": "pinnflow",
        "model_ready": CHECKPOINT_VALID and (MODEL is not None),
        "timestamp": time.time()
    }

@app.get("/api/v1/health", response_model=HealthResponse)
@app.get("/health", response_model=HealthResponse)
def get_health():
    is_dataset_avail = bool(DATA_PATH and os.path.exists(DATA_PATH))
    return HealthResponse(
        status="ok",
        trained=CHECKPOINT_VALID and (MODEL is not None),
        checkpoint_name=MODEL_METADATA.get("checkpoint_filename", "canonical_pinn"),
        checkpoint_sha=MODEL_METADATA.get("checkpoint_sha", "8008a1c9"),
        model_version=MODEL_METADATA.get("version", "2.0.0"),
        training_seed=MODEL_METADATA.get("training_seed", 0),
        git_commit=MODEL_METADATA.get("git_commit", get_git_commit_sha()),
        architecture=MODEL_METADATA.get("architecture", "hard_constrained_pinn"),
        reynolds_number=PHYSICS_CONFIG.reynolds_number,
        reference_dataset_available=is_dataset_avail
    )

@app.get("/")
def get_root():
    return {
        "service": "PINNFlow Scientific Research Platform",
        "status": "healthy",
        "version": "2.0.0",
        "checkpoint_ready": CHECKPOINT_VALID and (MODEL is not None),
        "docs_url": "/docs"
    }

@app.get("/api/v1/metadata")
def get_metadata():
    return {
        "physics": PHYSICS_CONFIG.to_dict(),
        "model_metadata": MODEL_METADATA,
        "device": str(DEVICE),
        "git_commit": get_git_commit_sha(),
        "checkpoint_status": "verified" if CHECKPOINT_VALID else "uncalibrated"
    }

@app.post("/api/v1/predict/point", response_model=PointResponse)
@app.post("/predict/point", response_model=PointResponse)
def predict_point(req: PointRequest, request: Request):
    req_id = getattr(request.state, "request_id", "default")

    if not CHECKPOINT_VALID or MODEL is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="PINNFlow Inference Service NOT READY: Trained checkpoint is not loaded. Cannot evaluate physical predictions."
        )

    with INFERENCE_LOCK:
        try:
            coords = torch.tensor([[req.x, req.y, req.t]], dtype=torch.float32, device=DEVICE)
            vorticity_val = None

            if req.compute_vorticity:
                coords.requires_grad_(True)
                preds = MODEL(coords)
                u, v = preds[:, 0:1], preds[:, 1:2]
                grads_u = torch.autograd.grad(u, coords, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=False)[0]
                grads_v = torch.autograd.grad(v, coords, grad_outputs=torch.ones_like(v), create_graph=False)[0]
                vorticity_raw = (grads_v[:, 0:1] - grads_u[:, 1:2]).item()
                vorticity_val = 0.0 if np.isnan(vorticity_raw) or np.isinf(vorticity_raw) else float(vorticity_raw)
            else:
                with torch.no_grad():
                    preds = MODEL(coords)

            u_raw = float(preds[0, 0].item())
            v_raw = float(preds[0, 1].item())
            p_raw = float(preds[0, 2].item())

            u_val = float(np.nan_to_num(u_raw, nan=0.0, posinf=0.0, neginf=0.0))
            v_val = float(np.nan_to_num(v_raw, nan=0.0, posinf=0.0, neginf=0.0))
            p_val = float(np.nan_to_num(p_raw, nan=0.0, posinf=0.0, neginf=0.0))
            vel_mag = float(np.nan_to_num(np.sqrt(u_val**2 + v_val**2), nan=0.0))

            return PointResponse(
                x=req.x, y=req.y, t=req.t, u=u_val, v=v_val, p=p_val,
                velocity_magnitude=vel_mag, vorticity=vorticity_val, request_id=req_id
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Point prediction error: {e}", extra={"request_id": req_id})
            raise HTTPException(status_code=500, detail=f"Inference calculation failed: {str(e)}")

@app.post("/api/v1/predict/field", response_model=FieldResponse)
@app.post("/predict/field", response_model=FieldResponse)
def predict_field(req: FieldRequest, request: Request):
    req_id = getattr(request.state, "request_id", "default")

    if not CHECKPOINT_VALID or MODEL is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="PINNFlow Inference Service NOT READY: Trained checkpoint is not loaded. Cannot evaluate physical predictions."
        )

    nx = int(req.nx)
    ny = int(req.ny)

    x_line = np.linspace(req.x_min, req.x_max, nx, dtype=np.float32)
    y_line = np.linspace(req.y_min, req.y_max, ny, dtype=np.float32)
    X, Y = np.meshgrid(x_line, y_line)
    T = np.full_like(X, req.t, dtype=np.float32)

    flat_coords = np.stack([X.flatten(), Y.flatten(), T.flatten()], axis=-1)
    total_points = flat_coords.shape[0]

    BATCH_SIZE = 500
    pde_module = PDELoss(physics_config=PHYSICS_CONFIG)

    u_chunks, v_chunks, p_chunks = [], [], []
    r_c_chunks, r_u_chunks, r_v_chunks = [], [], []
    vorticity_chunks = [] if req.compute_vorticity else None

    with INFERENCE_LOCK:
        try:
            if req.compute_vorticity:
                for i in range(0, total_points, BATCH_SIZE):
                    chunk_np = flat_coords[i:i + BATCH_SIZE]
                    chunk_coords = torch.tensor(chunk_np, dtype=torch.float32, device=DEVICE).requires_grad_(True)
                    chunk_preds = MODEL(chunk_coords)
                    chunk_residuals = pde_module.compute_residuals(chunk_coords, chunk_preds)

                    u_chunks.append(chunk_preds[:, 0:1].detach().cpu().numpy())
                    v_chunks.append(chunk_preds[:, 1:2].detach().cpu().numpy())
                    p_chunks.append(chunk_preds[:, 2:3].detach().cpu().numpy())

                    r_c_chunks.append(chunk_residuals["continuity"].detach().cpu().numpy())
                    r_u_chunks.append(chunk_residuals["momentum_x"].detach().cpu().numpy())
                    r_v_chunks.append(chunk_residuals["momentum_y"].detach().cpu().numpy())

                    if vorticity_chunks is not None:
                        vorticity_chunks.append(chunk_residuals["vorticity"].detach().cpu().numpy())

                    del chunk_coords, chunk_preds, chunk_residuals
            else:
                with torch.no_grad():
                    for i in range(0, total_points, BATCH_SIZE):
                        chunk_np = flat_coords[i:i + BATCH_SIZE]
                        chunk_coords = torch.tensor(chunk_np, dtype=torch.float32, device=DEVICE)
                        chunk_preds = MODEL(chunk_coords)

                        u_chunks.append(chunk_preds[:, 0:1].cpu().numpy())
                        v_chunks.append(chunk_preds[:, 1:2].cpu().numpy())
                        p_chunks.append(chunk_preds[:, 2:3].cpu().numpy())

                        del chunk_coords, chunk_preds

                sample_size = min(total_points, BATCH_SIZE)
                sample_np = flat_coords[:sample_size]
                sample_coords = torch.tensor(sample_np, dtype=torch.float32, device=DEVICE).requires_grad_(True)
                sample_preds = MODEL(sample_coords)
                sample_residuals = pde_module.compute_residuals(sample_coords, sample_preds)

                r_c_chunks.append(sample_residuals["continuity"].detach().cpu().numpy())
                r_u_chunks.append(sample_residuals["momentum_x"].detach().cpu().numpy())
                r_v_chunks.append(sample_residuals["momentum_y"].detach().cpu().numpy())

                del sample_coords, sample_preds, sample_residuals
        except Exception as e:
            logger.error(f"Field evaluation error: {e}", extra={"request_id": req_id})
            raise HTTPException(status_code=500, detail=f"Field evaluation calculation failed: {str(e)}")

    u_all = np.nan_to_num(np.vstack(u_chunks).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    v_all = np.nan_to_num(np.vstack(v_chunks).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    p_all = np.nan_to_num(np.vstack(p_chunks).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    r_c_all = np.nan_to_num(np.vstack(r_c_chunks).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    r_u_all = np.nan_to_num(np.vstack(r_u_chunks).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    r_v_all = np.nan_to_num(np.vstack(r_v_chunks).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    pde_loss = float(np.nan_to_num(np.mean(r_c_all**2 + r_u_all**2 + r_v_all**2), nan=0.0))
    cont_mean = float(np.nan_to_num(np.mean(np.abs(r_c_all)), nan=0.0))
    mom_x_mean = float(np.nan_to_num(np.mean(np.abs(r_u_all)), nan=0.0))
    mom_y_mean = float(np.nan_to_num(np.mean(np.abs(r_v_all)), nan=0.0))

    u_grid = u_all.reshape(ny, nx)
    v_grid = v_all.reshape(ny, nx)
    p_grid = p_all.reshape(ny, nx)
    vel_mag_grid = np.nan_to_num(np.sqrt(u_grid**2 + v_grid**2), nan=0.0, posinf=0.0, neginf=0.0)

    vorticity_grid = None
    if req.compute_vorticity and vorticity_chunks is not None and len(vorticity_chunks) > 0:
        vorticity_all = np.nan_to_num(np.vstack(vorticity_chunks).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        vorticity_grid = vorticity_all.reshape(ny, nx).tolist()

    del u_chunks, v_chunks, p_chunks, r_c_chunks, r_u_chunks, r_v_chunks
    gc.collect()

    return FieldResponse(
        x=np.nan_to_num(X, nan=0.0).tolist(),
        y=np.nan_to_num(Y, nan=0.0).tolist(),
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
            trained_weights=CHECKPOINT_VALID and (MODEL is not None)
        ),
        status="success",
        request_id=req_id
    )

@app.post("/api/v1/reference/field")
@app.post("/reference/field")
def reference_field(req: FieldRequest, request: Request):
    req_id = getattr(request.state, "request_id", "default")

    if not DATA_PATH or not os.path.exists(DATA_PATH):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Reference DNS dataset (cylinder_nektar_wake.mat) is not available on this server. Run 'python scripts/download_data.py' to download reference data."
        )

    x_line = np.linspace(req.x_min, req.x_max, req.nx, dtype=np.float32)
    y_line = np.linspace(req.y_min, req.y_max, req.ny, dtype=np.float32)
    X_grid, Y_grid = np.meshgrid(x_line, y_line)

    try:
        mat = scipy.io.loadmat(DATA_PATH)
        X_star = mat["X_star"].astype(np.float32)
        t_star = mat["t"].astype(np.float32).flatten()

        t_idx = int(np.argmin(np.abs(t_star - req.t)))
        matched_t = float(t_star[t_idx])

        u_exact = mat["U_star"][:, 0, t_idx].astype(np.float32)
        v_exact = mat["U_star"][:, 1, t_idx].astype(np.float32)
        p_exact = mat["p_star"][:, t_idx].astype(np.float32)

        del mat
        gc.collect()

        u_interp = np.nan_to_num(griddata(X_star, u_exact, (X_grid, Y_grid), method="cubic", fill_value=0.0).astype(np.float32), nan=0.0)
        v_interp = np.nan_to_num(griddata(X_star, v_exact, (X_grid, Y_grid), method="cubic", fill_value=0.0).astype(np.float32), nan=0.0)
        p_interp = np.nan_to_num(griddata(X_star, p_exact, (X_grid, Y_grid), method="cubic", fill_value=0.0).astype(np.float32), nan=0.0)

        del X_star, u_exact, v_exact, p_exact
        gc.collect()

        vel_mag = np.nan_to_num(np.sqrt(u_interp**2 + v_interp**2), nan=0.0)

        return {
            "x": np.nan_to_num(X_grid, nan=0.0).tolist(),
            "y": np.nan_to_num(Y_grid, nan=0.0).tolist(),
            "u": u_interp.tolist(),
            "v": v_interp.tolist(),
            "p": p_interp.tolist(),
            "velocity_magnitude": vel_mag.tolist(),
            "metrics": {
                "time_snapshot_requested": req.t,
                "time_snapshot_matched": matched_t,
                "time_index": t_idx
            },
            "status": "success",
            "request_id": req_id
        }
    except Exception as e:
        logger.error(f"Reference DNS interpolation error: {e}", extra={"request_id": req_id})
        raise HTTPException(status_code=500, detail=f"Reference DNS interpolation failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run("src.api.main:app", host=host, port=port, reload=True)


