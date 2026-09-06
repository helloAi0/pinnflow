import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
from contextlib import asynccontextmanager

from src.inference.predict import PINNInferenceEngine

# Global variable to hold the loaded inference engine
engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads the PINN model into memory on startup."""
    global engine
    print("Loading PINN Inference Engine...")
    try:
        engine = PINNInferenceEngine(model_path="final_pinn_model.pth")
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
    yield
    print("Shutting down API and clearing resources...")
    engine = None

app = FastAPI(
    title="PINNFlow Scientific API",
    description="REST API for querying PINN fluid dynamics predictions.",
    version="1.0.0",
    lifespan=lifespan
)

# FIX: Enable CORS for frontend integration while preventing credential/wildcard crashes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # Fixed: Must be False when using wildcard origin '*'
    allow_methods=["*"],
    allow_headers=["*"],
)

class PointQuery(BaseModel):
    x: float = Field(..., description="X coordinate")
    y: float = Field(..., description="Y coordinate")
    t: float = Field(..., description="Time stamp")
    compute_vorticity: bool = Field(False, description="Compute derived vorticity field via autograd")

class GridQuery(BaseModel):
    x_min: float = Field(-1.0, description="Minimum X boundary")
    x_max: float = Field(8.0, description="Maximum X boundary")
    y_min: float = Field(-2.0, description="Minimum Y boundary")
    y_max: float = Field(2.0, description="Maximum Y boundary")
    t: float = Field(10.0, description="Time stamp")
    nx: int = Field(50, description="Number of points along X axis")
    ny: int = Field(50, description="Number of points along Y axis")
    compute_vorticity: bool = Field(False, description="Compute derived vorticity field via autograd")

@app.post("/predict/point")
async def predict_point(query: PointQuery):
    """Predicts flow fields for a single (x, y, t) coordinate."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Model engine not initialized.")
        
    x_in = np.array([query.x])
    y_in = np.array([query.y])
    
    try:
        res = engine.predict_fields(x_in, y_in, query.t)
        
        out = {
            "u": float(res["u"][0]),
            "v": float(res["v"][0]),
            "p": float(res["p"][0]),
            "velocity_magnitude": float(res["velocity_magnitude"][0])
        }
        
        if query.compute_vorticity:
            w = engine.predict_vorticity(x_in, y_in, query.t)
            out["vorticity"] = float(w[0])
            
        return out
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/field")
async def predict_field(query: GridQuery):
    """Predicts flow fields across a generated spatial meshgrid for visualization."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Model engine not initialized.")
        
    x_linspace = np.linspace(query.x_min, query.x_max, query.nx)
    y_linspace = np.linspace(query.y_min, query.y_max, query.ny)
    X, Y = np.meshgrid(x_linspace, y_linspace)
    
    x_flat = X.flatten()
    y_flat = Y.flatten()
    
    try:
        res = engine.predict_fields(x_flat, y_flat, query.t)
        
        # Package 1D flat arrays back into 2D grid format for frontend heatmap plotting
        out = {
            "x": X.tolist(),
            "y": Y.tolist(),
            "u": res["u"].reshape(query.ny, query.nx).tolist(),
            "v": res["v"].reshape(query.ny, query.nx).tolist(),
            "p": res["p"].reshape(query.ny, query.nx).tolist(),
            "velocity_magnitude": res["velocity_magnitude"].reshape(query.ny, query.nx).tolist()
        }
        
        if query.compute_vorticity:
            w = engine.predict_vorticity(x_flat, y_flat, query.t)
            out["vorticity"] = w.reshape(query.ny, query.nx).tolist()
            
        return out
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
