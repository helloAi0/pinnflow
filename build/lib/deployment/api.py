import logging
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

from src.models.mlp import NavierStokesMLP

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# 1. Define Data Schemas
class Point3D(BaseModel):
    x: float
    y: float
    t: float

class PredictionResponse(BaseModel):
    u: float
    v: float
    p: float

class InferenceRequest(BaseModel):
    points: List[Point3D]

class InferenceResponse(BaseModel):
    predictions: List[PredictionResponse]

# 2. Initialize App and Model State
app = FastAPI(title="PINNFlow Inference API", version="1.0")
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@app.on_event("startup")
def load_model():
    global model
    model_path = 'final_pinn_model.pth'
    logging.info(f"Loading model from {model_path} onto {device}...")
    
    try:
        model = NavierStokesMLP().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        logging.info("Model loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        raise RuntimeError("Model initialization failed.")

# 3. Define Inference Endpoint
@app.post("/predict", response_model=InferenceResponse)
def predict_fluid_fields(request: InferenceRequest):
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    
    try:
        # Convert incoming JSON payload to PyTorch tensor
        input_data = [[pt.x, pt.y, pt.t] for pt in request.points]
        tensor_data = torch.tensor(input_data, dtype=torch.float32).to(device)
        
        # Execute forward pass
        with torch.no_grad():
            preds = model(tensor_data).cpu().numpy()
            
        # Format response
        results = []
        for row in preds:
            results.append(PredictionResponse(u=float(row[0]), v=float(row[1]), p=float(row[2])))
            
        return InferenceResponse(predictions=results)
        
    except Exception as e:
        logging.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("src.deployment.api:app", host="0.0.0.0", port=8000, reload=True)