# PINNFlow Production Deployment Guide (Render + Vercel)

This document provides the authoritative, step-by-step production deployment pipeline for **PINNFlow**:
- **Backend (FastAPI + PyTorch PINN)**: Deployed on **Render** (as a Web Service or Docker Container).
- **Frontend (React + Vite + Tailwind)**: Deployed on **Vercel** (with SPA rewrites and API injection).

---

## 1. Backend Deployment on Render

### Option A: Deploy via Blueprint (`render.yaml`) (Recommended)
1. Push all code to your GitHub repository: `https://github.com/helloAi0/pinnflow.git`.
2. In the [Render Dashboard](https://dashboard.render.com), click **New +** &rarr; **Blueprint**.
3. Select the `pinnflow` repository. Render will automatically parse [render.yaml](file:///c:/Users/user/Documents/pinnflow/render.yaml) and configure the services:
   - **Service Name**: `pinnflow-api`
   - **Runtime**: Python 3.10
   - **Build Command**: `pip install -r requirements.txt && pip install -e .`
   - **Start Command**: `uvicorn src.api.main:app --host 0.0.0.0 --port $PORT`
   - **Health Check Path**: `/api/v1/health`

### Option B: Manual Web Service Setup on Render
1. Click **New +** &rarr; **Web Service**.
2. Connect your Git repository.
3. Configure the following build & runtime settings:
   | Setting | Value |
   | :--- | :--- |
   | **Runtime** | `Python 3` |
   | **Build Command** | `pip install -r requirements.txt && pip install -e .` |
   | **Start Command** | `uvicorn src.api.main:app --host 0.0.0.0 --port $PORT` |
   | **Health Check Path** | `/api/v1/health` |
4. Set the **Environment Variables**:
   | Key | Value | Description |
   | :--- | :--- | :--- |
   | `PYTHON_VERSION` | `3.10.11` | Canonical Python runtime |
   | `CORS_ORIGINS` | `*` (or your Vercel URL) | Allowed frontend origins |
   | `PINNFLOW_CHECKPOINT` | `final_pinn_model.pth` | Canonical model weights |

5. Once deployed, copy your Render public URL (e.g. `https://pinnflow-api.onrender.com`).

---

## 2. Frontend Deployment on Vercel

### Deploying via Vercel Dashboard / CLI
1. Log in to [Vercel](https://vercel.com) and click **Add New...** &rarr; **Project**.
2. Import the `pinnflow` repository.
3. Configure the project settings:
   - **Framework Preset**: `Vite`
   - **Root Directory**: `./` (or `frontend` if deploying subfolder directly)
   - **Build Command**: `cd frontend && npm ci && npm run build` (or `npm run build` if Root Directory is `frontend`)
   - **Output Directory**: `frontend/dist` (or `dist` if Root Directory is `frontend`)
4. In **Environment Variables**, inject your backend URL:
   | Key | Value |
   | :--- | :--- |
   | `VITE_API_URL` | `https://pinnflow-api.onrender.com` (Your live Render backend URL) |
5. Click **Deploy**.

> [!NOTE]
> [vercel.json](file:///c:/Users/user/Documents/pinnflow/vercel.json) automatically handles SPA route rewrites (`/(.*) -> /index.html`) so refreshing or navigating tabs will never throw 404 errors.

---

## 3. End-to-End Production Verification Checklist

Run through this checklist to verify that both platforms are operating flawlessly in production:

- [ ] **1. Backend Health Check**:
  Navigate to `https://<YOUR-RENDER-URL>/api/v1/health`.
  Expected JSON:
  ```json
  {
    "status": "ok",
    "trained": true,
    "checkpoint_name": "final_pinn_model.pth",
    "checkpoint_sha": "0c29b4d666d6fd456b6407f33eea8b5ad0ae334c1622d83ddb1d40e2ffbe186d",
    "model_version": "2.0.0",
    "training_seed": 0,
    "git_commit": "...",
    "architecture": "hard_constrained_pinn",
    "reynolds_number": 100.0,
    "reference_dataset_available": true
  }
  ```
- [ ] **2. Swagger OpenAPI Documentation**:
  Navigate to `https://<YOUR-RENDER-URL>/docs` to test interactive endpoints (`/api/v1/predict/point`, `/api/v1/predict/field`, `/api/v1/reference/field`).
- [ ] **3. Vercel Frontend Connection**:
  Navigate to your live Vercel URL. The top-right badge should display **Checkpoint Verified** in glowing green with live Re = 100.0 telemetry.
- [ ] **4. Interactive Flow Field Evaluation**:
  Click **Evaluate Physical Field**. The flow canvas should render the 2D fluid velocity field $(|\mathbf{u}|, u, v, p, \omega)$ in under 50ms.
- [ ] **5. Hover Crosshair Probe**:
  Move your mouse over the flow field canvas. The real-time probe HUD should display exact coordinates, velocity vector components, static pressure, and vorticity.
- [ ] **6. Direct CSV / JSON Exports**:
  Click **Export Field CSV** and **Export Metrics JSON** to confirm client-side data bundles download accurately.
