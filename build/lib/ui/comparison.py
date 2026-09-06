import streamlit as st
import numpy as np
import h5py
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.models.mlp import NavierStokesMLP

st.set_page_config(page_title="PINNFlow Comparison Engine", layout="wide")

st.title("PINNFlow: Model vs Reference Comparison")
st.markdown("Side-by-side benchmark comparing PINN predictions against reference CFD flow fields.")

@st.cache_resource
def load_pinn_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NavierStokesMLP().to(device)
    model.load_state_dict(torch.load("final_pinn_model.pth", map_location=device))
    model.eval()
    return model, device

@st.cache_data
def load_reference_data():
    data_path = "datasets/raissi_cylinder/reference_cylinder.h5"
    with h5py.File(data_path, 'r') as f:
        # Squeeze arrays to handle both 1D and grid shapes consistently
        x = np.squeeze(f['x'][:])
        y = np.squeeze(f['y'][:])
        t = np.squeeze(f['t'][:])
        u = np.squeeze(f['u'][:])
        v = np.squeeze(f['v'][:])
        p = np.squeeze(f['p'][:])
    return x, y, t, u, v, p

try:
    model, device = load_pinn_model()
    x, y, t_ref, u_ref, v_ref, p_ref = load_reference_data()

    # Sidebar Controls
    st.sidebar.header("Benchmark Settings")
    time_idx = st.sidebar.slider("Time Step Index", min_value=0, max_value=len(t_ref)-1, value=100)
    field_var = st.sidebar.selectbox("Variable", ["u", "v", "p"])

    t_val = t_ref[time_idx]
    st.sidebar.info(f"Selected Timestamp: **t = {t_val:.2f}s**")

    # Select Reference Field
    if field_var == "u":
        ref_field = u_ref[:, time_idx] if u_ref.ndim > 1 else u_ref
    elif field_var == "v":
        ref_field = v_ref[:, time_idx] if v_ref.ndim > 1 else v_ref
    else:
        ref_field = p_ref[:, time_idx] if p_ref.ndim > 1 else p_ref

    # Model Inference
    t_flat = np.full_like(x, t_val)
    coords = np.stack([x, y, t_flat], axis=1)
    coords_tensor = torch.tensor(coords, dtype=torch.float32).to(device)

    with torch.no_grad():
        preds = model(coords_tensor).cpu().numpy()

    var_idx = {"u": 0, "v": 1, "p": 2}[field_var]
    pred_field = preds[:, var_idx]
    abs_error = np.abs(ref_field - pred_field)

    # Subplots Setup
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=(f"Reference {field_var.upper()}", f"PINN Predicted {field_var.upper()}", f"Absolute Error |{field_var.upper()}|"),
        horizontal_spacing=0.05
    )

    # Common bounds for unified color scales
    vmin = min(ref_field.min(), pred_field.min())
    vmax = max(ref_field.max(), pred_field.max())

    # Reference scatter
    fig.add_trace(
        go.Scattergl(x=x, y=y, mode='markers', marker=dict(color=ref_field, colorscale='RdBu_r', cmin=vmin, cmax=vmax, size=4)),
        row=1, col=1
    )

    # Prediction scatter
    fig.add_trace(
        go.Scattergl(x=x, y=y, mode='markers', marker=dict(color=pred_field, colorscale='RdBu_r', cmin=vmin, cmax=vmax, size=4)),
        row=1, col=2
    )

    # Error scatter
    fig.add_trace(
        go.Scattergl(x=x, y=y, mode='markers', marker=dict(color=abs_error, colorscale='Magma', size=4, showscale=True)),
        row=1, col=3
    )

    fig.update_layout(height=450, showlegend=False, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, width="stretch")

    # Metrics Summary
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("L1 Absolute Error", f"{np.mean(abs_error):.6f}")
    c2.metric("L2 Relative Error", f"{np.linalg.norm(pred_field - ref_field) / (np.linalg.norm(ref_field) + 1e-8):.6f}")
    c3.metric("Max Point Error", f"{np.max(abs_error):.6f}")
    c4.metric("MSE", f"{np.mean((pred_field - ref_field)**2):.6f}")

except Exception as e:
    st.error(f"Error initializing comparison module: {e}")