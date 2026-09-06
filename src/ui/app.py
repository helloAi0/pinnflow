import streamlit as st
import requests
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="PINNFlow Field Explorer", layout="wide")

st.title("PINNFlow: Interactive Fluid Field Explorer")
st.markdown("Real-time neural surrogate field solver for 2D cylinder wake flows.")

# Sidebar Controls
st.sidebar.header("Simulation Parameters")
t_val = st.sidebar.slider("Timestamp (t)", min_value=0.0, max_value=20.0, value=10.0, step=0.1)
field_type = st.sidebar.selectbox(
    "Field Variable", 
    ["velocity_magnitude", "u", "v", "p", "vorticity"]
)
grid_res = st.sidebar.slider("Grid Resolution (N x N)", min_value=30, max_value=120, value=60, step=10)

col1, col2 = st.sidebar.columns(2)
x_min = col1.number_input("x min", value=-1.0, step=0.5)
x_max = col2.number_input("x max", value=8.0, step=0.5)
y_min = col1.number_input("y min", value=-2.0, step=0.5)
y_max = col2.number_input("y max", value=2.0, step=0.5)

API_URL = "http://localhost:8000/predict/field"

payload = {
    "x_min": float(x_min),
    "x_max": float(x_max),
    "y_min": float(y_min),
    "y_max": float(y_max),
    "t": float(t_val),
    "nx": int(grid_res),
    "ny": int(grid_res),
    "compute_vorticity": (field_type == "vorticity")
}

# Fetch and Render Field
if st.sidebar.button("Refresh Field", type="primary"):
    st.rerun()

try:
    with st.spinner("Querying PINN REST API..."):
        response = requests.post(API_URL, json=payload, timeout=10)
        
    if response.status_code == 200:
        data = response.json()
        X = np.array(data["x"])
        Y = np.array(data["y"])
        Z = np.array(data[field_type])

        # Map appropriate color scales
        cmap = "RdBu_r" if field_type in ["u", "v", "vorticity"] else "Viridis"

        fig = go.Figure(data=go.Contour(
            z=Z, 
            x=X[0, :], 
            y=Y[:, 0],
            colorscale=cmap,
            contours_coloring='heatmap',
            line_smoothing=0.85,
            colorbar=dict(title=field_type)
        ))

        # Render physical cylinder boundary (Radius = 0.5 at origin)
        fig.add_shape(
            type="circle",
            xref="x", yref="y",
            x0=-0.5, y0=-0.5, x1=0.5, y1=0.5,
            fillcolor="grey",
            line_color="black",
        )

        fig.update_layout(
            title=f"PINN Predicted Field: {field_type.upper()} (t = {t_val:.1f}s)",
            xaxis_title="x",
            yaxis_title="y",
            yaxis=dict(scaleanchor="x", scaleratio=1),
            autosize=True,
            height=550
        )

        st.plotly_chart(fig, use_container_width=True)

        # Probing summary statistics
        st.subheader("Field Metrics")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Min Value", f"{np.min(Z):.4f}")
        m2.metric("Max Value", f"{np.max(Z):.4f}")
        m3.metric("Mean Value", f"{np.mean(Z):.4f}")
        m4.metric("Std Dev", f"{np.std(Z):.4f}")

    else:
        st.error(f"API Error ({response.status_code}): {response.text}")

except Exception as e:
        st.error(f"Failed to connect to FastAPI backend at `{API_URL}`.\nEnsure `python -m src.api.main` is active.\n\nError: {e}")