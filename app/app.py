import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time
import plotly.graph_objects as go
import plotly.express as px

# =========================================================
# Load model safely
# =========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model = joblib.load(os.path.join(BASE_DIR, "models", "rf_rul_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "models", "scaler.pkl"))

# =========================================================
# Page config
# =========================================================

st.set_page_config(
    page_title="Industrial Engine Health System",
    layout="wide",
    page_icon="✈️"
)

st.title("✈️ Industrial Predictive Maintenance System")

st.markdown(
    "AI-powered monitoring of engine health with real-time simulation, "
    "risk assessment, explainability, and maintenance recommendations."
)

# =========================================================
# Sidebar Inputs
# =========================================================

st.sidebar.header("⚙️ Engine Parameters")

cycle = st.sidebar.slider("Cycle", 1, 300, 100)

op1 = st.sidebar.slider("Operating Setting 1", -1.0, 1.0, 0.0)
op2 = st.sidebar.slider("Operating Setting 2", -1.0, 1.0, 0.0)
op3 = st.sidebar.slider("Operating Setting 3", -1.0, 1.0, 0.0)

sensor_list = [2,3,4,7,8,9,11,12,13,14,15,17,20,21]

sensor_values = []
for i in sensor_list:
    val = st.sidebar.slider(f"Sensor {i}", -50.0, 50.0, 0.0)
    sensor_values.append(val)

# =========================================================
# Prepare input
# =========================================================

input_data = pd.DataFrame([[
    cycle, op1, op2, op3, *sensor_values
]])

input_data.columns = [
    "cycle",
    "op_setting_1","op_setting_2","op_setting_3",
    "sensor_2","sensor_3","sensor_4","sensor_7",
    "sensor_8","sensor_9","sensor_11","sensor_12",
    "sensor_13","sensor_14","sensor_15",
    "sensor_17","sensor_20","sensor_21"
]

input_scaled = scaler.transform(input_data)

prediction = model.predict(input_scaled)[0]

# =========================================================
# Failure probability (Upgrade 2)
# =========================================================

failure_prob = max(0, min(1, 1 - prediction / 300))

# =========================================================
# Status classification
# =========================================================

if prediction > 100:
    status = "Healthy"
    color = "green"
    risk = "Low"
elif prediction > 50:
    status = "Warning"
    color = "orange"
    risk = "Medium"
else:
    status = "Critical"
    color = "red"
    risk = "High"

# =========================================================
# Top Metrics
# =========================================================

c1, c2, c3, c4 = st.columns(4)

c1.metric("Cycle", cycle)
c2.metric("Predicted RUL", f"{prediction:.1f}")
c3.metric("Health", status)
c4.metric("Failure Risk", f"{failure_prob*100:.1f}%")

# =========================================================
# Upgrade 1 — Real-Time Monitoring
# =========================================================

st.subheader("🟢 Live Engine Monitoring")

run = st.button("▶️ Start Simulation")

chart_placeholder = st.empty()
status_placeholder = st.empty()

if run:
    current_rul = prediction
    history = []

    while current_rul > 0:

        degradation = np.random.uniform(0.5, 2.0)
        current_rul = max(current_rul - degradation, 0)
        history.append(current_rul)

        if current_rul > 100:
            s, col = "Healthy", "green"
        elif current_rul > 50:
            s, col = "Warning", "orange"
        else:
            s, col = "Critical", "red"

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=history,
            mode="lines",
            line=dict(color=col, width=4)
        ))

        fig.update_layout(
            title="Remaining Useful Life — Live",
            yaxis=dict(range=[0, 300]),
            height=400
        )

        chart_placeholder.plotly_chart(fig, use_container_width=True)

        status_placeholder.markdown(
            f"### ⚠️ Status: **{s}**"
        )

        time.sleep(0.4)

# =========================================================
# Upgrade 2 — Risk Gauge
# =========================================================

st.subheader("⚠️ Failure Risk Meter")

gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=failure_prob * 100,
    title={'text': "Failure Probability (%)"},
    gauge={
        'axis': {'range': [0, 100]},
        'bar': {'color': color},
        'steps': [
            {'range': [0, 30], 'color': "#00cc96"},
            {'range': [30, 70], 'color': "#ffa500"},
            {'range': [70, 100], 'color': "#ff4b4b"}
        ]
    }
))

st.plotly_chart(gauge, use_container_width=True)

# =========================================================
# Upgrade 3 — Explainable AI Panel
# =========================================================

st.subheader("🔍 Feature Contribution (Approximate)")

importance = model.feature_importances_

feat_df = pd.DataFrame({
    "Feature": input_data.columns,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)

fig_imp = px.bar(
    feat_df.head(10),
    x="Importance",
    y="Feature",
    orientation="h"
)

st.plotly_chart(fig_imp, use_container_width=True)

# =========================================================
# Upgrade 4 — Upload Sensor File
# =========================================================

st.subheader("📂 Upload Sensor Data")

uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file:
    df_upload = pd.read_csv(uploaded_file)
    scaled = scaler.transform(df_upload)
    preds = model.predict(scaled)
    st.write("Predicted RUL:", preds)

# =========================================================
# Upgrade 5 — Maintenance Recommendation
# =========================================================

st.subheader("🛠️ Maintenance Recommendation")

if risk == "Low":
    rec = "Routine monitoring sufficient."
elif risk == "Medium":
    rec = "Schedule inspection soon."
else:
    rec = "Immediate maintenance required!"

st.success(rec)

# =========================================================
# Sensor Snapshot
# =========================================================

st.subheader("📈 Sensor Snapshot")

sensor_df = pd.DataFrame({
    "Sensor": [f"S{i}" for i in sensor_list],
    "Value": sensor_values
})

fig2 = px.bar(sensor_df, x="Sensor", y="Value")

st.plotly_chart(fig2, use_container_width=True)

# =========================================================
# Footer
# =========================================================

st.markdown("---")
st.markdown("Industrial Predictive Maintenance Demo 🇩🇪")