import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta
import os

# ML
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

st.set_page_config(layout="wide")

# ---------------- LOGO ----------------
logo_path = os.path.join(os.path.dirname(__file__), "Envision.png")
col1, col2, col3 = st.columns([1,2,1])
with col2:
    if os.path.exists(logo_path):
        st.image(logo_path, width=300)

st.title("Wind Farm Performance Analytics (GP + SPRT)")

# ---------------- CONSTANTS ----------------
RATED_POWER = 3400.0

# ---------------- SIDEBAR ----------------
uploaded_file = st.sidebar.file_uploader("Upload SCADA CSV", type=["csv"])

if uploaded_file is None:
    st.warning("Please upload SCADA file")
    st.stop()

# ---------------- LOAD SCADA ----------------
@st.cache_data
def load_scada(file):
    df = pd.read_csv(file, low_memory=False)
    df.columns = df.columns.str.strip()

    wind_col = [c for c in df.columns if "wind" in c.lower()][0]
    power_col = [c for c in df.columns if "power" in c.lower() or "active" in c.lower()][0]
    time_col = [c for c in df.columns if "time" in c.lower()][0]

    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df[wind_col] = pd.to_numeric(df[wind_col], errors="coerce")
    df[power_col] = pd.to_numeric(df[power_col], errors="coerce")

    df = df.dropna(subset=[wind_col, power_col, time_col])

    if "Name" not in df.columns:
        df["Name"] = "Turbine-1"

    df["Name"] = df["Name"].astype(str).str.strip()

    return df, wind_col, power_col, time_col

df, wind_col, power_col, time_col = load_scada(uploaded_file)

# ---------------- DATE FILTER ----------------
period = st.sidebar.selectbox("Period", ["Last 15 Days","Weekly","Monthly"])

end_date = df[time_col].max()

if period == "Last 15 Days":
    start_date = end_date - timedelta(days=15)
elif period == "Weekly":
    start_date = end_date - timedelta(days=7)
else:
    start_date = end_date - timedelta(days=30)

df = df[(df[time_col] >= start_date) & (df[time_col] <= end_date)]

# ---------------- GP MODEL ----------------
@st.cache_data
def train_gp_model(df_t, wind_col, power_col):

    X = df_t[[wind_col]].copy()

    # Optional features (if exist)
    optional_cols = ["Wind Direction", "Pitch", "Yaw", "Rotor Speed"]

    for col in optional_cols:
        if col in df_t.columns:
            X[col] = pd.to_numeric(df_t[col], errors="coerce")

    y = df_t[power_col]

    valid = X.notna().all(axis=1) & y.notna()
    X = X[valid]
    y = y[valid]

    kernel = RBF(length_scale=1.0) + WhiteKernel()

    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-2)
    gp.fit(X, y)

    return gp, X.columns.tolist()

# ---------------- SPRT ----------------
def sprt_test(residuals, mu0, sigma0, alpha=0.05, beta=0.05):

    A = np.log(beta / (1 - alpha))
    B = np.log((1 - beta) / alpha)

    mu1_up = mu0 + 3
    mu1_down = mu0 - 3
    sigma1 = sigma0 * 1.5

    logR_up = 0
    logR_down = 0

    for e in residuals:

        logR_up += ((e - mu1_up)**2 / (2*sigma1**2)) - ((e - mu0)**2 / (2*sigma0**2))
        logR_down += ((e - mu1_down)**2 / (2*sigma1**2)) - ((e - mu0)**2 / (2*sigma0**2))

        if logR_up >= B or logR_down >= B:
            return True

    return False

# ---------------- PROCESS TURBINE ----------------
def process_turbine(t):

    df_t = df[df["Name"] == t].copy()

    df_t = df_t[(df_t[wind_col] >= 3) & (df_t[wind_col] <= 25)]

    if len(df_t) < 50:
        return None

    gp, feature_cols = train_gp_model(df_t, wind_col, power_col)

    X_test = df_t[feature_cols]
    y_actual = df_t[power_col]

    y_pred = gp.predict(X_test)

    residuals = y_actual - y_pred

    mu0 = residuals.mean()
    sigma0 = residuals.std()

    alarm = sprt_test(residuals.values, mu0, sigma0)

    deviation = (residuals.mean() / RATED_POWER) * 100

    expected_points = ((end_date - start_date).total_seconds() / 600)
    availability = (len(df_t) / expected_points) * 100

    std_dev = y_actual.std()

    return df_t, y_pred, deviation, alarm, availability, std_dev

# ---------------- SUMMARY ----------------
results = []

for t in df["Name"].unique():
    res = process_turbine(t)
    if res:
        _,_,dev,alarm,_,_ = res
        results.append({
            "Turbine": t,
            "Deviation_%": dev,
            "Alarm": "Yes" if alarm else "No"
        })

results_df = pd.DataFrame(results)

# ---------------- HEADER ----------------
st.subheader(f"Performance ({start_date.date()} → {end_date.date()})")

# ---------------- BAR CHART ----------------
colors = ["red" if a=="Yes" else "green" for a in results_df["Alarm"]]

fig_bar = go.Figure()
fig_bar.add_trace(go.Bar(
    x=results_df["Turbine"],
    y=results_df["Deviation_%"],
    marker_color=colors
))

st.plotly_chart(fig_bar, use_container_width=True)

# ---------------- ALL TURBINES ----------------
st.subheader("Detailed Analysis")

cols = st.columns(2)
i = 0

for t in results_df["Turbine"]:

    df_f, y_pred, dev, alarm, availability, std_dev = process_turbine(t)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_f[wind_col],
        y=df_f[power_col],
        mode='markers',
        marker=dict(size=3, opacity=0.4),
        name="Actual"
    ))

    fig.add_trace(go.Scatter(
        x=df_f[wind_col],
        y=y_pred,
        mode='markers',
        marker=dict(size=3, color='green'),
        name="GP Prediction"
    ))

    # COMMENT
    comment = ""

    if alarm:
        comment += "🚨 SPRT Alarm Detected\n"

    if dev < -2:
        comment += "Underperformance\n"
    elif dev > 2:
        comment += "Overperformance\n"
    else:
        comment += "Normal\n"

    comment += f"\nAvailability: {round(availability,1)}%"
    comment += f"\nStd Dev: {round(std_dev,2)}"

    color = "red" if alarm else "green"

    fig.update_layout(
        title=f"{t} | Dev: {round(dev,1)}%",
        title_font=dict(color=color),
        height=350
    )

    cols[i%2].plotly_chart(fig, use_container_width=True)
    cols[i%2].markdown(f"```\n{comment}\n```")

    i += 1

# ---------------- TABLE ----------------
st.subheader("Ranking")
st.dataframe(results_df.sort_values("Deviation_%")
