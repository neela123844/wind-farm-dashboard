import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import savgol_filter
from datetime import timedelta
import os

st.set_page_config(layout="wide")

# LOGO
logo_path = os.path.join(os.path.dirname(__file__), "Envision.png")
col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.image(logo_path, width=300)

st.title("Wind Farm Performance Analytics Dashboard")

REF_FILE = "India site Standard & Theoretical PC data 123.xlsx"

BIN_SIZE = 0.5
RATED_SPEED = 10.0
AIR_DENSITY_STD = 1.225

# SIDEBAR
uploaded_file = st.sidebar.file_uploader("Upload SCADA CSV", type=["csv"])

if uploaded_file is None:
    st.warning("Upload SCADA file")
    st.stop()

site = st.sidebar.selectbox(
    "Select Site",
    ["CIP Hatalageri","JSW Tuljapur","Blupine Sagapara","Kalavad GJ"]
)

# LOAD SCADA
@st.cache_data
def load_scada(file):
    df = pd.read_csv(file, low_memory=False)
    df.columns = df.columns.str.strip()

    wind_col = [c for c in df.columns if "wind" in c.lower()][0]
    power_col = [c for c in df.columns if "power" in c.lower()][0]
    time_col = [c for c in df.columns if "time" in c.lower()][0]

    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df[wind_col] = pd.to_numeric(df[wind_col], errors="coerce")
    df[power_col] = pd.to_numeric(df[power_col], errors="coerce")

    df = df.dropna()
    df["Name"] = df["Name"].astype(str)

    return df, wind_col, power_col, time_col

df, wind_col, power_col, time_col = load_scada(uploaded_file)

# DATE FILTER
end_date = df[time_col].max()
start_date = end_date - timedelta(days=15)
df = df[(df[time_col] >= start_date) & (df[time_col] <= end_date)]

# FIXED REFERENCE FUNCTION ✅
@st.cache_data
def load_reference(site):
    ref_raw = pd.read_excel(REF_FILE, header=None)

    location = None
    for r in range(ref_raw.shape[0]):
        for c in range(ref_raw.shape[1]):
            if site.lower() in str(ref_raw.iloc[r, c]).lower():
                location = (r, c)
                break
        if location:
            break

    if location is None:
        st.error("Site not found")
        st.stop()

    r, c = location

    wind_col_ref = c - 1
    power_col_ref = c + 3

    ref = ref_raw.iloc[r+2:r+60, [wind_col_ref, power_col_ref]].copy()
    ref.columns = ["WindSpeed", "RefPower"]

    # CLEAN DATA (IMPORTANT FIX)
    ref["WindSpeed"] = pd.to_numeric(ref["WindSpeed"], errors="coerce")
    ref["RefPower"] = pd.to_numeric(ref["RefPower"], errors="coerce")
    ref = ref.dropna()

    if len(ref) < 5:
        st.error("Reference data invalid")
        st.stop()

    ref = ref.sort_values("WindSpeed")

    wind_bins = np.arange(3, 25.5, BIN_SIZE)

    ref_interp = np.interp(
        wind_bins,
        ref["WindSpeed"].values,
        ref["RefPower"].values
    )

    return pd.DataFrame({"WindBin": wind_bins, "RefPower": ref_interp})

ref_curve = load_reference(site)

# ANALYSIS
def process_turbine(turbine):
    df_t = df[df["Name"] == turbine].copy()

    df_t = df_t[(df_t[wind_col] >= 3) & (df_t[power_col] > 0)]

    if len(df_t) < 30:
        return None

    df_t["WindBin"] = (df_t[wind_col]/BIN_SIZE).round()*BIN_SIZE

    actual = df_t.groupby("WindBin").agg(AvgPower=(power_col,"mean")).reset_index()
    merged = ref_curve.merge(actual, on="WindBin", how="left")

    valid = merged["AvgPower"].notna()
    if valid.sum() > 7:
        merged.loc[valid,"AvgPower"] = savgol_filter(merged.loc[valid,"AvgPower"],7,2)

    merged["Deviation_%"] = ((merged["AvgPower"] - merged["RefPower"]) / merged["RefPower"]) * 100
    avg_dev = merged["Deviation_%"].mean(skipna=True)

    stall = merged[(merged["Deviation_%"] < -15) & (merged["WindBin"] < RATED_SPEED)]["WindBin"].tolist()

    return df_t, merged, avg_dev, stall

# SUMMARY
results = []
for t in df["Name"].unique():
    res = process_turbine(t)
    if res:
        _,_,d,_ = res
        results.append({"Turbine":t,"Deviation_%":d})

results_df = pd.DataFrame(results)

# BAR GRAPH
st.subheader("Deviation Overview")
fig_bar = go.Figure()
fig_bar.add_trace(go.Bar(x=results_df["Turbine"], y=results_df["Deviation_%"]))
st.plotly_chart(fig_bar, use_container_width=True)

# MODE
mode = st.radio("Mode", ["Single","Compare","All"])

# SINGLE
if mode == "Single":
    t = st.selectbox("Select Turbine", results_df["Turbine"])
    df_f, merged, avg_dev, stall = process_turbine(t)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_f[wind_col], y=df_f[power_col], mode='markers'))
    fig.add_trace(go.Scatter(x=merged["WindBin"], y=merged["AvgPower"], mode='lines+markers'))
    fig.add_trace(go.Scatter(x=merged["WindBin"], y=merged["RefPower"], mode='lines'))

    st.plotly_chart(fig, use_container_width=True)

    # Deviation Graph
    st.subheader("Deviation vs Wind Speed")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=merged["WindBin"], y=merged["Deviation_%"], mode='lines+markers'))
    fig2.add_hline(y=0)
    st.plotly_chart(fig2, use_container_width=True)

# ALL TURBINES (OLD STYLE BACK ✅)
else:
    cols = st.columns(2)
    i = 0

    for t in results_df["Turbine"]:
        df_f, merged, avg_dev, stall = process_turbine(t)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_f[wind_col], y=df_f[power_col],
                                 mode='markers', marker=dict(size=3, opacity=0.4)))
        fig.add_trace(go.Scatter(x=merged["WindBin"], y=merged["AvgPower"], mode='lines+markers'))
        fig.add_trace(go.Scatter(x=merged["WindBin"], y=merged["RefPower"],
                                 mode='lines', line=dict(dash='dash')))

        comment = ""
        if avg_dev < -10:
            comment = "❌ Underperformance"
        elif avg_dev > 10:
            comment = "⚡ Overperformance"

        if stall:
            comment += " | ⚠️ Stalling"

        fig.update_layout(
            title=f"{t} | Dev {round(avg_dev,1)}% {comment}",
            height=350
        )

        cols[i%2].plotly_chart(fig, use_container_width=True)
        i += 1

# TABLE
st.subheader("Ranking")
st.dataframe(results_df.sort_values("Deviation_%"))
