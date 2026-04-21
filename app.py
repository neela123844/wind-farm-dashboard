import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import savgol_filter
from datetime import timedelta
import os

st.set_page_config(layout="wide")

# ---------------- LOGO ----------------
logo_path = os.path.join(os.path.dirname(__file__), "Envision.png")
if os.path.exists(logo_path):
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.image(logo_path, width=300)

st.title("Wind Farm Performance Analytics Dashboard")

# ---------------- FILE ----------------
REF_FILE = os.path.join(os.path.dirname(__file__),
                       "India site Standard & Theoretical PC data 1234.xlsx")

BIN_SIZE = 0.5
TOLERANCE = 2.0
RATED_POWER = 3400.0

# ---------------- SIDEBAR ----------------
uploaded_file = st.sidebar.file_uploader("Upload SCADA CSV", type=["csv"])

if uploaded_file is None:
    st.warning("Please upload SCADA file")
    st.stop()

site = st.sidebar.selectbox(
    "Select Site",
    ["CIP Hatalageri","JSW Tuljapur","Blupine Sagapara","Kalavad GJ","Kalavad_PH2",
     "AMP_Energy","Wanki","CleanMax Motadevaliya","Ayana Amerli"]
)

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
    df["Name"] = df["Name"].astype(str).str.strip()

    status_cols = [c for c in df.columns if any(x in c.lower()
                    for x in ["status","alarm","derate","limit","curtail","temp","pitch"])]

    return df, wind_col, power_col, time_col, status_cols

df, wind_col, power_col, time_col, status_cols = load_scada(uploaded_file)

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

# ---------------- REFERENCE ----------------
@st.cache_data
def load_reference(site):
    ref_raw = pd.read_excel(REF_FILE, header=None)

    location=None
    for r in range(ref_raw.shape[0]):
        for c in range(ref_raw.shape[1]):
            if site.lower() in str(ref_raw.iloc[r,c]).lower():
                location=(r,c)
                break
        if location:
            break

    if location is None:
        st.error("Site not found in reference file")
        st.stop()

    r,c = location

    ref = ref_raw.iloc[r+2:r+60,[c-1,c+3]].copy()
    ref.columns=["WindSpeed","RefPower"]

    ref = ref.dropna()
    ref["WindSpeed"] = pd.to_numeric(ref["WindSpeed"])
    ref["RefPower"] = pd.to_numeric(ref["RefPower"])

    wind_bins = np.arange(3,25.5,BIN_SIZE)
    ref_interp = np.interp(wind_bins, ref["WindSpeed"], ref["RefPower"])

    return pd.DataFrame({"WindBin":wind_bins,"RefPower":ref_interp})

ref_curve = load_reference(site)

# ---------------- PROCESS ----------------
def process_turbine(t):
    df_t = df[df["Name"]==t].copy()

    df_t = df_t[(df_t[wind_col]>=3)&(df_t[wind_col]<=25)&(df_t[power_col]>0)]

    if len(df_t)<30:
        return None

    expected_points = ((end_date - start_date).total_seconds() / 600)
    availability = (len(df_t) / expected_points) * 100
    std_dev = df_t[power_col].std()

    df_t["WindBin"] = (df_t[wind_col]/BIN_SIZE).round()*BIN_SIZE

    actual = df_t.groupby("WindBin").agg(AvgPower=(power_col,"mean")).reset_index()
    merged = ref_curve.merge(actual,on="WindBin",how="left")

    valid = merged["AvgPower"].notna()
    if valid.sum()>7:
        merged.loc[valid,"AvgPower"] = savgol_filter(merged.loc[valid,"AvgPower"],7,2)

    merged["Deviation_%"] = ((merged["AvgPower"]-merged["RefPower"])/merged["RefPower"])*100
    avg_dev = merged["Deviation_%"].mean(skipna=True)

    return df_t, merged, avg_dev

# ---------------- COMPARISON GRAPH ----------------
st.subheader("Turbine Comparison (Single Graph)")

selected_turbines = st.multiselect(
    "Select turbines to compare",
    df["Name"].unique()
)

if selected_turbines:
    fig_compare = go.Figure()

    # Reference curve (only once)
    fig_compare.add_trace(go.Scatter(
        x=ref_curve["WindBin"],
        y=ref_curve["RefPower"],
        mode='lines',
        name='Reference',
        line=dict(dash='dash', color='black')
    ))

    for t in selected_turbines:
        res = process_turbine(t)
        if not res:
            continue

        df_t, merged, avg_dev = res

        # Scatter
        fig_compare.add_trace(go.Scatter(
            x=df_t[wind_col],
            y=df_t[power_col],
            mode='markers',
            name=f"{t} Raw",
            opacity=0.3
        ))

        # Avg curve
        fig_compare.add_trace(go.Scatter(
            x=merged["WindBin"],
            y=merged["AvgPower"],
            mode='lines+markers',
            name=f"{t} Avg"
        ))

    fig_compare.update_layout(
        height=500,
        title="Turbine Comparison Curve",
        xaxis_title="Wind Speed",
        yaxis_title="Power"
    )

    st.plotly_chart(fig_compare, use_container_width=True)
