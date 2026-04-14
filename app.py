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
TOLERANCE = 2.0
RATED_SPEED = 10.0
RATED_POWER = 3400.0
AIR_DENSITY_STD = 1.225

# SIDEBAR
st.sidebar.subheader("Upload SCADA File")
uploaded_file = st.sidebar.file_uploader("Upload Site CSV", type=["csv"])

if uploaded_file is None:
    st.warning("Please upload a SCADA CSV file")
    st.stop()

# SITE
site = st.sidebar.selectbox(
    "Select Site for Reference Curve",
    ["CIP Hatalageri","JSW Tuljapur","Blupine Sagapara","Kalavad GJ","Kalavad_PH2","AMP_Energy","Wanki","CleanMax Motadevaliya"]
)

# LOAD SCADA
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

    return df, wind_col, power_col, time_col

df, wind_col, power_col, time_col = load_scada(uploaded_file)

# DATE FILTER
period = st.sidebar.selectbox("Select Period", ["Last 15 Days","Weekly","Monthly"])

end_date = df[time_col].max()

if period == "Last 15 Days":
    start_date = end_date - timedelta(days=15)
elif period == "Weekly":
    start_date = end_date - timedelta(days=7)
else:
    start_date = end_date - timedelta(days=30)

df = df[(df[time_col] >= start_date) & (df[time_col] <= end_date)]

# LOAD REFERENCE
@st.cache_data
def load_reference(site):
    ref_raw = pd.read_excel(REF_FILE, header=None)

    for r in range(ref_raw.shape[0]):
        for c in range(ref_raw.shape[1]):
            if site.lower() in str(ref_raw.iloc[r,c]).lower():
                wind_ref_col = c-1
                power_ref_col = c+3

                ref = ref_raw.iloc[r+2:r+60,[wind_ref_col,power_ref_col]].copy()
                ref.columns=["WindSpeed","RefPower"]

                ref = ref.dropna().sort_values("WindSpeed")

                wind_bins = np.arange(3,25.5,BIN_SIZE)
                ref_interp = np.interp(wind_bins, ref["WindSpeed"], ref["RefPower"])

                return pd.DataFrame({"WindBin":wind_bins,"RefPower":ref_interp})

ref_curve = load_reference(site)

# ANALYSIS
def analyze_performance(merged):
    stalling = merged[(merged["Deviation_%"] < -15) & (merged["WindBin"] < RATED_SPEED)]
    return stalling["WindBin"].tolist()

# PROCESS
def process_turbine(turbine):
    df_t = df[df["Name"]==turbine].copy()

    df_t = df_t[(df_t[wind_col]>=3)&(df_t[wind_col]<=25)&(df_t[power_col]>0)]

    if len(df_t)<30:
        return None

    df_t["WindBin"] = (df_t[wind_col]/BIN_SIZE).round()*BIN_SIZE

    actual = df_t.groupby("WindBin").agg(AvgPower=(power_col,"mean")).reset_index()

    merged = ref_curve.merge(actual,on="WindBin",how="left")

    valid = merged["AvgPower"].notna()
    if valid.sum()>7:
        merged.loc[valid,"AvgPower"] = savgol_filter(merged.loc[valid,"AvgPower"],7,2)

    merged["Deviation_%"] = ((merged["AvgPower"]-merged["RefPower"])/merged["RefPower"])*100
    avg_dev = merged["Deviation_%"].mean(skipna=True)

    stall_bins = analyze_performance(merged)

    return df_t, merged, avg_dev, stall_bins

# SUMMARY
site_results=[]
for turbine in df["Name"].unique():
    res = process_turbine(turbine)
    if res:
        _,_,avg_dev,_ = res
        site_results.append({"Turbine":turbine,"Deviation_%":avg_dev})

results_df = pd.DataFrame(site_results)

# BAR
st.subheader("Deviation Overview")
fig_bar = go.Figure()
fig_bar.add_trace(go.Bar(x=results_df["Turbine"], y=results_df["Deviation_%"]))
st.plotly_chart(fig_bar, use_container_width=True)

# MODE
mode = st.radio("Display Mode", ["Single","Compare","All"])

# SINGLE
if mode=="Single":
    t = st.selectbox("Turbine", results_df["Turbine"])
    df_f, merged, avg_dev, stall = process_turbine(t)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_f[wind_col],y=df_f[power_col],mode='markers',name="SCADA"))
    fig.add_trace(go.Scatter(x=merged["WindBin"],y=merged["AvgPower"],mode='lines+markers',name="Actual"))
    fig.add_trace(go.Scatter(x=merged["WindBin"],y=merged["RefPower"],mode='lines',name="Reference"))

    st.plotly_chart(fig, use_container_width=True)

    # NEW GRAPH
    st.subheader("Deviation vs Wind Speed")
    dev_fig = go.Figure()
    dev_fig.add_trace(go.Scatter(x=merged["WindBin"],y=merged["Deviation_%"],mode='lines+markers'))
    dev_fig.add_hline(y=0)
    st.plotly_chart(dev_fig, use_container_width=True)

# COMPARE
elif mode=="Compare":
    t1 = st.selectbox("T1", results_df["Turbine"])
    t2 = st.selectbox("T2", results_df["Turbine"], index=1)

    _,m1,_,_ = process_turbine(t1)
    _,m2,_,_ = process_turbine(t2)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=m1["WindBin"],y=m1["AvgPower"],name=t1))
    fig.add_trace(go.Scatter(x=m2["WindBin"],y=m2["AvgPower"],name=t2))
    fig.add_trace(go.Scatter(x=m1["WindBin"],y=m1["RefPower"],name="Reference",line=dict(dash='dash')))
    st.plotly_chart(fig, use_container_width=True)

# ALL (🔥 YOUR OLD FEATURE BACK)
else:
    st.subheader("All Turbine Performance")

    cols = st.columns(2)
    i=0

    for turbine in results_df["Turbine"]:
        df_f, merged, avg_dev, stall = process_turbine(turbine)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_f[wind_col],y=df_f[power_col],
                                 mode='markers',marker=dict(size=3,opacity=0.4)))
        fig.add_trace(go.Scatter(x=merged["WindBin"],y=merged["AvgPower"],mode='lines+markers'))
        fig.add_trace(go.Scatter(x=merged["WindBin"],y=merged["RefPower"],
                                 mode='lines',line=dict(dash='dash')))

        comment = ""
        if avg_dev < -10:
            comment = "Underperformance"
        elif avg_dev > 10:
            comment = " Overperformance"

        if stall:
            comment += " |  Stalling"

        fig.update_layout(
            title=f"{turbine} | Dev {round(avg_dev,1)}% {comment}",
            height=350
        )

        cols[i%2].plotly_chart(fig, use_container_width=True)
        i+=1

# TABLE
st.subheader("Ranking")
st.dataframe(results_df.sort_values("Deviation_%"))
