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
AIR_DENSITY_SITE = 1.15   # assumed lower density vs standard

# SIDEBAR
st.sidebar.subheader("Upload SCADA File")

uploaded_file = st.sidebar.file_uploader("Upload Site CSV", type=["csv"])

if uploaded_file is None:
    st.warning("Please upload a SCADA CSV file")
    st.stop()

# SITE
site = st.sidebar.selectbox(
    "Select Site for Reference Curve",
    ["CIP Hatalageri","JSW Tuljapur","Blupine Sagapara","Kalavad GJ","Kalavad_PH2"]
)

# LOAD SCADA
@st.cache_data
def load_scada(file):
    df = pd.read_csv(file,low_memory=False)
    df.columns = df.columns.str.strip()

    wind_col = [c for c in df.columns if "wind" in c.lower()][0]
    power_col = [c for c in df.columns if "power" in c.lower() or "active" in c.lower()][0]
    time_col = [c for c in df.columns if "time" in c.lower()][0]

    df[time_col] = pd.to_datetime(df[time_col],errors="coerce")
    df[wind_col] = pd.to_numeric(df[wind_col],errors="coerce")
    df[power_col] = pd.to_numeric(df[power_col],errors="coerce")

    df = df.dropna(subset=[wind_col,power_col,time_col])
    df["Name"] = df["Name"].astype(str).str.strip()

    return df,wind_col,power_col,time_col

df,wind_col,power_col,time_col = load_scada(uploaded_file)

# DATE FILTER
st.sidebar.subheader("Date Filter")

period = st.sidebar.selectbox(
    "Select Period",
    ["Custom","Last 15 Days","Weekly","Monthly"]
)

end_date = df[time_col].max()

if period == "Last 15 Days":
    start_date = end_date - timedelta(days=15)
elif period == "Weekly":
    start_date = end_date - timedelta(days=7)
elif period == "Monthly":
    start_date = end_date - timedelta(days=30)
else:
    date_range = st.sidebar.date_input(
        "Custom Date Range",
        [df[time_col].min().date(),df[time_col].max().date()]
    )
    start_date = pd.to_datetime(date_range[0])
    end_date = pd.to_datetime(date_range[1])

df = df[(df[time_col] >= start_date) & (df[time_col] <= end_date)]

# LOAD REFERENCE (SAFE)
@st.cache_data
def load_reference(site):
    ref_raw = pd.read_excel(REF_FILE,header=None)

    location=None
    for r in range(ref_raw.shape[0]):
        for c in range(ref_raw.shape[1]):
            if site.lower() in str(ref_raw.iloc[r,c]).lower():
                location=(r,c)
                break
        if location:
            break

    if location is None:
        st.error("Site not found")
        st.stop()

    r,c = location
    ref = ref_raw.iloc[r+2:r+60,[c-1,c+3]].copy()
    ref.columns=["WindSpeed","RefPower"]

    ref["WindSpeed"] = pd.to_numeric(ref["WindSpeed"],errors="coerce")
    ref["RefPower"] = pd.to_numeric(ref["RefPower"],errors="coerce")
    ref = ref.dropna().sort_values("WindSpeed")

    wind_bins = np.arange(3,25.5,BIN_SIZE)
    ref_interp = np.interp(wind_bins, ref["WindSpeed"], ref["RefPower"])

    return pd.DataFrame({"WindBin":wind_bins,"RefPower":ref_interp})

ref_curve = load_reference(site)

# TURBINE PROCESS
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

    return df_t,merged,avg_dev

# SUMMARY
results=[]
for t in df["Name"].unique():
    res = process_turbine(t)
    if res:
        _,_,dev = res
        results.append({"Turbine":t,"Deviation_%":dev})

results_df = pd.DataFrame(results)

# STATUS TAG (±2%)
def get_status(dev):
    if dev < -2:
        return "🔴 Underperforming"
    elif dev > 2:
        return "🟡 Overperforming"
    else:
        return "🟢 Normal"

results_df["Status"] = results_df["Deviation_%"].apply(get_status)

# KPI
st.subheader(f"{site} Performance ({start_date.date()} → {end_date.date()})")

col1,col2,col3 = st.columns(3)
col1.metric("Total Turbines",len(results_df))
col2.metric("Avg Deviation",round(results_df["Deviation_%"].mean(),2))
col3.metric("Worst Deviation",round(results_df["Deviation_%"].min(),2))

# MODE
mode = st.radio("Display Mode",["Single","Compare","All"])

# ALL (MAIN UPGRADE)
if mode=="All":
    cols = st.columns(2)
    i=0

    for t in results_df["Turbine"]:
        df_f,merged,avg_dev = process_turbine(t)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_f[wind_col],y=df_f[power_col],
                                 mode='markers',marker=dict(size=3,opacity=0.4)))
        fig.add_trace(go.Scatter(x=merged["WindBin"],y=merged["AvgPower"],
                                 mode='lines+markers'))
        fig.add_trace(go.Scatter(x=merged["WindBin"],y=merged["RefPower"],
                                 mode='lines',line=dict(dash='dash')))

        # COLOR BASED TAG
        color = "green"
        if avg_dev < -2:
            color = "red"
        elif avg_dev > 2:
            color = "orange"

        # AIR DENSITY COMMENT
        comment = ""
        if avg_dev < -2:
            comment = "Low power due to lower air density / stalling region"
        elif avg_dev > 2:
            comment = "Higher than expected output (possible measurement variation)"
        else:
            comment = "Performance within expected range"

        fig.update_layout(
            title=f"{t} | Dev {round(avg_dev,1)}% | {comment}",
            height=350,
            title_font=dict(color=color)
        )

        cols[i%2].plotly_chart(fig,use_container_width=True)
        i+=1

# TABLE
st.subheader("Turbine Ranking")
st.dataframe(results_df.sort_values("Deviation_%"))
