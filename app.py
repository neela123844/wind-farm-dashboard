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
AIR_DENSITY_STD = 1.225  # kg/m3

# SIDEBAR
st.sidebar.subheader("Upload SCADA File")

uploaded_file = st.sidebar.file_uploader("Upload Site CSV", type=["csv"])

if uploaded_file is None:
    st.warning("Please upload a SCADA CSV file")
    st.stop()

# SITE SELECTOR
site = st.sidebar.selectbox(
    "Select Site for Reference Curve",
    ["CIP Hatalageri","JSW Tuljapur","Blupine Sagapara","Kalavad GJ","Kalavad_PH2","AMP_Energy","Wanki","CleanMax Motadevaliya",
     "Ayana Amerli","Mahadev PH1","Blupine-I, Ambada-GJ","ACME Shapar","FP_Kudligi","Sprng TN","Otha Pithalpur-GJ",
     "AMGEPL,Kurnool AP","ReNew1_Gadag","partner Ottapidaum","Cleanmax Motadevaliya","Cleanmax SANATHALI","Cleanmax Babra",
     "RenfraEnergy Trichy","RENEW-03 Sholapur","Renew2 Chandwad","ReNew-4 Patoda","Clean max Jagalur","Sembcorp Tuticorin",
     "Renew-4 Kudligi","Renew Otha","Cleanmax Honavad"," Blueleaf Agar","JSW_Sandur","India_Hero_Doni"]
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
st.sidebar.subheader("Immediate Date Filter")

period = st.sidebar.selectbox("Select Period", ["Custom","Last 15 Days","Weekly","Monthly"])

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
        [df[time_col].min().date(), df[time_col].max().date()]
    )
    start_date = pd.to_datetime(date_range[0])
    end_date = pd.to_datetime(date_range[1])

df = df[(df[time_col] >= start_date) & (df[time_col] <= end_date)]

# LOAD REFERENCE
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

    wind_ref_col = c-1
    power_ref_col = c+3

    ref = ref_raw.iloc[r+2:r+60,[wind_ref_col,power_ref_col]].copy()
    ref.columns=["WindSpeed","RefPower"]

    ref["WindSpeed"] = pd.to_numeric(ref["WindSpeed"],errors="coerce")
    ref["RefPower"] = pd.to_numeric(ref["RefPower"],errors="coerce")

    ref = ref.dropna().sort_values("WindSpeed")

    wind_bins = np.arange(3,25.5,BIN_SIZE)

    ref_interp = np.interp(wind_bins, ref["WindSpeed"], ref["RefPower"])

    ref_curve = pd.DataFrame({
        "WindBin":wind_bins,
        "RefPower":ref_interp
    })

    return ref_curve

ref_curve = load_reference(site)

# NEW: ANALYSIS FUNCTION
def analyze_performance(merged):
    stalling_zone = merged[(merged["Deviation_%"] < -15) & (merged["WindBin"] < RATED_SPEED)]
    stall_bins = stalling_zone["WindBin"].tolist()

    reason = []
    if len(stall_bins) > 0:
        reason.append("Possible aerodynamic stalling or pitch issue")

    high_dev = merged["Deviation_%"].mean()
    if high_dev < -10:
        reason.append("Low output vs reference → possible air density or blade issue")

    return stall_bins, ", ".join(reason)

# TURBINE PROCESSING
def process_turbine(turbine):
    df_t = df[df["Name"]==turbine].copy()

    df_t = df_t[
        (df_t[wind_col]>=3)&
        (df_t[wind_col]<=25)&
        (df_t[power_col]>0)
    ]

    if len(df_t)<30:
        return None

    df_t["WindBin"] = (df_t[wind_col]/BIN_SIZE).round()*BIN_SIZE

    actual = (
        df_t.groupby("WindBin")
        .agg(AvgPower=(power_col,"mean"))
        .reset_index()
    )

    merged = ref_curve.merge(actual,on="WindBin",how="left")

    valid = merged["AvgPower"].notna()

    if valid.sum()>7:
        merged.loc[valid,"AvgPower"] = savgol_filter(
            merged.loc[valid,"AvgPower"],7,2
        )

    merged["Deviation_%"] = (
        (merged["AvgPower"]-merged["RefPower"])
        /merged["RefPower"]
    )*100

    avg_dev = merged["Deviation_%"].mean(skipna=True)

    # NEW ANALYSIS
    stall_bins, reason = analyze_performance(merged)

    return df_t, merged, avg_dev, stall_bins, reason

# SITE SUMMARY
site_results=[]

for turbine in df["Name"].unique():
    result = process_turbine(turbine)

    if result:
        _,_,avg_dev,_,_ = result
        site_results.append({"Turbine":turbine,"Deviation_%":avg_dev})

results_df = pd.DataFrame(site_results)

if results_df.empty:
    st.warning("No sufficient data available")
    st.stop()

results_df["Status"] = np.where(
    results_df["Deviation_%"]<-2,"Underperforming",
    np.where(results_df["Deviation_%"]>2,"Overperforming","Within Limit")
)

# KPI
st.subheader(f"{site} Site Performance ({start_date.date()} to {end_date.date()})")

col1,col2,col3,col4 = st.columns(4)
col1.metric("Total Turbines",len(results_df))
col2.metric("Underperforming",len(results_df[results_df["Status"]=="Underperforming"]))
col3.metric("Overperforming",len(results_df[results_df["Status"]=="Overperforming"]))
col4.metric("Within Limit",len(results_df[results_df["Status"]=="Within Limit"]))

# BAR CHART (NEW)
st.subheader("Deviation Overview")
bar_fig = go.Figure()
bar_fig.add_trace(go.Bar(
    x=results_df["Turbine"],
    y=results_df["Deviation_%"]
))
st.plotly_chart(bar_fig,use_container_width=True)

# MODE
mode = st.radio(
    "Display Mode",
    ["Show Single Turbine","Compare Two Turbines","Show All Turbines"]
)

# SINGLE
if mode=="Show Single Turbine":
    selected = st.selectbox("Select Turbine",results_df["Turbine"])
    df_filtered,merged,avg_dev,stall_bins,reason = process_turbine(selected)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_filtered[wind_col],y=df_filtered[power_col],mode='markers',name="SCADA"))
    fig.add_trace(go.Scatter(x=merged["WindBin"],y=merged["AvgPower"],mode='lines+markers',name="Actual"))
    fig.add_trace(go.Scatter(x=merged["WindBin"],y=merged["RefPower"],mode='lines',name="Reference"))

    st.plotly_chart(fig,use_container_width=True)

    # INSIGHTS (NEW)
    st.subheader("Performance Insights")

    st.write(f" Data Period: {start_date.date()} → {end_date.date()}")
    st.write(f" Average Deviation: {round(avg_dev,2)} %")

    if stall_bins:
        st.error(f" Stalling observed at wind speeds: {stall_bins}")

    if reason:
        st.warning(f"Reason: {reason}")

    # AIR DENSITY EFFECT
    st.info(f"Air density assumption: {AIR_DENSITY_STD} kg/m³. Lower density → lower power output → negative deviation.")

# TABLE
st.subheader("Turbine Ranking")
results_df = results_df.sort_values("Deviation_%")
st.dataframe(results_df)
