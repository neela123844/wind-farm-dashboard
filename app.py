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

REF_FILE = "India site Standard & Theoretical PC data 1234.xlsx"

BIN_SIZE = 0.5
TOLERANCE = 2.0
RATED_POWER = 3400.0

# SIDEBAR
uploaded_file = st.sidebar.file_uploader("Upload SCADA CSV", type=["csv"])

if uploaded_file is None:
    st.warning("Please upload SCADA file")
    st.stop()

site = st.sidebar.selectbox(
    "Select Site for Reference Curve",
    ["CIP Hatalageri","JSW Tuljapur","Blupine Sagapara","Kalavad GJ","Kalavad_PH2"]
)

# ---------------- DISPLAY MODE ----------------
mode = st.sidebar.radio(
    "Select View",
    ["Single Turbine", "Compare Turbines", "Show All Turbines"]
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

    df = df.dropna(subset=[wind_col,power_col,time_col])
    df["Name"] = df["Name"].astype(str).str.strip()

    return df, wind_col, power_col, time_col

df, wind_col, power_col, time_col = load_scada(uploaded_file)

# DATE FILTER
period = st.sidebar.selectbox("Period", ["Last 15 Days","Weekly","Monthly"])
end_date = df[time_col].max()
start_date = end_date - timedelta(days=15 if period=="Last 15 Days" else 7 if period=="Weekly" else 30)
df = df[(df[time_col]>=start_date)&(df[time_col]<=end_date)]

# REFERENCE
@st.cache_data
def load_reference(site):
    ref_raw = pd.read_excel(REF_FILE, header=None)

    for r in range(ref_raw.shape[0]):
        for c in range(ref_raw.shape[1]):
            if site.lower() in str(ref_raw.iloc[r,c]).lower():
                ref = ref_raw.iloc[r+2:r+60,[c-1,c+3]].copy()
                ref.columns=["WindSpeed","RefPower"]
                ref = ref.dropna()
                ref["WindSpeed"]=pd.to_numeric(ref["WindSpeed"])
                ref["RefPower"]=pd.to_numeric(ref["RefPower"])
                wind_bins = np.arange(4,10,BIN_SIZE)
                ref_interp = np.interp(wind_bins, ref["WindSpeed"], ref["RefPower"])
                return pd.DataFrame({"WindBin":wind_bins,"RefPower":ref_interp})

    st.error("Site not found")
    st.stop()

ref_curve = load_reference(site)

# PROCESS
def process_turbine(t):
    df_t = df[df["Name"]==t].copy()
    df_t = df_t[(df_t[wind_col]>=3)&(df_t[wind_col]<=25)&(df_t[power_col]>0)]

    if len(df_t)<30:
        return None

    expected_points = ((end_date-start_date).total_seconds()/600)
    availability = (len(df_t)/expected_points)*100
    std_dev = df_t[power_col].std()

    df_t["WindBin"] = (df_t[wind_col]/BIN_SIZE).round()*BIN_SIZE

    actual = df_t.groupby("WindBin").agg(AvgPower=(power_col,"mean")).reset_index()
    merged = ref_curve.merge(actual,on="WindBin",how="left")

    valid = merged["AvgPower"].notna()
    if valid.sum()>7:
        merged.loc[valid,"AvgPower"] = savgol_filter(merged.loc[valid,"AvgPower"],7,2)

    merged["Deviation_%"] = ((merged["AvgPower"]-merged["RefPower"])/merged["RefPower"])*100
    avg_dev = merged["Deviation_%"].mean(skipna=True)

    return df_t, merged, avg_dev, availability, std_dev

# SUMMARY
results=[]
for t in df["Name"].unique():
    res = process_turbine(t)
    if res:
        _,_,dev,_,_ = res
        results.append({"Turbine":t,"Deviation_%":dev})

results_df = pd.DataFrame(results)

# STATUS
def get_status(dev):
    if dev < -2: return "Under"
    elif dev > 2: return "Over"
    else: return "Normal"

results_df["Status"] = results_df["Deviation_%"].apply(get_status)

# SORT ORDER
order = {"Normal":0,"Under":1,"Over":2}
results_df["order"] = results_df["Status"].map(order)
results_df = results_df.sort_values("order")

# ================= SINGLE =================
if mode=="Single Turbine":
    t = st.selectbox("Select Turbine", results_df["Turbine"])
    df_t, merged, dev, avail, std = process_turbine(t)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_t[wind_col],y=df_t[power_col],mode='markers'))
    fig.add_trace(go.Scatter(x=merged["WindBin"],y=merged["AvgPower"],mode='lines'))
    fig.add_trace(go.Scatter(x=merged["WindBin"],y=merged["RefPower"],mode='lines',line=dict(dash='dash')))
    st.plotly_chart(fig,use_container_width=True)

    st.write(f"Deviation: {round(dev,2)}%")
    st.write(f"Std Dev %: {round((std/RATED_POWER)*100,2)}%")

# ================= COMPARISON =================
elif mode=="Compare Turbines":
    t1 = st.selectbox("Turbine 1", results_df["Turbine"])
    t2 = st.selectbox("Turbine 2", results_df["Turbine"], key="t2")

    r1 = process_turbine(t1)
    r2 = process_turbine(t2)

    df1,m1,d1,a1,s1 = r1
    df2,m2,d2,a2,s2 = r2

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=m1["WindBin"],y=m1["AvgPower"],name=t1))
    fig.add_trace(go.Scatter(x=m2["WindBin"],y=m2["AvgPower"],name=t2))
    fig.add_trace(go.Scatter(x=ref_curve["WindBin"],y=ref_curve["RefPower"],line=dict(dash='dash'),name="Ref"))
    st.plotly_chart(fig,use_container_width=True)

    better = t1 if d1>d2 else t2
    worse = t2 if d1>d2 else t1

    st.markdown(f"""
