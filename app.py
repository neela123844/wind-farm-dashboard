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
    if os.path.exists(logo_path):
        st.image(logo_path, width=300)

st.title("Wind Farm Performance Analytics Dashboard")

REF_FILE = "India site Standard & Theoretical PC data 1234.xlsx"

BIN_SIZE = 0.5
RATED_POWER = 3400.0

# SIDEBAR
uploaded_file = st.sidebar.file_uploader("Upload SCADA CSV", type=["csv"])

if uploaded_file is None:
    st.warning("Please upload SCADA file")
    st.stop()

#  SITE LIST (NO [...])
site = st.sidebar.selectbox(
    "Select Site for Reference Curve",
    ["CIP Hatalageri","JSW Tuljapur","Blupine Sagapara","Kalavad GJ","Kalavad_PH2","AMP_Energy","Wanki",
     "CleanMax Motadevaliya","Ayana Amerli","Mahadev PH1","Blupine-I, Ambada-GJ","ACME Shapar",
     "FP_Kudligi","Sprng TN","Otha Pithalpur-GJ","AMGEPL,Kurnool AP","ReNew1_Gadag",
     "partner Ottapidaum","Cleanmax SANATHALI","Cleanmax Babra","RenfraEnergy Trichy",
     "RENEW-03 Sholapur","Renew2 Chandwad","ReNew-4 Patoda","Clean max Jagalur",
     "Sembcorp Tuticorin","Renew-4 Kudligi","Renew Otha","Cleanmax Honavad",
     "Blueleaf Agar","JSW_Sandur","India_Hero_Doni"]
)

# extra safety
if not isinstance(site, str):
    st.error("Invalid site selection")
    st.stop()

mode = st.sidebar.radio(
    "Select View",
    ["Single Turbine", "Compare Turbines", "Show All Turbines"]
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

    df = df.dropna(subset=[wind_col,power_col,time_col])
    df["Name"] = df["Name"].astype(str).str.strip()

    return df, wind_col, power_col, time_col

df, wind_col, power_col, time_col = load_scada(uploaded_file)

# DATE FILTER
period = st.sidebar.selectbox("Period", ["Last 15 Days","Weekly","Monthly"])
end_date = df[time_col].max()
start_date = end_date - timedelta(days=15 if period=="Last 15 Days" else 7 if period=="Weekly" else 30)
df = df[(df[time_col]>=start_date)&(df[time_col]<=end_date)]

# ---------------- LOAD REFERENCE (FIXED) ----------------
@st.cache_data
def load_reference(site):
    ref_raw = pd.read_excel(REF_FILE, header=None)

    for r in range(ref_raw.shape[0]):
        for c in range(ref_raw.shape[1]):
            cell = str(ref_raw.iloc[r,c])

            if isinstance(site, str) and site.lower() in cell.lower():
                ref = ref_raw.iloc[r+2:r+60,[c-1,c+3]].copy()
                ref.columns=["WindSpeed","RefPower"]
                ref = ref.dropna()

                ref["WindSpeed"]=pd.to_numeric(ref["WindSpeed"], errors="coerce")
                ref["RefPower"]=pd.to_numeric(ref["RefPower"], errors="coerce")

                wind_bins = np.arange(4,10,BIN_SIZE)
                ref_interp = np.interp(wind_bins, ref["WindSpeed"], ref["RefPower"])

                return pd.DataFrame({"WindBin":wind_bins,"RefPower":ref_interp})

    st.error(" Site not found in reference file")
    st.stop()

ref_curve = load_reference(site)

# ---------------- PROCESS ----------------
def process_turbine(t):
    df_t = df[df["Name"]==t].copy()
    df_t = df_t[(df_t[wind_col]>=3)&(df_t[wind_col]<=25)&(df_t[power_col]>0)]

    if len(df_t)<30:
        return None

    std_dev = df_t[power_col].std()

    df_t["WindBin"] = (df_t[wind_col]/BIN_SIZE).round()*BIN_SIZE
    actual = df_t.groupby("WindBin").agg(AvgPower=(power_col,"mean")).reset_index()

    merged = ref_curve.merge(actual,on="WindBin",how="left")

    valid = merged["AvgPower"].notna()
    if valid.sum()>7:
        merged.loc[valid,"AvgPower"] = savgol_filter(merged.loc[valid,"AvgPower"],7,2)

    merged["Deviation_%"] = ((merged["AvgPower"]-merged["RefPower"])/merged["RefPower"])*100
    avg_dev = merged["Deviation_%"].mean(skipna=True)

    return df_t, merged, avg_dev, std_dev

# ---------------- GRAPH ----------------
def plot_graph(df_t, merged, title, dev):
    color = "green" if -2 <= dev <= 2 else "orange" if dev < -2 else "red"

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_t[wind_col],y=df_t[power_col],
                             mode='markers',marker=dict(size=3,opacity=0.4)))
    fig.add_trace(go.Scatter(x=merged["WindBin"],y=merged["AvgPower"],
                             mode='lines+markers'))
    fig.add_trace(go.Scatter(x=merged["WindBin"],y=merged["RefPower"],
                             mode='lines',line=dict(dash='dash')))

    fig.update_layout(title=dict(text=title, font=dict(color=color)))
    return fig

# ---------------- COMMENT ----------------
def generate_comment(dev):
    if dev < -10:
        return "🔴 Severe underperformance → Blade / Yaw issue"
    elif dev < -2:
        return "🟠 Underperformance → Control / availability issue"
    elif dev > 8:
        return "🟢 High overperformance → Sensor issue"
    elif dev > 2:
        return "🟢 Slight overperformance"
    else:
        return "🟢 Normal performance"

# ---------------- DISPLAY ----------------
cols = st.columns(2)
i = 0

for t in df["Name"].unique():
    res = process_turbine(t)
    if not res:
        continue

    df_t, merged, dev, std = res

    with cols[i%2]:
        st.plotly_chart(plot_graph(df_t, merged, t, dev), use_container_width=True)
        st.markdown("** Analysis**")
        st.code(generate_comment(dev))

    i += 1
