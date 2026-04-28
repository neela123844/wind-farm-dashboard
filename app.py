import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import savgol_filter
from datetime import timedelta
import os
import zipfile
import io

st.set_page_config(layout="wide")

# ---------------- SAFE KALEIDO CHECK ----------------
try:
    import kaleido
    KALEIDO_AVAILABLE = True
except:
    KALEIDO_AVAILABLE = False

# ---------------- LOGO ----------------
logo_path = os.path.join(os.path.dirname(__file__), "Envision.png")
col1, col2, col3 = st.columns([1,2,1])
with col2:
    if os.path.exists(logo_path):
        st.image(logo_path, width=300)

# ---------------- SITE CAPACITY ----------------
SITE_CAPACITY = {
    "CIP Hatalageri": 3.3,
    "JSW Tuljapur": 3.3,
    "Blupine Sagapara": 3.3,
    "Kalavad GJ": 3.3,
    "Kalavad_PH2": 3.3,
    "AMP_Energy": 3.3,
    "Wanki": 3.3,
    "CleanMax Motadevaliya": 3.3,
    "Ayana Amerli": 3.3,
    "Mahadev PH1": 3.3,
    "Blupine-I, Ambada-GJ": 3.3,
    "ACME Shapar": 3.3,
    "FP_Kudligi": 3.3,
    "Sprng TN": 3.3,
    "Otha Pithalpur-GJ": 3.3,
    "AMGEPL,Kurnool AP": 3.3,
    "ReNew1_Gadag": 3.3,
    "partner Ottapidaum": 3.3,
    "Cleanmax SANATHALI": 3.3,
    "Cleanmax Babra": 3.3,
    "RenfraEnergy Trichy": 3.3,
    "RENEW-03 Sholapur": 3.3,
    "Renew2 Chandwad": 3.3,
    "ReNew-4 Patoda": 3.3,
    "Clean max Jagalur": 3.3,
    "Sembcorp Tuticorin": 3.3,
    "Renew-4 Kudligi": 3.3,
    "Renew Otha": 3.3,
    "Cleanmax Honavad": 3.3,
    "Blueleaf Agar": 3.3,
    "JSW_Sandur": 3.3,
    "India_Hero_Doni": 3.3
}

REF_FILE = "India site Standard & Theoretical PC data 1234.xlsx"
BIN_SIZE = 0.5
RATED_POWER = 3400.0

# ---------------- SIDEBAR ----------------
st.sidebar.subheader("Upload SCADA File")
uploaded_file = st.sidebar.file_uploader("Upload SCADA CSV", type=["csv"])

if uploaded_file is None:
    st.warning("Please upload SCADA file")
    st.stop()

site = st.sidebar.selectbox("Select Site", list(SITE_CAPACITY.keys()))

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

# ---------------- DATE FILTER ----------------
st.sidebar.markdown("Select Date Range")

min_date = df[time_col].min()
max_date = df[time_col].max()

start_date = st.sidebar.date_input("Start Date", value=max_date - timedelta(days=15))
end_date = st.sidebar.date_input("End Date", value=max_date)

start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date) + pd.Timedelta(days=1)

df = df[(df[time_col] >= start_date) & (df[time_col] <= end_date)]

# ---------------- HEADER ----------------
num_turbines = df["Name"].nunique()
capacity_per_turbine = SITE_CAPACITY.get(site, 3.3)
total_capacity = num_turbines * capacity_per_turbine

st.title(f"{site} | {num_turbines} Turbines | {capacity_per_turbine} MW Each | Total: {round(total_capacity,2)} MW")
st.markdown(f"📅 Date Range: {start_date.date()} → {end_date.date()}")

# ---------------- LOAD REFERENCE ----------------
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

    st.error("Site not found")
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
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_t[wind_col],y=df_t[power_col],mode='markers'))
    fig.add_trace(go.Scatter(x=merged["WindBin"],y=merged["AvgPower"],mode='lines+markers'))
    fig.add_trace(go.Scatter(x=merged["WindBin"],y=merged["RefPower"],mode='lines',line=dict(dash='dash')))
    return fig

# ---------------- DISPLAY ----------------
results = []
zip_buffer = io.BytesIO()
zip_file = zipfile.ZipFile(zip_buffer, "w")

for t in df["Name"].unique():
    res = process_turbine(t)
    if not res:
        continue

    df_t, merged, dev, std = res
    fig = plot_graph(df_t, merged, t, dev)

    st.plotly_chart(fig, use_container_width=True)

    # SAFE IMAGE EXPORT
    if KALEIDO_AVAILABLE:
        try:
            img_bytes = fig.to_image(format="png")
            zip_file.writestr(f"{t}.png", img_bytes)
        except:
            pass

    results.append({
        "Turbine": t,
        "Deviation_%": round(dev,2)
    })

# ---------------- TABLE ----------------
results_df = pd.DataFrame(results)
st.dataframe(results_df)

# SAVE CSV
zip_file.writestr("report.csv", results_df.to_csv(index=False))
zip_file.close()

# ---------------- DOWNLOAD ----------------
st.download_button(
    label="Download Full Report (ZIP)",
    data=zip_buffer.getvalue(),
    file_name="WindFarm_Report.zip",
    mime="application/zip"
)
