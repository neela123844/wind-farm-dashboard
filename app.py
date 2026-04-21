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
RATED_SPEED = 10.0
RATED_POWER = 3400.0

# SIDEBAR
uploaded_file = st.sidebar.file_uploader("Upload SCADA CSV", type=["csv"])

if uploaded_file is None:
    st.warning("Please upload SCADA file")
    st.stop()

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

    wind_bins = np.arange(4,10,BIN_SIZE)

    ref_interp = np.interp(
        wind_bins,
        ref["WindSpeed"].values,
        ref["RefPower"].values
    )

    return pd.DataFrame({"WindBin":wind_bins,"RefPower":ref_interp})

ref_curve = load_reference(site)

# PROCESS
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

    stall_bins = merged[
        (merged["WindBin"] >= 4) &
        (merged["WindBin"] <= 10) &
        (merged["Deviation_%"] <= -40) &
        (merged["Deviation_%"] >= -72)
    ]["WindBin"].tolist()

    stall_flag = len(stall_bins) >= 3

    return df_t, merged, avg_dev, stall_flag, stall_bins, availability, std_dev

# SUMMARY
results=[]
for t in df["Name"].unique():
    res = process_turbine(t)
    if res:
        _,_,dev,_,_,_,_ = res
        results.append({"Turbine":t,"Deviation_%":dev})

results_df = pd.DataFrame(results)

# STATUS
def get_status(dev):
    if dev < -2:
        return "Under"
    elif dev > 2:
        return "Over"
    else:
        return "Normal"

results_df["Status"] = results_df["Deviation_%"].apply(get_status)

# HEADER
st.subheader(f"{site} Performance ({start_date.date()} → {end_date.date()})")

# BAR GRAPH
colors = ["red" if d < -2 else "orange" if d > 2 else "green" for d in results_df["Deviation_%"]]
fig_bar = go.Figure()
fig_bar.add_trace(go.Bar(x=results_df["Turbine"], y=results_df["Deviation_%"], marker_color=colors))
st.plotly_chart(fig_bar, use_container_width=True)

# -------- SINGLE TURBINE --------
st.subheader("Single Turbine View")
t_sel = st.selectbox("Select Turbine", results_df["Turbine"])

if t_sel:
    df_f, merged, avg_dev, *_ = process_turbine(t_sel)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_f[wind_col], y=df_f[power_col], mode='markers', marker=dict(size=3, opacity=0.4)))
    fig.add_trace(go.Scatter(x=merged["WindBin"], y=merged["AvgPower"], mode='lines+markers', name="Actual"))
    fig.add_trace(go.Scatter(x=merged["WindBin"], y=merged["RefPower"], mode='lines', line=dict(dash='dash'), name="Reference"))

    st.plotly_chart(fig, use_container_width=True)

# -------- COMPARISON --------
st.subheader("Compare Two Turbines")

t1 = st.selectbox("Turbine 1", results_df["Turbine"], key="c1")
t2 = st.selectbox("Turbine 2", results_df["Turbine"], key="c2")

if t1 and t2:
    r1 = process_turbine(t1)
    r2 = process_turbine(t2)

    if r1 and r2:
        df1, m1, d1, _, _, _, s1 = r1
        df2, m2, d2, _, _, _, s2 = r2

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=m1["WindBin"], y=m1["AvgPower"], mode='lines+markers', name=t1))
        fig.add_trace(go.Scatter(x=m2["WindBin"], y=m2["AvgPower"], mode='lines+markers', name=t2))
        fig.add_trace(go.Scatter(x=ref_curve["WindBin"], y=ref_curve["RefPower"],
                                 mode='lines', line=dict(dash='dash'), name="Reference"))

        st.plotly_chart(fig, use_container_width=True)

        # Percentile
        stds = [process_turbine(t)[6] for t in results_df["Turbine"] if process_turbine(t)]
        p1 = (sum(x <= s1 for x in stds)/len(stds))*100
        p2 = (sum(x <= s2 for x in stds)/len(stds))*100

        better = t1 if d1 > d2 else t2

        st.markdown(f"""
        """)

# -------- ORIGINAL ALL TURBINES --------
st.subheader("All Turbine Analysis")

cols = st.columns(2)
i=0

for t in results_df["Turbine"]:
    df_f, merged, avg_dev, stall_flag, stall_bins, availability, std_dev = process_turbine(t)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_f[wind_col],y=df_f[power_col], mode='markers',marker=dict(size=3,opacity=0.4)))
    fig.add_trace(go.Scatter(x=merged["WindBin"],y=merged["AvgPower"],mode='lines+markers'))
    fig.add_trace(go.Scatter(x=merged["WindBin"],y=merged["RefPower"], mode='lines',line=dict(dash='dash')))

    comment = f"Dev: {round(avg_dev,1)}%\nAvailability: {round(availability,1)}%\nStd Dev: {round(std_dev,2)}"

    cols[i%2].plotly_chart(fig,use_container_width=True)
    cols[i%2].markdown(f"```\n{comment}\n```")
    i+=1

# TABLE
st.subheader("Ranking")
st.dataframe(results_df.sort_values("Deviation_%"))
