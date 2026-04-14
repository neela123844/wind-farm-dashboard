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

# SIDEBAR
uploaded_file = st.sidebar.file_uploader("Upload SCADA CSV", type=["csv"])

if uploaded_file is None:
    st.warning("Please upload SCADA file")
    st.stop()

site = st.sidebar.selectbox(
    "Select Site",
    ["CIP Hatalageri","JSW Tuljapur","Blupine Sagapara","Kalavad GJ","Kalavad_PH2"]
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

    wind_bins = np.arange(3,25.5,BIN_SIZE)

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

    df_t["WindBin"] = (df_t[wind_col]/BIN_SIZE).round()*BIN_SIZE

    actual = df_t.groupby("WindBin").agg(AvgPower=(power_col,"mean")).reset_index()
    merged = ref_curve.merge(actual,on="WindBin",how="left")

    valid = merged["AvgPower"].notna()
    if valid.sum()>7:
        merged.loc[valid,"AvgPower"] = savgol_filter(merged.loc[valid,"AvgPower"],7,2)

    merged["Deviation_%"] = ((merged["AvgPower"]-merged["RefPower"])/merged["RefPower"])*100
    avg_dev = merged["Deviation_%"].mean(skipna=True)

    return df_t, merged, avg_dev

# SUMMARY
results=[]
for t in df["Name"].unique():
    res = process_turbine(t)
    if res:
        _,_,dev = res
        results.append({"Turbine":t,"Deviation_%":dev})

results_df = pd.DataFrame(results)

# STATUS
def get_status(dev):
    if dev < -2:
        return " Under"
    elif dev > 2:
        return " Over"
    else:
        return " Normal"

results_df["Status"] = results_df["Deviation_%"].apply(get_status)

# HEADER
st.subheader(f"{site} Performance ({start_date.date()} → {end_date.date()})")

# BAR GRAPH WITH COLOR
st.subheader("Deviation Overview")

colors = []
for d in results_df["Deviation_%"]:
    if d < -2:
        colors.append("red")
    elif d > 2:
        colors.append("orange")
    else:
        colors.append("green")

fig_bar = go.Figure()
fig_bar.add_trace(go.Bar(
    x=results_df["Turbine"],
    y=results_df["Deviation_%"],
    marker_color=colors
))
st.plotly_chart(fig_bar, use_container_width=True)

# MODE
mode = st.radio("Mode", ["Single","Compare","All"])

# SINGLE
if mode=="Single":
    t = st.selectbox("Select Turbine", results_df["Turbine"])
    df_f, merged, avg_dev = process_turbine(t)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_f[wind_col],y=df_f[power_col],mode='markers',name="SCADA"))
    fig.add_trace(go.Scatter(x=merged["WindBin"],y=merged["AvgPower"],mode='lines+markers',name="Actual"))
    fig.add_trace(go.Scatter(x=merged["WindBin"],y=merged["RefPower"],mode='lines',name="Reference"))
    st.plotly_chart(fig,use_container_width=True)

    st.subheader("Deviation vs Wind Speed")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=merged["WindBin"],y=merged["Deviation_%"],mode='lines+markers'))
    fig2.add_hline(y=0)
    st.plotly_chart(fig2,use_container_width=True)

# COMPARE
elif mode=="Compare":
    t1 = st.selectbox("T1", results_df["Turbine"])
    t2 = st.selectbox("T2", results_df["Turbine"], index=1)

    df1,m1,dev1 = process_turbine(t1)
    df2,m2,dev2 = process_turbine(t2)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=m1["WindBin"],y=m1["AvgPower"],name=t1))
    fig.add_trace(go.Scatter(x=m2["WindBin"],y=m2["AvgPower"],name=t2))
    fig.add_trace(go.Scatter(x=m1["WindBin"],y=m1["RefPower"],name="Reference",line=dict(dash='dash')))
    st.plotly_chart(fig,use_container_width=True)

    better = t1 if dev1 > dev2 else t2
    worse = t2 if better == t1 else t1

    stall1 = m1[(m1["Deviation_%"] < -15) & (m1["WindBin"] < RATED_SPEED)]["WindBin"].tolist()
    stall2 = m2[(m2["Deviation_%"] < -15) & (m2["WindBin"] < RATED_SPEED)]["WindBin"].tolist()

    st.subheader("Comparison Insight")
    st.write(f" {better} performs better than {worse}")

    st.write(f"{t1} → Dev: {round(dev1,2)}%")
    if stall1:
        st.write(f" Stalling at: {stall1}")

    st.write(f"{t2} → Dev: {round(dev2,2)}%")
    if stall2:
        st.write(f" Stalling at: {stall2}")

# ALL
else:
    cols = st.columns(2)
    i=0

    for t in results_df["Turbine"]:
        df_f, merged, avg_dev = process_turbine(t)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_f[wind_col],y=df_f[power_col],
                                 mode='markers',marker=dict(size=3,opacity=0.4)))
        fig.add_trace(go.Scatter(x=merged["WindBin"],y=merged["AvgPower"],mode='lines+markers'))
        fig.add_trace(go.Scatter(x=merged["WindBin"],y=merged["RefPower"],
                                 mode='lines',line=dict(dash='dash')))

        stall = merged[(merged["Deviation_%"] < -15) & (merged["WindBin"] < RATED_SPEED)]["WindBin"].tolist()

        if avg_dev < -2:
            comment = "Underperformance\n- Low air density\n- Stalling\n- Blade issue"
            color = "red"
        elif avg_dev > 2:
            comment = "Overperformance\n- Wind variation\n- Sensor issue"
            color = "orange"
        else:
            comment = "Normal performance"
            color = "green"

        if stall:
            comment += f"\n Stalling at: {stall}"

        fig.update_layout(
            title=f"{t} | Dev: {round(avg_dev,1)}%",
            title_font=dict(color=color),
            height=350
        )

        cols[i%2].plotly_chart(fig,use_container_width=True)
        cols[i%2].markdown(f"```\n{comment}\n```")
        i+=1

# TABLE
st.subheader("Ranking")
st.dataframe(results_df.sort_values("Deviation_%"))
