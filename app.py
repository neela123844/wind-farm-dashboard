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
    ["CIP Hatalageri","JSW Tuljapur","Blupine Sagapara","Kalavad GJ","Kalavad_PH2","AMP_Energy","Wanki",
     "CleanMax Motadevaliya","Ayana Amerli","Mahadev PH1","Blupine-I, Ambada-GJ","ACME Shapar",
     "FP_Kudligi","Sprng TN","Otha Pithalpur-GJ","AMGEPL,Kurnool AP","ReNew1_Gadag",
     "partner Ottapidaum","Cleanmax SANATHALI","Cleanmax Babra","RenfraEnergy Trichy",
     "RENEW-03 Sholapur","Renew2 Chandwad","ReNew-4 Patoda","Clean max Jagalur",
     "Sembcorp Tuticorin","Renew-4 Kudligi","Renew Otha","Cleanmax Honavad",
     "Blueleaf Agar","JSW_Sandur","India_Hero_Doni"]
)

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

# LOAD REFERENCE
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

    stall_bins = merged[(merged["Deviation_%"]<=-40)&(merged["Deviation_%"]>=-72)]["WindBin"].tolist()
    stall_flag = len(stall_bins) >= 3

    derating_flag = False
    for col in df_t.columns:
        if any(x in col.lower() for x in ["derate","limit","curtail","temp","pitch"]):
            if df_t[col].astype(str).str.contains("derat|limit|curtail|temp|pitch", case=False, na=False).any():
                derating_flag = True

    return df_t, merged, avg_dev, availability, std_dev, stall_flag, stall_bins, derating_flag

# GRAPH
def plot_graph(df_t, merged, title, dev):
    color = "green" if -2 <= dev <= 2 else "orange" if dev < -2 else "red"

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_t[wind_col],y=df_t[power_col],
                             mode='markers',marker=dict(size=3,opacity=0.4),name="SCADA"))
    fig.add_trace(go.Scatter(x=merged["WindBin"],y=merged["AvgPower"],
                             mode='lines+markers',name="Actual"))
    fig.add_trace(go.Scatter(x=merged["WindBin"],y=merged["RefPower"],
                             mode='lines',line=dict(dash='dash'),name="Reference"))

    fig.update_layout(title=dict(text=title, font=dict(color=color)))
    return fig

# COMMENT
def generate_comment(dev, std, avail, merged, stall_flag, stall_bins, derating_flag):
    if stall_flag:
        return "🔴 Stall issue detected", "Stall"
    elif derating_flag:
        return "🟡 Derating detected", "Derating"
    elif dev < -2:
        return "🔻 Underperformance", "Underperformance"
    elif dev > 8:
        return "🟢 High overperformance", "Overperformance"
    elif dev > 2:
        return "🟢 Slight overperformance", "Overperformance"
    else:
        return "🟢 Normal", "Normal"

# ================= GRAPH =================
if mode == "Single Turbine":
    turbine = st.selectbox("Select Turbine", df["Name"].unique())
    res = process_turbine(turbine)

    if res:
        df_t, merged, dev, avail, std, stall_flag, stall_bins, derating_flag = res
        st.plotly_chart(plot_graph(df_t, merged, turbine, dev), use_container_width=True)

        comment, _ = generate_comment(dev, std, avail, merged, stall_flag, stall_bins, derating_flag)
        st.markdown("**📝 Analysis**")
        st.code(comment)

elif mode == "Compare Turbines":
    turbines = st.multiselect("Select Turbines", df["Name"].unique())
    cols = st.columns(2)

    for i, t in enumerate(turbines):
        res = process_turbine(t)
        if not res:
            continue

        df_t, merged, dev, avail, std, stall_flag, stall_bins, derating_flag = res
        comment, _ = generate_comment(dev, std, avail, merged, stall_flag, stall_bins, derating_flag)

        with cols[i % 2]:
            st.plotly_chart(plot_graph(df_t, merged, t, dev), use_container_width=True)
            st.markdown("** Analysis**")
            st.code(comment)

else:
    cols = st.columns(2)

    for i, t in enumerate(df["Name"].unique()):
        res = process_turbine(t)
        if not res:
            continue

        df_t, merged, dev, avail, std, stall_flag, stall_bins, derating_flag = res
        comment, _ = generate_comment(dev, std, avail, merged, stall_flag, stall_bins, derating_flag)

        with cols[i % 2]:
            st.plotly_chart(plot_graph(df_t, merged, t, dev), use_container_width=True)
            st.markdown("** Analysis**")
            st.code(comment)

# ================= TABLE =================
st.subheader("Turbine Ranking")

results = []

for t in df["Name"].unique():
    res = process_turbine(t)

    if not res:
        results.append({
            "Turbine": t,
            "Deviation_%": None,
            "Std_Dev_%": None,
            "Status": "Issue",
            "Reason": "Data Not Available"
        })
        continue

    _, merged, dev, avail, std, stall_flag, stall_bins, derating_flag = res
    _, reason = generate_comment(dev, std, avail, merged, stall_flag, stall_bins, derating_flag)

    if -2 <= dev <= 2:
        status = "Normal"
    elif 2 < dev <= 8:
        status = "Slight Over"
    elif dev > 8:
        status = "High Over"
    elif -10 <= dev < -2:
        status = "Under"
    elif dev < -10:
        status = "High Under"
    else:
        status = "Issue"

    results.append({
        "Turbine": t,
        "Deviation_%": round(dev, 2),
        "Std_Dev_%": round((std / RATED_POWER) * 100, 2),
        "Status": status,
        "Reason": reason
    })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="Deviation_%", ascending=False, na_position='last')

# TOP 5 WORST
st.markdown("###  Top 5 Worst Performing Turbines")
st.table(results_df.sort_values(by="Deviation_%").head(5)[["Turbine","Deviation_%","Status"]])

# COLOR TABLE
def color_row(row):
    if row["Status"] == "Normal":
        return ['background-color: #ccffcc'] * len(row)
    elif row["Status"] == "Slight Over":
        return ['background-color: #66ff66'] * len(row)
    elif row["Status"] == "High Over":
        return ['background-color: #009933'] * len(row)
    elif row["Status"] == "Under":
        return ['background-color: #ffcc66'] * len(row)
    elif row["Status"] == "High Under":
        return ['background-color: #ff6666'] * len(row)
    else:
        return ['background-color: #cccccc'] * len(row)

st.dataframe(results_df.style.apply(color_row, axis=1), use_container_width=True)

# DOWNLOAD
st.download_button(
    label="Download Report (CSV)",
    data=results_df.to_csv(index=False).encode('utf-8'),
    file_name='turbine_report.csv',
    mime='text/csv'
)
