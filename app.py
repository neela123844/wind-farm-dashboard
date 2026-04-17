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

    stall_df = merged[
        (merged["WindBin"]>=4)&(merged["WindBin"]<=10)&
        (merged["Deviation_%"]<=-40)&(merged["Deviation_%"]>=-72)
    ]

    stall_bins = stall_df["WindBin"].tolist()
    stall_flag = (len(stall_bins)>=3) and (availability>=99) and (std_dev<1)

    derating_flag = False
    derating_sources = []

    for col in status_cols:
        if df_t[col].astype(str).str.contains("derat|limit|curtail|temp|pitch|overheat", case=False, na=False).any():
            derating_flag = True
            derating_sources.append(col)

    measurement_flag = len(df_t[(df_t[wind_col]<4)&(df_t[power_col]>0.2*RATED_POWER)]) > 10

    return df_t, merged, avg_dev, stall_flag, stall_bins, availability, std_dev, derating_flag, derating_sources, measurement_flag

# ---------------- SUMMARY ----------------
results = []

for t in df["Name"].unique():
    res = process_turbine(t)
    if not res:
        continue

    df_t, merged, avg_dev, stall_flag, stall_bins, availability, std_dev, derating_flag, derating_sources, measurement_flag = res

    # -------- COMMENT ENGINE --------
    comment = ""

    if stall_flag:
        comment += " STALL DETECTED\n"

    elif avg_dev < -2:
        comment += " UNDERPERFORMANCE\n"

    if avg_dev > 8:
        comment += "\n HIGH OVERPERFORMANCE\n"
        comment += "Measurement issue\n"
        comment += "o NTF\n"
        comment += "o Sensor alignment\n"
        comment += "o IPC sensor not active\n"

    elif avg_dev > 2:
        comment += "\n SLIGHT OVERPERFORMANCE\n"

    if -TOLERANCE <= avg_dev <= TOLERANCE:
        comment += "\n NORMAL PERFORMANCE\n"

    if derating_flag:
        comment += f"\n DERATING ACTIVE: {', '.join(derating_sources)}"

    if measurement_flag:
        comment += "\n MEASUREMENT ISSUE (low wind high power)"

    comment += f"\nAvailability: {round(availability,1)}%"
    comment += f"\nStd Dev: {round(std_dev,2)}"

    if stall_flag:
        status = "STALL"
    elif avg_dev < -2:
        status = "UNDER"
    elif avg_dev > 8:
        status = "OVER"
    else:
        status = "NORMAL"

    results.append({
        "Turbine": t,
        "Deviation_%": avg_dev,
        "Status": status,
        "Remarks": comment
    })

results_df = pd.DataFrame(results)

# ---------------- CUSTOM SORT ----------------
order = {"NORMAL":0, "OVER":1, "UNDER":2, "STALL":3}
results_df["order"] = results_df["Status"].map(order)
results_df = results_df.sort_values("order").drop(columns=["order"])

# ---------------- BAR ----------------
colors = ["red" if d < -2 else "orange" if d > 2 else "green" for d in results_df["Deviation_%"]]

fig_bar = go.Figure()
fig_bar.add_trace(go.Bar(
    x=results_df["Turbine"],
    y=results_df["Deviation_%"],
    marker_color=colors
))

st.plotly_chart(fig_bar, use_container_width=True)

# ---------------- TABLE ----------------
st.subheader("Ranking")

def color_rows(row):
    if row["Status"] == "STALL":
        return ['background-color: #ff4d4d'] * len(row)
    elif row["Status"] == "UNDER":
        return ['background-color: #ff9999'] * len(row)
    elif row["Status"] == "OVER":
        return ['background-color: #ffcc66'] * len(row)
    else:
        return ['background-color: #99ff99'] * len(row)

styled_df = results_df.style.apply(color_rows, axis=1)

st.dataframe(styled_df, use_container_width=True)
