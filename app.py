import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import savgol_filter
from datetime import timedelta
import os
import zipfile
import io
from openpyxl import Workbook
from openpyxl.styles import PatternFill

st.set_page_config(layout="wide")

# ---------------- SAFE KALEIDO ----------------
try:
    import kaleido
    KALEIDO_AVAILABLE = True
except:
    KALEIDO_AVAILABLE = False

# ---------------- TITLE ----------------
st.title("Power Curve Analytics Report")

# ---------------- LOGO ----------------
logo_path = os.path.join(os.path.dirname(__file__), "Envision.png")
col1, col2, col3 = st.columns([1,2,1])
with col2:
    if os.path.exists(logo_path):
        st.image(logo_path, width=300)

# ---------------- SITE ----------------
SITE_CAPACITY = {site:3.3 for site in [
"CIP Hatalageri","JSW Tuljapur","Blupine Sagapara","Kalavad GJ","Kalavad_PH2",
"AMP_Energy","Wanki","CleanMax Motadevaliya","Ayana Amerli","Mahadev PH1",
"Blupine-I, Ambada-GJ","ACME Shapar","FP_Kudligi","Sprng TN",
"Otha Pithalpur-GJ","AMGEPL,Kurnool AP","ReNew1_Gadag","partner Ottapidaum",
"Cleanmax SANATHALI","Cleanmax Babra","RenfraEnergy Trichy","RENEW-03 Sholapur",
"Renew2 Chandwad","ReNew-4 Patoda","Clean max Jagalur","Sembcorp Tuticorin",
"Renew-4 Kudligi","Renew Otha","Cleanmax Honavad","Blueleaf Agar",
"JSW_Sandur","India_Hero_Doni"
]}

REF_FILE = "India site Standard & Theoretical PC data 1234.xlsx"
BIN_SIZE = 0.5
RATED_POWER = 3400.0

# ---------------- SIDEBAR ----------------
st.sidebar.subheader("Upload SCADA File")
uploaded_file = st.sidebar.file_uploader("Upload SCADA CSV", type=["csv"])

if uploaded_file is None:
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

# ---------------- DATE ----------------
st.sidebar.markdown("Select Date Range")
max_date = df[time_col].max()

start_date = st.sidebar.date_input("Start Date", value=max_date - timedelta(days=15))
end_date = st.sidebar.date_input("End Date", value=max_date)

start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date) + pd.Timedelta(days=1)

df = df[(df[time_col] >= start_date) & (df[time_col] <= end_date)]

# ---------------- HEADER ----------------
num_turbines = df["Name"].nunique()
cap = SITE_CAPACITY.get(site, 3.3)
total_cap = num_turbines * cap

st.subheader(f"{site} | {num_turbines} Turbines | {cap} MW Each | Total: {round(total_cap,2)} MW")
st.markdown(f"Date Range: {start_date.date()} → {end_date.date()}")

# ---------------- REF ----------------
@st.cache_data
def load_reference(site):
    ref_raw = pd.read_excel(REF_FILE, header=None)
    for r in range(ref_raw.shape[0]):
        for c in range(ref_raw.shape[1]):
            if site.lower() in str(ref_raw.iloc[r,c]).lower():
                ref = ref_raw.iloc[r+2:r+60,[c-1,c+3]].copy()
                ref.columns=["WindSpeed","RefPower"]
                ref = ref.dropna()
                ref["WindSpeed"]=pd.to_numeric(ref["WindSpeed"], errors="coerce")
                ref["RefPower"]=pd.to_numeric(ref["RefPower"], errors="coerce")

                wind_bins = np.arange(4,10,BIN_SIZE)
                ref_interp = np.interp(wind_bins, ref["WindSpeed"], ref["RefPower"])

                return pd.DataFrame({"WindBin":wind_bins,"RefPower":ref_interp})
    st.stop()

ref_curve = load_reference(site)

# ---------------- PROCESS ----------------
def process_turbine(t):
    df_t = df[df["Name"]==t].copy()
    df_t = df_t[(df_t[wind_col]>=3)&(df_t[wind_col]<=25)&(df_t[power_col]>0)]
    if len(df_t)<30:
        return None

    std = df_t[power_col].std()

    df_t["WindBin"] = (df_t[wind_col]/BIN_SIZE).round()*BIN_SIZE
    actual = df_t.groupby("WindBin").agg(AvgPower=(power_col,"mean")).reset_index()

    merged = ref_curve.merge(actual,on="WindBin",how="left")

    valid = merged["AvgPower"].notna()
    if valid.sum()>7:
        merged.loc[valid,"AvgPower"] = savgol_filter(merged.loc[valid,"AvgPower"],7,2)

    merged["Deviation_%"] = ((merged["AvgPower"]-merged["RefPower"])/merged["RefPower"])*100
    dev = merged["Deviation_%"].mean(skipna=True)

    return df_t, merged, dev, std

# ---------------- COMMENT ----------------
def generate_comment(dev):
    if dev < -10:
        return "High Under"
    elif dev < -2:
        return "Under"
    elif dev > 8:
        return "High Over"
    elif dev > 2:
        return "Slight Over"
    else:
        return "Normal"

# ---------------- GRAPH ----------------
def plot_graph(df_t, merged, t, dev):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_t[wind_col],y=df_t[power_col],mode='markers'))
    fig.add_trace(go.Scatter(x=merged["WindBin"],y=merged["AvgPower"],mode='lines+markers'))
    fig.add_trace(go.Scatter(x=merged["WindBin"],y=merged["RefPower"],mode='lines',line=dict(dash='dash')))
    fig.update_layout(title=f"{t} (Dev: {round(dev,2)}%)")
    return fig

# ---------------- DISPLAY ----------------
results = []
zip_buffer = io.BytesIO()
zip_file = zipfile.ZipFile(zip_buffer, "w")

cols = st.columns(2)
i = 0

for t in df["Name"].unique():
    res = process_turbine(t)
    if not res:
        continue

    df_t, merged, dev, std = res
    fig = plot_graph(df_t, merged, t, dev)

    with cols[i%2]:
        st.plotly_chart(fig, use_container_width=True)

    if KALEIDO_AVAILABLE:
        try:
            zip_file.writestr(f"{t}.png", fig.to_image(format="png"))
        except:
            pass

    status = generate_comment(dev)

    results.append({
        "Turbine": t,
        "Deviation_%": round(dev,2),
        "Status": status
    })

    i += 1

# ---------------- TABLE ----------------
st.subheader("Turbine Ranking")

results_df = pd.DataFrame(results).sort_values(by="Deviation_%")

def color_excel(cell):
    if cell == "Normal":
        return PatternFill(start_color="CCFFCC", fill_type="solid")
    elif cell == "Slight Over":
        return PatternFill(start_color="66FF66", fill_type="solid")
    elif cell == "High Over":
        return PatternFill(start_color="009933", fill_type="solid")
    elif cell == "Under":
        return PatternFill(start_color="FFCC66", fill_type="solid")
    else:
        return PatternFill(start_color="FF6666", fill_type="solid")

st.dataframe(results_df, use_container_width=True)

# ---------------- EXCEL EXPORT ----------------
wb = Workbook()
ws = wb.active
ws.title = "Report"

# header
ws.append(["Power Curve Analytics Report"])
ws.append([f"{site} | {num_turbines} Turbines"])
ws.append([f"Date: {start_date.date()} to {end_date.date()}"])
ws.append([])

ws.append(["Turbine","Deviation %","Status"])

for row in results:
    ws.append(list(row.values()))

# apply color
for r in range(6, 6+len(results)):
    status = ws[f"C{r}"].value
    ws[f"C{r}"].fill = color_excel(status)

excel_buffer = io.BytesIO()
wb.save(excel_buffer)

# add excel to zip
zip_file.writestr("Full_Report.xlsx", excel_buffer.getvalue())

# CSV
zip_file.writestr("report.csv", results_df.to_csv(index=False))

zip_file.close()

# ---------------- DOWNLOAD ----------------
st.download_button(
    "Download Full Dashboard (ZIP)",
    data=zip_buffer.getvalue(),
    file_name="WindFarm_Full_Report.zip"
)
