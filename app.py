import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import savgol_filter
from datetime import timedelta
import os
import io

st.set_page_config(layout="wide")

# SAFE KALEIDO CHECK
try:
    import kaleido
    KALEIDO_AVAILABLE = True
except:
    KALEIDO_AVAILABLE = False

# SAFE DOCX CHECK
try:
    from docx import Document
    from docx.shared import Inches
    DOCX_AVAILABLE = True
except:
    DOCX_AVAILABLE = False

# TITLE
st.title("Power Curve Analytics Report")

# LOGO
logo_path = os.path.join(os.path.dirname(__file__), "Envision.png")
col1, col2, col3 = st.columns([1,2,1])
with col2:
    if os.path.exists(logo_path):
        st.image(logo_path, width=300)

# SITE CAPACITY
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

# SIDEBAR
st.sidebar.subheader("Upload SCADA File")
uploaded_file = st.sidebar.file_uploader("Upload SCADA CSV", type=["csv"])

if uploaded_file is None:
    st.stop()

site = st.sidebar.selectbox("Select Site", list(SITE_CAPACITY.keys()))

# LOAD DATA
@st.cache_data
def load_scada(file):
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()

    wind_col = [c for c in df.columns if "wind" in c.lower()][0]
    power_col = [c for c in df.columns if "power" in c.lower() or "active" in c.lower()][0]
    time_col = [c for c in df.columns if "time" in c.lower()][0]

    df[time_col] = pd.to_datetime(df[time_col])
    df[wind_col] = pd.to_numeric(df[wind_col])
    df[power_col] = pd.to_numeric(df[power_col])

    df["Name"] = df["Name"].astype(str)
    return df, wind_col, power_col, time_col

df, wind_col, power_col, time_col = load_scada(uploaded_file)

# REFERENCE
@st.cache_data
def load_reference(site):
    ref_raw = pd.read_excel(REF_FILE, header=None)
    for r in range(ref_raw.shape[0]):
        for c in range(ref_raw.shape[1]):
            if site.lower() in str(ref_raw.iloc[r,c]).lower():
                ref = ref_raw.iloc[r+2:r+60,[c-1,c+3]]
                ref.columns=["WindSpeed","RefPower"]
                ref = ref.dropna()
                wind_bins = np.arange(4,10,BIN_SIZE)
                ref_interp = np.interp(wind_bins, ref["WindSpeed"], ref["RefPower"])
                return pd.DataFrame({"WindBin":wind_bins,"RefPower":ref_interp})

ref_curve = load_reference(site)

# PROCESS
def process_turbine(t):
    df_t = df[df["Name"]==t]
    df_t["WindBin"] = (df_t[wind_col]/BIN_SIZE).round()*BIN_SIZE
    actual = df_t.groupby("WindBin").agg(AvgPower=(power_col,"mean")).reset_index()
    merged = ref_curve.merge(actual,on="WindBin",how="left")

    merged["Deviation_%"] = ((merged["AvgPower"]-merged["RefPower"])/merged["RefPower"])*100
    dev = merged["Deviation_%"].mean()

    return df_t, merged, dev

# GRAPH
def plot_graph(df_t, merged, name, dev):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_t[wind_col],y=df_t[power_col],mode='markers'))
    fig.add_trace(go.Scatter(x=merged["WindBin"],y=merged["AvgPower"],mode='lines'))
    fig.add_trace(go.Scatter(x=merged["WindBin"],y=merged["RefPower"],mode='lines',line=dict(dash='dash')))
    fig.update_layout(title=f"{name} (Dev: {round(dev,2)}%)")
    return fig

# DISPLAY + STORE IMAGES
images = []
results = []

for t in df["Name"].unique():
    res = process_turbine(t)
    if not res:
        continue

    df_t, merged, dev = res
    fig = plot_graph(df_t, merged, t, dev)

    st.plotly_chart(fig)

    if KALEIDO_AVAILABLE:
        try:
            img = fig.to_image(format="png", width=1000, height=500)
            img_stream = io.BytesIO(img)
            img_stream.seek(0)  # ⭐ IMPORTANT FIX
            images.append((t, img_stream))
        except:
            pass

    results.append([t, round(dev,2)])

# TABLE
results_df = pd.DataFrame(results, columns=["Turbine","Deviation_%"])
st.dataframe(results_df)

# WORD EXPORT (FIXED)
if DOCX_AVAILABLE:
    doc = Document()
    doc.add_heading("Power Curve Analytics Report", 0)

    for name, img_stream in images:
        doc.add_heading(name, 1)
        doc.add_picture(img_stream, width=Inches(6))  # ⭐ FIXED SIZE
        doc.add_paragraph(" ")

    doc.add_heading("Turbine Ranking", 1)
    doc.add_paragraph(results_df.to_string())

    buffer = io.BytesIO()
    doc.save(buffer)

    st.download_button(
        "Download Word Report",
        buffer.getvalue(),
        file_name="Report.docx"
    )
