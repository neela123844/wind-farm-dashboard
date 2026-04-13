import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

st.set_page_config(layout="wide")
st.title("Wind Farm Performance Analytics Dashboard")

# FILE PATHS (Local VS Code)
SCADA_FILE = "EN-156-3.3_20260120_20260228_ten_minutes_3.csv"
REF_FILE = "India site Standard & Theoretical PC data 123.xlsx"

BIN_SIZE = 0.5
MIN_POINTS = 30
DEVIATION_LIMIT = 1  # ±1%



# LOAD SCADA DATA

@st.cache_data
def load_scada():
    df = pd.read_csv(SCADA_FILE, low_memory=False)
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


df, wind_col, power_col, time_col = load_scada()


# LOAD REFERENCE CURVE
@st.cache_data
def load_reference():

    ref_raw = pd.read_excel(REF_FILE, header=None)

    location = None
    for r in range(ref_raw.shape[0]):
        for c in range(ref_raw.shape[1]):
            if "wanki" in str(ref_raw.iloc[r, c]).lower():
                location = (r, c)
                break
        if location:
            break

    if location is None:
        st.error("Wanki site not found in reference file")
        st.stop()

    r, c = location
    wind_col_ref = c - 1
    power_col_ref = c + 3

    ref = ref_raw.iloc[r+2:r+60, [wind_col_ref, power_col_ref]].copy()
    ref.columns = ["WindSpeed", "RefPower"]

    ref["WindSpeed"] = pd.to_numeric(ref["WindSpeed"], errors="coerce")
    ref["RefPower"] = pd.to_numeric(ref["RefPower"], errors="coerce")

    ref = ref.dropna().sort_values("WindSpeed")

    wind_bins = np.arange(3, 25.5, BIN_SIZE)

    ref_interp = np.interp(
        wind_bins,
        ref["WindSpeed"],
        ref["RefPower"]
    )

    ref_curve = pd.DataFrame({
        "WindBin": wind_bins,
        "RefPower": ref_interp
    })

    return ref_curve


ref_curve = load_reference()


# SIDEBAR FILTER
st.sidebar.header("Filters")

date_range = st.sidebar.date_input(
    "Date Range",
    value=(df[time_col].min().date(), df[time_col].max().date())
)

# Safe handling
if isinstance(date_range, tuple) and len(date_range) == 2:
    start = pd.to_datetime(date_range[0])
    end = pd.to_datetime(date_range[1])
else:
    start = pd.to_datetime(date_range[0])
    end = start

df_filtered = df[
    (df[time_col] >= start) &
    (df[time_col] <= end)
]
# TURBINE PROCESSING
def process_turbine(df_input, turbine):

    df_t = df_input[df_input["Name"] == turbine].copy()

    df_t = df_t[
        (df_t[power_col] > 0) &
        (df_t[wind_col] >= 3) &
        (df_t[wind_col] <= 25)
    ]

    if len(df_t) < MIN_POINTS:
        return None

    df_t["WindBin"] = (df_t[wind_col] / BIN_SIZE).round() * BIN_SIZE

    actual = (
        df_t.groupby("WindBin")
        .agg(AvgPower=(power_col, "mean"))
        .reset_index()
    )

    merged = ref_curve.merge(actual, on="WindBin", how="left")

    valid = merged["AvgPower"].notna()

    if valid.sum() > 7:
        merged.loc[valid, "AvgPower"] = savgol_filter(
            merged.loc[valid, "AvgPower"], 7, 2
        )

    merged["Deviation_%"] = (
        (merged["AvgPower"] - merged["RefPower"])
        / merged["RefPower"]
    ) * 100

    efficiency = (
        merged["AvgPower"].sum() /
        merged["RefPower"].sum()
    ) * 100

    avg_dev = merged["Deviation_%"].mean()
    tolerance = merged["Deviation_%"].std()

    return merged, efficiency, avg_dev, tolerance

# FLEET ANALYSIS
turbines = sorted(df_filtered["Name"].unique())
summary = []

for t in turbines:
    res = process_turbine(df_filtered, t)
    if res is None:
        continue

    merged, eff, dev, tol = res
    summary.append([t, eff, dev, tol])

summary_df = pd.DataFrame(
    summary,
    columns=["Turbine", "Efficiency_%", "Deviation_%", "Tolerance_%"]
)

# KPI Calculations
total_turbines = len(summary_df)
under_perf = summary_df[summary_df["Deviation_%"] < -DEVIATION_LIMIT]
over_perf = summary_df[summary_df["Deviation_%"] > DEVIATION_LIMIT]
within_limit = summary_df[
    (summary_df["Deviation_%"] >= -DEVIATION_LIMIT) &
    (summary_df["Deviation_%"] <= DEVIATION_LIMIT)
]

fleet_efficiency = summary_df["Efficiency_%"].mean()
eff_spread = summary_df["Efficiency_%"].max() - summary_df["Efficiency_%"].min()
fleet_tolerance = summary_df["Tolerance_%"].mean()


# KPI DISPLAY
st.subheader("Fleet Performance Summary")

k1, k2, k3, k4, k5 = st.columns(5)

k1.metric("Total Turbines", total_turbines)
k2.metric("Underperforming", len(under_perf))
k3.metric("Overperforming", len(over_perf))
k4.metric("Within Limit", len(within_limit))
k5.metric("Fleet Efficiency (%)", f"{fleet_efficiency:.2f}")

st.write(f"Efficiency Spread: {eff_spread:.2f} %")
st.write(f"Average Fleet Tolerance: {fleet_tolerance:.2f} %")


# POWER CURVES
st.subheader("Avg Actual vs Reference Power")

cols = st.columns(3)
i = 0

for t in turbines:

    res = process_turbine(df_filtered, t)
    if res is None:
        continue

    merged, _, _, _ = res

    fig, ax = plt.subplots(figsize=(4,3))

    ax.plot(merged["WindBin"], merged["AvgPower"], label="Actual", linewidth=2)
    ax.plot(merged["WindBin"], merged["RefPower"], label="Reference", linewidth=2.5)

    ax.set_title(t, fontsize=9)
    ax.set_xlabel("Wind Speed (m/s)", fontsize=8)
    ax.set_ylabel("Power (kW)", fontsize=8)
    ax.legend(fontsize=7)
    ax.grid(True, linestyle="--", alpha=0.5)

    cols[i % 3].pyplot(fig)
    i += 1

# RANKING TABLE
st.subheader("Turbine Ranking")

summary_df = summary_df.sort_values("Deviation_%")
st.dataframe(summary_df)