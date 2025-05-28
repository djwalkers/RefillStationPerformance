import streamlit as st
import pandas as pd
import requests
from io import BytesIO
from PIL import Image

# --- APP CONFIGURATION ---
st.set_page_config(page_title="Refill Station Performance Dashboard", layout="wide")

MAIN_BG = "#DA362C"
st.markdown(
    f"""
    <style>
        body, .stApp {{
            background-color: {MAIN_BG} !important;
        }}
        .stDataFrame thead tr th {{
            background-color: {MAIN_BG} !important;
            color: #FFF !important;
        }}
        .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {{
            background-color: #b2281c !important;
            color: #FFF !important;
        }}
        .stTabs [data-baseweb="tab-list"] button[aria-selected="false"] {{
            color: #FFF !important;
        }}
        .stTabs [data-baseweb="tab-list"] {{
            background-color: {MAIN_BG} !important;
        }}
        .stSelectbox > div:first-child {{
            color: #DA362C !important;
            font-weight: bold;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

# --- LOGO ---
logo_path = "The Roc.png"
try:
    logo = Image.open(logo_path)
    st.image(logo, width=120)
except Exception:
    st.markdown("<h1 style='color:white'>Refill Station Performance Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# --- GITHUB CONFIGURATION ---
GITHUB_USER = "djwalkers"
GITHUB_REPO = "RefillStationPerformance"
GITHUB_BRANCH = "main"
DATA_FOLDER = "Data"
FILES_FOLDER = "Files"

@st.cache_data(ttl=300, show_spinner=False)
def list_github_csv_files():
    api_url = f"https://api.github.com/repos/{GITHUB_USER}/{GITHUB_REPO}/contents/{DATA_FOLDER}?ref={GITHUB_BRANCH}"
    r = requests.get(api_url)
    if r.status_code != 200:
        st.error("Error loading file list from GitHub.")
        return []
    contents = r.json()
    files = [file["download_url"] for file in contents if file["name"].lower().endswith(".csv")]
    return files

@st.cache_data(ttl=300, show_spinner=False)
def load_reference_tables():
    # Date Dimension Table
    date_dim_url = f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/{GITHUB_BRANCH}/{FILES_FOLDER}/Date%20Dimension%20Table.xlsx"
    station_url = f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/{GITHUB_BRANCH}/{FILES_FOLDER}/Station%20Standard.xlsx"
    date_dim = pd.read_excel(date_dim_url, sheet_name="Sheet1")
    station = pd.read_excel(station_url, sheet_name="Station")
    return date_dim, station

@st.cache_data(ttl=120, show_spinner=False)
def load_data():
    csv_files = list_github_csv_files()
    dfs = []
    for url in csv_files:
        df = pd.read_csv(url)
        dfs.append(df)
    if not dfs:
        st.error("No data files found in GitHub repo.")
        return pd.DataFrame()
    data = pd.concat(dfs, ignore_index=True)
    # Rename columns to match Power Query logic
    column_map = {
        "textBox9": "Station Id",
        "textBox10": "User",
        "textBox13": "Drawers Counted",
        "textBox17": "Damaged Drawers Processed",
        "textBox19": "Damaged Products Processed",
        "textBox21": "Rogues Processed"
    }
    for col in column_map:
        if col in data.columns:
            data = data.rename(columns={col: column_map[col]})
    # Remove unwanted columns
    drop_cols = [
        "textBox14", "textBox15", "textBox16", "textBox34", "textBox31",
        "textBox23", "textBox24", "textBox25", "textBox26", "textBox28", "textBox29", "textBox30"
    ]
    for col in drop_cols:
        data = data.drop(columns=[col], errors="ignore")
    # Extract Date and Time from Source.Name
    data["Date"] = pd.to_datetime(data["Source.Name"].str[:10], format="%d-%m-%Y", errors="coerce")
    data["Time"] = data["Source.Name"].str[11:16].str.replace("-", ":")
    data["Users"] = data["User"].astype(str).str.split(" (").str[0].str.replace("*", "").str.strip()
    data["Users"] = data["Users"].str.strip()
    # Add Date Dimension
    date_dim, station = load_reference_tables()
    data = data.merge(date_dim[["Date", "Year", "Month", "Week"]], how="left", on="Date")
    data = data.merge(station[["Station", "Type", "Drawer Avg", "KPI"]], how="left", left_on="Station Id", right_on="Station")
    data = data.rename(columns={"Type": "Station Type", "Drawer Avg": "Drawer Avg", "KPI": "Station KPI"})
    # Calculate Carts Counted Per Hour
    data["Drawers Counted"] = pd.to_numeric(data["Drawers Counted"], errors="coerce").fillna(0)
    data["Drawer Avg"] = pd.to_numeric(data["Drawer Avg"], errors="coerce").fillna(1)
    data["Carts Counted Per Hour"] = (data["Drawers Counted"] / data["Drawer Avg"]).fillna(0)
    # Remove ATLAS BOX & BOND BAGS from station type in breakdowns
    return data

data = load_data()

# --- TABS ---
tabs = st.tabs([
    "Hourly Dashboard",
    "Weekly Dashboard",
    "Monthly Dashboard",
    "High Performers"
])

# --- SHARED HELPER FUNCTION ---
def assign_shift(time_str):
    try:
        if pd.isnull(time_str):
            return None
        h = int(str(time_str)[:2])
        if 6 <= h < 14:
            return "AM"
        elif 14 <= h < 22:
            return "PM"
        else:
            return "Night"
    except Exception:
        return None

# --- HOURLY DASHBOARD ---
with tabs[0]:
    st.header("Hourly Dashboard")

    filtered = data.copy()
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        unique_months = sorted(filtered["Month"].dropna().unique())
        month_sel = st.selectbox("Month", ["All"] + unique_months, key="hour_month")
        if month_sel != "All":
            filtered = filtered[filtered["Month"] == month_sel]
    with col2:
        unique_dates = sorted(filtered["Date"].dt.strftime('%d-%m-%Y').unique())
        date_sel = st.selectbox("Date", ["All"] + list(unique_dates), key="hour_date")
        if date_sel != "All":
            filtered = filtered[filtered["Date"].dt.strftime('%d-%m-%Y') == date_sel]
    with col3:
        unique_times = sorted(filtered["Time"].unique())
        time_sel = st.selectbox("Time", ["All"] + [t for t in unique_times if pd.notnull(t)], key="hour_time")
        if time_sel != "All":
            filtered = filtered[filtered["Time"] == time_sel]
    with col4:
        unique_stations = sorted(filtered["Station Type"].dropna().unique())
        station_sel = st.selectbox("Station Type", ["All"] + unique_stations, key="hour_station")
        if station_sel != "All":
            filtered = filtered[filtered["Station Type"] == station_sel]

    st.subheader("Carts Counted Per Hour")
    summary = (
        filtered.groupby("Users", as_index=False)["Carts Counted Per Hour"].sum()
    )
    summary = summary[summary["Carts Counted Per Hour"] > 0]
    summary = summary.sort_values("Carts Counted Per Hour", ascending=False)
    st.dataframe(summary.rename(columns={"Users": "User"}), use_container_width=True)

    st.subheader("Rogues Processed")
    rogues = (
        filtered.groupby("Users", as_index=False)["Rogues Processed"].sum()
    )
    rogues = rogues[rogues["Rogues Processed"] > 0]
    st.dataframe(rogues.rename(columns={"Users": "User"}), use_container_width=True)

    st.subheader("Damaged Drawers Processed")
    drawers = (
        filtered.groupby("Users", as_index=False)["Damaged Drawers Processed"].sum()
    )
    drawers = drawers[drawers["Damaged Drawers Processed"] > 0]
    st.dataframe(drawers.rename(columns={"Users": "User"}), use_container_width=True)

    st.subheader("Damaged Products Processed")
    products = (
        filtered.groupby("Users", as_index=False)["Damaged Products Processed"].sum()
    )
    products = products[products["Damaged Products Processed"] > 0]
    st.dataframe(products.rename(columns={"Users": "User"}), use_container_width=True)

# --- WEEKLY DASHBOARD ---
with tabs[1]:
    st.header("Weekly Dashboard")

    filtered = data.copy()
    col1, col2 = st.columns(2)
    with col1:
        unique_weeks = sorted(filtered["Week"].dropna().unique())
        week_sel = st.selectbox("Week Number", ["All"] + [str(w) for w in unique_weeks], key="week_week")
        if week_sel != "All":
            filtered = filtered[filtered["Week"].astype(str) == week_sel]
    with col2:
        unique_stations = sorted(filtered["Station Type"].dropna().unique())
        station_sel = st.selectbox("Station Type", ["All"] + unique_stations, key="week_station")
        if station_sel != "All":
            filtered = filtered[filtered["Station Type"] == station_sel]

    st.subheader("Carts Counted Per Week")
    weekly = (
        filtered.groupby("Users", as_index=False)["Carts Counted Per Hour"].sum()
    )
    weekly = weekly[weekly["Carts Counted Per Hour"] > 0]
    st.dataframe(weekly.rename(columns={"Users": "User"}), use_container_width=True)

# --- MONTHLY DASHBOARD ---
with tabs[2]:
    st.header("Monthly Dashboard")

    filtered = data.copy()
    col1, col2 = st.columns(2)
    with col1:
        unique_months = sorted(filtered["Month"].dropna().unique())
        month_sel = st.selectbox("Month", ["All"] + unique_months, key="month_month")
        if month_sel != "All":
            filtered = filtered[filtered["Month"] == month_sel]
    with col2:
        unique_stations = sorted(filtered["Station Type"].dropna().unique())
        station_sel = st.selectbox("Station Type", ["All"] + unique_stations, key="month_station")
        if station_sel != "All":
            filtered = filtered[filtered["Station Type"] == station_sel]

    st.subheader("Carts Counted Per Month")
    monthly = (
        filtered.groupby("Users", as_index=False)["Carts Counted Per Hour"].sum()
    )
    monthly = monthly[monthly["Carts Counted Per Hour"] > 0]
    st.dataframe(monthly.rename(columns={"Users": "User"}), use_container_width=True)

# --- HIGH PERFORMERS TAB ---
with tabs[3]:
    st.header("High Performers")

    # ----- FILTERS -----
    filtered = data.copy()
    col1, col2, col3 = st.columns(3)
    with col1:
        unique_months = sorted(filtered["Month"].dropna().unique())
        month_sel = st.selectbox("Select Month", ["All"] + unique_months, key="hf_month")
        if month_sel != "All":
            filtered = filtered[filtered["Month"] == month_sel]
    with col2:
        unique_days = sorted(filtered["Date"].dt.strftime('%d-%m-%Y').unique())
        day_sel = st.selectbox("Select Date", ["All"] + list(unique_days), key="hf_day")
        if day_sel != "All":
            filtered = filtered[filtered["Date"].dt.strftime('%d-%m-%Y') == day_sel]
    with col3:
        unique_stations = sorted(filtered["Station Type"].dropna().unique())
        station_sel = st.selectbox("Select Station Type", ["All"] + unique_stations, key="hf_station")
        if station_sel != "All":
            filtered = filtered[filtered["Station Type"] == station_sel]

    # ---- TOP PICKER PER DAY (WHOLE DAY) ----
    st.subheader("Top Picker Per Day (Whole Day)")
    top_per_day = (
        filtered.groupby(["Date", "Users", "Station Type"], as_index=False)["Carts Counted Per Hour"].sum()
    )
    idx = (
        top_per_day.groupby("Date")["Carts Counted Per Hour"].idxmax()
    )
    top_day_df = top_per_day.loc[idx].sort_values("Date")
    top_day_df = top_day_df.rename(columns={
        "Users": "Top Picker",
        "Carts Counted Per Hour": "Total Carts Counted",
        "Station Type": "Station Type"
    })
    top_day_df["Date"] = pd.to_datetime(top_day_df["Date"]).dt.strftime('%d-%m-%Y')
    top_day_df = top_day_df[["Date", "Top Picker", "Station Type", "Total Carts Counted"]]
    st.dataframe(top_day_df, use_container_width=True)

    # ---- TOP PICKER PER SHIFT ----
    st.subheader("Top Picker Per Shift (Carts Counted Per Hour)")
    filtered["Shift"] = filtered["Time"].apply(assign_shift)
    shift_data = (
        filtered.groupby(["Date", "Shift", "Users", "Station Type"], as_index=False)["Carts Counted Per Hour"].sum()
    )
    idx = shift_data.groupby(["Date", "Shift"])["Carts Counted Per Hour"].idxmax()
    top_shift_df = shift_data.loc[idx].sort_values(["Date", "Shift"])
    top_shift_df = top_shift_df.rename(columns={
        "Users": "Top Picker",
        "Carts Counted Per Hour": "Total Carts Counted",
        "Station Type": "Station Type"
    })
    top_shift_df["Date"] = pd.to_datetime(top_shift_df["Date"]).dt.strftime('%d-%m-%Y')
    shift_order = pd.CategoricalDtype(["AM", "PM", "Night"], ordered=True)
    top_shift_df["Shift"] = top_shift_df["Shift"].astype(shift_order)
    top_shift_df = top_shift_df.sort_values(["Date", "Shift"])
    top_shift_df = top_shift_df[["Date", "Shift", "Top Picker", "Station Type", "Total Carts Counted"]]
    st.dataframe(top_shift_df, use_container_width=True)

    # ---- TOTAL CARTS PICKED PER SHIFT (PER DAY) ----
    st.subheader("Total Carts Picked Per Shift (per day)")
    carts_per_shift = (
        filtered.pivot_table(
            index="Date", 
            columns="Shift", 
            values="Carts Counted Per Hour",
            aggfunc="sum"
        )
        .fillna(0)
        .reset_index()
    )
    for shift in ["AM", "PM", "Night"]:
        if shift not in carts_per_shift.columns:
            carts_per_shift[shift] = 0
    carts_per_shift["Date"] = pd.to_datetime(carts_per_shift["Date"]).dt.strftime('%d-%m-%Y')
    carts_per_shift = carts_per_shift[["Date", "AM", "PM", "Night"]]
    st.dataframe(carts_per_shift, use_container_width=True)

    # ---- BREAKDOWN BY STATION TYPE AND SHIFT ----
    st.subheader("Carts Counted Per Hour by Station Type & Shift")
    breakdown = (
        filtered[filtered["Station Type"].astype(str).str.upper() != "ATLAS BOX & BOND BAGS"]
        .pivot_table(
            index="Station Type", 
            columns="Shift", 
            values="Carts Counted Per Hour",
            aggfunc="sum"
        )
        .fillna(0)
        .reset_index()
    )
    breakdown = breakdown[breakdown["Station Type"].notna()]
    breakdown = breakdown.sort_values("Station Type")
    st.dataframe(breakdown, use_container_width=True)
