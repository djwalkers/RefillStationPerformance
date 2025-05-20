import streamlit as st
import pandas as pd
import requests
from io import BytesIO
import matplotlib.pyplot as plt
from urllib.parse import quote
import re

# --- CONFIG ---
GITHUB_USER = "djwalkers"
REPO_NAME = "RefillStationPerformance"
DATA_FOLDER = "Data"
FILES_FOLDER = "Files"

st.set_page_config(page_title="Refill Station Performance Dashboard", layout="wide")
st.title("Refill Station Performance Dashboard")

# --- FILE URLS ---
date_dim_url = f"https://raw.githubusercontent.com/{GITHUB_USER}/{REPO_NAME}/main/{FILES_FOLDER}/Date%20Dimension%20Table.xlsx"
station_url = f"https://raw.githubusercontent.com/{GITHUB_USER}/{REPO_NAME}/main/{FILES_FOLDER}/Station%20Standard.xlsx"

# --- FILE ACCESSIBILITY TEST ---
st.info("🔎 Checking access to reference Excel files on GitHub...")
for url, label in [
    (date_dim_url, "Date Dimension Table.xlsx"),
    (station_url, "Station Standard.xlsx")
]:
    response = requests.get(url)
    if response.status_code == 200 and len(response.content) > 0:
        st.success(f"✅ {label} found! ({len(response.content)} bytes)")
    else:
        st.error(f"❌ {label} NOT found at expected URL. Check the file name, folder, or GitHub branch.")
        st.stop()

# --- 1. LOAD RAW DATA FILES FROM GITHUB ---
@st.cache_data(show_spinner="Loading raw data from GitHub...")
def load_raw_data():
    api_url = f"https://api.github.com/repos/{GITHUB_USER}/{REPO_NAME}/contents/{DATA_FOLDER}"
    files_api = requests.get(api_url).json()
    data_files = [f['name'] for f in files_api if f['name'].endswith('.csv')]
    dfs = []
    for fname in data_files:
        file_url = f"https://raw.githubusercontent.com/{GITHUB_USER}/{REPO_NAME}/main/{DATA_FOLDER}/{quote(fname)}"
        try:
            df = pd.read_csv(file_url)
            df['Source.Name'] = fname
            dfs.append(df)
        except Exception as e:
            st.warning(f"Error loading {fname}: {e}")
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    else:
        return pd.DataFrame()

# --- 2. LOAD DIMENSION TABLES ---
@st.cache_data(show_spinner="Loading reference tables...")
def load_reference_tables():
    date_dim = pd.read_excel(date_dim_url, sheet_name=0)
    station = pd.read_excel(station_url, sheet_name=0)
    return date_dim, station

raw_data = load_raw_data()
date_dim, station = load_reference_tables()

if raw_data.empty:
    st.error("No data files found in the Data folder. Please check your GitHub repository.")
    st.stop()

# --- 3. DATA PROCESSING / POWER QUERY LOGIC ---
data = raw_data.copy()

# Standardize column names if needed (edit as your columns change)
data = data.rename(columns={
    'textBox9': 'Station Id',
    'textBox10': 'User',
    'textBox13': 'Drawers Counted',
    'textBox17': 'Damaged Drawers Processed',
    'textBox19': 'Damaged Products Processed',
    'textBox21': 'Rogues Processed',
})

# Drop unused columns if present
drop_cols = [
    "textBox14", "textBox15", "textBox16", "textBox34", "textBox31", "textBox23",
    "textBox24", "textBox25", "textBox26", "textBox28", "textBox29", "textBox30"
]
data = data.drop(columns=[c for c in drop_cols if c in data.columns], errors='ignore')

# Parse Date from filename
data['Date'] = data['Source.Name'].str[:10]
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')

# --- Improved Time Extraction from filename ---
def extract_time_from_filename(fname):
    # Expects filenames like '14-02-2025 07-00.csv'
    m = re.search(r"\s?(\d{2})-(\d{2})\.csv$", fname)
    if m:
        hour, minute = m.groups()
        return f"{hour}:{minute}"
    else:
        # Try to get the 11:16 slice, fallback
        try:
            s = fname[11:16].replace("-", ":").strip()
            return s if len(s) == 5 and ':' in s else None
        except:
            return None

data['Time'] = data['Source.Name'].apply(extract_time_from_filename)

# (Optional) Show filename and parsed time for debug/confirmation
st.subheader("Filename and Parsed Time Preview")
st.dataframe(data[['Source.Name', 'Time']].head(20))

# Clean Users column (from Power Query logic)
data['Users'] = data['User'].astype(str).apply(lambda x: x.partition(' (')[0].replace('*','').strip())
data = data.drop(columns=['User'])

# Merge with Date Dimension
if ('Date' in date_dim.columns) and all(x in date_dim.columns for x in ['Year', 'Month', 'Week']):
    data = data.merge(date_dim[['Date', 'Year', 'Month', 'Week']], on='Date', how='left')
else:
    st.warning("Could not find 'Date', 'Year', 'Month', 'Week' columns in the Date Dimension Table.")

# Merge with Station table
if 'Station' in station.columns:
    data = data.merge(station.rename(columns={'Station': 'Station Id'}), on='Station Id', how='left')
elif 'Station Id' in station.columns:
    data = data.merge(station, on='Station Id', how='left')
else:
    st.warning("Could not find 'Station' or 'Station Id' column in Station Standard.")

# Rename for clarity
data = data.rename(columns={
    'Type': 'Station Type',
    'Drawer Avg': 'Drawer Avg',
    'KPI': 'Station KPI'
})

# Calculate Carts Counted Per Hour
if 'Drawers Counted' in data.columns and 'Drawer Avg' in data.columns:
    data['Carts Counted Per Hour'] = (data['Drawers Counted'] / data['Drawer Avg']).round(2)
else:
    data['Carts Counted Per Hour'] = 0

# --- 4. HOURLY DASHBOARD TAB ---
tab1, tab2 = st.tabs(["Hourly Dashboard", "Raw Data"])

with tab1:
    st.header("Hourly Dashboard")
    filter_cols = []

    # Filters
    # Month
    if "Month" in data.columns:
        months = data["Month"].dropna().unique()
        month_sel = st.selectbox("Select Month:", ["All"] + sorted(months.tolist()))
        if month_sel != "All":
            data = data[data["Month"] == month_sel]
            filter_cols.append(f"Month={month_sel}")

    # Date
    if "Date" in data.columns:
        dates = data["Date"].dropna().unique()
        try:
            dates = pd.to_datetime(dates)
            dates = sorted(dates)
        except Exception:
            dates = sorted(dates.tolist())
        date_sel = st.selectbox("Select Date:", ["All"] + [str(d)[:10] for d in dates])
        if date_sel != "All":
            data = data[data["Date"].astype(str).str[:10] == date_sel]
            filter_cols.append(f"Date={date_sel}")

    # Time
    if "Time" in data.columns:
        times = data["Time"].dropna().unique()
        times = sorted([str(t) for t in times if t is not None])
        time_sel = st.selectbox("Select Time:", ["All"] + times)
        if time_sel != "All":
            data = data[data["Time"].astype(str) == time_sel]
            filter_cols.append(f"Time={time_sel}")

    # Station Type
    if "Station Type" in data.columns:
        station_types = data["Station Type"].dropna().unique()
        station_types = sorted([str(t) for t in station_types])
        station_sel = st.selectbox("Select Station Type:", ["All"] + station_types)
        if station_sel != "All":
            data = data[data["Station Type"].astype(str) == station_sel]
            filter_cols.append(f"Station Type={station_sel}")

    st.write("**Filters applied:**", ", ".join(filter_cols) if filter_cols else "None")

    # Leaderboard (Users by Carts Counted Per Hour)
    leaderboard = (
        data.groupby("Users", as_index=False)["Carts Counted Per Hour"]
        .sum()
        .sort_values("Carts Counted Per Hour", ascending=True)
    )
    st.subheader("Leaderboard: Carts Counted Per Hour by User")

    # Bar chart
    fig, ax = plt.subplots(figsize=(8, max(5, len(leaderboard) * 0.35)))
    ax.barh(leaderboard["Users"], leaderboard["Carts Counted Per Hour"])
    ax.set_xlabel("Carts Counted Per Hour")
    ax.set_ylabel("User")
    ax.set_title("Carts Counted Per Hour by User")
    st.pyplot(fig)

    # Table
    st.dataframe(leaderboard, use_container_width=True)

    # Download
    st.download_button(
        "Download Leaderboard as CSV",
        leaderboard.to_csv(index=False),
        "hourly_dashboard_leaderboard.csv"
    )

with tab2:
    st.header("Raw Data")
    st.dataframe(data, use_container_width=True)
