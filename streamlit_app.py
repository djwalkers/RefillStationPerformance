import streamlit as st
import pandas as pd
import requests
from io import BytesIO
import matplotlib.pyplot as plt
from urllib.parse import quote
import re
from datetime import datetime

# ---- COLOR STYLES ----
PRIMARY_COLOR = "#DA362C"
BG_COLOR = "#DA362C"
FG_COLOR = "#FFFFFF"
AXIS_COLOR = "#333333"

st.set_page_config(page_title="Refill Station Performance Dashboard", layout="wide")

# ---- Custom CSS for background, fonts, Streamlit widgets
st.markdown(f"""
    <style>
    .stApp {{
        background-color: {BG_COLOR} !important;
        color: {FG_COLOR};
    }}
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {{
        color: {FG_COLOR};
    }}
    .st-bq {{
        background-color: {PRIMARY_COLOR} !important;
        color: {FG_COLOR};
    }}
    .stButton > button {{
        background-color: {PRIMARY_COLOR};
        color: {FG_COLOR};
        border: 1px solid {PRIMARY_COLOR};
    }}
    .stSelectbox, .stTextInput, .stDataFrame, .stTable {{
        background-color: #fff;
        color: {AXIS_COLOR};
    }}
    </style>
""", unsafe_allow_html=True)

st.title("Refill Station Performance Dashboard")

# --- CONFIG ---
GITHUB_USER = "djwalkers"
REPO_NAME = "RefillStationPerformance"
DATA_FOLDER = "Data"
FILES_FOLDER = "Files"

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
    m = re.search(r"\s?(\d{2})-(\d{2})\.csv$", fname)
    if m:
        hour, minute = m.groups()
        return f"{hour}:{minute}"
    else:
        try:
            s = fname[11:16].replace("-", ":").strip()
            return s if len(s) == 5 and ':' in s else None
        except:
            return None

data['Time'] = data['Source.Name'].apply(extract_time_from_filename)

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

# --- HELPER for displaying current week/month ---
def get_current_week_and_month():
    today = datetime.today()
    week = today.isocalendar()[1]
    month = today.strftime('%B')
    return week, month

current_week, current_month = get_current_week_and_month()

def make_leaderboard(df, value_column):
    """Aggregate by Users, ignore zeros, return sorted df."""
    temp = (
        df.groupby("Users", as_index=False)[value_column]
        .sum()
        .query(f"`{value_column}` != 0")
        .sort_values(value_column, ascending=True)
    )
    return temp

def show_bar_chart(df, x, y, title):
    fig, ax = plt.subplots(figsize=(8, max(5, len(df) * 0.35)))
    ax.barh(df[y], df[x], color=PRIMARY_COLOR, edgecolor=PRIMARY_COLOR)
    ax.set_xlabel(x.replace('_', ' '), color=FG_COLOR, weight="bold")
    ax.set_ylabel(y.replace('_', ' '), color=FG_COLOR, weight="bold")
    ax.set_title(title, color=FG_COLOR, weight="bold")
    ax.tick_params(axis='x', colors=FG_COLOR)
    ax.tick_params(axis='y', colors=FG_COLOR)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    plt.tight_layout()
    st.pyplot(fig)

def show_all_leaderboards(df, tag):
    st.subheader(f"Leaderboard: Carts Counted Per Hour by User ({tag})")
    leaderboard = make_leaderboard(df, "Carts Counted Per Hour")
    show_bar_chart(leaderboard, "Carts Counted Per Hour", "Users", "Carts Counted Per Hour by User")
    st.dataframe(leaderboard, use_container_width=True)
    st.download_button(
        f"Download {tag} Leaderboard (Carts) as CSV",
        leaderboard.to_csv(index=False),
        f"{tag.lower()}_dashboard_leaderboard_carts.csv"
    )
    # Rogues Processed
    st.subheader(f"Leaderboard: Rogues Processed by User ({tag})")
    rogues = make_leaderboard(df, "Rogues Processed")
    show_bar_chart(rogues, "Rogues Processed", "Users", "Rogues Processed by User")
    st.dataframe(rogues, use_container_width=True)
    st.download_button(
        f"Download {tag} Leaderboard (Rogues) as CSV",
        rogues.to_csv(index=False),
        f"{tag.lower()}_dashboard_leaderboard_rogues.csv"
    )
    # Damaged Drawers
    st.subheader(f"Leaderboard: Damaged Drawers Processed by User ({tag})")
    drawers = make_leaderboard(df, "Damaged Drawers Processed")
    show_bar_chart(drawers, "Damaged Drawers Processed", "Users", "Damaged Drawers Processed by User")
    st.dataframe(drawers, use_container_width=True)
    st.download_button(
        f"Download {tag} Leaderboard (Damaged Drawers) as CSV",
        drawers.to_csv(index=False),
        f"{tag.lower()}_dashboard_leaderboard_drawers.csv"
    )
    # Damaged Products
    st.subheader(f"Leaderboard: Damaged Products Processed by User ({tag})")
    products = make_leaderboard(df, "Damaged Products Processed")
    show_bar_chart(products, "Damaged Products Processed", "Users", "Damaged Products Processed by User")
    st.dataframe(products, use_container_width=True)
    st.download_button(
        f"Download {tag} Leaderboard (Damaged Products) as CSV",
        products.to_csv(index=False),
        f"{tag.lower()}_dashboard_leaderboard_products.csv"
    )

# --- 4. DASHBOARD TABS ---
tab1, tab2, tab3, tab4 = st.tabs([
    "Hourly Dashboard", 
    "Weekly Dashboard",
    "Monthly Dashboard",
    "Raw Data"
])

with tab1:
    st.header("Hourly Dashboard")
    filter_cols = []

    # Filters
    df = data.copy()
    # Month
    if "Month" in df.columns:
        months = df["Month"].dropna().unique()
        month_sel = st.selectbox("Select Month:", ["All"] + sorted(months.tolist()), key="hour_month")
        if month_sel != "All":
            df = df[df["Month"] == month_sel]
            filter_cols.append(f"Month={month_sel}")

    # Date
    if "Date" in df.columns:
        dates = df["Date"].dropna().unique()
        try:
            dates = pd.to_datetime(dates)
            dates = sorted(dates)
        except Exception:
            dates = sorted(dates.tolist())
        date_sel = st.selectbox("Select Date:", ["All"] + [str(d)[:10] for d in dates], key="hour_date")
        if date_sel != "All":
            df = df[df["Date"].astype(str).str[:10] == date_sel]
            filter_cols.append(f"Date={date_sel}")

    # Time
    if "Time" in df.columns:
        times = df["Time"].dropna().unique()
        times = sorted([str(t) for t in times if t is not None])
        time_sel = st.selectbox("Select Time:", ["All"] + times, key="hour_time")
        if time_sel != "All":
            df = df[df["Time"].astype(str) == time_sel]
            filter_cols.append(f"Time={time_sel}")

    # Station Type
    if "Station Type" in df.columns:
        station_types = df["Station Type"].dropna().unique()
        station_types = sorted([str(t) for t in station_types])
        station_sel = st.selectbox("Select Station Type:", ["All"] + station_types, key="hour_station")
        if station_sel != "All":
            df = df[df["Station Type"].astype(str) == station_sel]
            filter_cols.append(f"Station Type={station_sel}")

    st.write("**Filters applied:**", ", ".join(filter_cols) if filter_cols else "None")
    show_all_leaderboards(df, "Hourly")

with tab2:
    st.header("Weekly Dashboard")
    st.markdown(f"**Current Week Number:** {current_week}")
    filter_cols = []

    df = data.copy()
    # Week number filter
    if "Week" in df.columns:
        weeks = df["Week"].dropna().unique()
        weeks = sorted([str(int(w)) for w in weeks if pd.notnull(w)])
        week_sel = st.selectbox("Select Week Number:", ["All"] + weeks, key="week_week")
        if week_sel != "All":
            df = df[df["Week"].astype(str) == week_sel]
            filter_cols.append(f"Week={week_sel}")

    # Station Type
    if "Station Type" in df.columns:
        station_types = df["Station Type"].dropna().unique()
        station_types = sorted([str(t) for t in station_types])
        station_sel = st.selectbox("Select Station Type:", ["All"] + station_types, key="week_station")
        if station_sel != "All":
            df = df[df["Station Type"].astype(str) == station_sel]
            filter_cols.append(f"Station Type={station_sel}")

    st.write("**Filters applied:**", ", ".join(filter_cols) if filter_cols else "None")
    show_all_leaderboards(df, "Weekly")

with tab3:
    st.header("Monthly Dashboard")
    st.markdown(f"**Current Month:** {current_month}")
    filter_cols = []

    df = data.copy()
    # Month filter
    if "Month" in df.columns:
        months = df["Month"].dropna().unique()
        months = sorted(months.tolist())
        month_sel = st.selectbox("Select Month:", ["All"] + months, key="month_month")
        if month_sel != "All":
            df = df[df["Month"] == month_sel]
            filter_cols.append(f"Month={month_sel}")

    # Station Type
    if "Station Type" in df.columns:
        station_types = df["Station Type"].dropna().unique()
        station_types = sorted([str(t) for t in station_types])
        station_sel = st.selectbox("Select Station Type:", ["All"] + station_types, key="month_station")
        if station_sel != "All":
            df = df[df["Station Type"].astype(str) == station_sel]
            filter_cols.append(f"Station Type={station_sel}")

    st.write("**Filters applied:**", ", ".join(filter_cols) if filter_cols else "None")
    show_all_leaderboards(df, "Monthly")

with tab4:
    st.header("Raw Data")
    st.dataframe(data, use_container_width=True)
