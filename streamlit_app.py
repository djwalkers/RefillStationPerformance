import streamlit as st
import pandas as pd
import numpy as np
import requests
import base64
import matplotlib.pyplot as plt
from urllib.parse import quote
import re
from datetime import datetime

PRIMARY_COLOR = "#DA362C"
BG_COLOR = "#DA362C"
FG_COLOR = "#FFFFFF"
AXIS_COLOR = "#333333"
BAR_COLOR = "#FFFFFF"
BAR_EDGE = "#8B1A12"

st.set_page_config(page_title="Refill Station Performance Dashboard", layout="wide")

st.markdown("""
    <style>
    html, body, [data-testid="stAppViewContainer"], .stApp {
        background-color: #DA362C !important;
    }
    label, .stSelectbox label, .stTextInput label {
        color: #FFF !important;
        background-color: #DA362C !important;
        font-weight: bold !important;
        padding: 4px 8px !important;
        border-radius: 6px !important;
        display: inline-block !important;
    }
    </style>
""", unsafe_allow_html=True)

logo_url = "https://raw.githubusercontent.com/djwalkers/RefillStationPerformance/main/The%20Roc.png"
st.image(logo_url, width=180)
st.title("Refill Station Performance Dashboard")

# ---- Sticky Tab Navigation ----
tab_labels = [
    "Hourly Dashboard", 
    "Weekly Dashboard",
    "Monthly Dashboard",
    "High Performers"
]
if "active_tab" not in st.session_state:
    st.session_state["active_tab"] = tab_labels[0]

active_tab = st.selectbox(
    "Navigation",
    tab_labels,
    index=tab_labels.index(st.session_state["active_tab"]),
    key="main_tab"
)
st.session_state["active_tab"] = active_tab

# ---- FILENAME VALIDATION ----
def is_valid_filename(filename):
    pattern = r"^\d{2}-\d{2}-\d{4} \d{2}-\d{2}\.csv$"
    return bool(re.match(pattern, filename))

# ---- Automated File Uploader (GitHub API) ----
st.markdown("### Upload new data file to GitHub")
uploaded_file = st.file_uploader(
    "Add a CSV file to be pushed directly to the GitHub Data folder.",
    type="csv",
    help="Uploads to the Data folder in the GitHub repo using the GitHub API."
)

if uploaded_file is not None:
    if not is_valid_filename(uploaded_file.name):
        st.error(
            "❌ **Filename must be in the format 'DD-MM-YYYY HH-MM.csv'** "
            "(e.g., '14-02-2025 07-00.csv').\n\n"
            "Please rename your file and try again."
        )
        st.stop()
    else:
        try:
            github_token = st.secrets["github_token"]
            repo = "djwalkers/RefillStationPerformance"
            branch = "main"
            upload_path = f"Data/{uploaded_file.name}"

            api_url = f"https://api.github.com/repos/{repo}/contents/{upload_path}"

            headers = {
                "Authorization": f"Bearer {github_token}",
                "Accept": "application/vnd.github.v3+json"
            }
            r = requests.get(api_url, headers=headers)
            if r.status_code == 200 and "sha" in r.json():
                sha = r.json()["sha"]
            else:
                sha = None

            content_base64 = base64.b64encode(uploaded_file.getvalue()).decode()
            payload = {
                "message": f"Upload {uploaded_file.name} via Streamlit",
                "content": content_base64,
                "branch": branch
            }
            if sha:
                payload["sha"] = sha

            put_response = requests.put(api_url, headers=headers, json=payload)
            if put_response.status_code in [200, 201]:
                st.success(f"File '{uploaded_file.name}' uploaded to GitHub Data folder!")
                st.info("Refreshing dashboard to include the new data...")
                st.rerun()
            else:
                st.error(f"GitHub upload failed: {put_response.json().get('message', put_response.text)}")
        except Exception as ex:
            st.error(f"Uploader failed: {ex}")

# --- CONFIG ---
GITHUB_USER = "djwalkers"
REPO_NAME = "RefillStationPerformance"
DATA_FOLDER = "Data"
FILES_FOLDER = "Files"

# --- FILE URLS ---
date_dim_url = f"https://raw.githubusercontent.com/{GITHUB_USER}/{REPO_NAME}/main/{FILES_FOLDER}/Date%20Dimension%20Table.xlsx"
station_url = f"https://raw.githubusercontent.com/{GITHUB_USER}/{REPO_NAME}/main/{FILES_FOLDER}/Station%20Standard.xlsx"

# --- 1. LOAD RAW DATA FILES FROM GITHUB ---
@st.cache_data(show_spinner="Loading raw data from GitHub...")
def load_raw_data():
    github_token = st.secrets["github_token"]  # Same as your uploader
    headers = {
      "Authorization": f"Bearer {github_token}",
      "Accept": "application/vnd.github.v3+json"
}
response = requests.get(api_url, headers=headers)

    try:
        files_api = response.json()
    except Exception:
        st.error("Could not decode response from GitHub.")
        return pd.DataFrame()
    if not isinstance(files_api, list):
        msg = files_api.get('message', 'Unknown error')
        st.error(f"Error loading file list from GitHub: {msg}")
        return pd.DataFrame()
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

# --- 3. DATA PROCESSING ---
data = raw_data.copy()
data = data.rename(columns={
    'textBox9': 'Station Id',
    'textBox10': 'User',
    'textBox13': 'Drawers Counted',
    'textBox17': 'Damaged Drawers Processed',
    'textBox19': 'Damaged Products Processed',
    'textBox21': 'Rogues Processed',
})
drop_cols = [
    "textBox14", "textBox15", "textBox16", "textBox34", "textBox31", "textBox23",
    "textBox24", "textBox25", "textBox26", "textBox28", "textBox29", "textBox30"
]
data = data.drop(columns=[c for c in drop_cols if c in data.columns], errors='ignore')

# Parse Date and Time from filename
data['Date'] = data['Source.Name'].str[:10]
data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y', errors='coerce')
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
data['Users'] = data['User'].astype(str).apply(lambda x: x.partition(' (')[0].replace('*','').strip())
data = data.drop(columns=['User'])

# Remove bad usernames before grouping
data = data[
    (data['Users'].astype(str).str.strip() != "") &
    (data['Users'].astype(str).str.strip() != "0")
]

# Merge with Date Dimension
if ('Date' in date_dim.columns) and all(x in date_dim.columns for x in ['Year', 'Month', 'Week']):
    data = data.merge(date_dim[['Date', 'Year', 'Month', 'Week']], on='Date', how='left')
if 'Station' in station.columns:
    data = data.merge(station.rename(columns={'Station': 'Station Id'}), on='Station Id', how='left')
elif 'Station Id' in station.columns:
    data = data.merge(station, on='Station Id', how='left')
data = data.rename(columns={
    'Type': 'Station Type',
    'Drawer Avg': 'Drawer Avg',
    'KPI': 'Station KPI'
})

# Safe calculation to avoid division by zero/infinity
if 'Drawers Counted' in data.columns and 'Drawer Avg' in data.columns:
    drawer_avg = pd.to_numeric(data['Drawer Avg'], errors='coerce').replace(0, pd.NA)
    data['Carts Counted Per Hour'] = (
        pd.to_numeric(data['Drawers Counted'], errors='coerce') / drawer_avg
    ).fillna(0).round(2).astype(float)
else:
    data['Carts Counted Per Hour'] = 0.0

def get_current_week_and_month():
    today = datetime.today()
    week = today.isocalendar()[1]
    month = today.strftime('%B')
    return week, month
current_week, current_month = get_current_week_and_month()

def clean_grouped_users(df, value_column):
    temp = (
        df.groupby("Users", as_index=False)[value_column]
        .sum()
    )
    temp = temp[
        (temp[value_column] > 0) &
        (temp["Users"].astype(str).str.strip() != "") &
        (temp["Users"].astype(str).str.strip() != "0")
    ]
    temp = temp.sort_values(value_column, ascending=False)
    return temp

def show_bar_chart(df, x, y, title, figsize=(10, 5), label_fontsize=10, axis_fontsize=11):
    if df.empty or x not in df.columns or y not in df.columns:
        st.info("No data to display for this selection.")
        return

    total = df[x].sum()
    if 0 < total < 1:
        total_label = f"{total:.2f}"
    else:
        total_label = f"{int(round(total))}"
    st.markdown(
        f"<div style='color:{FG_COLOR}; font-size:18px; font-weight:bold; margin-bottom:10px'>Total {x.replace('_',' ')}: {total_label}</div>",
        unsafe_allow_html=True,
    )

    df = df.sort_values(by=x, ascending=True)
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(df[y], df[x], color=BAR_COLOR, edgecolor=BAR_EDGE, linewidth=2)
    for bar in bars:
        width = bar.get_width()
        if width > 0:
            if 0 < width < 1:
                label = f"{width:.2f}"
            else:
                label = f"{int(round(width))}"
            ax.annotate(label,
                        xy=(width, bar.get_y() + bar.get_height() / 2),
                        xytext=(3, 0), textcoords="offset points",
                        ha='left', va='center', fontsize=label_fontsize, color=FG_COLOR, fontweight="bold")
    ax.set_xlabel(x.replace('_', ' '), color=FG_COLOR, weight="bold", fontsize=axis_fontsize)
    ax.set_ylabel(y.replace('_', ' '), color=FG_COLOR, weight="bold", fontsize=axis_fontsize)
    ax.set_title(title, color=FG_COLOR, weight="bold", fontsize=axis_fontsize+1)
    ax.tick_params(axis='y', colors=FG_COLOR, labelsize=label_fontsize-1)
    ax.tick_params(axis='x', colors=FG_COLOR, labelsize=label_fontsize)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    plt.tight_layout()
    st.pyplot(fig)

def dashboard_tab(df, tag, time_filters=True, week_filter=False, month_filter=False):
    filter_cols = []
    if month_filter and "Month" in df.columns:
        months = df["Month"].dropna().unique()
        month_sel = st.selectbox("Select Month:", ["All"] + sorted(months.tolist()), key=f"{tag}_month")
        if month_sel != "All":
            df = df[df["Month"] == month_sel]
            filter_cols.append(f"Month={month_sel}")
    if week_filter and "Week" in df.columns:
        weeks = df["Week"].dropna().unique()
        weeks = sorted([str(int(w)) for w in weeks if pd.notnull(w)])
        week_sel = st.selectbox("Select Week Number:", ["All"] + weeks, key=f"{tag}_week")
        if week_sel != "All":
            df = df[df["Week"].astype(str) == week_sel]
            filter_cols.append(f"Week={week_sel}")
    if time_filters and "Date" in df.columns:
        # ----- DD-MM-YYYY formatting for date filters -----
        dates = df["Date"].dropna().unique()
        try:
            dates_dt = pd.to_datetime(dates)
            dates_fmt = [d.strftime('%d-%m-%Y') for d in sorted(dates_dt)]
        except Exception:
            dates_fmt = [str(d)[:10] for d in sorted(dates.tolist())]
        date_sel = st.selectbox("Select Date:", ["All"] + dates_fmt, key=f"{tag}_date")
        if date_sel != "All":
            df = df[df["Date"].dt.strftime('%d-%m-%Y') == date_sel]
            filter_cols.append(f"Date={date_sel}")
    if time_filters and "Time" in df.columns:
        times = df["Time"].dropna().unique()
        times = sorted([str(t) for t in times if t is not None])
        time_sel = st.selectbox("Select Time:", ["All"] + times, key=f"{tag}_time")
        if time_sel != "All":
            df = df[df["Time"].astype(str) == time_sel]
            filter_cols.append(f"Time={time_sel}")
    if "Station Type" in df.columns:
        station_types = df["Station Type"].dropna().unique()
        station_types = sorted([str(t) for t in station_types])
        station_sel = st.selectbox("Select Station Type:", ["All"] + station_types, key=f"{tag}_station")
        if station_sel != "All":
            df = df[df["Station Type"].astype(str) == station_sel]
            filter_cols.append(f"Station Type={station_sel}")

    st.write("**Filters applied:**", ", ".join(filter_cols) if filter_cols else "None")

    if tag == "hourly":
        titles = {
            "main": "Carts Counted Per Hour",
            "rogues": "Rogues Processed",
            "drawers": "Damaged Drawers Processed",
            "products": "Damaged Products Processed",
        }
    elif tag == "weekly":
        titles = {
            "main": "Carts Counted Per Week",
            "rogues": "Rogues Processed",
            "drawers": "Damaged Drawers Processed",
            "products": "Damaged Products Processed",
        }
    elif tag == "monthly":
        titles = {
            "main": "Carts Counted Per Month",
            "rogues": "Rogues Processed",
            "drawers": "Damaged Drawers Processed",
            "products": "Damaged Products Processed",
        }
    else:
        titles = {
            "main": "Carts Counted Per Hour by User",
            "rogues": "Rogues Processed by User",
            "drawers": "Damaged Drawers Processed by User",
            "products": "Damaged Products Processed by User",
        }

    # --- ROW 1: Full-width main chart ---
    show_bar_chart(
        clean_grouped_users(df, "Carts Counted Per Hour"),
        "Carts Counted Per Hour", "Users", titles["main"], figsize=(14, 6), label_fontsize=11, axis_fontsize=12
    )

    # --- ROW 2: Three charts side by side below ---
    col1, col2, col3 = st.columns(3)
    with col1:
        show_bar_chart(
            clean_grouped_users(df, "Rogues Processed"),
            "Rogues Processed", "Users", titles["rogues"], figsize=(6, 6), label_fontsize=9, axis_fontsize=10
        )
    with col2:
        show_bar_chart(
            clean_grouped_users(df, "Damaged Drawers Processed"),
            "Damaged Drawers Processed", "Users", titles["drawers"], figsize=(6, 6), label_fontsize=9, axis_fontsize=10
        )
    with col3:
        show_bar_chart(
            clean_grouped_users(df, "Damaged Products Processed"),
            "Damaged Products Processed", "Users", titles["products"], figsize=(6, 6), label_fontsize=9, axis_fontsize=10
        )

# --------- TABS IMPLEMENTATION ---------
if active_tab == "Hourly Dashboard":
    st.header("Hourly Dashboard")
    dashboard_tab(data.copy(), "hourly", time_filters=True, week_filter=False, month_filter=True)

elif active_tab == "Weekly Dashboard":
    st.header("Weekly Dashboard")
    st.markdown(f"**Current Week Number:** {current_week}")
    dashboard_tab(data.copy(), "weekly", time_filters=False, week_filter=True, month_filter=False)

elif active_tab == "Monthly Dashboard":
    st.header("Monthly Dashboard")
    st.markdown(f"**Current Month:** {current_month}")
    dashboard_tab(data.copy(), "monthly", time_filters=False, week_filter=False, month_filter=True)

elif active_tab == "High Performers":
    st.header("High Performers")

    # ---- Month and Day Filters ----
    month_options = sorted(data['Date'].dt.strftime('%B').dropna().unique())
    month_sel = st.selectbox("Filter by Month:", ["All"] + month_options, key="summary_month")
    filtered_data = data.copy()
    if month_sel != "All":
        filtered_data = filtered_data[filtered_data['Date'].dt.strftime('%B') == month_sel]

    day_options = sorted(filtered_data['Date'].dt.strftime('%d-%m-%Y').dropna().unique())
    day_sel = st.selectbox("Filter by Day:", ["All"] + day_options, key="summary_day")
    if day_sel != "All":
        filtered_data = filtered_data[filtered_data['Date'].dt.strftime('%d-%m-%Y') == day_sel]

    # Assign shift based on time
    def assign_shift(time_str):
        if pd.isna(time_str):
            return "Unknown"
        try:
            hour, minute = map(int, str(time_str).split(":"))
        except:
            return "Unknown"
        total_minutes = hour * 60 + minute
        if 6*60 <= total_minutes <= 14*60:
            return "AM"
        elif 14*60 < total_minutes <= 22*60:
            return "PM"
        else:
            return "Night"
    filtered_data['Shift'] = filtered_data['Time'].apply(assign_shift)

    def ensure_shift_columns(df, index_col="Date"):
        for shift in ["AM", "PM", "Night"]:
            if shift not in df.columns:
                df[shift] = 0
        cols = [index_col, "AM", "PM", "Night"]
        df = df[[c for c in cols if c in df.columns]]
        for shift in ["AM", "PM", "Night"]:
            if shift in df.columns:
                df[shift] = df[shift].fillna(0)
        return df

    trophy = "🏆 "

    # --- Top Picker Per Day (Carts Counted Per Hour) with Station Type ---
    top_carts_day = (
        filtered_data.groupby(['Date', 'Users', 'Station Type'], as_index=False)['Carts Counted Per Hour'].sum()
    )
    idx = top_carts_day.groupby('Date')['Carts Counted Per Hour'].idxmax()
    top_picker_per_day = top_carts_day.loc[idx].reset_index(drop=True)
    top_picker_per_day = top_picker_per_day.rename(columns={'Users': 'Top Picker', 'Carts Counted Per Hour': 'Total Carts Counted Per Hour'})
    if not top_picker_per_day.empty:
        top_picker_per_day['Date'] = pd.to_datetime(top_picker_per_day['Date']).dt.strftime('%d-%m-%Y')
        top_picker_per_day['Top Picker'] = trophy + top_picker_per_day['Top Picker'].astype(str)
        top_picker_per_day['Total Carts Counted Per Hour'] = top_picker_per_day['Total Carts Counted Per Hour'].apply(
            lambda x: f"{x:.2f}" if 0 < x < 1 else f"{int(round(x))}"
        )
    st.subheader("Top Picker Per Day (Carts Counted Per Hour)")
    st.dataframe(top_picker_per_day[['Date', 'Top Picker', 'Station Type', 'Total Carts Counted Per Hour']], use_container_width=True, hide_index=True)

    # --- Top Picker Per Shift (Carts Counted Per Hour) with Station Type ---
    top_carts_shift = (
        filtered_data.groupby(['Date', 'Shift', 'Users', 'Station Type'], as_index=False)['Carts Counted Per Hour'].sum()
    )
    idx_shift = top_carts_shift.groupby(['Date', 'Shift'])['Carts Counted Per Hour'].idxmax()
    top_picker_per_shift = top_carts_shift.loc[idx_shift].reset_index(drop=True)
    top_picker_per_shift = top_picker_per_shift.rename(columns={'Users': 'Top Picker', 'Carts Counted Per Hour': 'Total Carts Counted Per Hour'})
    if not top_picker_per_shift.empty:
        top_picker_per_shift['Date'] = pd.to_datetime(top_picker_per_shift['Date']).dt.strftime('%d-%m-%Y')
        top_picker_per_shift['Top Picker'] = trophy + top_picker_per_shift['Top Picker'].astype(str)
        top_picker_per_shift['Total Carts Counted Per Hour'] = top_picker_per_shift['Total Carts Counted Per Hour'].apply(
            lambda x: f"{x:.2f}" if 0 < x < 1 else f"{int(round(x))}"
        )
    st.subheader("Top Picker Per Shift (Carts Counted Per Hour)")
    st.dataframe(top_picker_per_shift[['Date', 'Shift', 'Top Picker', 'Station Type', 'Total Carts Counted Per Hour']], use_container_width=True, hide_index=True)

    # --- Total Carts Counted Per Hour Per Shift (per day) ---
    carts_per_shift = (
        filtered_data.groupby(['Date', 'Shift'], as_index=False)['Carts Counted Per Hour'].sum()
        .rename(columns={'Carts Counted Per Hour': 'Carts Counted'})
        .pivot(index='Date', columns='Shift', values='Carts Counted')
        .reset_index()
    )
    carts_per_shift = ensure_shift_columns(carts_per_shift, index_col="Date")
    if not carts_per_shift.empty:
        carts_per_shift['Date'] = pd.to_datetime(carts_per_shift['Date']).dt.strftime('%d-%m-%Y')
    st.subheader("Total Carts Counted Per Hour Per Shift (per day)")
    st.dataframe(carts_per_shift, use_container_width=True, hide_index=True)

    # --- Breakdown by Station Type and Shift ---
    breakdown = (
        filtered_data.groupby(['Station Type', 'Shift'], as_index=False)['Carts Counted Per Hour'].sum()
        .rename(columns={'Carts Counted Per Hour': 'Carts Counted'})
        .pivot(index='Station Type', columns='Shift', values='Carts Counted')
        .reset_index()
    )
    breakdown = ensure_shift_columns(breakdown, index_col="Station Type")
    st.subheader("Carts Counted Per Hour by Station Type & Shift")
    st.dataframe(breakdown, use_container_width=True, hide_index=True)
