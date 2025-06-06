import streamlit as st
import pandas as pd
import numpy as np
import requests
import base64
import matplotlib.pyplot as plt
from urllib.parse import quote
import re
from datetime import datetime, timedelta
import time
import random

# --- CONFIG ---
GITHUB_USER = "djwalkers"
REPO_NAME = "RefillStationPerformance"
DATA_FOLDER = "Data"
FILES_FOLDER = "Files"

date_dim_url = f"https://raw.githubusercontent.com/{GITHUB_USER}/{REPO_NAME}/main/{FILES_FOLDER}/Date%20Dimension%20Table.xlsx"
station_url = f"https://raw.githubusercontent.com/{GITHUB_USER}/{REPO_NAME}/main/{FILES_FOLDER}/Station%20Standard.xlsx"

# --- AUTO-REFRESH ---
def auto_refresh(interval_sec=300):
    if 'last_refresh' not in st.session_state:
        st.session_state['last_refresh'] = time.time()
        st.session_state['cachebuster'] = random.randint(0, int(1e8))
    if time.time() - st.session_state['last_refresh'] > interval_sec:
        st.session_state['last_refresh'] = time.time()
        st.session_state['cachebuster'] = random.randint(0, int(1e8))
        st.rerun()
auto_refresh(300)
if 'cachebuster' not in st.session_state:
    st.session_state['cachebuster'] = random.randint(0, int(1e8))

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

tab_labels = [
    "Hourly Dashboard", 
    "Weekly Dashboard",
    "Monthly Dashboard",
    "High Performers",
    "Low Performers"
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

# --- File upload to GitHub ---
def is_valid_filename(filename):
    pattern = r"^\d{2}-\d{2}-\d{4} \d{2}-\d{2}\.csv$"
    return bool(re.match(pattern, filename))

st.markdown("### Upload new data file to GitHub")
uploaded_file = st.file_uploader(
    "Add a CSV file.",
    type="csv",
    help="Uploads to the Data folder"
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
                st.session_state['cachebuster'] = random.randint(0, int(1e8))
                st.experimental_rerun()
            else:
                st.error(f"GitHub upload failed: {put_response.json().get('message', put_response.text)}")
        except Exception as ex:
            st.error(f"Uploader failed: {ex}")

# --- Data Loading ---
@st.cache_data(show_spinner="Loading data")
def load_raw_data(cachebuster=0):
    api_url = f"https://api.github.com/repos/{GITHUB_USER}/{REPO_NAME}/contents/{DATA_FOLDER}"
    github_token = st.secrets["github_token"]
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

@st.cache_data(show_spinner="Loading reference tables...")
def load_reference_tables():
    date_dim = pd.read_excel(date_dim_url, sheet_name=0)
    station = pd.read_excel(station_url, sheet_name=0)
    return date_dim, station

raw_data = load_raw_data(st.session_state['cachebuster'])
date_dim, station = load_reference_tables()

if raw_data.empty:
    st.error("No data files found in the Data folder. Please check your GitHub repository.")
    st.stop()

# --- Data Processing ---
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

# --- Shift Assignment Function (KEY) ---
def assign_shift_and_shiftdate(row):
    time_str = row['Time']
    date = row['Date']
    try:
        hour, minute = map(int, str(time_str).split(":"))
    except:
        return pd.Series({"Shift": "Unknown", "ShiftDate": date})
    total_minutes = hour * 60 + minute
    if 6*60 <= total_minutes < 14*60:
        return pd.Series({"Shift": "AM", "ShiftDate": date})
    elif 14*60 <= total_minutes < 22*60:
        return pd.Series({"Shift": "PM", "ShiftDate": date})
    else:
        # Night shift: 22:00-23:59 + 00:00-05:59 (should be previous day)
        if 22*60 <= total_minutes <= 23*60+59:
            shiftdate = date
        else:  # 00:00–05:59
            shiftdate = date - pd.Timedelta(days=1)
        return pd.Series({"Shift": "Night", "ShiftDate": shiftdate})
data[['Shift', 'ShiftDate']] = data.apply(assign_shift_and_shiftdate, axis=1)

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

if 'Drawers Counted' in data.columns and 'Drawer Avg' in data.columns:
    drawer_avg = pd.to_numeric(data['Drawer Avg'], errors='coerce').replace(0, pd.NA)
    data['Carts Counted Per Hour'] = (
        pd.to_numeric(data['Drawers Counted'], errors='coerce') / drawer_avg
    ).fillna(0).round(2).astype(float)
    data = data.infer_objects(copy=False)
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

def show_bar_chart(df, x, y, title, figsize=(10, 5), label_fontsize=10, axis_fontsize=11, default_top_n=20):
    if df.empty or x not in df.columns or y not in df.columns:
        st.info("No data to display for this selection.")
        return
    max_n = min(50, len(df))
    n_users = st.slider("How many top users to display?", 5, max_n, default_top_n, key=f"slider_{title}")
    df_sorted = df.sort_values(by=x, ascending=False).head(n_users)
    users = df[y].unique()
    selected_users = st.multiselect("Or search/select users to display:", users, key=f"multiselect_{title}")
    if selected_users:
        df_sorted = df[df[y].isin(selected_users)]
    total = df[x].sum()
    total_label = f"{total:.2f}" if 0 < total < 1 else f"{int(round(total))}"
    st.markdown(
        f"<div style='color:{FG_COLOR}; font-size:18px; font-weight:bold; margin-bottom:10px'>Total {x.replace('_',' ')}: {total_label}</div>",
        unsafe_allow_html=True,
    )
    df_sorted = df_sorted.sort_values(by=x, ascending=True)
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(df_sorted[y], df_sorted[x], color=BAR_COLOR, edgecolor=BAR_EDGE, linewidth=2)
    for bar in bars:
        width = bar.get_width()
        if width > 0:
            label = f"{width:.2f}" if 0 < width < 1 else f"{int(round(width))}"
            ax.annotate(label,
                        xy=(width, bar.get_y() + bar.get_height() / 2),
                        xytext=(3, 0), textcoords="offset points",
                        ha='left', va='center', fontsize=label_fontsize, color=FG_COLOR, fontweight="bold")
    ax.set_xlabel(x.replace('_', ' '), color=FG_COLOR, weight="bold", fontsize=axis_fontsize)
    ax.set_ylabel(y.replace('_', ' '), color=FG_COLOR, weight="bold", fontsize=axis_fontsize)
    ax.set_title(title, color=FG_COLOR, weight="bold", fontsize=axis_fontsize+1)
    ax.tick_params(axis='y', colors=FG_COLOR, labelsize=max(label_fontsize-2, 8))
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
    if time_filters and "ShiftDate" in df.columns:
        dates = df["ShiftDate"].dropna().unique()
        try:
            dates_dt = pd.to_datetime(dates)
            dates_fmt = [d.strftime('%d-%m-%Y') for d in sorted(dates_dt)]
        except Exception:
            dates_fmt = [str(d)[:10] for d in sorted(dates.tolist())]
        date_sel = st.selectbox("Select ShiftDate (Reporting Day):", ["All"] + dates_fmt, key=f"{tag}_date")
        if date_sel != "All":
            df = df[df["ShiftDate"].dt.strftime('%d-%m-%Y') == date_sel]
            filter_cols.append(f"ShiftDate={date_sel}")
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

    titles = {
        "main": "Carts Counted Per Hour",
        "rogues": "Rogues Processed",
        "drawers": "Damaged Drawers Processed",
        "products": "Damaged Products Processed",
    }

    show_bar_chart(
        clean_grouped_users(df, "Carts Counted Per Hour"),
        "Carts Counted Per Hour", "Users", titles["main"], figsize=(14, 6), label_fontsize=11, axis_fontsize=12
    )
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
    month_options = sorted(data['ShiftDate'].dt.strftime('%B').dropna().unique())
    month_sel = st.selectbox("Filter by Month:", ["All"] + month_options, key="summary_month")
    filtered_data = data.copy()
    if month_sel != "All":
        filtered_data = filtered_data[filtered_data['ShiftDate'].dt.strftime('%B') == month_sel]
    day_options = sorted(filtered_data['ShiftDate'].dt.strftime('%d-%m-%Y').dropna().unique())
    day_sel = st.selectbox("Filter by Shift Date:", ["All"] + day_options, key="summary_day")
    if day_sel != "All":
        filtered_data = filtered_data[filtered_data['ShiftDate'].dt.strftime('%d-%m-%Y') == day_sel]

    trophy = "🏆 "
    def ensure_shift_columns(df, index_col="ShiftDate"):
        for shift in ["AM", "PM", "Night"]:
            if shift not in df.columns:
                df[shift] = 0
        cols = [index_col, "AM", "PM", "Night"]
        df = df[[c for c in cols if c in df.columns]]
        for shift in ["AM", "PM", "Night"]:
            if shift in df.columns:
                df[shift] = df[shift].fillna(0)
        return df

    # --- Top Picker Per ShiftDate (All Hours) ---
    top_carts_day = (
        filtered_data.groupby(['ShiftDate', 'Users', 'Station Type'], as_index=False)['Carts Counted Per Hour'].sum()
    )
    idx = top_carts_day.groupby('ShiftDate')['Carts Counted Per Hour'].idxmax()
    top_picker_per_day = top_carts_day.loc[idx].reset_index(drop=True)
    top_picker_per_day = top_picker_per_day.rename(columns={
        'Users': 'Top Picker',
        'Carts Counted Per Hour': 'Total Carts Counted'
    })

    if not top_picker_per_day.empty:
        station_types_per_user = (
            filtered_data.groupby(['ShiftDate', 'Users'])['Station Type']
            .apply(lambda sts: ', '.join(sorted(set(map(str, sts)))))
            .reset_index()
            .rename(columns={'Users': 'Top Picker', 'Station Type': 'All Station Types'})
        )
        top_picker_per_day = top_picker_per_day.merge(
            station_types_per_user,
            left_on=['ShiftDate', 'Top Picker'],
            right_on=['ShiftDate', 'Top Picker'],
            how='left'
        )
        top_picker_per_day['ShiftDate'] = pd.to_datetime(top_picker_per_day['ShiftDate']).dt.strftime('%d-%m-%Y')
        top_picker_per_day['Top Picker'] = trophy + top_picker_per_day['Top Picker'].astype(str)
        top_picker_per_day['Total Carts Counted'] = top_picker_per_day['Total Carts Counted'].apply(
            lambda x: f"{x:.2f}" if 0 < x < 1 else f"{int(round(x))}"
        )

    st.subheader("Top Picker Per Day (All Hours, by ShiftDate)")
    st.dataframe(
        top_picker_per_day[['ShiftDate', 'Top Picker', 'All Station Types', 'Total Carts Counted']],
        use_container_width=True,
        hide_index=True
    )

    # --- Drilldown selection ---
    drilldown_row = None
    if not top_picker_per_day.empty:
        drilldown_options = top_picker_per_day[['ShiftDate', 'Top Picker']].apply(lambda row: f"{row['ShiftDate']} - {row['Top Picker']}", axis=1)
        drilldown_selection = st.selectbox(
            "Drill down into picker details:",
            ["None"] + list(drilldown_options),
            index=0,
            key="drilldown_picker"
        )
        if drilldown_selection != "None":
            selected_date, selected_picker = drilldown_selection.split(" - ", 1)
            selected_picker = selected_picker.replace("🏆 ", "")
            drilldown_row = (selected_date, selected_picker)
    if drilldown_row:
        detailed_rows = data[
            (data['ShiftDate'].dt.strftime('%d-%m-%Y') == drilldown_row[0]) &
            (data['Users'] == drilldown_row[1])
        ].copy()
        if not detailed_rows.empty:
            st.markdown(f"### Details for {drilldown_row[1]} on {drilldown_row[0]} (ShiftDate)")
            show_cols = [
                "ShiftDate", "Time", "Station Id", "Station Type", "Drawers Counted",
                "Damaged Drawers Processed", "Damaged Products Processed", "Rogues Processed",
                "Carts Counted Per Hour", "Shift"
            ]
            show_cols = [c for c in show_cols if c in detailed_rows.columns]
            st.dataframe(
                detailed_rows[show_cols].sort_values("Time"),
                use_container_width=True, hide_index=True
            )
        else:
            st.info("No detailed records found for this picker on that day.")

elif active_tab == "Low Performers":
    st.header("Low Performers")
    month_options = sorted(data['ShiftDate'].dt.strftime('%B').dropna().unique())
    month_sel = st.selectbox("Filter by Month:", ["All"] + month_options, key="low_month")
    filtered_data = data.copy()
    if month_sel != "All":
        filtered_data = filtered_data[filtered_data['ShiftDate'].dt.strftime('%B') == month_sel]
    day_options = sorted(filtered_data['ShiftDate'].dt.strftime('%d-%m-%Y').dropna().unique())
    day_sel = st.selectbox("Filter by Shift Date:", ["All"] + day_options, key="low_day")
    if day_sel != "All":
        filtered_data = filtered_data[filtered_data['ShiftDate'].dt.strftime('%d-%m-%Y') == day_sel]

    turtle = "🐢 "
    def ensure_shift_columns(df, index_col="ShiftDate"):
        for shift in ["AM", "PM", "Night"]:
            if shift not in df.columns:
                df[shift] = 0
        cols = [index_col, "AM", "PM", "Night"]
        df = df[[c for c in cols if c in df.columns]]
        for shift in ["AM", "PM", "Night"]:
            if shift in df.columns:
                df[shift] = df[shift].fillna(0)
        return df

    low_carts_day = (
        filtered_data.groupby(['ShiftDate', 'Users', 'Station Type'], as_index=False)['Carts Counted Per Hour'].sum()
    )
    low_carts_day = low_carts_day[low_carts_day['Carts Counted Per Hour'] > 0]
    idx = low_carts_day.groupby('ShiftDate')['Carts Counted Per Hour'].idxmin()
    low_picker_per_day = low_carts_day.loc[idx].reset_index(drop=True)
    low_picker_per_day = low_picker_per_day.rename(columns={
        'Users': 'Low Picker',
        'Carts Counted Per Hour': 'Total Carts Counted'
    })

    if not low_picker_per_day.empty:
        station_types_per_user = (
            filtered_data.groupby(['ShiftDate', 'Users'])['Station Type']
            .apply(lambda sts: ', '.join(sorted(set(map(str, sts)))))
            .reset_index()
            .rename(columns={'Users': 'Low Picker', 'Station Type': 'All Station Types'})
        )
        low_picker_per_day = low_picker_per_day.merge(
            station_types_per_user,
            left_on=['ShiftDate', 'Low Picker'],
            right_on=['ShiftDate', 'Low Picker'],
            how='left'
        )
        low_picker_per_day['ShiftDate'] = pd.to_datetime(low_picker_per_day['ShiftDate']).dt.strftime('%d-%m-%Y')
        low_picker_per_day['Low Picker'] = turtle + low_picker_per_day['Low Picker'].astype(str)
        low_picker_per_day['Total Carts Counted'] = low_picker_per_day['Total Carts Counted'].apply(
            lambda x: f"{x:.2f}" if 0 < x < 1 else f"{int(round(x))}"
        )

    st.subheader("Lowest Picker Per Day (All Hours, by ShiftDate)")
    st.dataframe(
        low_picker_per_day[['ShiftDate', 'Low Picker', 'All Station Types', 'Total Carts Counted']],
        use_container_width=True,
        hide_index=True
    )

    drilldown_row = None
    if not low_picker_per_day.empty:
        drilldown_options = low_picker_per_day[['ShiftDate', 'Low Picker']].apply(lambda row: f"{row['ShiftDate']} - {row['Low Picker']}", axis=1)
        drilldown_selection = st.selectbox(
            "Drill down into picker details:",
            ["None"] + list(drilldown_options),
            index=0,
            key="low_drilldown_picker"
        )
        if drilldown_selection != "None":
            selected_date, selected_picker = drilldown_selection.split(" - ", 1)
            selected_picker = selected_picker.replace("🐢 ", "")
            drilldown_row = (selected_date, selected_picker)
    if drilldown_row:
        detailed_rows = data[
            (data['ShiftDate'].dt.strftime('%d-%m-%Y') == drilldown_row[0]) &
            (data['Users'] == drilldown_row[1])
        ].copy()
        if not detailed_rows.empty:
            st.markdown(f"### Details for {drilldown_row[1]} on {drilldown_row[0]} (ShiftDate)")
            show_cols = [
                "ShiftDate", "Time", "Station Id", "Station Type", "Drawers Counted",
                "Damaged Drawers Processed", "Damaged Products Processed", "Rogues Processed",
                "Carts Counted Per Hour", "Shift"
            ]
            show_cols = [c for c in show_cols if c in detailed_rows.columns]
            st.dataframe(
                detailed_rows[show_cols].sort_values("Time"),
                use_container_width=True, hide_index=True
            )
        else:
            st.info("No detailed records found for this picker on that day.")
