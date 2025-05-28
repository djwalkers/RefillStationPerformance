import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from datetime import datetime

# ---- PAGE CONFIG ----
st.set_page_config(
    page_title="Refill Station Performance Dashboard",
    layout="wide",
    page_icon=":bar_chart:"
)

# ---- COLOR SCHEME ----
MAIN_COLOR = "#DA362C"
WHITE = "#FFF"
st.markdown(
    f"""
    <style>
    body, .stApp {{ background-color: {MAIN_COLOR} !important; }}
    .block-container {{
        padding-top: 1rem;
    }}
    .stDataFrame .e1eexj620, .stDataFrame .e1eexj610 {{
        color: {MAIN_COLOR};
        font-weight: bold;
    }}
    .stTabs [data-baseweb="tab-list"] {{
        background: {MAIN_COLOR};
    }}
    .stTabs [data-baseweb="tab"] {{
        color: {WHITE};
    }}
    .stTabs [aria-selected="true"] {{
        color: {MAIN_COLOR};
        background: {WHITE};
        border-radius: 10px 10px 0 0;
    }}
    .stSelectbox label, .st-mr, .stTextInput label, .stMarkdown {{
        color: {MAIN_COLOR};
        font-weight: bold;
    }}
    .st-bj, .st-bs, .st-cc, .st-ci {{
        background: {WHITE} !important;
        color: {MAIN_COLOR} !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---- LOGO ----
st.image("The Roc.png", width=120)

# ---- GITHUB DATA ----
DATA_FOLDER_URL = "https://raw.githubusercontent.com/djwalkers/RefillStationPerformance/main/Data/"
FILES_FOLDER_URL = "https://raw.githubusercontent.com/djwalkers/RefillStationPerformance/main/Files/"

def get_file_list_from_github(folder_url):
    api_url = "https://api.github.com/repos/djwalkers/RefillStationPerformance/contents/Data"
    response = requests.get(api_url)
    files = []
    if response.status_code == 200:
        for file in response.json():
            if file["name"].endswith(".csv"):
                files.append(file["name"])
    return sorted(files)

@st.cache_data(ttl=300)
def load_data():
    file_list = get_file_list_from_github(DATA_FOLDER_URL)
    dfs = []
    for filename in file_list:
        # Validate file name: dd-mm-yyyy hh-mm.csv (strict)
        try:
            base = filename.replace(".csv", "")
            dt = datetime.strptime(base, "%d-%m-%Y %H-%M")
        except ValueError:
            continue  # skip non-matching files
        url = DATA_FOLDER_URL + filename.replace(" ", "%20")
        try:
            df = pd.read_csv(url)
            df["Source.Name"] = filename.replace(".csv", "")
            dfs.append(df)
        except Exception as e:
            st.warning(f"Error loading {filename}: {e}")
    if not dfs:
        return pd.DataFrame()
    data = pd.concat(dfs, ignore_index=True)
    # Clean and rename columns
    rename_map = {
        "textBox9": "Station Id", "textBox10": "User",
        "textBox13": "Drawers Counted", "textBox17": "Damaged Drawers Processed",
        "textBox19": "Damaged Products Processed", "textBox21": "Rogues Processed"
    }
    data = data.rename(columns=rename_map)
    # Drop unwanted columns
    drop_cols = [col for col in data.columns if col.startswith("textBox") and col not in rename_map]
    data = data.drop(columns=drop_cols, errors="ignore")
    # Fix User (*** FIXED regex=False ***)
    data["Users"] = data["User"].astype(str).str.split(" (", n=1, regex=False).str[0].str.replace("*", "").str.strip()
    # Date and Time from filename
    data["Date"] = data["Source.Name"].str[:10]
    data["Time"] = data["Source.Name"].str[11:16].str.replace("-", ":")
    data["Date"] = pd.to_datetime(data["Date"], format="%d-%m-%Y", errors="coerce")
    data["Time"] = data["Time"].replace("nan", np.nan)
    # Remove null/empty users
    data = data[data["Users"].notna() & (data["Users"] != "")]
    # More cleaning
    for col in ["Drawers Counted", "Damaged Drawers Processed", "Damaged Products Processed", "Rogues Processed"]:
        data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0).astype(int)
    # ---- Load Reference Tables ----
    # Date Dimension
    date_dim_url = FILES_FOLDER_URL + "Date%20Dimension%20Table.xlsx"
    date_dim = pd.read_excel(date_dim_url, sheet_name="Sheet1")
    date_dim["Date"] = pd.to_datetime(date_dim["Date"])
    # Station Table
    station_url = FILES_FOLDER_URL + "Station%20Standard.xlsx"
    station = pd.read_excel(station_url, sheet_name="Station")
    # Merge Date Dim
    data = pd.merge(data, date_dim[["Date", "Year", "Month", "Week"]], on="Date", how="left")
    # Merge Station Info
    data = pd.merge(data, station[["Station", "Type", "Drawer Avg", "KPI"]], left_on="Station Id", right_on="Station", how="left")
    data = data.rename(columns={
        "Type": "Station Type",
        "Drawer Avg": "Drawer Avg",
        "KPI": "Station KPI"
    })
    # Carts Counted Per Hour calculation
    data["Carts Counted Per Hour"] = np.where(
        data["Drawer Avg"].fillna(0) > 0,
        np.round(data["Drawers Counted"] / data["Drawer Avg"], 2),
        0
    )
    # Remove excluded users
    excluded_users = [
        "AARON DAVIES", "ADAM DAVENHILL", "ANDY WALKER", "ANNA DZIEDZIC-WIDZICKA", "BEN NORBURY",
        "CHARLOTTE BIBBY", "DANIEL ROGERSON", "DOMINIC PASKIN", "GEORGE SMITH", "JEFFERY FLETCHER",
        "MARCIN SZABLINSKI", "MARCOS CHAINIUK", "MARK BANHAM", "MAUREEN OUGHTON", "MAX CHAPPEL",
        "MICHAEL RUSHTON", "MICHAL ROWDO", "PIETER DAVIDS", "ROGER COULSON", "ROXANNE HAYNES",
        "SAM BENNETT", "STUART FOWLES", "TAMMY HITCHMOUGH", "TAYLOR MADDOCK", "TAYLOR MADDOX", "VASELA VELKOVA"
    ]
    data = data[~data["Users"].isin(excluded_users)]
    data = data.reset_index(drop=True)
    return data

data = load_data()

# ---- UPLOAD SECTION ----
with st.sidebar:
    st.header("Upload New Data File")
    uploaded = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded is not None:
        # Filename validation: dd-mm-yyyy hh-mm.csv
        filename = uploaded.name
        try:
            datetime.strptime(filename.replace(".csv", ""), "%d-%m-%Y %H-%M")
            is_valid = True
        except ValueError:
            is_valid = False
        if not is_valid:
            st.error("❌ Filename must be in format: DD-MM-YYYY HH-MM.csv")
        else:
            st.success("✅ File accepted. (Manual add to GitHub required)")
    st.markdown("---")
    st.markdown(
        "<div style='color: #FFF; background: #DA362C; border-radius: 4px; padding: 8px; font-size: 15px; text-align: center;'>"
        "<b>Refill Station Performance Report</b></div>",
        unsafe_allow_html=True
    )

# ---- TAB NAVIGATION ----
tabs = ["Hourly Dashboard", "Weekly Dashboard", "Monthly Dashboard", "High Performers"]
active_tab = st.selectbox("Select Dashboard Tab", tabs, key="main_tabs")

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

def dashboard_tab(df, period="hourly", **filter_args):
    df = df.copy()
    # ---- Filters ----
    filter_cols = st.columns([1, 1, 1, 1])
    with filter_cols[0]:
        month_options = sorted(df['Month'].dropna().unique(), key=lambda x: str(x))
        month_sel = st.selectbox("Month", ["All"] + month_options, key=f"{period}_month")
    with filter_cols[1]:
        week_options = sorted(df['Week'].dropna().unique())
        week_sel = st.selectbox("Week", ["All"] + [str(w) for w in week_options], key=f"{period}_week")
    with filter_cols[2]:
        date_options = sorted(df['Date'].dt.strftime('%d-%m-%Y').dropna().unique())
        date_sel = st.selectbox("Date", ["All"] + date_options, key=f"{period}_date")
    with filter_cols[3]:
        station_type_options = sorted(df['Station Type'].dropna().astype(str).unique())
        station_type_sel = st.selectbox("Station Type", ["All"] + station_type_options, key=f"{period}_station_type")
    # Apply filters
    if month_sel != "All":
        df = df[df['Month'] == month_sel]
    if week_sel != "All":
        df = df[df['Week'].astype(str) == week_sel]
    if date_sel != "All":
        df = df[df['Date'].dt.strftime('%d-%m-%Y') == date_sel]
    if station_type_sel != "All":
        df = df[df['Station Type'] == station_type_sel]
    # Assign shift for breakdowns
    df['Shift'] = df['Time'].apply(assign_shift)

    # ---- Layout: 1 chart full width, 3 charts below ----
    st.markdown("### Carts Counted Per Hour")
    fig, ax = plt.subplots(figsize=(14, 4.5))
    agg = (
        df.groupby('Users', as_index=False)['Carts Counted Per Hour'].sum()
        .query("`Carts Counted Per Hour` > 0")
        .sort_values('Carts Counted Per Hour', ascending=True)
    )
    ax.barh(agg['Users'], agg['Carts Counted Per Hour'], color=MAIN_COLOR, edgecolor=WHITE)
    for i, v in enumerate(agg['Carts Counted Per Hour']):
        val_str = f"{v:.2f}" if 0 < v < 1 else f"{int(round(v))}"
        ax.text(v + 0.05, i, val_str, color=WHITE, va='center', fontsize=11, fontweight='bold')
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_yticklabels(agg['Users'], fontsize=10)
    ax.set_facecolor(WHITE)
    ax.tick_params(axis='x', colors=MAIN_COLOR)
    ax.tick_params(axis='y', colors=MAIN_COLOR)
    plt.tight_layout()
    st.pyplot(fig)

    # ---- 3 charts below in columns ----
    cols = st.columns(3)
    # Rogues Processed
    with cols[0]:
        st.markdown("#### Rogues Processed")
        fig1, ax1 = plt.subplots(figsize=(5, 3.5))
        rogues = (
            df.groupby('Users', as_index=False)['Rogues Processed'].sum()
            .query("`Rogues Processed` > 0")
            .sort_values('Rogues Processed', ascending=True)
        )
        ax1.barh(rogues['Users'], rogues['Rogues Processed'], color=MAIN_COLOR, edgecolor=WHITE)
        for i, v in enumerate(rogues['Rogues Processed']):
            ax1.text(v + 0.05, i, int(v), color=WHITE, va='center', fontsize=11, fontweight='bold')
        ax1.set_yticklabels(rogues['Users'], fontsize=9)
        ax1.set_facecolor(WHITE)
        ax1.tick_params(axis='x', colors=MAIN_COLOR)
        ax1.tick_params(axis='y', colors=MAIN_COLOR)
        plt.tight_layout()
        st.pyplot(fig1)
    # Damaged Drawers Processed
    with cols[1]:
        st.markdown("#### Damaged Drawers")
        fig2, ax2 = plt.subplots(figsize=(5, 3.5))
        damaged = (
            df.groupby('Users', as_index=False)['Damaged Drawers Processed'].sum()
            .query("`Damaged Drawers Processed` > 0")
            .sort_values('Damaged Drawers Processed', ascending=True)
        )
        ax2.barh(damaged['Users'], damaged['Damaged Drawers Processed'], color=MAIN_COLOR, edgecolor=WHITE)
        for i, v in enumerate(damaged['Damaged Drawers Processed']):
            ax2.text(v + 0.05, i, int(v), color=WHITE, va='center', fontsize=11, fontweight='bold')
        ax2.set_yticklabels(damaged['Users'], fontsize=9)
        ax2.set_facecolor(WHITE)
        ax2.tick_params(axis='x', colors=MAIN_COLOR)
        ax2.tick_params(axis='y', colors=MAIN_COLOR)
        plt.tight_layout()
        st.pyplot(fig2)
    # Damaged Products Processed
    with cols[2]:
        st.markdown("#### Damaged Products")
        fig3, ax3 = plt.subplots(figsize=(5, 3.5))
        damaged_p = (
            df.groupby('Users', as_index=False)['Damaged Products Processed'].sum()
            .query("`Damaged Products Processed` > 0")
            .sort_values('Damaged Products Processed', ascending=True)
        )
        ax3.barh(damaged_p['Users'], damaged_p['Damaged Products Processed'], color=MAIN_COLOR, edgecolor=WHITE)
        for i, v in enumerate(damaged_p['Damaged Products Processed']):
            ax3.text(v + 0.05, i, int(v), color=WHITE, va='center', fontsize=11, fontweight='bold')
        ax3.set_yticklabels(damaged_p['Users'], fontsize=9)
        ax3.set_facecolor(WHITE)
        ax3.tick_params(axis='x', colors=MAIN_COLOR)
        ax3.tick_params(axis='y', colors=MAIN_COLOR)
        plt.tight_layout()
        st.pyplot(fig3)

# ---- DASHBOARD TABS ----
if active_tab == "Hourly Dashboard":
    st.header("Hourly Dashboard")
    dashboard_tab(data.copy(), "hourly")

elif active_tab == "Weekly Dashboard":
    st.header("Weekly Dashboard")
    current_week = data["Week"].dropna().max()
    st.markdown(f"**Current Week:** {current_week}")
    dashboard_tab(data.copy(), "weekly")

elif active_tab == "Monthly Dashboard":
    st.header("Monthly Dashboard")
    current_month = data["Month"].dropna().astype(str).max()
    st.markdown(f"**Current Month:** {current_month}")
    dashboard_tab(data.copy(), "monthly")

elif active_tab == "High Performers":
    st.header("High Performers")
    # Month and Day Filters
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
    filtered_data['Shift'] = filtered_data['Time'].apply(assign_shift)

    trophy = "🏆 "

    # --- Top Picker Per Day (Carts Counted Per Hour) with Station Type ---
    daily_totals = (
        filtered_data.groupby(['Date', 'Users'], as_index=False)['Carts Counted Per Hour'].sum()
    )
    idx = daily_totals.groupby('Date')['Carts Counted Per Hour'].idxmax()
    top_picker_per_day = daily_totals.loc[idx].reset_index(drop=True)
    top_picker_per_day = top_picker_per_day.rename(
        columns={'Users': 'Top Picker', 'Carts Counted Per Hour': 'Total Carts Counted Per Hour'}
    )
    # Get main station type for the day
    main_station = (
        filtered_data.groupby(['Date', 'Users', 'Station Type'], as_index=False)['Carts Counted Per Hour'].sum()
    )
    main_station_idx = main_station.groupby(['Date', 'Users'])['Carts Counted Per Hour'].idxmax()
    main_station = main_station.loc[main_station_idx][['Date', 'Users', 'Station Type']]
    top_picker_per_day = top_picker_per_day.merge(
        main_station,
        left_on=['Date', 'Top Picker'],
        right_on=['Date', 'Users'],
        how='left'
    ).drop(columns=['Users'])

    if not top_picker_per_day.empty:
        top_picker_per_day['Date'] = pd.to_datetime(top_picker_per_day['Date']).dt.strftime('%d-%m-%Y')
        top_picker_per_day['Top Picker'] = trophy + top_picker_per_day['Top Picker'].astype(str)
        top_picker_per_day['Total Carts Counted Per Hour'] = top_picker_per_day['Total Carts Counted Per Hour'].apply(
            lambda x: f"{x:.2f}" if 0 < x < 1 else f"{int(round(x))}"
        )
    st.subheader("Top Picker Per Day (All Hours, Calendar Day)")
    st.markdown(
        "*Note: This table sums all picks by each user within the full calendar day, regardless of shift boundaries. "
        "A user’s total may differ from the sum of their per-shift totals if their activity crosses shift times.*",
        unsafe_allow_html=True
    )
    top_picker_per_day_display = top_picker_per_day[['Date', 'Top Picker', 'Station Type', 'Total Carts Counted Per Hour']].reset_index(drop=True)
    st.dataframe(
        top_picker_per_day_display,
        use_container_width=True,
        hide_index=True
    )

    # --- Top Picker Per Shift (Carts Counted Per Hour) ---
    shift_totals = (
        filtered_data.groupby(['Date', 'Shift', 'Users', 'Station Type'], as_index=False)['Carts Counted Per Hour'].sum()
    )
    idx = shift_totals.groupby(['Date', 'Shift'])['Carts Counted Per Hour'].idxmax()
    top_picker_per_shift = shift_totals.loc[idx].reset_index(drop=True)
    shift_order = ['AM', 'PM', 'Night']
    top_picker_per_shift['Shift'] = pd.Categorical(top_picker_per_shift['Shift'], categories=shift_order, ordered=True)
    top_picker_per_shift = top_picker_per_shift.sort_values(['Date', 'Shift'])
    if not top_picker_per_shift.empty:
        top_picker_per_shift['Date'] = pd.to_datetime(top_picker_per_shift['Date']).dt.strftime('%d-%m-%Y')
        top_picker_per_shift['Top Picker'] = trophy + top_picker_per_shift['Users'].astype(str)
        top_picker_per_shift['Total Carts Counted Per Hour'] = top_picker_per_shift['Carts Counted Per Hour'].apply(
            lambda x: f"{x:.2f}" if 0 < x < 1 else f"{int(round(x))}"
        )
    st.subheader("Top Picker Per Shift (Carts Counted Per Hour)")
    st.dataframe(
        top_picker_per_shift[['Date', 'Shift', 'Top Picker', 'Station Type', 'Total Carts Counted Per Hour']],
        use_container_width=True,
        hide_index=True
    )

    # --- Total Carts Picked Per Shift (per day) ---
    carts_per_shift = (
        filtered_data
        .groupby(['Date', 'Shift'], as_index=False)[['Drawers Counted', 'Drawer Avg']].sum(min_count=1)
        .assign(**{
            'AM': lambda df: np.where(df['Shift'] == "AM", np.round(df['Drawers Counted'] / df['Drawer Avg'], 2), 0),
            'PM': lambda df: np.where(df['Shift'] == "PM", np.round(df['Drawers Counted'] / df['Drawer Avg'], 2), 0),
            'Night': lambda df: np.where(df['Shift'] == "Night", np.round(df['Drawers Counted'] / df['Drawer Avg'], 2), 0),
        })
        .groupby('Date', as_index=False)[['AM', 'PM', 'Night']].sum()
    )
    for shift in ["AM", "PM", "Night"]:
        if shift not in carts_per_shift.columns:
            carts_per_shift[shift] = 0
    carts_per_shift = carts_per_shift[["Date", "AM", "PM", "Night"]]
    carts_per_shift['Date'] = pd.to_datetime(carts_per_shift['Date']).dt.strftime('%d-%m-%Y')
    st.subheader("Total Carts Picked Per Shift (per day)")
    st.dataframe(carts_per_shift, use_container_width=True, hide_index=True)

    # --- Carts Counted Per Hour by Station Type & Shift (excludes "Atlas Box & Bond Bags") ---
    filtered_station = filtered_data[
        (filtered_data['Station Type'].notna()) &
        (filtered_data['Station Type'].str.strip().str.lower() != "atlas box & bond bags".lower()) &
        (filtered_data['Station Type'].str.strip().str.lower() != "nan")
    ]
    breakdown = (
        filtered_station
        .groupby(['Station Type', 'Shift'], as_index=False)
        .agg({'Drawers Counted': 'sum', 'Drawer Avg': 'sum'})
        .assign(Carts_Per_Hour=lambda x: np.where(
            x['Drawer Avg'] > 0, np.round(x['Drawers Counted'] / x['Drawer Avg'], 2), 0
        ))
        .query("Carts_Per_Hour > 0")
        .sort_values(['Station Type', 'Shift'])
        .reset_index(drop=True)
    )
    st.subheader("Carts Counted Per Hour by Station Type & Shift")
    st.dataframe(breakdown[['Station Type', 'Shift', 'Carts_Per_Hour']], use_container_width=True, hide_index=True)
