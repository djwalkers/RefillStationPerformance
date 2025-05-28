import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests

st.set_page_config(page_title="Refill Station Performance Report", layout="wide")

# -- THEME --
st.markdown("""
    <style>
    .stApp { background-color: #DA362C !important; }
    .stTabs [data-baseweb="tab"] {
        background-color: #DA362C !important;
        color: white !important;
        font-weight: bold;
        border-radius: 10px 10px 0 0 !important;
        border: none !important;
    }
    .stTabs [data-baseweb="tab-list"] { background: #DA362C !important; }
    .stSelectbox label, .stSlider label { color: #fff !important; font-weight:bold; }
    .stDataFrame th, .stDataFrame td { color: #23272F !important; }
    .st-bx, .stButton>button { background-color: #DA362C !important; color: #fff !important; }
    </style>
""", unsafe_allow_html=True)

MAIN_COLOR = "#DA362C"
WHITE = "#FFFFFF"

# --- LOGO & TITLE ---
row_logo, row_title = st.columns([1,5])
with row_logo:
    st.image("The Roc.png", use_column_width=True)
with row_title:
    st.title("Refill Station Performance Report")

# --- LOAD DATA FROM GITHUB ---
@st.cache_data(show_spinner=True)
def load_data():
    github_repo = "https://raw.githubusercontent.com/djwalkers/RefillStationPerformance/main/"
    api_url = "https://api.github.com/repos/djwalkers/RefillStationPerformance/contents/Data"
    file_list = requests.get(api_url).json()
    csv_files = [f["download_url"] for f in file_list if f["name"].endswith(".csv")]
    dfs = []
    for url in csv_files:
        df = pd.read_csv(url)
        dfs.append(df)
    if not dfs:
        st.error("No data files found in GitHub repo.")
        return pd.DataFrame()
    data = pd.concat(dfs, ignore_index=True)

    # Date Dimension Table
    date_dim_url = github_repo + "Files/Date%20Dimension%20Table.xlsx"
    date_dim = pd.read_excel(date_dim_url, sheet_name="Sheet1")
    # Station Table
    station_url = github_repo + "Files/Station%20Standard.xlsx"
    station = pd.read_excel(station_url, sheet_name="Station")

    # --- Power Query Style Cleanup ---
    data.rename(columns={
        "textBox9": "Station Id",
        "textBox10": "User",
        "textBox13": "Drawers Counted",
        "textBox17": "Damaged Drawers Processed",
        "textBox19": "Damaged Products Processed",
        "textBox21": "Rogues Processed",
    }, inplace=True)

    # Drop extra columns
    keep_cols = [
        "Source.Name", "Station Id", "User", "Drawers Counted",
        "Damaged Drawers Processed", "Damaged Products Processed", "Rogues Processed"
    ]
    for col in data.columns:
        if col not in keep_cols:
            data = data.drop(columns=[col], errors="ignore")

    # Extract Date and Time from Source.Name
    data["Date"] = pd.to_datetime(data["Source.Name"].str[:10], format="%d-%m-%Y", errors='coerce')
    data["Time"] = data["Source.Name"].str[11:16].str.replace("-", ":")
    data["Users"] = data["User"].astype(str).str.split(" (").str[0].str.replace("*", "", regex=False).str.strip()

    # Remove staff/test accounts
    excluded_users = [
        "AARON DAVIES", "ADAM DAVENHILL", "ANDY WALKER", "ANNA DZIEDZIC-WIDZICKA",
        "BEN NORBURY", "CHARLOTTE BIBBY", "DANIEL ROGERSON", "DOMINIC PASKIN",
        "GEORGE SMITH", "JEFFERY FLETCHER", "MARCIN SZABLINSKI", "MARCOS CHAINIUK",
        "MARK BANHAM", "MAUREEN OUGHTON", "MAX CHAPPEL", "MICHAEL RUSHTON",
        "MICHAL ROWDO", "PIETER DAVIDS", "ROGER COULSON", "ROXANNE HAYNES",
        "SAM BENNETT", "STUART FOWLES", "TAMMY HITCHMOUGH", "TAYLOR MADDOCK",
        "TAYLOR MADDOX", "VASELA VELKOVA"
    ]
    data = data[~data["Users"].isin(excluded_users)]

    # Merge with dimension tables
    data = pd.merge(data, date_dim, on="Date", how="left")
    data = pd.merge(data, station, left_on="Station Id", right_on="Station", how="left")

    # Compute Carts Counted Per Hour
    data["Drawer Avg"] = pd.to_numeric(data["Drawer Avg"], errors="coerce").fillna(1)
    data["Drawers Counted"] = pd.to_numeric(data["Drawers Counted"], errors="coerce").fillna(0)
    data["Carts Counted Per Hour"] = (data["Drawers Counted"] / data["Drawer Avg"]).replace([np.inf, -np.inf], 0).round(2)

    # Remove "Atlas Box & Bond Bags" and NaN station types
    data = data[~(data["Station Type"].str.upper() == "ATLAS BOX & BOND BAGS")]
    data = data[~data["Station Type"].isna()]

    # Drop NaNs in main columns
    data = data.dropna(subset=["Users", "Station Id", "Date"])

    # Ensure columns present for plotting
    needed = [
        "Date", "Users", "Drawers Counted", "Drawer Avg",
        "Damaged Drawers Processed", "Damaged Products Processed", "Rogues Processed",
        "Station Type", "Carts Counted Per Hour", "Time"
    ]
    for col in needed:
        if col not in data.columns:
            data[col] = 0

    # Format dates
    data["Date"] = data["Date"].dt.strftime("%d-%m-%Y")
    return data

data = load_data()

# -- SEARCH + LIMIT CHART HELPER --
def horizontal_bar_chart(df, category_col, value_col, chart_title, color=MAIN_COLOR):
    # Search + limit widgets
    cols = st.columns([2,1])
    search = cols[0].text_input(f"Search {category_col}...", key=chart_title+"search")
    max_n = min(50, len(df))
    n_items = cols[1].slider(f"How many to show?", min_value=5, max_value=max_n, value=min(20,max_n), step=1, key=chart_title+"n")
    if search.strip():
        mask = df[category_col].astype(str).str.contains(search.strip(), case=False, na=False)
        filtered = df[mask]
    else:
        filtered = df
    agg = (
        filtered[[category_col, value_col]]
        .groupby(category_col, as_index=False)
        .sum()
        .query(f"`{value_col}` > 0")
        .sort_values(value_col, ascending=False)
        .head(n_items)
        .sort_values(value_col, ascending=True)
    )
    if agg.empty:
        st.info("No data to display.")
        return
    fig, ax = plt.subplots(figsize=(12, 0.45 * len(agg) + 2))
    ax.barh(agg[category_col], agg[value_col], color=color, edgecolor=WHITE)
    for i, v in enumerate(agg[value_col]):
        val_str = f"{v:.2f}" if 0 < v < 1 else f"{int(round(v))}"
        ax.text(v + 0.05, i, val_str, color=WHITE, va='center', fontsize=11, fontweight='bold')
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_yticklabels(agg[category_col], fontsize=10)
    ax.set_facecolor(WHITE)
    ax.tick_params(axis='x', colors=MAIN_COLOR)
    ax.tick_params(axis='y', colors=MAIN_COLOR)
    plt.tight_layout()
    st.markdown(f"#### {chart_title}")
    st.pyplot(fig)

# --- TABS ---
tabs = st.tabs([
    "Hourly Dashboard",
    "Weekly Dashboard",
    "Monthly Dashboard",
    "High Performers"
])

# ---- HOURLY DASHBOARD ----
with tabs[0]:
    st.markdown("## Hourly Dashboard")
    col1, col2, col3, col4 = st.columns(4)
    month_filter = col1.selectbox("Month", options=["All"] + sorted(data["Date"].str[3:10].unique().tolist()))
    date_filter = col2.selectbox("Date", options=["All"] + sorted(data["Date"].unique().tolist()))
    time_filter = col3.selectbox("Time", options=["All"] + sorted(data["Time"].astype(str).unique().tolist()))
    station_type_filter = col4.selectbox("Station Type", options=["All"] + sorted(data["Station Type"].astype(str).unique().tolist()))
    filtered = data.copy()
    if month_filter != "All":
        filtered = filtered[filtered["Date"].str[3:10] == month_filter]
    if date_filter != "All":
        filtered = filtered[filtered["Date"] == date_filter]
    if time_filter != "All":
        filtered = filtered[filtered["Time"].astype(str) == time_filter]
    if station_type_filter != "All":
        filtered = filtered[filtered["Station Type"].astype(str) == station_type_filter]
    horizontal_bar_chart(filtered, "Users", "Carts Counted Per Hour", "Carts Counted Per Hour")
    colA, colB, colC = st.columns(3)
    with colA:
        horizontal_bar_chart(filtered, "Users", "Rogues Processed", "Rogues Processed")
    with colB:
        horizontal_bar_chart(filtered, "Users", "Damaged Drawers Processed", "Damaged Drawers Processed")
    with colC:
        horizontal_bar_chart(filtered, "Users", "Damaged Products Processed", "Damaged Products Processed")

# ---- WEEKLY DASHBOARD ----
with tabs[1]:
    st.markdown("## Weekly Dashboard")
    station_type_filter = st.selectbox("Station Type", options=["All"] + sorted(data["Station Type"].astype(str).unique().tolist()))
    filtered = data.copy()
    if station_type_filter != "All":
        filtered = filtered[filtered["Station Type"].astype(str) == station_type_filter]
    horizontal_bar_chart(filtered, "Users", "Carts Counted Per Hour", "Carts Counted Per Week")
    colA, colB, colC = st.columns(3)
    with colA:
        horizontal_bar_chart(filtered, "Users", "Rogues Processed", "Rogues Processed")
    with colB:
        horizontal_bar_chart(filtered, "Users", "Damaged Drawers Processed", "Damaged Drawers Processed")
    with colC:
        horizontal_bar_chart(filtered, "Users", "Damaged Products Processed", "Damaged Products Processed")

# ---- MONTHLY DASHBOARD ----
with tabs[2]:
    st.markdown("## Monthly Dashboard")
    month_filter = st.selectbox("Month", options=["All"] + sorted(data["Date"].str[3:10].unique().tolist()))
    station_type_filter = st.selectbox("Station Type", options=["All"] + sorted(data["Station Type"].astype(str).unique().tolist()))
    filtered = data.copy()
    if month_filter != "All":
        filtered = filtered[filtered["Date"].str[3:10] == month_filter]
    if station_type_filter != "All":
        filtered = filtered[filtered["Station Type"].astype(str) == station_type_filter]
    horizontal_bar_chart(filtered, "Users", "Carts Counted Per Hour", "Carts Counted Per Month")
    colA, colB, colC = st.columns(3)
    with colA:
        horizontal_bar_chart(filtered, "Users", "Rogues Processed", "Rogues Processed")
    with colB:
        horizontal_bar_chart(filtered, "Users", "Damaged Drawers Processed", "Damaged Drawers Processed")
    with colC:
        horizontal_bar_chart(filtered, "Users", "Damaged Products Processed", "Damaged Products Processed")

# ---- HIGH PERFORMERS ----
with tabs[3]:
    st.markdown("## High Performers")
    df = data.copy()
    # Top picker per day (sum of all picks in calendar day)
    top_picker_per_day = (
        df.groupby(["Date", "Users", "Station Type"], as_index=False)["Carts Counted Per Hour"].sum()
        .sort_values(["Date", "Carts Counted Per Hour"], ascending=[True, False])
        .groupby("Date").head(1)
    )
    st.markdown("### Top Picker Per Day (Carts Counted Per Hour)")
    st.dataframe(top_picker_per_day[["Date", "Users", "Station Type", "Carts Counted Per Hour"]], use_container_width=True, hide_index=True)

    # Top picker per shift (AM, PM, Night)
    def shift_label(time_str):
        try:
            h, m = map(int, str(time_str).split(":"))
            mins = h*60 + m
            if 6*60 <= mins <= 14*60: return "AM"
            elif 14*60+1 <= mins <= 22*60: return "PM"
            else: return "Night"
        except:
            return "Unknown"
    df["Shift"] = df["Time"].apply(shift_label)
    shift_order = ["AM", "PM", "Night"]
    top_picker_per_shift = (
        df.groupby(["Date", "Shift", "Users", "Station Type"], as_index=False)["Carts Counted Per Hour"].sum()
        .sort_values(["Date", "Shift", "Carts Counted Per Hour"], ascending=[True, True, False])
        .groupby(["Date", "Shift"]).head(1)
    )
    st.markdown("### Top Picker Per Shift (Carts Counted Per Hour)")
    st.dataframe(top_picker_per_shift[["Date", "Shift", "Users", "Station Type", "Carts Counted Per Hour"]]
        .sort_values(["Date", "Shift"], key=lambda x: x.map({k:i for i,k in enumerate(shift_order)} if x.name=="Shift" else None))
        , use_container_width=True, hide_index=True)
