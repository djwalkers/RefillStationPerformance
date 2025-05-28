import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---- THEME ----
st.set_page_config(
    page_title="Refill Station Performance Report",
    page_icon=":bar_chart:",
    layout="wide"
)

# --- Custom CSS for DA362C color theme and NO sidebar ---
st.markdown("""
    <style>
    .stApp { background-color: #DA362C !important; }
    .st-emotion-cache-10trblm, .st-emotion-cache-1v0mbdj, .st-emotion-cache-16txtl3,
    .st-emotion-cache-1d391kg, .css-1q8dd3e, .st-cg, .st-emotion-cache-bm2z3a, .st-emotion-cache-zt5igj, .st-emotion-cache-1n76uvr, .st-emotion-cache-1avcm0n {
        color: #fff !important;
    }
    .st-emotion-cache-1jicfl2 {background-color: #DA362C !important;}
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

# ---- TOP: LOGO AND UPLOAD ----
cols = st.columns([1,5])
with cols[0]:
    st.image("The Roc.png", use_column_width=True)
with cols[1]:
    st.title("Refill Station Performance Report")
    uploaded_file = st.file_uploader("Upload New Data File (CSV)", type="csv", label_visibility="visible")

# ---- DATA LOAD ----
@st.cache_data
def load_data():
    # Simulated dummy data, replace with your real load/clean code!
    data = pd.DataFrame({
        "Date": pd.date_range("2025-05-01", periods=60, freq='D').repeat(8),
        "Users": np.tile(["USER"+str(i) for i in range(1,9)], 60),
        "Drawers Counted": np.random.randint(10, 100, 480),
        "Drawer Avg": np.random.randint(1, 10, 480),
        "Rogues Processed": np.random.randint(0, 10, 480),
        "Damaged Drawers Processed": np.random.randint(0, 10, 480),
        "Damaged Products Processed": np.random.randint(0, 10, 480),
        "Station Type": np.tile(["TypeA", "TypeB", "Atlas Box & Bond Bags", "TypeC"], 120),
        "Time": np.random.choice([f"{h:02d}:00" for h in range(6,23)], 480),
    })
    data["Carts Counted Per Hour"] = data["Drawers Counted"] / data["Drawer Avg"]
    return data

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    # Do your cleaning/calculated columns here, e.g.:
    data["Carts Counted Per Hour"] = data["Drawers Counted"] / data["Drawer Avg"]
else:
    data = load_data()

# --- Helper for chart search & limit ---
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
    month_filter = col1.selectbox("Month", options=["All"] + sorted(data["Date"].astype(str).str[:7].unique().tolist()))
    date_filter = col2.selectbox("Date", options=["All"] + sorted(data["Date"].astype(str).unique().tolist()))
    time_filter = col3.selectbox("Time", options=["All"] + sorted(data["Time"].astype(str).unique().tolist()))
    station_type_filter = col4.selectbox("Station Type", options=["All"] + sorted(data["Station Type"].astype(str).unique().tolist()))
    filtered = data.copy()
    if month_filter != "All":
        filtered = filtered[filtered["Date"].astype(str).str.startswith(month_filter)]
    if date_filter != "All":
        filtered = filtered[filtered["Date"].astype(str) == date_filter]
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
    week_filter = st.selectbox("Week", options=["All"] + ["2025-W19","2025-W20"])
    station_type_filter = st.selectbox("Station Type", options=["All"] + sorted(data["Station Type"].astype(str).unique().tolist()))
    filtered = data.copy()
    if station_type_filter != "All":
        filtered = filtered[filtered["Station Type"].astype(str) == station_type_filter]
    # Weekly groupby aggregation if needed
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
    month_filter = st.selectbox("Month", options=["All"] + sorted(data["Date"].astype(str).str[:7].unique().tolist()))
    station_type_filter = st.selectbox("Station Type", options=["All"] + sorted(data["Station Type"].astype(str).unique().tolist()))
    filtered = data.copy()
    if month_filter != "All":
        filtered = filtered[filtered["Date"].astype(str).str.startswith(month_filter)]
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
    df["Carts Counted Per Hour"] = df["Drawers Counted"] / df["Drawer Avg"]
    top_picker_per_day = (
        df.groupby(["Date", "Users"], as_index=False)["Carts Counted Per Hour"].sum()
        .sort_values(["Date", "Carts Counted Per Hour"], ascending=[True, False])
        .groupby("Date").head(1)
    )
    st.markdown("### Top Picker Per Day (Carts Counted Per Hour)")
    st.dataframe(top_picker_per_day, use_container_width=True, hide_index=True)

