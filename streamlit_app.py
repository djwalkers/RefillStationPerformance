import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Page config & theme ---
st.set_page_config(
    page_title="Refill Station Performance Report",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for full color branding ---
st.markdown(
    """
    <style>
    .stApp { background-color: #DA362C !important; }
    .st-bf, .st-emotion-cache-1d391kg { background-color: #23272F !important; }
    .css-1q8dd3e, .st-cg, .st-emotion-cache-10trblm, .st-emotion-cache-1v0mbdj, .st-emotion-cache-16txtl3 {
        color: #FFFFFF !important;
    }
    .stSelectbox label, .stSlider label { color: #DA362C !important; font-weight:bold; }
    .stDataFrame th, .stDataFrame td { color: #23272F !important; }
    .st-bx, .stButton>button { background-color: #DA362C !important; color: #FFFFFF !important; }
    </style>
    """,
    unsafe_allow_html=True
)

# --- CONSTANTS ---
MAIN_COLOR = "#DA362C"
WHITE = "#FFFFFF"

# --- Sidebar logo ---
st.sidebar.image("The Roc.png", use_column_width=True)
st.sidebar.header("Upload New Data File")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type="csv")
st.sidebar.markdown("---")
st.sidebar.button("Refill Station Performance Report")

# --- Dummy data loading for example (Replace with your own) ---
@st.cache_data
def load_data():
    # Example dummy data, replace with your actual loading code!
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
    # Add derived/calculated columns here
    data["Carts Counted Per Hour"] = data["Drawers Counted"] / data["Drawer Avg"]
    return data

if uploaded_file:
    data = pd.read_csv(uploaded_file)
else:
    data = load_data()

# --- Helper: Filter data by search ---
def filter_search(df, col, search):
    if search.strip():
        mask = df[col].str.contains(search.strip(), case=False, na=False)
        return df[mask]
    return df

# --- Helper: Charting with limit and search ---
def horizontal_bar_chart(df, category_col, value_col, chart_title, color=MAIN_COLOR):
    # Search + limit widgets
    cols = st.columns([2,1])
    search = cols[0].text_input(f"Search {category_col}...", key=chart_title+"search")
    max_n = min(50, len(df))
    n_items = cols[1].slider(f"How many to show?", min_value=5, max_value=max_n, value=min(20,max_n), step=1, key=chart_title+"n")
    filtered = filter_search(df, category_col, search)
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

# --- MAIN LAYOUT with TABS ---
tabs = st.tabs([
    "Hourly Dashboard",
    "Weekly Dashboard",
    "Monthly Dashboard",
    "High Performers"
])

# ---- HOURLY DASHBOARD ----
with tabs[0]:
    st.markdown("## Hourly Dashboard")
    # Filters (dummy examples)
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
    # Main bar charts with search and limit
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
    # You'd aggregate by week here for real data
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
    # Aggregate by month for real data
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
    # Example: Show top pickers per day
    df = data.copy()
    df["Carts Counted Per Hour"] = df["Drawers Counted"] / df["Drawer Avg"]
    top_picker_per_day = (
        df.groupby(["Date", "Users"], as_index=False)["Carts Counted Per Hour"].sum()
        .sort_values(["Date", "Carts Counted Per Hour"], ascending=[True, False])
        .groupby("Date").head(1)
    )
    st.markdown("### Top Picker Per Day (Carts Counted Per Hour)")
    st.dataframe(top_picker_per_day, use_container_width=True, hide_index=True)
    # ...add more summary tables or charts as needed...

