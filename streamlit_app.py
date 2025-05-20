import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Refill Station Performance Dashboard", layout="wide")
st.title("Refill Station Performance Dashboard")

# === Load the Excel file ===
excel_path = "/mnt/data/Refill Station Performance Report.xlsx"

# Read the Carts Per Hour data
df = pd.read_excel(excel_path, sheet_name="Carts Per Hour")

# Clean up column names for easier use
df.columns = [str(col).strip() for col in df.columns]

# Try to guess typical names for filters, adjust if needed
month_col = next((col for col in df.columns if "month" in col.lower()), None)
date_col = next((col for col in df.columns if "date" in col.lower()), None)
time_col = next((col for col in df.columns if "time" in col.lower()), None)
station_col = next((col for col in df.columns if "station" in col.lower()), None)
user_col = next((col for col in df.columns if "row label" in col.lower() or "user" in col.lower()), None)
value_col = next((col for col in df.columns if "carts counted per hour" in col.lower()), None)

# Add new tabs (update your other tabs as needed)
tab1, tab2, tab3, tab4 = st.tabs([
    "Full Data",
    "Weekly User Summary",
    "Total Carts Hourly",
    "Hourly Dashboard"
])

with tab4:
    st.header("Hourly Dashboard")

    # === Filters ===
    filter_cols = []

    # Month filter
    if month_col:
        months = df[month_col].dropna().unique()
        month_sel = st.selectbox("Select Month:", ["All"] + sorted(months.tolist()))
        if month_sel != "All":
            df = df[df[month_col] == month_sel]
            filter_cols.append(f"{month_col}={month_sel}")

    # Date filter
    if date_col:
        dates = df[date_col].dropna().unique()
        # Try to convert to date type for better sorting
        try:
            dates = pd.to_datetime(dates)
            dates = sorted(dates)
        except Exception:
            dates = sorted(dates.tolist())
        date_sel = st.selectbox("Select Date:", ["All"] + [str(d) for d in dates])
        if date_sel != "All":
            df = df[df[date_col].astype(str) == date_sel]
            filter_cols.append(f"{date_col}={date_sel}")

    # Time filter
    if time_col:
        times = df[time_col].dropna().unique()
        times = sorted(times.tolist())
        time_sel = st.selectbox("Select Time:", ["All"] + [str(t) for t in times])
        if time_sel != "All":
            df = df[df[time_col].astype(str) == time_sel]
            filter_cols.append(f"{time_col}={time_sel}")

    # Station Type filter
    if station_col:
        stations = df[station_col].dropna().unique()
        stations = sorted(stations.tolist())
        station_sel = st.selectbox("Select Station Type:", ["All"] + [str(s) for s in stations])
        if station_sel != "All":
            df = df[df[station_col].astype(str) == station_sel]
            filter_cols.append(f"{station_col}={station_sel}")

    st.write("**Filters applied:**", ", ".join(filter_cols) if filter_cols else "None")

    # === Bar Chart ===
    # Sort for plotting
    if user_col and value_col and not df.empty:
        df_sorted = df.sort_values(value_col, ascending=True)

        fig, ax = plt.subplots(figsize=(8, max(5, len(df_sorted) * 0.35)))
        ax.barh(df_sorted[user_col], df_sorted[value_col])
        ax.set_xlabel("Carts Counted Per Hour")
        ax.set_ylabel("User")
        ax.set_title("Carts Counted Per Hour by User")
        st.pyplot(fig)

        # Show the filtered data table
        st.dataframe(df_sorted[[user_col, value_col] + (
            [month_col] if month_col else []
        ) + (
            [date_col] if date_col else []
        ) + (
            [time_col] if time_col else []
        ) + (
            [station_col] if station_col else []
        )], use_container_width=True)
    else:
        st.warning("No data available with selected filters or column names not found.")

    # Download filtered data
    st.download_button(
        "Download Filtered Table as CSV",
        df.to_csv(index=False),
        "hourly_dashboard_filtered.csv"
    )
