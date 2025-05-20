import streamlit as st
import pandas as pd
import requests
from urllib.parse import quote

st.set_page_config(page_title="Refill Station Performance Dashboard", layout="wide")

st.title("Refill Station Performance Dashboard")

# === 1. Define GitHub locations ===
GITHUB_USER = "djwalkers"
REPO_NAME = "RefillStationPerformance"
DATA_FOLDER = "Data"
FILES_FOLDER = "Files"

# === 2. Helper: Load all data files from GitHub ===
@st.cache_data(show_spinner="Fetching raw data from GitHub...")
def load_raw_data():
    api_url = f"https://api.github.com/repos/{GITHUB_USER}/{REPO_NAME}/contents/{DATA_FOLDER}"
    files_api = requests.get(api_url).json()
    data_files = [f['name'] for f in files_api if f['name'].endswith('.csv') or f['name'].endswith('.xlsx')]
    dfs = []
    for fname in data_files:
        file_url = f"https://raw.githubusercontent.com/{GITHUB_USER}/{REPO_NAME}/main/{DATA_FOLDER}/{quote(fname)}"
        try:
            if fname.endswith('.csv'):
                df = pd.read_csv(file_url)
            elif fname.endswith('.xlsx'):
                df = pd.read_excel(file_url)
            else:
                continue
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
    date_dim_url = f"https://raw.githubusercontent.com/{GITHUB_USER}/{REPO_NAME}/main/{FILES_FOLDER}/Date%20Dimension%20Table.xlsx"
    station_url = f"https://raw.githubusercontent.com/{GITHUB_USER}/{REPO_NAME}/main/{FILES_FOLDER}/Station%20Standard.xlsx"

    # DEBUG: Show all available sheet names in each file
    date_excel = pd.ExcelFile(date_dim_url)
    station_excel = pd.ExcelFile(station_url)
    st.write("Date Dimension Table.xlsx sheets:", date_excel.sheet_names)
    st.write("Station Standard.xlsx sheets:", station_excel.sheet_names)

    # Try to load a default sheet (likely to fail, but prints available sheets)
    # Replace 'Sheet1' and 'Station' after you see what is printed above!
    try:
        date_dim = pd.read_excel(date_dim_url, sheet_name="Sheet1")
    except Exception as e:
        st.error(f"Could not load 'Sheet1' from Date Dimension Table.xlsx: {e}")
        date_dim = None
    try:
        station = pd.read_excel(station_url, sheet_name="Station")
    except Exception as e:
        st.error(f"Could not load 'Station' from Station Standard.xlsx: {e}")
        station = None
    return date_dim, station

raw_data = load_raw_data()
date_dim, station = load_reference_tables()

if raw_data.empty:
    st.error("No data files found in the Data folder. Please check your GitHub repository.")
    st.stop()

if (date_dim is None) or (station is None):
    st.error("Please update the sheet names in the code as printed above and rerun the app.")
    st.stop()

# === 3. Data transformation steps (Power Query logic) ===

data = raw_data.copy()

# Rename columns
data = data.rename(columns={
    'textBox9': 'Station Id',
    'textBox10': 'USer',
    'textBox13': 'Drawers Counted',
    'textBox17': 'Damaged Drawers Processed',
    'textBox19': 'Damaged Products Processed',
    'textBox21': 'Rogues Processed',
})

# Remove unnecessary columns
drop_cols = [
    "textBox14", "textBox15", "textBox16", "textBox34", "textBox31", "textBox23",
    "textBox24", "textBox25", "textBox26", "textBox28", "textBox29", "textBox30"
]
data = data.drop(columns=[c for c in drop_cols if c in data.columns], errors='ignore')

# Rename 'USer' to 'User'
if 'USer' in data.columns:
    data = data.rename(columns={'USer': 'User'})

# Extract Date and Time from Source.Name
data['Date'] = data['Source.Name'].str[:10]
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
data['Time'] = data['Source.Name'].str[10:15].str.replace('-', ':')
try:
    data['Time'] = pd.to_datetime(data['Time'], format='%H:%M', errors='coerce').dt.time
except Exception:
    pass

# Clean Users column
data['Users'] = data['User'].astype(str).str.split(' (').str[0].str.replace('*', '', regex=False).str.strip()
data = data.drop(columns=['User'])

# Merge with DateDimensionTable on 'Date'
if ('Date' in date_dim.columns) and all(x in date_dim.columns for x in ['Year', 'Month', 'Week']):
    data = data.merge(date_dim[['Date', 'Year', 'Month', 'Week']], on='Date', how='left')
else:
    st.warning("Could not find 'Date', 'Year', 'Month', 'Week' columns in the Date Dimension Table. Check your reference file.")

# Merge with Station on 'Station Id'
if 'Station' in station.columns:
    data = data.merge(station.rename(columns={'Station': 'Station Id'}), on='Station Id', how='left')
elif 'Station Id' in station.columns:
    data = data.merge(station, on='Station Id', how='left')
else:
    st.warning("Could not find 'Station' or 'Station Id' column in Station Standard. Check your reference file.")

# Rename Station columns
data = data.rename(columns={
    'Type': 'Station Type',
    'Drawer Avg': 'Drawer Avg',
    'KPI': 'Station KPI'
})

# Carts Counted Per Hour
if 'Drawers Counted' in data.columns and 'Drawer Avg' in data.columns:
    data['Carts Counted Per Hour'] = (data['Drawers Counted'] / data['Drawer Avg']).round(2)
    data['Drawers Counted'] = data['Drawers Counted'].fillna(0)
    data['Login Count'] = data['Drawers Counted'].apply(lambda x: 1 if x > 0 else 0)
else:
    data['Carts Counted Per Hour'] = 0
    data['Login Count'] = 0

# Exclude certain users
exclude_users = [
    "AARON DAVIES", "ADAM DAVENHILL", "ANDY WALKER", "ANNA DZIEDZIC-WIDZICKA", "BEN NORBURY",
    "CHARLOTTE BIBBY", "DANIEL ROGERSON", "DOMINIC PASKIN", "GEORGE SMITH", "JEFFERY FLETCHER",
    "MARCIN SZABLINSKI", "MARCOS CHAINIUK", "MARK BANHAM", "MAUREEN OUGHTON", "MAX CHAPPEL",
    "MICHAEL RUSHTON", "MICHAL ROWDO", "PIETER DAVIDS", "ROGER COULSON", "ROXANNE HAYNES",
    "SAM BENNETT", "STUART FOWLES", "TAMMY HITCHMOUGH", "TAYLOR MADDOCK", "TAYLOR MADDOX", "VASELA VELKOVA"
]
if 'Users' in data.columns:
    data = data[~data['Users'].isin(exclude_users)].reset_index(drop=True)

data['Index'] = data.index

main_cols = [
    'Index', 'Year', 'Date', 'Month', 'Week', 'Time', 'Station Id', 'Users',
    'Drawers Counted', 'Damaged Drawers Processed', 'Damaged Products Processed',
    'Rogues Processed', 'Station Type', 'Drawer Avg', 'Station KPI',
    'Carts Counted Per Hour', 'Login Count'
]
main_cols = [col for col in main_cols if col in data.columns]
data = data[main_cols]

# === 4. Build outputs for your dashboard ===

# Output 1: Weekly summary by user
if 'Users' in data.columns and 'Week' in data.columns and 'Drawers Counted' in data.columns:
    weekly = (
        data.groupby(['Users', 'Week'], as_index=False)['Drawers Counted']
        .sum()
        .rename(columns={'Drawers Counted': 'Drawers Weekly'})
    )
    weekly['Index'] = weekly.index
    weekly = weekly[['Index', 'Users', 'Week', 'Drawers Weekly']]
else:
    weekly = pd.DataFrame()

# Output 2: Total carts hourly summary
if all(x in data.columns for x in ['Station Type', 'Time', 'Station Id', 'Carts Counted Per Hour', 'Date']):
    hourly = (
        data.groupby(['Station Type', 'Time', 'Station Id'], as_index=False)['Carts Counted Per Hour']
        .sum()
        .rename(columns={'Carts Counted Per Hour': 'Total Carts Counted'})
    )
    hourly['Total Carts Counted'] = hourly['Total Carts Counted'].fillna(0)
    recent_dates = (
        data.groupby(['Station Type', 'Time', 'Station Id'])['Date']
        .max()
        .reset_index()
    )
    hourly = hourly.merge(recent_dates, on=['Station Type', 'Time', 'Station Id'], how='left')
    hourly = hourly[['Date', 'Station Type', 'Time', 'Station Id', 'Total Carts Counted']]
else:
    hourly = pd.DataFrame()

# === 5. Streamlit Tabs ===
tab1, tab2, tab3 = st.tabs(["Full Data", "Weekly User Summary", "Total Carts Hourly"])

with tab1:
    st.subheader("Full Cleaned Data")
    st.dataframe(data, use_container_width=True)

with tab2:
    st.subheader("Drawers Counted Per User, Per Week")
    st.dataframe(weekly, use_container_width=True)

with tab3:
    st.subheader("Total Carts Counted Per Hour")
    st.dataframe(hourly, use_container_width=True)

# === Optional: Download buttons ===
with st.expander("Download outputs as CSV"):
    st.download_button("Download Full Data CSV", data.to_csv(index=False), "full_data.csv")
    st.download_button("Download Weekly User Summary CSV", weekly.to_csv(index=False), "weekly_user_summary.csv")
    st.download_button("Download Total Carts Hourly CSV", hourly.to_csv(index=False), "total_carts_hourly.csv")
