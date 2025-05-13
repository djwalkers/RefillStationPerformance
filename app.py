import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Streamlit app title
st.title('Refill Station Performance Dashboard')

# File uploaders for the first dataset (CSV)
uploaded_file_1 = st.file_uploader("Upload the first CSV file (Performance data)", type=["csv"])

# Correct GitHub URL for the Excel file
excel_file_url = "https://raw.githubusercontent.com/djwalkers/RefillStationPerformance/main/Station%20Standard.xlsx"

if uploaded_file_1 is not None:
    # Load the first dataset (performance data)
    data = pd.read_csv(uploaded_file_1)

    # Load the second dataset (Excel file from GitHub)
    mapping_data = pd.read_excel(excel_file_url, usecols=[0, 1])  # Read columns A (Resource Identity) and B (Resource Name)
    mapping_data.columns = ['Resource Identity', 'Resource Name']  # Rename columns for clarity

    # Rename columns in the first dataset (performance data)
    data.columns = [
        'Resource Identity', 'Username', 'User Active', 'Drawers Counted', 
        'Drawers Counted Per Hour', 'Drawers Refilled', 'Drawers Refilled Per Hour', 
        'Damaged Drawers Processed', 'Damaged Products Processed', 'Rogues Processed', 
        'Total Time Station Active', 'Total Drawers Counted', 'Total Drawers Counted Per Hour', 
        'Total Drawers Refilled', 'Total Drawers Refilled Per Hour', 'Total Damaged Drawers Processed', 
        'Total Damaged Products Processed', 'Total Rogues Processed'
    ]

    # Merge the two datasets on "Resource Identity"
    merged_data = pd.merge(data, mapping_data, on="Resource Identity", how="left")

    # Interactive filters
    st.sidebar.header("Filters")
    
    # Username filter
    username_filter = st.sidebar.multiselect(
        "Select Username(s)", merged_data['Username'].unique(), default=merged_data['Username'].unique())
    
    # Resource Name filter (based on Column B from Excel)
    resource_name_filter = st.sidebar.multiselect(
        "Select Resource Name(s)", merged_data['Resource Name'].unique(), default=merged_data['Resource Name'].unique())

    # Drawers Counted filter (range slider)
    drawers_min = merged_data['Drawers Counted'].min()
    drawers_max = merged_data['Drawers Counted'].max()
    drawers_filter = st.sidebar.slider(
        "Select Drawers Counted Range", min_value=drawers_min, max_value=drawers_max, 
        value=(drawers_min, drawers_max))

    # Apply the filters
    filtered_data = merged_data[
        (merged_data['Username'].isin(username_filter)) & 
        (merged_data['Resource Name'].isin(resource_name_filter)) &
        (merged_data['Drawers Counted'] >= drawers_filter[0]) & 
        (merged_data['Drawers Counted'] <= drawers_filter[1])
    ]

    # Exclude rows where any of the key metrics have 0 values
    filtered_data = filtered_data[
        (filtered_data['Drawers Counted'] > 0) & 
        (filtered_data['Rogues Processed'] > 0) & 
        (filtered_data['Damaged Drawers Processed'] > 0) & 
        (filtered_data['Damaged Products Processed'] > 0)
    ]

    # Plotting the Total Drawers Counted based on "Resource Name"
    total_drawers_by_resource_name = filtered_data.groupby('Resource Name')['Drawers Counted'].sum().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    total_drawers_by_resource_name.plot(kind='bar', ax=ax, color='purple')
    ax.set_title('Total Drawers Counted by Resource Name')
    ax.set_xlabel('Resource Name')
    ax.set_ylabel('Total Drawers Counted')
    st.pyplot(fig)

    # Top 10 and Bottom 10 Drawers Counted by Username (excluding 0s)
    top_10_by_username = filtered_data.groupby('Username')['Drawers Counted'].sum().sort_values(ascending=False).head(10)
    bottom_10_by_username = filtered_data.groupby('Username')['Drawers Counted'].sum().sort_values().head(10)

    # Plotting the Top 10 and Bottom 10 charts based on "Drawers Counted"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot Top 10 "Drawers Counted" by Username
    top_10_by_username.plot(kind='bar', ax=ax1, color='green')
    ax1.set_title('Top 10 Drawers Counted by Username')
    ax1.set_xlabel('Username')
    ax1.set_ylabel('Drawers Counted')

    # Plot Bottom 10 "Drawers Counted" by Username excluding 0 values
    bottom_10_by_username.plot(kind='bar', ax=ax2, color='red')
    ax2.set_title('Bottom 10 Drawers Counted by Username (Excluding 0)')
    ax2.set_xlabel('Username')
    ax2.set_ylabel('Drawers Counted')

    # Display the plots
    plt.tight_layout()
    st.pyplot(fig)

    # Other charts (all metrics excluding 0s for additional metrics)
    metrics = ['Rogues Processed', 'Damaged Drawers Processed', 'Damaged Products Processed']
    
    for metric in metrics:
        # Grouping the data by Username and summing the metric values (no exclusion for 0s)
        metric_by_username = filtered_data.groupby('Username')[metric].sum().sort_values(ascending=False)
        
        # Plotting the data for each metric
        fig, ax = plt.subplots(figsize=(10, 6))
        metric_by_username.plot(kind='bar', ax=ax, color='skyblue')
        ax.set_title(f'{metric} by Username')
        ax.set_xlabel('Username')
        ax.set_ylabel(metric)
        st.pyplot(fig)

else:
    st.write("Please upload the first CSV file to get started.")
