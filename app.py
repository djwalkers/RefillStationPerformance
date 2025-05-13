import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Streamlit app title
st.title('Refill Station Performance Dashboard')

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded CSV file
    data = pd.read_csv(uploaded_file)

    # Rename columns to more meaningful names
    data.columns = [
        'Resource Identity', 'Username', 'User Active', 'Drawers Counted', 
        'Drawers Counted Per Hour', 'Drawers Refilled', 'Drawers Refilled Per Hour', 
        'Damaged Drawers Processed', 'Damaged Products Processed', 'Rogues Processed', 
        'Total Time Station Active', 'Total Drawers Counted', 'Total Drawers Counted Per Hour', 
        'Total Drawers Refilled', 'Total Drawers Refilled Per Hour', 'Total Damaged Drawers Processed', 
        'Total Damaged Products Processed', 'Total Rogues Processed'
    ]

    # Interactive filters
    st.sidebar.header("Filters")
    
    # Username filter
    username_filter = st.sidebar.multiselect(
        "Select Username(s)", data['Username'].unique(), default=data['Username'].unique())
    
    # Drawers Counted filter (range slider)
    drawers_min = data['Drawers Counted'].min()
    drawers_max = data['Drawers Counted'].max()
    drawers_filter = st.sidebar.slider(
        "Select Drawers Counted Range", min_value=drawers_min, max_value=drawers_max, 
        value=(drawers_min, drawers_max))

    # Apply the filters
    filtered_data = data[
        (data['Username'].isin(username_filter)) & 
        (data['Drawers Counted'] >= drawers_filter[0]) & 
        (data['Drawers Counted'] <= drawers_filter[1])
    ]

    # Exclude rows where "Drawers Counted" is 0 for Top/Bottom charts
    filtered_data_for_top_bottom = filtered_data[
        (filtered_data['Drawers Counted'] > 0)
    ]

    # Top 10 and Bottom 10 Drawers Counted by Username (excluding 0s)
    top_10_by_username = filtered_data_for_top_bottom.groupby('Username')['Drawers Counted'].sum().sort_values(ascending=False).head(10)
    bottom_10_by_username = filtered_data_for_top_bottom.groupby('Username')['Drawers Counted'].sum().sort_values().head(10)

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
    st.write("Please upload a CSV file to get started.")
