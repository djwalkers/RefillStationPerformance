import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Streamlit app title
st.title('Refill Station Performance Dashboard')

# File uploader outside any conditional block
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

# Only proceed if the file is uploaded
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

    # Time filter (Total Time Station Active)
    time_filter = st.sidebar.slider(
        "Select Time Range (in hrs)", 
        min_value=0, 
        max_value=24, 
        value=(0, 24))

    # Apply the filters
    filtered_data = data[
        (data['Username'].isin(username_filter)) & 
        (data['Drawers Counted'] >= drawers_filter[0]) & 
        (data['Drawers Counted'] <= drawers_filter[1]) & 
        (data['Total Time Station Active'].apply(lambda x: float(x.split()[0])) >= time_filter[0]) &
        (data['Total Time Station Active'].apply(lambda x: float(x.split()[0])) <= time_filter[1])
    ]

    # Exclude rows where any of the metrics have 0 values
    filtered_data = filtered_data[
        (filtered_data['Drawers Counted'] > 0) & 
        (filtered_data['Rogues Processed'] > 0) & 
        (filtered_data['Damaged Drawers Processed'] > 0) & 
        (filtered_data['Damaged Products Processed'] > 0)
    ]

    # Display the filtered data
    st.subheader("Filtered Data")
    st.write(filtered_data)

    # Additional analysis - Summary statistics
    st.subheader("Summary Statistics")
    st.write(filtered_data.describe())

    # Generating the requested charts for metrics by Username
    metrics = ['Drawers Counted', 'Rogues Processed', 'Damaged Drawers Processed', 'Damaged Products Processed']
    
    for metric in metrics:
        # Grouping the data by Username and summing the metric values
        metric_by_username = filtered_data.groupby('Username')[metric].sum().sort_values(ascending=False)
        
        # Plotting the data
        fig, ax = plt.subplots(figsize=(10, 6))
        metric_by_username.plot(kind='bar', ax=ax, color='skyblue')
        ax.set_title(f'{metric} by Username')
        ax.set_xlabel('Username')
        ax.set_ylabel(metric)
        st.pyplot(fig)

else:
    st.write("Please upload a CSV file to get started.")
