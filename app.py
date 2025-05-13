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

    # Total Time Station Active filter (you can use a range slider)
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

    # Display the filtered data
    st.subheader("Filtered Data")
    st.write(filtered_data)

    # Additional analysis - Summary statistics
    st.subheader("Summary Statistics")
    st.write(filtered_data.describe())

    # Generate the Top 10 and Bottom 10 charts
    top_10_by_username = filtered_data.groupby('Username').sum().sort_values(by='Drawers Counted', ascending=False).head(10)
    filtered_bottom_10 = filtered_data[filtered_data['Drawers Counted'] > 0].groupby('Username').sum().sort_values(by='Drawers Counted').head(10)

    # Plotting the Top 10 and Bottom 10 charts
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot Top 10 "Drawers Counted" by Username
    top_10_by_username.plot(kind='bar', y='Drawers Counted', ax=ax1, color='green')
    ax1.set_title('Top 10 Drawers Counted by Username')
    ax1.set_xlabel('Username')
    ax1.set_ylabel('Drawers Counted')

    # Plot Bottom 10 "Drawers Counted" by Username excluding 0 values
    filtered_bottom_10.plot(kind='bar', y='Drawers Counted', ax=ax2, color='red')
    ax2.set_title('Bottom 10 Drawers Counted by Username (Excluding 0)')
    ax2.set_xlabel('Username')
    ax2.set_ylabel('Drawers Counted')

    # Display the plots
    plt.tight_layout()
    st.pyplot(fig)

    # Additional chart: Drawers Counted vs. Time Station Active (scatter plot)
    st.subheader("Scatter Plot: Drawers Counted vs. Time Station Active")
    fig, ax = plt.subplots(figsize=(8, 6))
    filtered_data['Time (hrs)'] = filtered_data['Total Time Station Active'].apply(lambda x: float(x.split()[0]))
    ax.scatter(filtered_data['Time (hrs)'], filtered_data['Drawers Counted'], color='purple')
    ax.set_xlabel('Time Station Active (hrs)')
    ax.set_ylabel('Drawers Counted')
    ax.set_title('Scatter Plot: Drawers Counted vs. Time Station Active')
    st.pyplot(fig)
    
else:
    st.write("Please upload a CSV file to get started.")
