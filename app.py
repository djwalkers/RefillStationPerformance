
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Load data from CSV
file_path = '/mnt/data/RefillStationPerformanceReport.csv'  # Update with the correct path if needed
data = pd.read_csv(file_path)

# Rename columns to more meaningful names
data.columns = [
    'Station', 'Operator', 'TimeSpent', 'QtyRefilled', 'CostPerUnit', 
    'TotalCost', 'Efficiency', 'Downtime', 'Maintenance', 'BatchSize', 
    'Duration', 'NumRefills', 'AvgDuration', 'MaxDuration', 'OtherMetric1', 
    'OtherMetric2', 'OtherMetric3', 'OtherMetric4'
]

# Streamlit app title
st.title('Refill Station Performance Dashboard')

# Filter by Station
station = st.selectbox('Select a Station:', data['Station'].unique())

# Filter by Time Spent (hrs)
time_spent_min = float(data['TimeSpent'].min().split()[0])
time_spent_max = float(data['TimeSpent'].max().split()[0])

time_spent = st.slider('Select Time Spent Range (hrs)', 
                       min_value=time_spent_min, 
                       max_value=time_spent_max, 
                       value=(time_spent_min, time_spent_max))

# Apply filters based on user input
filtered_data = data[(data['Station'] == station) & 
                     (data['TimeSpent'].apply(lambda x: float(x.split()[0])) >= time_spent[0]) & 
                     (data['TimeSpent'].apply(lambda x: float(x.split()[0])) <= time_spent[1])]

# Display filtered data
st.write(filtered_data)

# Create a bar chart of Quantity Refilled per Station
st.subheader('Total Quantity Refilled per Station')

filtered_data_grouped = filtered_data.groupby('Station')['QtyRefilled'].sum()
fig, ax = plt.subplots()
filtered_data_grouped.plot(kind='bar', ax=ax, figsize=(10, 6))
ax.set_title('Total Quantity Refilled per Station')
ax.set_xlabel('Station')
ax.set_ylabel('Total Quantity Refilled')
st.pyplot(fig)
