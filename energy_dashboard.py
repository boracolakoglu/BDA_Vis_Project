import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Page configuration should be the first Streamlit command
st.set_page_config(layout="wide")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('cleaned_energy_data.csv', parse_dates=['time'])
    return df

df = load_data()

# Create a new 'date' column for daily grouping
df['date'] = df['time'].dt.date

# Select the energy columns (the ones containing '[kW]')
energy_columns = df.columns[df.columns.str.contains('[kW]')].tolist()

# Remove the 'use [kW]', 'gen [kW]', and 'House overall [kW]' from the appliance list for Top 5 Energy Consuming Appliances
appliance_columns = [col for col in energy_columns if col not in ['use [kW]', 'gen [kW]', 'House overall [kW]']]

# Filters at the top
st.title('Energy Dashboard')
col1, col2 = st.columns(2)

with col1:
    # Date filter
    start_date = df['date'].min()
    end_date = df['date'].max()
    date_range = st.date_input('Select Date Range', [start_date, end_date], min_value=start_date, max_value=end_date)

with col2:
    # Appliance filter
    selected_appliances = st.multiselect("Select Appliances", appliance_columns, default=appliance_columns[:5])

# Filter the dataframe based on the selected date range
filtered_df = df[(df['date'] >= pd.to_datetime(date_range[0]).date()) & 
                 (df['date'] <= pd.to_datetime(date_range[1]).date())]

# Aggregating energy consumption by day, summing up the values
daily_data = filtered_df.groupby('date')[energy_columns].sum()

# Create a new daily comparison DataFrame for selected appliances
daily_comparison = daily_data[['use [kW]'] + selected_appliances]

# Apply line smoothing using rolling average (smoothing with a window of 7 days)
smoothing_window = 7
daily_comparison_smoothed = daily_comparison.rolling(window=smoothing_window).mean()

# Convert the dates to numerical format (Unix timestamps)
daily_comparison_smoothed['date_num'] = pd.to_datetime(daily_comparison_smoothed.index).astype(int) / 10**9

# 2x2 Grid layout
col1, col2 = st.columns(2)

# Graph 1: Energy Consumption Trend Over Time
with col1:
    st.subheader('Energy Consumption Trend Over Time')

    # Checkbox to toggle 'use [kW]' visibility
    show_use_kw = st.checkbox('Show Use [kW]', value=True)

    # Scatter plot + regression plot for selected appliances
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette("tab10", len(selected_appliances) + 1)

    # Plotting 'use [kW]' if checked
    if show_use_kw:
        ax.scatter(daily_comparison_smoothed['date_num'], daily_comparison_smoothed['use [kW]'], 
                   color=colors[0], alpha=0.5, label='Use [kW] Scatter')
        sns.regplot(x=daily_comparison_smoothed['date_num'], y=daily_comparison_smoothed['use [kW]'], 
                    scatter=False, line_kws={'color': colors[0]}, order=4, ax=ax)

    # Scatter and regplot for each selected appliance
    for i, appliance in enumerate(selected_appliances):
        ax.scatter(daily_comparison_smoothed['date_num'], daily_comparison_smoothed[appliance], 
                   color=colors[i + 1], alpha=0.5, label=f'{appliance} Scatter')
        sns.regplot(x=daily_comparison_smoothed['date_num'], y=daily_comparison_smoothed[appliance], 
                    scatter=False, line_kws={'color': colors[i + 1]}, order=4, ax=ax)

    ax.set_title('Energy Consumption Trend Over Time', fontsize=16)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Energy Consumption [kW]', fontsize=12)
    ax.set_xticks(daily_comparison_smoothed['date_num'][::10])
    ax.set_xticklabels(pd.to_datetime(daily_comparison_smoothed.index[::10]).strftime('%Y-%m-%d'), rotation=45)
    ax.legend(title="Appliance", bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(fig)

# Graph 2: Weather vs Energy Consumption
with col2:
    st.subheader('Weather vs Energy Consumption')

    # Sample data to reduce scatter points and prevent overcrowding
    sampled_df = filtered_df.sample(frac=0.05, random_state=42)  # Only 5% of the data

    # Scatter plot for Temperature vs. Total Energy Consumption
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.regplot(x=sampled_df['temperature'], y=sampled_df['use [kW]'], scatter_kws={'s': 10}, 
                line_kws={'color': 'red'}, order=4)
    ax.set_title('Weather vs Energy Consumption', fontsize=16)
    ax.set_xlabel('Temperature [°C]', fontsize=12)
    ax.set_ylabel('Energy Consumption [kW]', fontsize=12)
    st.pyplot(fig)

# Graph 3: Solar Energy vs Household Energy Consumption
col1, col2 = st.columns(2)

with col1:
    st.subheader('Solar Energy vs Household Energy Consumption')

    # Display solar energy generation and house energy consumption
    total_solar = filtered_df['Solar [kW]'].sum()
    total_house_energy = filtered_df['House overall [kW]'].sum()
    net_energy_consumption = total_house_energy - total_solar

    st.metric("Total Solar Energy Production [kW]", f"{total_solar:.2f}")
    st.metric("Total Household Energy Consumption [kW]", f"{total_house_energy:.2f}")
    st.metric("Net Energy Consumption [kW]", f"{net_energy_consumption:.2f}")

# Graph 4: Top 5 Energy Consuming Appliances (Month-by-Month)
with col2:
    st.subheader('Top 5 Energy Consuming Appliances (Month-by-Month)')

    # Group data by month
    filtered_df['month'] = filtered_df['time'].dt.to_period('M')

    # Sum energy usage for each appliance per month
    monthly_usage = filtered_df.groupby(['month'])[energy_columns].sum()

    # Exclude 'use [kW]', 'gen [kW]', and 'House overall [kW]'
    monthly_usage = monthly_usage.drop(columns=['use [kW]', 'gen [kW]', 'House overall [kW]'])

    # Get the top 5 appliances with the highest total energy consumption
    top_5_monthly = monthly_usage[selected_appliances].sum().sort_values(ascending=False).head(5)

    # Plot the top 5 energy-consuming appliances month by month
    fig, ax = plt.subplots(figsize=(10, 6))
    monthly_usage[top_5_monthly.index].plot(kind='line', marker='o', ax=ax)

    ax.set_title('Top 5 Energy Consuming Appliances (Month-by-Month)', fontsize=16)
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Energy Consumption [kW]', fontsize=12)
    ax.legend(top_5_monthly.index, title="Appliance", bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(fig)
