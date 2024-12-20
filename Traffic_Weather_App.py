import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
from xgboost import XGBRegressor

# Load models and datasets
with open("whyNotDevelop\\Predicting-Ho-Chi-Minh-City-Traffic-Flow-Based-on-Weather-Conditions\\xgb_model.pkl", "rb") as file:
    rf_model = pickle.load(file)

daily_data_path = r"whyNotDevelop\\Predicting-Ho-Chi-Minh-City-Traffic-Flow-Based-on-Weather-Conditions\\daily_traffic_weather.csv"
full_data_path = r"whyNotDevelop\\Predicting-Ho-Chi-Minh-City-Traffic-Flow-Based-on-Weather-Conditions\\cleaned_traffic_weather_dataset.csv"

daily_df = pd.read_csv(daily_data_path)
full_df = pd.read_csv(full_data_path)

# Function to remove outliers
def remove_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

if 'velocity' in full_df.columns:
    full_df = remove_outliers(full_df, 'velocity')

# Ensure KMeans labels match data used
clustering_features = ['rain', 'max', 'min', 'humidi', 'cloud', 'pressure', 'length', 'wind', 'mean_velocity']
clustering_data = daily_df[clustering_features]
clustering_data = clustering_data.dropna()  # Drop missing values

# Streamlit app title
st.title("Traffic and Weather Analysis")

# Sidebar navigation
menu = st.sidebar.radio("Navigation", ["Analysis", "Predict and Suggest"])

if menu == "Analysis":
    # Dataset selection
    data_option = st.radio("Select Dataset:", ["Full Dataset", "Daily Aggregated Data"])

    if data_option == "Full Dataset":
        st.subheader("Full Dataset")
        st.write("Sample Data:")
        st.write(full_df.head())

        # Velocity distribution
        if 'velocity' in full_df.columns:
            st.subheader("Velocity Distribution")
            st.write("This plot shows how traffic velocity is distributed across the dataset. Peaks indicate common velocity ranges, which help identify trends in traffic flow.")
            fig, ax = plt.subplots()
            sns.histplot(full_df['velocity'], kde=True, ax=ax, color='blue')
            ax.set_title("Distribution of Traffic Velocity")
            st.pyplot(fig)

        # Velocity by street level
        if 'street_level' in full_df.columns:
            st.subheader("Box Plot: Velocity by Street Level")
            st.write("This plot illustrates variations in traffic velocity across street levels, offering insights into how road hierarchy shapes travel speeds.")
            fig, ax = plt.subplots()
            sns.boxplot(data=full_df, x='street_level', y='velocity', ax=ax)
            ax.set_title("Traffic Velocity by Street Level")
            st.pyplot(fig)

        # Geospatial visualization
        if 'lat' in full_df.columns and 'long' in full_df.columns:
            st.subheader("Geospatial Visualization of Velocity")
            st.write("This map visualizes traffic velocity geographically, showing areas in Ho Chi Minh City with consistently high or low traffic speeds.")
            fig, ax = plt.subplots(figsize=(8, 5))
            scatter = ax.scatter(full_df['long'], full_df['lat'], c=full_df['velocity'], cmap='coolwarm', alpha=0.7)
            fig.colorbar(scatter, ax=ax, label='Velocity')
            ax.set_title('Velocity Across Geographical Locations')
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            plt.grid(True)
            st.pyplot(fig)

    elif data_option == "Daily Aggregated Data":
        st.subheader("Daily Aggregated Data")
        st.write("Sample Data:")
        st.write(daily_df.head())

        # Time series: Mean Velocity
        if 'date' in daily_df.columns and 'mean_velocity' in daily_df.columns:
            st.subheader("Time Series: Mean Velocity")
            st.write("This plot tracks the average traffic velocity over time, helping identify patterns or anomalies in traffic flow. Overall this plot shows a drop in velocty indicating traffic congested traffic conditions in the Ho Chi Minh City.")
            daily_df['date'] = pd.to_datetime(daily_df['date'], errors='coerce')
            fig, ax = plt.subplots()
            sns.lineplot(data=daily_df, x='date', y='mean_velocity', ax=ax)
            ax.set_title("Mean Velocity Over Time")
            plt.xticks(rotation=45)
            st.pyplot(fig)

        # Time series: Rain and Humidity
        if 'rain' in daily_df.columns and 'humidi' in daily_df.columns:
            st.subheader("Time Series: Rain and Humidity")
            st.write("This plot illustrates changes in rainfall and humidity over time, which are key factors affecting traffic conditions.")
            fig, ax = plt.subplots()
            sns.lineplot(data=daily_df, x='date', y='rain', ax=ax, label='Rain (mm)', color='blue')
            sns.lineplot(data=daily_df, x='date', y='humidi', ax=ax, label='Humidity (%)', color='green')
            ax.set_title("Rain and Humidity Over Time")
            ax.set_xlabel("Date")
            ax.set_ylabel("Values")
            plt.xticks(rotation=45)
            st.pyplot(fig)

        # Min-Max time series plot
        if 'date' in daily_df.columns and 'max' in daily_df.columns and 'min' in daily_df.columns:
            st.subheader("Time Series: Min and Max Temperature")
            st.write("This plot showcases temperature fluctuations, which can affect traffic through extreme heat or cold conditions.")
            fig, ax = plt.subplots()
            daily_df['date'] = pd.to_datetime(daily_df['date'], errors='coerce')
            sns.lineplot(data=daily_df, x='date', y='max', ax=ax, label='Max Temperature', color='red')
            sns.lineplot(data=daily_df, x='date', y='min', ax=ax, label='Min Temperature', color='blue')
            ax.set_title("Min and Max Temperature Over Time")
            ax.set_xlabel("Date")
            ax.set_ylabel("Temperature (°C)")
            plt.xticks(rotation=45)
            st.pyplot(fig)

elif menu == "Predict and Suggest":
    st.subheader("Predict Traffic Conditions")

    # Input sliders for features (excluding velocity)
    rain = st.slider("Rain (mm)", min_value=0.0, max_value=100.0, step=0.1)
    humidity = st.slider("Humidity (%)", min_value=0, max_value=100, step=1)
    cloud = st.slider("Cloud Cover (%)", min_value=0, max_value=100, step=1)
    pressure = st.slider("Pressure (hPa)", min_value=950, max_value=1050, step=1)
    min_temp = st.slider("Min Temperature (°C)", min_value=-10, max_value=50, step=1)
    max_temp = st.slider("Max Temperature (°C)", min_value=-10, max_value=50, step=1)
    length = st.slider("Length", min_value=0, max_value=100, step=1)
    wind = st.slider("Wind", min_value=0, max_value=100, step=1)

    # Prepare input data for velocity prediction
    input_data_xgb = [[rain, humidity, cloud, pressure, min_temp, max_temp, length, wind]]  # 8 features

    # Predict velocity using XGBoost model
    predicted_velocity = rf_model.predict(input_data_xgb)[0]

    # Display results
    st.write(f"Predicted Traffic Velocity: {predicted_velocity:.2f} km/h")

    # Add dynamic text summary
    st.subheader("Traffic Insights")
    if predicted_velocity < 20:
        st.write("The traffic is highly congested. Expect significant delays.")
    elif 20 <= predicted_velocity < 40:
        st.write("Moderate traffic flow. Plan for some delays.")
    else:
        st.write("Smooth traffic conditions. Enjoy your journey!")
