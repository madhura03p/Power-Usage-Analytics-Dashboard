# Power-Usage-Analytics-Dashboard
Developed during my internship at Tata Motors, this interactive dashboard monitors, analyzes and forecasts electrical power usage. The tool, built using Dash and Plotly, processes user-uploaded CSV files to generate detailed, interactive visualizations. Users can filter data by stations, shifts, and custom date ranges for detailed insights.

## Overview

Power-Usage-Analytics-Dashboard is an interactive web application developed during my internship at Tata Motors to monitor, analyze, and forecast electrical power consumption. The tool, built using *Dash* and *Plotly*, processes user-uploaded CSV files to generate detailed, interactive visualizations, helping stakeholders track power usage, identify trends, and make data-driven decisions.

This dashboard is designed to provide:
- Shift-wise power consumption analysis
- Trend and seasonality detection
- 7-day power consumption forecasting using SARIMA


## Key Features
- **Interactive Data Upload:** Users can upload a CSV file containing electrical power consumption data. The data is automatically unpivoted and processed using Python’s `pandas` library.
- **Comprehensive Data Visualizations:** The dashboard offers several interactive graphs for in-depth analysis:
  - **Power Consumption Over Time (Line Graph)**
  - **Station-wise Hourly Consumption (Stacked Area Graph)**
  - **Shift-wise Power Consumption Distribution (Pie Chart)**
  - **Shift-wise Min/Max/Average Consumption (Clustered Bar Graph)**
  - **Hourly Average Consumption by Day (Bar Graph)**
  - **Trend & Seasonality Analysis (Line Graph)**
  - **7-day Forecast (Line Graph and Table)**
- **Custom Filtering:** Users can filter data by stations, shifts, and custom date ranges.

## Graphical Analysis

### 1. **Power Consumption Over Time (Line Graph)**
This line graph illustrates power consumption over time, providing a continuous visual representation that makes it easy to observe trends, peaks, and troughs. 

### 2. **Station-wise Hourly Consumption (Stacked Area Graph)**
This stacked area graph shows the contribution of each station’s power consumption relative to the total consumption, scaled to 100%. Each station is represented by a distinct color, filling its proportional area in the graph over time. 

### 3. **Shift-wise Power Consumption (Pie Chart)**
The pie chart represents the share of total energy consumed during each operational shift (Shift 1, Shift 2, Shift 3). 

### 4. **Shift-wise Minimum, Maximum, and Average Consumption (Clustered Bar Graph)**
This clustered bar graph compares the minimum, maximum, and average power consumption for each shift. An overall average line is also included. 

### 5. **Hourly Average Power Consumption by Day (Bar Graph)**
This bar graph shows the average power consumption for each hour of the day, broken down by day of the week (e.g., Monday, Tuesday). 

### 6. **Trend and Seasonality Analysis (Line Graph)**
This graph presents a time series analysis, breaking down power consumption into its *trend* and *seasonal* components. A 3-day moving average is used to smooth short-term fluctuations and reveal long-term trends. The seasonal component highlights recurring patterns, such as daily or weekly cycles, that might influence power usage.

### 7. **Next 7-day Prediction (Line Graph and Table)**
The 7-day SARIMA forecast provides a predictive model for power consumption over the next week. A line graph visualizes the predicted power consumption, including upper and lower bounds for uncertainty. A table below the graph provides the exact forecast values (in KWH).

## Data Processing Workflow
The CSV data uploaded by the user undergoes several processing steps:

1. **Data Upload & Unpivoting:**
   - The CSV file is parsed, and the data is unpivoted to convert it into a long format, making it easier to analyze over time. This transformation ensures that the data is structured for further processing and visualization.

2. **Error Handling:**
   - Data processing is wrapped in a `try-else` block to handle errors gracefully, ensuring that invalid or incomplete data doesn’t crash the application. If errors are detected, users are notified and asked to upload a valid CSV file.

3. **Data Cleaning & Transformation:**
   - Missing values are handled, and the data is cleaned and transformed for analysis. This includes removing unwanted sections, standarizing date formats and standarizing shift names, assigning stations for each column, and creating derived columns (e.g., shifts, date range, stations).

## Technical Highlights
- **Dash & Plotly:** Built with *Dash* for an interactive web interface, using *Plotly* for dynamic, rich visualizations.
- **Data Processing & Forecasting:** Implemented with *Python*, *Pandas*, and *Statsmodels* to process and analyze power consumption data. The *SARIMA* model forecasts future power consumption based on historical data.
- **Error Handling:** Comprehensive error handling ensures smooth operation.

## How It Works

1. **Data Upload:**
   - Users upload a CSV file that contains historical power usage data.
   - The data is unpivoted and processed into a format suitable for analysis.

2. **Data Analysis & Visualization:**
   - Once processed, the data is displayed using interactive graphs (line, bar, stacked area, pie charts, etc).
   - Users can filter by stations, shifts, and date ranges.

3. **Trend & Seasonality Analysis:**
   - The time series analysis component identifies long-term trends and seasonal patterns in power usage.

4. **Forecasting:**
   - A **SARIMA** model predicts power consumption for the next 7 days. These predictions are visualized in both a line graph and table.

## Technologies Used
- **Frontend:** Dash, Plotly
- **Backend:** Python
- **Libraries:** Pandas, NumPy, Statsmodels, Plotly Express
- **Statistical Modeling:** SARIMA (Seasonal ARIMA) for forecasting
