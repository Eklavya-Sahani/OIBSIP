# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA

# Load the sales data
data = pd.read_csv('sales_data.csv', parse_dates=['Date'], index_col='Date')

# Display the first few rows of the dataset
print(data.head())

# Basic statistics
print(data.describe())

# Check for missing values
print(data.isnull().sum())

# Plot the sales data over time
plt.figure(figsize=(12, 6))
sns.lineplot(data=data, x='Date', y='Sales', marker='o')
plt.title('Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.grid(True)
plt.show()

# Decompose the time series to check its components
result = seasonal_decompose(data['Sales'], model='additive', period=12)
result.plot()
plt.show()

# Fit an ARIMA model
model = ARIMA(data['Sales'], order=(5, 1, 0))
model_fit = model.fit()

# Summary of the model
print(model_fit.summary())

# Forecast future sales
forecast_steps = 12  # Number of periods to forecast
forecast = model_fit.forecast(steps=forecast_steps)

# Create a DataFrame to hold the forecasted values
forecast_dates = pd.date_range(start=data.index[-1], periods=forecast_steps + 1, freq='M')[1:]
forecast_df = pd.DataFrame(forecast, index=forecast_dates, columns=['Forecast'])

# Plot the forecasted values along with historical sales data
plt.figure(figsize=(12, 6))
plt.plot(data['Sales'], label='Historical Sales')
plt.plot(forecast_df, label='Forecast', color='red')
plt.title('Sales Forecast')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.grid(True)
plt.show()
