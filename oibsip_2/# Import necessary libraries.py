# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the unemployment data
data = pd.read_csv('unemployment_data.csv', parse_dates=['Date'])

# Display the first few rows of the dataset
print(data.head())

# Basic statistics
print(data.describe())

# Check for missing values
print(data.isnull().sum())

# Plot the unemployment rate over time
plt.figure(figsize=(12, 6))
sns.lineplot(data=data, x='Date', y='Unemployment_Rate', marker='o')
plt.title('Unemployment Rate Over Time')
plt.xlabel('Date')
plt.ylabel('Unemployment Rate (%)')
plt.grid(True)
plt.show()

# Moving average to smooth out the series
data['Unemployment_Rate_MA'] = data['Unemployment_Rate'].rolling(window=12).mean()

# Plot the moving average
plt.figure(figsize=(12, 6))
sns.lineplot(data=data, x='Date', y='Unemployment_Rate_MA', marker='o', label='12-Month Moving Average')
plt.title('Unemployment Rate (12-Month Moving Average)')
plt.xlabel('Date')
plt.ylabel('Unemployment Rate (%)')
plt.legend()
plt.grid(True)
plt.show()

# Seasonal decomposition
from statsmodels.tsa.seasonal import seasonal_decompose

result = seasonal_decompose(data['Unemployment_Rate'], model='additive', period=12)
result.plot()
plt.show()

# Optional: Predict future unemployment rates using ARIMA
from statsmodels.tsa.arima.model import ARIMA

# Fit the ARIMA model
model = ARIMA(data['Unemployment_Rate'], order=(5, 1, 0))
model_fit = model.fit()

# Summary of the model
print(model_fit.summary())

# Forecast future values
forecast = model_fit.forecast(steps=12)
print(forecast)

# Plot the forecast
plt.figure(figsize=(12, 6))
plt.plot(data['Date'], data['Unemployment_Rate'], label='Historical Data')
plt.plot(pd.date_range(start=data['Date'].iloc[-1], periods=12, freq='M'), forecast, label='Forecast', color='red')
plt.title('Unemployment Rate Forecast')
plt.xlabel('Date')
plt.ylabel('Unemployment Rate (%)')
plt.legend()
plt.grid(True)
plt.show()
