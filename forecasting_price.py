
# Import necessary libraries
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Load the historical data
file_path = 'XOM.csv'
rebar_data = pd.read_csv(file_path)

# Convert the 'Date' column to datetime and sort the data
rebar_data['Date'] = pd.to_datetime(rebar_data['Date'])
rebar_data.sort_values('Date', inplace=True)

# Set the date as the index
rebar_data.set_index('Date', inplace=True)

# Focus on the 'Price' column for time series analysis
price_series = rebar_data['Price']

# Decompose the time series
decomposition = seasonal_decompose(price_series, model='additive', period=30)

# Function to fit ARIMA model and forecast future values
def forecast_with_arima(series, n_periods):
    # Fit the ARIMA model
    model = ARIMA(series, order=(5,1,0))
    model_fit = model.fit()

    # Forecast future values
    forecast = model_fit.forecast(steps=n_periods)
    return forecast

# Forecasting the next 12 months
n_months = 12
forecasted_values = forecast_with_arima(price_series, n_months)

# Creating a date range for the forecasted period
last_date = price_series.index[-1]
forecasted_dates = pd.date_range(start=last_date, periods=n_months + 1, closed='right')

# Creating a DataFrame for the forecasted values
forecasted_df = pd.DataFrame({
    'Date': forecasted_dates,
    'Forecasted Price': forecasted_values
})
forecasted_df.set_index('Date', inplace=True)

# Plotting the historical and forecasted values
plt.figure(figsize=(12, 6))
plt.plot(price_series, label='Historical Prices', color='blue')
plt.plot(forecasted_df['Forecasted Price'], label='Forecasted Prices', color='red', linestyle='--')
plt.title('Steel Rebar Price Forecast for the Next 12 Months')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
