import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Load dataset
file_path = r"D:\Self_Learning\Data Analysics Projects\Sales Data Analystic\cleaned_data.csv"  # Update path if necessary
df = pd.read_csv(file_path)

# Convert Order_Date to datetime
df["Order_Date"] = pd.to_datetime(df["Order_Date"])

# Aggregate monthly sales
df["Year-Month"] = df["Order_Date"].dt.to_period("M")
monthly_sales = df.groupby("Year-Month")["Sales"].sum().reset_index()

# Convert to datetime for modeling
monthly_sales["Year-Month"] = monthly_sales["Year-Month"].astype(str)
monthly_sales["Year-Month"] = pd.to_datetime(monthly_sales["Year-Month"])

# Fit ARIMA Model
model = ARIMA(monthly_sales["Sales"], order=(2, 1, 2))  # ARIMA(p=2, d=1, q=2)
model_fit = model.fit()

# Forecast next 12 months
forecast_steps = 12
future_dates = pd.date_range(start=monthly_sales["Year-Month"].max(), periods=forecast_steps+1, freq="M")[1:]
forecast = model_fit.forecast(steps=forecast_steps)

# Create forecast dataframe
forecast_df = pd.DataFrame({"Year-Month": future_dates, "Forecasted_Sales": forecast})

# Plot actual vs forecasted sales
plt.figure(figsize=(12, 5))
plt.plot(monthly_sales["Year-Month"], monthly_sales["Sales"], marker="o", linestyle="-", color="blue", label="Actual Sales")
plt.plot(forecast_df["Year-Month"], forecast_df["Forecasted_Sales"], marker="o", linestyle="--", color="red", label="Forecasted Sales")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.title("Sales Forecast for Next 12 Months")
plt.legend()
plt.grid()
plt.show()

# Save forecast results
forecast_df.to_csv("sales_forecast.csv", index=False)
print("Forecast saved as 'sales_forecast.csv'")