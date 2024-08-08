import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from gurobipy import Model, GRB

# Load the dataset
file_path = 'Car Sales.xlsx - car_data.csv'
car_sales_data = pd.read_csv(file_path)

# Data preprocessing
car_sales_data['Date'] = pd.to_datetime(car_sales_data['Date'])
car_sales_data['Month'] = car_sales_data['Date'].dt.to_period('M')
monthly_sales = car_sales_data.groupby(['Month', 'Model'])['Price ($)'].sum().unstack().fillna(0)

# Display the first few rows of the processed data
print("Processed Data Head:")
print(monthly_sales.head())

# Plotting monthly sales for each model
plt.figure(figsize=(15, 10))
for model in monthly_sales.columns:
    plt.plot(monthly_sales.index.to_timestamp(), monthly_sales[model], label=model)
plt.title('Monthly Sales for Each Model')
plt.xlabel('Month')
plt.ylabel('Total Sales')
plt.legend(title='Model', loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('monthly_sales_trends.png')
plt.show()

# Select a model for forecasting
selected_model = monthly_sales.columns[0]  # Replace with the model name you want to forecast

# ARIMA model
print("Running ARIMA model...")
model = SARIMAX(monthly_sales[selected_model], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
results = model.fit()
forecast = results.get_forecast(steps=12).predicted_mean
print("ARIMA model completed.")

# Prophet model
print("Running Prophet model...")
df = monthly_sales.reset_index()[['Month', selected_model]]
df.columns = ['ds', 'y']
df['ds'] = df['ds'].dt.to_timestamp()
prophet_model = Prophet()
prophet_model.fit(df)
future = prophet_model.make_future_dataframe(periods=12, freq='M')
forecast_prophet = prophet_model.predict(future)
print("Prophet model completed.")

# Save the forecast results
forecast_results_file = 'forecast_results.csv'
forecast_prophet[['ds', 'yhat']].to_csv(forecast_results_file, index=False)
print(f"\nForecast results saved to {forecast_results_file}")

# Define the model
print("Running optimization model...")
m = Model('Inventory Optimization')

# Parameters
demand = forecast.values
holding_cost = 2
ordering_cost = 50
shortage_cost = 5
lead_time = 1
safety_stock = 0.1 * demand

# Variables
order_qty = m.addVars(len(demand), vtype=GRB.INTEGER, name="order_qty")
inventory = m.addVars(len(demand), vtype=GRB.INTEGER, name="inventory")

# Objective function
m.setObjective(
    ordering_cost * order_qty.sum() +
    holding_cost * inventory.sum() +
    shortage_cost * (demand - inventory).sum(), GRB.MINIMIZE)

# Constraints
for t in range(1, len(demand)):
    m.addConstr(inventory[t] == inventory[t-1] + order_qty[t-1] - demand[t-1], name=f"balance_{t}")
    m.addConstr(order_qty[t] >= safety_stock[t], name=f"safety_stock_{t}")

# Optimize
m.optimize()
print("Optimization model completed.")

# Save optimization results
optimization_results = pd.DataFrame({
    'Month': pd.date_range(start='2023-01-01', periods=12, freq='M'),
    'Demand': demand,
    'Order Quantity': [order_qty[t].x for t in range(len(demand))],
    'Inventory': [inventory[t].x for t in range(len(demand))]
})
optimization_results_file = 'optimization_results_advanced.csv'
optimization_results.to_csv(optimization_results_file, index=False)
print(f"\nOptimization results saved to {optimization_results_file}")
