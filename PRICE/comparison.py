import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime, timedelta
import pickle

# Load the dataset
crop_price_data = pd.read_csv('crop_price_data.csv')

# Preprocess the data
crop_price_data['Date'] = pd.to_datetime(crop_price_data['Date'])
grouped = crop_price_data.groupby(['Crop', 'Market'])

# Function to create features for future dates
def create_future_features(future_dates, crop_market_combinations):
    future_features = pd.DataFrame({'Date': future_dates})
    future_features['year'] = future_features['Date'].dt.year
    future_features['month'] = future_features['Date'].dt.month
    future_features['day'] = future_features['Date'].dt.day

    # Create columns for each crop and market combination
    for _, row in crop_market_combinations.iterrows():
        crop = row['Crop']
        market = row['Market']
        future_features[f'Crop_{crop}'] = 0
        future_features[f'Market_{market}'] = 0

    # Drop the "Date" column
    future_features.drop(columns=['Date'], inplace=True)

    return future_features

# Get all crop-market combinations seen during training
crop_market_combinations = crop_price_data[['Crop', 'Market']].drop_duplicates()

# Lists to collect actual and predicted prices for all groups
all_actual_prices = []
all_predicted_prices = []

# Train Random Forest Regression and make future predictions
for group_name, group in grouped:
    group = group.sort_values('Date').set_index('Date')

    # Check if there are enough samples for splitting
    if len(group) < 3:
        continue

    # Feature engineering
    group['year'] = group.index.year
    group['month'] = group.index.month
    group['day'] = group.index.day
    X = pd.get_dummies(group[['District', 'Crop', 'Market', 'year', 'month', 'day']], drop_first=True)
    y = group['Price (INR/quintal)']

    # Split into train and test sets
    tscv = TimeSeriesSplit(n_splits=2)
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Train Random Forest Regression
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)

        # Evaluate on test set
        y_pred = rf_model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        print(f"Random Forest Regression - Group: {group_name}, MAE: {mae:.2f}, MSE: {mse:.2f}")

        # Collect actual and predicted prices
        all_actual_prices.extend(y_test)
        all_predicted_prices.extend(y_pred)

# Plot overall actual vs. predicted prices
plt.figure(figsize=(10, 6))
plt.plot(all_actual_prices, label='Actual Price', color='blue')
plt.plot(all_predicted_prices, label='Predicted Price', color='red', linestyle='--')
plt.xlabel('Sample Index')
plt.ylabel('Price (INR/quintal)')
plt.title('Overall Actual vs Predicted Prices')
plt.legend()
plt.show()

# Save the model and crop-market combinations
with open("rf_model.pkl", "wb") as f:
    pickle.dump(rf_model, f)

with open("crop_market_combinations.pkl", "wb") as f:
    pickle.dump(crop_market_combinations, f)
