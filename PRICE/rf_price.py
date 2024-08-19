import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime, timedelta
import pickle

# Load the dataset
crop_price_data = pd.read_csv("crop_price_data.csv")

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
        if f'Crop_{crop}' in future_features.columns:
            future_features[f'Crop_{crop}'] = 0
        if f'Market_{market}' in future_features.columns:
            future_features[f'Market_{market}'] = 0

    # Drop the "Date" column
    future_features.drop(columns=['Date'], inplace=True)

    return future_features


# Get all crop-market combinations seen during training
crop_market_combinations = crop_price_data[['Crop', 'Market']].drop_duplicates()

# User input
district = input("Enter District: ")
crop = input("Enter Crop: ")
market = input("Enter Market: ")
input_date = input("Enter the date (YYYY-MM-DD) for prediction: ")

# Convert input date to datetime object
input_date = datetime.strptime(input_date, "%Y-%m-%d")

# Track maximum price and its date
max_price = -float('inf')
max_price_date = None

# Train Random Forest Regression and make future predictions
for group_name, group in grouped:
    if group_name[0] != crop or group_name[1] != market:
        continue

    group = group.sort_values('Date').set_index('Date')

    # Check if there are enough samples for splitting
    if len(group) < 3:
        print(f"Not enough samples for splitting in group: {group_name}")
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

    # Generate future dates for the next 14 days
    future_dates = [input_date + timedelta(days=i) for i in range(1, 15)]

    # Create features for future dates
    future_features = create_future_features(future_dates, crop_market_combinations)

    # Predict future prices with Random Forest model
    future_predictions_rf = rf_model.predict(future_features)
    print(f"Future Price Predictions for {group_name} using Random Forest for the next 14 days:")
    for date, price in zip(future_dates, future_predictions_rf):
        print(f"Date: {date.strftime('%Y-%m-%d')}, Predicted Price: {price:.2f}")
        if price > max_price:
            max_price = price
            max_price_date = date

# Print maximum crop price and its date
print(f"\nMaximum Crop Price: {max_price:.2f}, Date: {max_price_date.strftime('%Y-%m-%d')}")
with open("rf_model.pkl", "wb") as f:
    pickle.dump(rf_model, f)

crop_market_combinations = crop_price_data[['Crop', 'Market']].drop_duplicates()

with open("crop_market_combinations.pkl", "wb") as f:
    pickle.dump(crop_market_combinations, f) 