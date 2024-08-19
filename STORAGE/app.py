from flask import Flask, request, render_template
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta
import pickle

app = Flask(__name__)

# Load the trained model
with open("rf_model.pkl", "rb") as f:
    rf_model = pickle.load(f)

# Function to create features for future dates
def create_future_features(future_dates):
    future_features = pd.DataFrame({'Date': future_dates})
    future_features['year'] = future_features['Date'].dt.year
    future_features['month'] = future_features['Date'].dt.month
    future_features['day'] = future_features['Date'].dt.day
    future_features.drop(columns=['Date'], inplace=True)
    return future_features

# Data structure to store the number of days each crop can be stored
crop_storage_days = {
    "Wheat": 270,
    "Barley": 270,
    "Onion": 180,
    "Bajra": 180,
    "Chilli": 60,
    "Coriander": 30,
    "Citrus": 60,
    "Cotton": 180,
    "Fennel": 180,
    "Fenugreek": 30,
    "Garlic": 150,
    "Gram": 180,
    "Guava": 14,
    "Maize": 180,
    "Mango": 14,
    "Mustard": 180,
    "Oilseeds": 180,
    "Opium": 180,
    "Pulses": 180,
    "Sugarcane": 1,
    "Tomato": 14
}
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    district = request.form.get("District")
    crop = request.form.get("Crop")
    market = request.form.get("Market")
    input_date = request.form.get("Date")
    num_days = int(request.form.get("NumberOfDays"))

    input_date = datetime.strptime(input_date, "%Y-%m-%d")
    future_dates = [input_date + timedelta(days=i*6) for i in range(1, (num_days // 6) + 2)]

    future_features = create_future_features(future_dates)
    future_predictions_rf = rf_model.predict(future_features)

    max_price_index = future_predictions_rf.argmax()
    max_price_date = future_dates[max_price_index].strftime('%Y-%m-%d')
    max_price = future_predictions_rf[max_price_index]

    predictions = [{"Date": date.strftime('%Y-%m-%d'), "Predicted_Price": price} for date, price in zip(future_dates, future_predictions_rf)]

    # Check if max price date is within the storage period for the selected crop
    # Check if max price date is within the storage period for the selected crop
    if crop in crop_storage_days:
        storage_days = crop_storage_days[crop]
        if datetime.strptime(max_price_date, "%Y-%m-%d") <= input_date + timedelta(days=storage_days):
            max_profit_date = (input_date + timedelta(days=storage_days)).strftime('%Y-%m-%d')
            max_profit_message = f"On {max_price_date}, you can sell your {crop} to maximize profit."
        else:
            max_price_date = (input_date + timedelta(days=storage_days)).strftime('%Y-%m-%d')
            max_profit_message = f"The max price date falls outside the storage period for {crop}. Adjusted max price date is {max_price_date}."
    else:
        max_profit_message = "No information available for storage period of the selected crop."
    return render_template('index.html', predictions=predictions, max_price=max_price, max_price_date=max_price_date,
                           num_days=num_days, max_profit_message=max_profit_message)

if __name__ == '__main__':
    app.run(debug=True)
