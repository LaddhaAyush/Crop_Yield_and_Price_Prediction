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
    future_dates = [input_date + timedelta(days=i) for i in range(1, num_days + 1)]

    future_features = create_future_features(future_dates)
    future_predictions_rf = rf_model.predict(future_features)

    max_price_index = future_predictions_rf.argmax()
    max_price_date = future_dates[max_price_index].strftime('%Y-%m-%d')
    max_price = future_predictions_rf[max_price_index]

    predictions = [{"Date": date.strftime('%Y-%m-%d'), "Predicted_Price": price} for date, price in zip(future_dates, future_predictions_rf)]

    return render_template('index.html', predictions=predictions, max_price=max_price, max_price_date=max_price_date, num_days=num_days)

if __name__ == '__main__':
    app.run(debug=True)