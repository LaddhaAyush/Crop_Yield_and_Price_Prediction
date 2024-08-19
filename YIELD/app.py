from flask import Flask, request, render_template, jsonify
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the CatBoost model
with open('catboost_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

def apply_water_penalty(water_avail, yield_pred):
    lower_threshold = 6000
    upper_threshold = 11000
    if water_avail < lower_threshold:
        water_penalty = np.exp(-(lower_threshold - water_avail) ** 2 / (2 * (950 ** 2)))
    elif water_avail > upper_threshold:
        water_penalty = np.exp(-(water_avail - upper_threshold) ** 2 / (2 * (950 ** 2)))
    else:
        water_penalty = 1.0
    return yield_pred * water_penalty

def adjust_yield_ph(ph_level, yield_pred):
    ph_penalty = np.where(
        (ph_level <= 5) | (ph_level >= 10),
        np.exp(-(ph_level - 7) ** 2 / (2 * (1 ** 2))),
        1.0
    )
    return yield_pred * ph_penalty

def adjust_yield_irrigation(irrigation_method, yield_pred):
    if irrigation_method == 'Drip':
        return yield_pred * 1.1
    elif irrigation_method == 'Sprinkler':
        return yield_pred * 1.0
    elif irrigation_method == 'Canal':
        return yield_pred * 0.95
    elif irrigation_method == 'Tube Well':
        return yield_pred * 0.90
    else:
        return yield_pred

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    user_df = pd.DataFrame([data])
    y_user_pred = model.predict(user_df)
    water_avail = float(user_df['Water Availability (liters/hectare)'])
    y_user_pred_with_penalty = apply_water_penalty(water_avail, y_user_pred)
    ph_level = float(user_df['pH Level'])
    y_user_pred_with_penalty = adjust_yield_ph(ph_level, y_user_pred_with_penalty)
    irrigation_method = user_df['Irrigation Method'].iloc[0]
    y_user_pred_with_penalty = adjust_yield_irrigation(irrigation_method, y_user_pred_with_penalty)
    return jsonify({"predicted_yield": y_user_pred_with_penalty[0]})

if __name__ == '__main__':
    app.run(debug=True)
