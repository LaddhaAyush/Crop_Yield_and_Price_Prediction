import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
import pickle

# Your existing functions
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

# Load data
df_crop_production = pd.read_csv("crop_production_data.csv")
df_soil_analysis = pd.read_csv("soil_analysis_data.csv")
df_water_usage = pd.read_csv("water_usage_data.csv")

# Merge data
merge_soil_crop_production = df_crop_production.merge(df_soil_analysis, on='District')
merge_water_soil_crop_production = merge_soil_crop_production.merge(df_water_usage, on=['District', 'Crop'])
df_agro = merge_water_soil_crop_production.copy()

# Drop unnecessary columns
df_agro = df_agro.drop(columns=['Production (metric tons)', 'Water Consumption (liters/hectare)'], axis=1)

# Define features and target
X = df_agro.drop('Yield (quintals)', axis=1)
y = df_agro['Yield (quintals)']

# Split data
X_train, X_, y_train, y_ = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_, y_, test_size=0.7, random_state=42)

# Specify categorical columns explicitly for CatBoost
cat_cols = ['District', 'Crop', 'Season', 'Soil Type', 'Irrigation Method']

# Model training
catboost_model = CatBoostRegressor(random_state=42, cat_features=cat_cols)
catboost_model.fit(X_train, y_train, verbose=False)

# Save the model using pickle
with open('catboost_model.pkl', 'wb') as model_file:
    pickle.dump(catboost_model, model_file)
