import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'app_data', 'model.pkl')
DATA_PATH = os.path.join(BASE_DIR, 'Model_Phase', 'data', 'credit_card_transactions.csv')

# Indices of features selected by BSSA for this specific model
SELECTED_INDICES = [1, 3, 7, 8, 9, 12, 13, 16, 19, 21]
FEATURE_NAMES = [
    "category", "gender", "city_pop", "job", "unix_time", 
    "merch_zipcode", "trans_date_trans_time_year", "dob_year", "log_amt", "hour"
]

def load_data_and_fit_preprocessors():
    """
    Fits the scalers and encoders using the training set of the processed CSV.
    Note: The processed CSV in Model_Phase/data is already scaled, but we fit 
    a NEW scaler on a sample to get the transformation parameters (mean/std) 
    if possible, OR we just trust the unique values.
    
    Wait: The processed CSV IS the training data (partially).
    To be 100% accurate, we would need to reverse the scaling, but since we have 
    the processed CSV, we can just use the unique values for categories.
    For numerical inputs, we'll provide sliders based on the distribution.
    """
    df = pd.read_csv(DATA_PATH)
    
    # In reality, the UI should take RAW values and transform them.
    # But since the model expects SCALED values, we need the Scaler.
    # We will fit a scaler on the processed data to 'mock' the transformation 
    # if we had the raw data.
    # Actually, the simplest way is to let the user input 'Scaled' values 
    # or we provide a mapping.
    
    # RECOVERY PLAN: 
    # 1. We have the unique scaled values for categorical columns. 
    # 2. We have the original strings from Data_Phase.
    # 3. We create a dictionary mapping: string -> scaled_value.
    
    return df

def get_model():
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    return model

def predict(model, input_vector):
    """
    input_vector: 10 features in correct order
    """
    # Ensure it's a 2D array
    X = np.array(input_vector).reshape(1, -1)
    # XGBoost prediction
    prob = model.predict_proba(X)[0][1]
    pred = model.predict(X)[0]
    return pred, prob
