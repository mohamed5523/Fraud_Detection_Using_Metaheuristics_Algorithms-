"""
Central Configuration for Fraud Detection System
Merges settings from Data_Phase and Model_Phase
"""

import os
from pathlib import Path

# ==================== PATHS ====================
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / 'datasets'
OUTPUT_DIR = BASE_DIR / 'output'
RESULTS_DIR = BASE_DIR / 'results'
LOGS_DIR = BASE_DIR / 'logs'

# Ensure directories exist
for directory in [DATA_DIR, OUTPUT_DIR, RESULTS_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)

# ==================== DATASET CONFIGURATIONS ====================
DATASETS = {
    'European': {
        'filename': 'European.csv',
        'target_column': 'Class',
        'numerical_columns': ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)],
        'categorical_columns': [],
        'date_columns': [],
        'drop_columns': []
    },
    'Statlog': {
        'filename': 'australian.csv',
        'target_column': 'Y',
        'numerical_columns': ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X9', 
                            'X10', 'X11', 'X12', 'X13', 'X14'],
        'categorical_columns': [],
        'date_columns': [],
        'drop_columns': []
    },
    'Paysim': {
        'filename': 'paysim.csv',
        'target_column': 'isFraud',
        'numerical_columns': ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
                            'oldbalanceDest', 'newbalanceDest'],
        'categorical_columns': ['type'],
        'date_columns': [],
        'drop_columns': ['nameOrig', 'nameDest', 'isFlaggedFraud']
    },
    'CCTF': {
        'filename': 'credit_card_transactions.csv',
        'target_column': 'is_fraud',
        'numerical_columns': ['amt', 'lat', 'long', 'city_pop', 'unix_time', 
                            'merch_lat', 'merch_long'],
        'categorical_columns': ['merchant', 'category', 'gender', 'state', 'job'],
        'date_columns': ['trans_date_trans_time', 'dob'],
        'drop_columns': ['Unnamed: 0', 'cc_num', 'first', 'last', 'street', 
                        'city', 'zip', 'trans_num']
    }
}

# ==================== PREPROCESSING ====================
MISSING_VALUE_THRESHOLD = 0.5
OUTLIER_METHOD = 'IQR'
OUTLIER_THRESHOLD = 3
SCALING_METHOD = 'standard'
RANDOM_STATE = 42
TEST_SIZE = 0.2

# ==================== OPTIMIZATION ====================
METAHEURISTICS = [
    'BPSO', 'BHHO', 'BWOA', 'BAT', 'BGOA' # Add others as needed
]

META_PARAMS = {
    'population_size': 30,
    'max_iterations': 50,
    'fitness_weight': 0.99,  # Accuracy weight
    'feature_weight': 0.01   # Feature reduction weight
}

# ==================== CLASSIFIERS ====================
CLASSIFIERS = {
    'XGBoost': {
        'name': 'XGBoost',
        'params': {
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'random_state': RANDOM_STATE,
            'eval_metric': 'logloss'
        }
    },
    'KNN': {
        'name': 'K-Nearest Neighbors',
        'params': {'n_neighbors': 5}
    },
    # Add other classifiers here
}

# ==================== LOGGING ====================
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
