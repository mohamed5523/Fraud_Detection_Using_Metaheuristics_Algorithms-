"""
Configuration file for fraud detection project
Contains all project settings and parameters
"""

import os
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'datasets'
OUTPUT_DIR = BASE_DIR / 'output'
PROCESSED_DATA_DIR = OUTPUT_DIR / 'processed_data'
VISUALIZATIONS_DIR = OUTPUT_DIR / 'visualizations'
BALANCED_DATA_DIR = OUTPUT_DIR / 'balanced_data'

# Create directories if they don't exist
for dir_path in [DATA_DIR, OUTPUT_DIR, PROCESSED_DATA_DIR, VISUALIZATIONS_DIR, BALANCED_DATA_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Dataset configurations
DATASETS = {
    'CCTF': {
        'filename': 'credit_card_transactions.csv',
        'target_column': 'is_fraud',
        'date_columns': ['trans_date_trans_time', 'dob'],
        'categorical_columns': ['merchant', 'category', 'gender', 'state', 'job'],
        'numerical_columns': ['amt', 'lat', 'long', 'city_pop', 'unix_time', 
                            'merch_lat', 'merch_long'],
        'drop_columns': ['Unnamed: 0', 'cc_num', 'first', 'last', 'street', 
                        'city', 'zip', 'trans_num']
    },
    'Statlog': {
        'filename': 'australian.csv',
        'target_column': 'Y',
        'date_columns': [],
        'categorical_columns': [],
        'numerical_columns': ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X9', 
                            'X10', 'X11', 'X12', 'X13', 'X14'],
        'drop_columns': []
    },
    'European': {
        'filename': 'European.csv',
        'target_column': 'Class',
        'date_columns': [],
        'categorical_columns': [],
        'numerical_columns': ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)],
        'drop_columns': []
    },
    'Paysim': {
        'filename': 'paysim dataset.csv',
        'target_column': 'isFraud',
        'date_columns': [],
        'categorical_columns': ['type'],
        'numerical_columns': ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
                            'oldbalanceDest', 'newbalanceDest'],
        'drop_columns': ['nameOrig', 'nameDest', 'isFlaggedFraud']
    }
}

# Preprocessing parameters
MISSING_VALUE_THRESHOLD = 0.5  # Drop columns with more than 50% missing values
OUTLIER_METHOD = 'IQR'  # Options: 'IQR', 'Z-score'
OUTLIER_THRESHOLD = 3  # For Z-score method
SCALING_METHOD = 'standard'  # Options: 'standard', 'minmax', 'robust'

# Visualization parameters
FIGURE_SIZE = (12, 8)
COLOR_PALETTE = 'Set2'
PLOT_STYLE = 'seaborn'
DPI = 100
SAVE_PLOTS = True

# Class balancing parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Sampling strategies
SAMPLING_STRATEGIES = {
    'rus': {
        'sampling_strategy': 'auto',
        'random_state': RANDOM_STATE,
        'replacement': False
    },
    'smote': {
        'sampling_strategy': 'auto',
        'random_state': RANDOM_STATE,
        'k_neighbors': 5,
        'n_jobs': -1
    }
}

# Logging settings
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'