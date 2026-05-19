"""
Configuration file for Feature Selection Optimization using Metaheuristics
"""

import os

# ==================== PATHS ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
MODELS_DIR = os.path.join(RESULTS_DIR, 'models')
METRICS_DIR = os.path.join(RESULTS_DIR, 'metrics')
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')

# Create directories if they don't exist
for directory in [DATA_DIR, RESULTS_DIR, MODELS_DIR, METRICS_DIR, PLOTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# ==================== DATASET CONFIGURATIONS ====================
DATASETS = {
    'dataset1': {
        'file': 'European.csv',
        'target': 'Class',
        'name': 'European'
    },
    'dataset2': {
        'file': 'Australian.csv',
        'target': 'Y',
        'name': 'Australian'
    },
    'dataset3': {
        'file': 'paysim.csv',
        'target': 'isFraud',
        'name': 'paysim'
    },
    'dataset4': {
        'file': 'credit_card_transactions.csv',
        'target': 'is_fraud',
        'name': 'credit_card_transactions'
    }
}

# ==================== METAHEURISTIC ALGORITHMS ====================
METAHEURISTICS = [
    'BHHO',   # Binary Harris Hawks Optimization
    'BHGSO',  # Binary Henry Gas Solubility Optimization
    'BASO',   # Binary Atom Search Optimization
    'BSSA',   # Binary Salp Swarm Algorithm
    'BAO',    # Binary Aquila Optimizer
    'BAVO',   # Binary African Vultures Optimization
    'BMOA',   # Binary Mayfly Optimization Algorithm
    'BPSO',   # Binary Particle Swarm Optimization
    'BGOA',   # Binary Grasshopper Optimization Algorithm
    'BBA',    # Binary Bat Algorithm
    'BWOA'    # Binary Whale Optimization Algorithm
]

# ==================== CLASSIFIERS ====================
CLASSIFIERS = {
    'KNN': {
        'name': 'K-Nearest Neighbors',
        'params': {'n_neighbors': 5}
    },
    'DecisionTree': {
        'name': 'Decision Tree',
        'params': {'max_depth': 10, 'random_state': 42}
    },
    'XGBoost': {
        'name': 'XGBoost',
        'params': {
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'random_state': 42,
            'eval_metric': 'logloss'
        }
    }
}

# ==================== METAHEURISTIC PARAMETERS ====================
META_PARAMS = {
    'population_size': 30,
    'max_iterations': 50,
    'num_features_min': 5,
    'num_features_max': None,  # Will be set based on dataset
    'fitness_weight': 0.99,  # Weight for accuracy in fitness
    'feature_weight': 0.01   # Weight for feature reduction in fitness
}

# ==================== CROSS-VALIDATION SETTINGS ====================
CV_SETTINGS = {
    'n_splits': 5,
    'shuffle': True,
    'random_state': 42
}

# ==================== TRAIN-TEST SPLIT ====================
SPLIT_SETTINGS = {
    'test_size': 0.2,
    'random_state': 42,
    'stratify': True
}

# ==================== RANDOM SEED ====================
RANDOM_SEED = 42

# ==================== METRICS TO COMPUTE ====================
METRICS_LIST = [
    'accuracy',
    'precision',
    'recall',
    'f1_score',
    'roc_auc',
    'confusion_matrix',
    'classification_report'
]

# ==================== VISUALIZATION SETTINGS ====================
PLOT_SETTINGS = {
    'figsize': (12, 8),
    'dpi': 300,
    'style': 'seaborn-v0_8-darkgrid',
    'save_format': 'png'
}
