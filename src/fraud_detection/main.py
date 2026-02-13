"""
Main Entry Point
Command-line interface for the Fraud Detection system.
"""

import argparse
import logging
import sys
import pandas as pd
import warnings
from pathlib import Path

# Add src to path if running directly
sys.path.append(str(Path(__file__).resolve().parent.parent))

from fraud_detection.config import DATASETS, LOG_LEVEL
from fraud_detection.data.loader import DataLoader
from fraud_detection.data.preprocessing import DataPreprocessor
from fraud_detection.data.balancing import DataBalancer
from fraud_detection.optimization.pipeline import OptimizationPipeline
from fraud_detection.visualization.plots import Visualizer

warnings.filterwarnings('ignore')
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_exploration(dataset_name: str):
    """Run data exploration pipeline."""
    logger.info(f"Starting exploration for {dataset_name}...")
    loader = DataLoader()
    df = loader.load_dataset(dataset_name)
    if df is None:
        return
        
    viz = Visualizer()
    target = DATASETS[dataset_name].get('target_column')
    
    if target and target in df.columns:
        viz.plot_class_distribution(df[target], title=f"{dataset_name} Class Distribution")
        
    viz.plot_correlation_matrix(df, title=f"{dataset_name} Correlation Matrix")
    logger.info("Exploration completed.")

def run_training(dataset_name: str, algorithm: str, classifier: str, balance: str = None):
    """Run training and optimization pipeline."""
    logger.info(f"Starting training on {dataset_name} with {algorithm} + {classifier}...")
    
    # 1. Load
    loader = DataLoader()
    df = loader.load_dataset(dataset_name)
    if df is None:
        return
        
    config = DATASETS[dataset_name]
    target_col = config.get('target_column')
    
    # 2. Preprocess
    preprocessor = DataPreprocessor()
    X, y = preprocessor.preprocess(df, config)
    
    if y is None:
        logger.error("Target column not found.")
        return

    # 3. Balance (Optional, but usually done on train set only)
    # Note: splitting should happen BEFORE balancing to avoid leakage, 
    # but for simple pipeline demonstration we might balance first or inside loop.
    # Standard practice: Split -> Balance Train -> Train.
    
    X_train, X_test, y_train, y_test = loader.split_data(pd.concat([X, y], axis=1), target_col)
    
    if balance:
        balancer = DataBalancer()
        X_train, y_train = balancer.balance_dataset(X_train, y_train, method=balance)
        
    # 4. Optimization & Training
    # Convert feature names
    feature_names = X.columns.tolist()
    
    pipeline = OptimizationPipeline(
        dataset_name, 
        X_train.values, y_train.values, 
        X_test.values, y_test.values, 
        feature_names
    )
    
    result = pipeline.run_experiment(algorithm, classifier)
    
    # 5. Visualize Results
    viz = Visualizer()
    viz.plot_confusion_matrix(result['confusion_matrix'], title=f"CM {algorithm}-{classifier}")
    viz.plot_convergence(result['convergence'], title=f"Convergence {algorithm}-{classifier}")
    viz.plot_feature_importance(result['selected_features'], len(feature_names), title=f"Features {algorithm}-{classifier}")
    
    logger.info("Training completed successfully.")

def main():
    parser = argparse.ArgumentParser(description="Fraud Detection System CLI")
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Explore command
    explore_parser = subparsers.add_parser('explore', help='Explore a dataset')
    explore_parser.add_argument('dataset', choices=list(DATASETS.keys()), help='Dataset name')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train and optimize a model')
    train_parser.add_argument('dataset', choices=list(DATASETS.keys()), help='Dataset name')
    train_parser.add_argument('--algorithm', type=str, default='BPSO', help='Metaheuristic algorithm')
    train_parser.add_argument('--classifier', type=str, default='XGBoost', help='Classifier to use')
    train_parser.add_argument('--balance', type=str, choices=['smote', 'rus'], help='Balancing method')
    
    args = parser.parse_args()
    
    if args.command == 'explore':
        run_exploration(args.dataset)
    elif args.command == 'train':
        run_training(args.dataset, args.algorithm, args.classifier, args.balance)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
