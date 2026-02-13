"""
Optimization Pipeline Module
Orchestrates the feature selection and training process.
"""

import numpy as np
import pandas as pd
import time
import logging
from typing import Dict, List, Any
from ..config import METAHEURISTICS, CLASSIFIERS, META_PARAMS
from ..models.classifiers import ClassifierWrapper
from ..models.metrics import MetricsCalculator
from .algorithms import get_algorithm

logger = logging.getLogger(__name__)

class OptimizationPipeline:
    """Pipeline for feature selection optimization."""
    
    def __init__(self, dataset_name: str, X_train: np.ndarray, y_train: np.ndarray, 
                 X_test: np.ndarray, y_test: np.ndarray, feature_names: List[str]):
        self.dataset_name = dataset_name
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.feature_names = feature_names
        self.n_features = X_train.shape[1]
        self.results = {}
        
    def run_experiment(self, algorithm_name: str, classifier_name: str) -> Dict[str, Any]:
        """Run a single optimization experiment."""
        logger.info(f"Running {algorithm_name} + {classifier_name} on {self.dataset_name}")
        start_time = time.time()
        
        # Setup Classifier wrapping
        clf_wrapper = ClassifierWrapper(classifier_name, self.X_train, self.y_train)
        
        # Fitness function expecting binary mask
        def fitness_func(mask):
            return clf_wrapper.evaluate_features(mask)
            
        # Initialize Algorithm
        algorithm = get_algorithm(algorithm_name, self.n_features)
        
        # Run Optimization
        best_solution, best_fitness, convergence = algorithm.optimize(fitness_func)
        duration = time.time() - start_time
        
        # Select features
        if np.sum(best_solution) == 0:
            logger.warning("No features selected! Forcing all features.")
            best_solution = np.ones(self.n_features, dtype=int)
            
        selected_indices = np.where(best_solution == 1)[0]
        selected_names = [self.feature_names[i] for i in selected_indices]
        
        logger.info(f"Selected {len(selected_indices)}/{self.n_features} features.")
        
        # Final Training
        clf_wrapper.train(best_solution)
        y_pred = clf_wrapper.predict(self.X_test, best_solution)
        y_prob = clf_wrapper.predict_proba(self.X_test, best_solution)
        
        # Metrics
        calc = MetricsCalculator(self.y_test, y_pred, y_prob)
        metrics = calc.calculate_all_metrics()
        
        result = {
            'dataset': self.dataset_name,
            'algorithm': algorithm_name,
            'classifier': classifier_name,
            'n_features': len(selected_indices),
            'selected_features': selected_names,
            'best_fitness': best_fitness,
            'optimization_time': duration,
            'convergence': convergence,
            **metrics
        }
        
        self.results[f"{algorithm_name}_{classifier_name}"] = result
        return result

    def get_summary(self) -> pd.DataFrame:
        """Return summary of all experiments."""
        if not self.results:
            return pd.DataFrame()
        return pd.DataFrame(self.results.values())
