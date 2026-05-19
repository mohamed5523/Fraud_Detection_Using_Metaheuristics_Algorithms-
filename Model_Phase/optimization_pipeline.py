"""
Optimization Pipeline
Orchestrates feature selection using metaheuristics
"""

import numpy as np
import pandas as pd
import time
import logging
from metaheuristics_algorithms import get_algorithm
from classifiers import ClassifierWrapper, get_fitness_function
from metrics import MetricsCalculator
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OptimizationPipeline:
    """Pipeline for feature selection optimization"""
    
    def __init__(self, dataset_name, data_dict):
        """
        Initialize optimization pipeline
        
        Args:
            dataset_name: Name of dataset
            data_dict: Dictionary containing data arrays
        """
        self.dataset_name = dataset_name
        self.X_train = data_dict['X_train']
        self.X_test = data_dict['X_test']
        self.y_train = data_dict['y_train']
        self.y_test = data_dict['y_test']
        self.feature_names = data_dict['feature_names']
        self.n_features = data_dict['n_features']
        
        self.results = {}
        
    def run_single_experiment(self, algorithm_name, classifier_name):
        """
        Run single optimization experiment
        
        Args:
            algorithm_name: Name of metaheuristic algorithm
            classifier_name: Name of classifier
            
        Returns:
            Dictionary with experiment results
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"Running: {algorithm_name} - {classifier_name} - {self.dataset_name}")
        logger.info(f"{'='*70}")
        
        start_time = time.time()
        
        # Initialize classifier
        classifier_wrapper = ClassifierWrapper(
            classifier_name, self.X_train, self.y_train
        )
        
        # Get fitness function
        fitness_func = get_fitness_function(classifier_wrapper)
        
        # Initialize metaheuristic algorithm
        algorithm = get_algorithm(
            algorithm_name,
            self.n_features,
            config.META_PARAMS['population_size'],
            config.META_PARAMS['max_iterations']
        )
        
        # Run optimization
        logger.info("Starting optimization...")
        best_solution, best_fitness, convergence_curve = algorithm.optimize(fitness_func)
        
        optimization_time = time.time() - start_time
        logger.info(f"Optimization completed in {optimization_time:.2f} seconds")
        
        # Get selected features
        selected_features = best_solution
        n_selected = np.sum(selected_features)
        selected_indices = np.where(selected_features == 1)[0]
        
        logger.info(f"Selected {n_selected}/{self.n_features} features")
        
        # Train model with selected features
        logger.info("Training final model...")
        model = classifier_wrapper.train(selected_features)
        
        # Make predictions
        y_pred = classifier_wrapper.predict(self.X_test, selected_features)
        y_pred_proba = classifier_wrapper.predict_proba(self.X_test, selected_features)
        
        # Calculate metrics
        logger.info("Calculating metrics...")
        metrics_calc = MetricsCalculator(self.y_test, y_pred, y_pred_proba)
        metrics = metrics_calc.calculate_all_metrics()
        
        # Prepare results
        results = {
            'dataset': self.dataset_name,
            'algorithm': algorithm_name,
            'classifier': classifier_name,
            'n_features_total': self.n_features,
            'n_features_selected': n_selected,
            'feature_selection_ratio': n_selected / self.n_features,
            'selected_features': selected_features,
            'selected_feature_indices': selected_indices,
            'selected_feature_names': [self.feature_names[i] for i in selected_indices],
            'best_fitness': best_fitness,
            'convergence_curve': convergence_curve,
            'optimization_time': optimization_time,
            'model': model,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'feature_names': self.feature_names
        }
        
        # Add all metrics
        results.update(metrics)
        
        # Print summary
        metrics_calc.print_summary()
        
        logger.info(f"\nExperiment completed successfully!")
        logger.info(f"Best Fitness: {best_fitness:.4f}")
        logger.info(f"Test Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Test F1-Score: {metrics['f1_score']:.4f}")
        
        return results
    
    def run_all_combinations(self, algorithms=None, classifiers=None):
        """
        Run all combinations of algorithms and classifiers
        
        Args:
            algorithms: List of algorithm names (None for all)
            classifiers: List of classifier names (None for all)
            
        Returns:
            Dictionary with all results
        """
        if algorithms is None:
            algorithms = config.METAHEURISTICS
        
        if classifiers is None:
            classifiers = list(config.CLASSIFIERS.keys())
        
        total_experiments = len(algorithms) * len(classifiers)
        experiment_count = 0
        
        logger.info(f"\n{'#'*70}")
        logger.info(f"Starting {total_experiments} experiments for {self.dataset_name}")
        logger.info(f"{'#'*70}\n")
        
        for classifier_name in classifiers:
            for algorithm_name in algorithms:
                experiment_count += 1
                logger.info(f"\nExperiment {experiment_count}/{total_experiments}")
                
                try:
                    results = self.run_single_experiment(algorithm_name, classifier_name)
                    
                    # Store results
                    key = f"{algorithm_name}_{classifier_name}"
                    self.results[key] = results
                    
                except Exception as e:
                    logger.error(f"Error in experiment: {str(e)}")
                    import traceback
                    traceback.print_exc()
        
        logger.info(f"\n{'#'*70}")
        logger.info(f"All experiments completed for {self.dataset_name}")
        logger.info(f"{'#'*70}\n")
        
        return self.results
    
    def get_results_summary(self):
        """Get summary DataFrame of all results"""
        if not self.results:
            logger.warning("No results available")
            return None
        
        summary_data = []
        
        for key, result in self.results.items():
            row = {
                'Experiment': key,
                'Dataset': result['dataset'],
                'Algorithm': result['algorithm'],
                'Classifier': result['classifier'],
                'Features_Selected': result['n_features_selected'],
                'Features_Total': result['n_features_total'],
                'Selection_Ratio': result['feature_selection_ratio'],
                'Best_Fitness': result['best_fitness'],
                'Accuracy': result['accuracy'],
                'Balanced_Accuracy': result.get('balanced_accuracy', 0),
                'Precision': result['precision'],
                'Recall': result['recall'],
                'F1_Score': result['f1_score'],
                'MCC': result['mcc'],
                'Kappa': result['kappa'],
                'Optimization_Time': result['optimization_time']
            }
            
            # Add optional metrics
            if 'sensitivity' in result:
                row['Sensitivity'] = result['sensitivity']
                row['Specificity'] = result['specificity']
            
            if 'roc_auc' in result and result['roc_auc'] is not None:
                row['ROC_AUC'] = result['roc_auc']
                row['PR_AUC'] = result['pr_auc']
            
            summary_data.append(row)
        
        df = pd.DataFrame(summary_data)
        return df
    
    def get_best_result(self, metric='f1_score'):
        """
        Get best result based on metric
        
        Args:
            metric: Metric to use for comparison
            
        Returns:
            Best result dictionary
        """
        if not self.results:
            return None
        
        best_key = None
        best_value = -np.inf
        
        for key, result in self.results.items():
            if metric in result and result[metric] > best_value:
                best_value = result[metric]
                best_key = key
        
        return self.results[best_key] if best_key else None


class MultiDatasetPipeline:
    """Pipeline for running experiments across multiple datasets"""
    
    def __init__(self, datasets_dict):
        """
        Initialize multi-dataset pipeline
        
        Args:
            datasets_dict: Dictionary with dataset names and data
        """
        self.datasets_dict = datasets_dict
        self.pipelines = {}
        self.all_results = {}
        
    def run_all_datasets(self, algorithms=None, classifiers=None):
        """
        Run optimization for all datasets
        
        Args:
            algorithms: List of algorithm names
            classifiers: List of classifier names
            
        Returns:
            Dictionary with all results
        """
        for dataset_name, dataset_info in self.datasets_dict.items():
            logger.info(f"\n{'*'*80}")
            logger.info(f"PROCESSING DATASET: {dataset_name}")
            logger.info(f"{'*'*80}\n")
            
            # Create pipeline for dataset
            pipeline = OptimizationPipeline(dataset_name, dataset_info['data'])
            
            # Run all combinations
            results = pipeline.run_all_combinations(algorithms, classifiers)
            
            # Store pipeline and results
            self.pipelines[dataset_name] = pipeline
            self.all_results[dataset_name] = results
        
        return self.all_results
    
    def get_combined_summary(self):
        """Get combined summary for all datasets"""
        all_summaries = []
        
        for dataset_name, pipeline in self.pipelines.items():
            summary = pipeline.get_results_summary()
            if summary is not None:
                all_summaries.append(summary)
        
        if all_summaries:
            combined_df = pd.concat(all_summaries, ignore_index=True)
            return combined_df
        
        return None
    
    def get_best_per_dataset(self, metric='f1_score'):
        """Get best result for each dataset"""
        best_results = {}
        
        for dataset_name, pipeline in self.pipelines.items():
            best_result = pipeline.get_best_result(metric)
            if best_result:
                best_results[dataset_name] = best_result
        
        return best_results
