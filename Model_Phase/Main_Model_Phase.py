"""
Main Execution Script
Run complete feature selection optimization pipeline
"""

import os
import numpy as np
import pandas as pd
import logging
import time
from datetime import datetime

# Import all modules
import config
from data_loader import load_all_datasets
from optimization_pipeline import OptimizationPipeline, MultiDatasetPipeline
from visualizations import Visualizer
from results_saver import ResultsSaver
from metrics import compare_results, get_best_result

# Setup logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/execution_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def run_single_dataset_experiment(dataset_name, algorithms=None, classifiers=None):
    """
    Run experiment for a single dataset

    Args:
        dataset_name: Name of dataset
        algorithms: List of algorithms (None for all)
        classifiers: List of classifiers (None for all)
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"SINGLE DATASET EXPERIMENT: {dataset_name}")
    logger.info(f"{'='*80}\n")

    # Load datasets
    datasets = load_all_datasets()

    if dataset_name not in datasets:
        logger.error(f"Dataset {dataset_name} not found!")
        return None

    data_obj = datasets[dataset_name]['data']

    # Create pipeline for this dataset
    pipeline = OptimizationPipeline(dataset_name, data_obj)

    # Run experiments (returns a dict of results keyed by experiment id/name)
    results = pipeline.run_all_combinations(algorithms, classifiers)

    # Get summary dataframe (per-pipeline summary)
    summary_df = pipeline.get_results_summary()

    # Save results
    saver = ResultsSaver()
    visualizer = Visualizer()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save summary table if present
    if summary_df is not None:
        saver.save_summary_table(summary_df, f"{dataset_name}_summary_{timestamp}.csv")

    # Save individual experiment results (results is expected to be dict)
    if isinstance(results, dict):
        for key, result in results.items():
            prefix = f"{dataset_name}_{key}_{timestamp}"
            try:
                saver.save_experiment_results(result, prefix)
            except Exception as e:
                logger.warning(f"Failed to save results for {prefix}: {e}")

            # Create per-experiment visualization/dashboard if possible
            try:
                visualizer.create_results_dashboard(
                    result, dataset_name,
                    result.get('algorithm'), result.get('classifier'),
                    prefix
                )
            except Exception as e:
                logger.debug(f"Visualizer skipped for {prefix}: {e}")

    # Get best result from pipeline if supported
    try:
        best_result = pipeline.get_best_result('f1_score')
        if best_result:
            logger.info(f"\nBest Result (F1-Score): {best_result.get('algorithm')} - {best_result.get('classifier')}")
            logger.info(f"F1-Score: {best_result.get('f1_score', float('nan')):.4f}")
            logger.info(f"Accuracy: {best_result.get('accuracy', float('nan')):.4f}")
    except Exception as e:
        logger.debug(f"Could not get best result: {e}")
        best_result = None

    logger.info("\n" + "="*80)
    logger.info("SINGLE DATASET EXPERIMENT COMPLETED")
    logger.info("="*80 + "\n")

    return {
        'pipeline': pipeline,
        'results': results,
        'summary': summary_df,
        'best_result': best_result
    }


def run_all_datasets_experiment(algorithms=None, classifiers=None):
    """
    Run experiments for all datasets

    Args:
        algorithms: List of algorithms (None for all)
        classifiers: List of classifiers (None for all)
    """
    logger.info(f"\n{'#'*80}")
    logger.info(f"MULTI-DATASET EXPERIMENT")
    logger.info(f"{'#'*80}\n")

    start_time = time.time()

    # Load all datasets
    datasets = load_all_datasets()

    if not datasets:
        logger.error("No datasets loaded!")
        return None

    # Create multi-dataset pipeline
    multi_pipeline = MultiDatasetPipeline(datasets)

    # Run all experiments across datasets (returns dict: {dataset_name: {exp_key: result, ...}, ...})
    all_results = multi_pipeline.run_all_datasets(algorithms, classifiers)

    # Get combined summary (pandas DataFrame)
    combined_summary = multi_pipeline.get_combined_summary()

    total_time = time.time() - start_time
    logger.info(f"\nTotal execution time: {total_time/60:.2f} minutes")

    # Save results and visualizations
    saver = ResultsSaver()
    visualizer = Visualizer()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if combined_summary is not None:
        try:
            saver.save_summary_table(combined_summary, f"all_datasets_summary_{timestamp}.csv")
        except Exception as e:
            logger.warning(f"Failed to save combined summary: {e}")

    # Save all individual results and create dashboards
    if isinstance(all_results, dict):
        for ds_name, ds_results in all_results.items():
            if not isinstance(ds_results, dict):
                continue
            for key, result in ds_results.items():
                prefix = f"{ds_name}_{key}_{timestamp}"
                try:
                    saver.save_experiment_results(result, prefix)
                except Exception as e:
                    logger.debug(f"Failed saving {prefix}: {e}")

                try:
                    visualizer.create_results_dashboard(
                        result, ds_name,
                        result.get('algorithm'), result.get('classifier'),
                        prefix
                    )
                except Exception as e:
                    logger.debug(f"Visualizer skipped for {prefix}: {e}")

    # Feature-selection and algorithm/classifier comparisons across datasets (if combined_summary exists)
    if combined_summary is not None:
        try:
            algo_comparison = combined_summary.groupby('Algorithm').agg({
                'Accuracy': 'mean',
                'F1_Score': 'mean',
                'Precision': 'mean',
                'Recall': 'mean'
            }).reset_index()

            visualizer.plot_metrics_comparison(
                algo_comparison,
                ['Accuracy', 'F1_Score', 'Precision', 'Recall'],
                'Algorithm Comparison (Average across all datasets)',
                f"algorithm_comparison_{timestamp}.png"
            )

            clf_comparison = combined_summary.groupby('Classifier').agg({
                'Accuracy': 'mean',
                'F1_Score': 'mean',
                'Precision': 'mean',
                'Recall': 'mean'
            }).reset_index()

            visualizer.plot_metrics_comparison(
                clf_comparison,
                ['Accuracy', 'F1_Score', 'Precision', 'Recall'],
                'Classifier Comparison (Average across all datasets)',
                f"classifier_comparison_{timestamp}.png"
            )
        except Exception as e:
            logger.debug(f"Failed to create aggregated comparisons: {e}")

        # Feature selection counts per algorithm (from all_results)
        try:
            feature_counts = {}
            for ds_name, ds_results in all_results.items():
                for key, result in (ds_results or {}).items():
                    algo_name = result.get('algorithm')
                    n_selected = result.get('n_features_selected')
                    if algo_name is None or n_selected is None:
                        continue
                    feature_counts.setdefault(algo_name, []).append(n_selected)

            avg_feature_counts = {algo: float(np.mean(counts)) for algo, counts in feature_counts.items() if counts}
            visualizer.plot_selected_features_count(
                avg_feature_counts,
                'Average Number of Selected Features by Algorithm',
                f"feature_selection_counts_{timestamp}.png"
            )
        except Exception as e:
            logger.debug(f"Failed to compute/plot feature counts: {e}")

    # Best per dataset
    try:
        best_per_dataset = multi_pipeline.get_best_per_dataset('f1_score')
        saver.save_best_results(best_per_dataset, f"best_per_dataset_{timestamp}.csv")
    except Exception as e:
        logger.debug(f"Could not compute/save best_per_dataset: {e}")
        best_per_dataset = None

    logger.info("\n" + "="*80)
    logger.info("MULTI-DATASET EXPERIMENT COMPLETED")
    logger.info("="*80 + "\n")

    return {
        'multi_pipeline': multi_pipeline,
        'all_results': all_results,
        'combined_summary': combined_summary,
        'best_per_dataset': best_per_dataset
    }


def run_custom_experiment(dataset_names, algorithm_names, classifier_names):
    """
    Run custom experiment with specific configurations

    Args:
        dataset_names: List of dataset names
        algorithm_names: List of algorithm names
        classifier_names: List of classifier names
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"CUSTOM EXPERIMENT")
    logger.info(f"Datasets: {dataset_names}")
    logger.info(f"Algorithms: {algorithm_names}")
    logger.info(f"Classifiers: {classifier_names}")
    logger.info(f"{'='*80}\n")

    # Load datasets
    all_datasets = load_all_datasets()

    # Filter selected datasets
    selected_datasets = {name: all_datasets[name] for name in dataset_names if name in all_datasets}

    if not selected_datasets:
        logger.error("No valid datasets selected!")
        return None

    # Create multi-dataset pipeline
    multi_pipeline = MultiDatasetPipeline(selected_datasets)

    # Run experiments
    all_results = multi_pipeline.run_all_datasets(algorithm_names, classifier_names)

    # Get combined summary
    combined_summary = multi_pipeline.get_combined_summary()

    # Save and visualize (similar to run_all_datasets_experiment)
    saver = ResultsSaver()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if combined_summary is not None:
        try:
            saver.save_summary_table(combined_summary, f"custom_summary_{timestamp}.csv")
        except Exception as e:
            logger.debug(f"Failed to save custom summary: {e}")

    logger.info("CUSTOM EXPERIMENT COMPLETED")
    return {
        'multi_pipeline': multi_pipeline,
        'results': all_results,
        'summary': combined_summary
    }


def analyze_results(combined_summary):
    """
    Perform detailed analysis of results

    Args:
        combined_summary: Combined results DataFrame
    """
    if combined_summary is None:
        logger.warning("No combined summary provided to analyze.")
        return

    logger.info("\n" + "="*80)
    logger.info("RESULTS ANALYSIS")
    logger.info("="*80 + "\n")

    # Best algorithm per dataset
    logger.info("Best Algorithm per Dataset (by F1-Score):")
    for dataset in combined_summary['Dataset'].unique():
        dataset_df = combined_summary[combined_summary['Dataset'] == dataset]
        if dataset_df.empty:
            continue
        best_row = dataset_df.loc[dataset_df['F1_Score'].idxmax()]
        logger.info(f"\n  {dataset}:")
        logger.info(f"    Algorithm: {best_row['Algorithm']}")
        logger.info(f"    Classifier: {best_row['Classifier']}")
        logger.info(f"    F1-Score: {best_row['F1_Score']:.4f}")
        logger.info(f"    Accuracy: {best_row['Accuracy']:.4f}")
        logger.info(f"    Features: {best_row.get('Features_Selected', 'N/A')}/{best_row.get('Features_Total', 'N/A')}")

    # Best classifier overall
    logger.info("\n\nBest Classifier (Average Performance):")
    clf_stats = combined_summary.groupby('Classifier').agg({
        'Accuracy': 'mean',
        'F1_Score': 'mean',
        'Precision': 'mean',
        'Recall': 'mean'
    })
    logger.info(f"\n{clf_stats.to_string()}")

    # Best algorithm overall
    logger.info("\n\nBest Algorithm (Average Performance):")
    algo_stats = combined_summary.groupby('Algorithm').agg({
        'Accuracy': 'mean',
        'F1_Score': 'mean',
        'Precision': 'mean',
        'Recall': 'mean'
    })
    logger.info(f"\n{algo_stats.to_string()}")

    # Feature selection efficiency
    logger.info("\n\nFeature Selection Efficiency:")
    efficiency = combined_summary.groupby('Algorithm').agg({
        'Selection_Ratio': 'mean',
        'Features_Selected': 'mean',
        'F1_Score': 'mean'
    }).sort_values('F1_Score', ascending=False)
    logger.info(f"\n{efficiency.to_string()}")

    logger.info("\n" + "="*80)


if __name__ == "__main__":
    # Create logs directory (already created above but keep for clarity)
    os.makedirs('logs', exist_ok=True)

    logger.info("\n" + "#"*80)
    logger.info("FEATURE SELECTION OPTIMIZATION USING METAHEURISTICS")
    logger.info("#"*80 + "\n")

    # Print configuration
    try:
        logger.info("Configuration:")
        logger.info(f"  Datasets: {list(config.DATASETS.keys())}")
        logger.info(f"  Algorithms: {config.METAHEURISTICS}")
        logger.info(f"  Classifiers: {list(config.CLASSIFIERS.keys())}")
        logger.info(f"  Population Size: {config.META_PARAMS['population_size']}")
        logger.info(f"  Max Iterations: {config.META_PARAMS['max_iterations']}")
        logger.info(f"  CV Folds: {config.CV_SETTINGS['n_splits']}\n")
    except Exception as e:
        logger.debug(f"Could not print full config: {e}")

    # Choose experiment mode
    MODE = "all"  # Options: "single", "all", "custom"

    try:
        if MODE == "single":
            result = run_single_dataset_experiment(
                dataset_name='dataset1',
                algorithms=['BPSO', 'BHHO', 'BWOA'],  # Subset for testing
                classifiers=['KNN', 'XGBoost']  # Subset for testing
            )
            # Optionally analyze single-summary
            if result and result.get('summary') is not None:
                analyze_results(result['summary'])

        elif MODE == "all":
            result = run_all_datasets_experiment()
            # Analyze results if available
            if result and result.get('combined_summary') is not None:
                analyze_results(result['combined_summary'])

        elif MODE == "custom":
            result = run_custom_experiment(
                dataset_names=['dataset1', 'dataset2'],
                algorithm_names=['BPSO', 'BHHO', 'BWOA', 'BBA'],
                classifier_names=['KNN', 'DecisionTree', 'XGBoost']
            )
            if result and result.get('summary') is not None:
                analyze_results(result['summary'])

        else:
            logger.error(f"Unknown MODE: {MODE}")

    except Exception as e:
        logger.exception(f"Execution failed: {e}")

    logger.info("\n" + "#"*80)
    logger.info("EXECUTION COMPLETED")
    logger.info("#"*80 + "\n")
