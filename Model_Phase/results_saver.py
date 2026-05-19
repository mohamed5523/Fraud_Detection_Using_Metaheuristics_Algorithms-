"""
Results Saver Module
Save all results, models, and metrics to files
"""

import pandas as pd
import numpy as np
import pickle
import json
import os
import config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResultsSaver:
    """Save experiment results to various formats"""
    
    def __init__(self):
        """Initialize results saver"""
        self.results_dir = config.RESULTS_DIR
        self.models_dir = config.MODELS_DIR
        self.metrics_dir = config.METRICS_DIR
        
        # Ensure directories exist
        for directory in [self.results_dir, self.models_dir, self.metrics_dir]:
            os.makedirs(directory, exist_ok=True)
    
    def save_model(self, model, filename):
        """
        Save trained model to file
        
        Args:
            model: Trained model object
            filename: Output filename
        """
        filepath = os.path.join(self.models_dir, filename)
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"Model saved: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return None
    
    def save_results_dict(self, results_dict, filename):
        """
        Save results dictionary to pickle file
        
        Args:
            results_dict: Dictionary with results
            filename: Output filename
        """
        filepath = os.path.join(self.results_dir, filename)
        
        try:
            # Remove non-serializable objects
            clean_dict = {}
            for key, value in results_dict.items():
                if isinstance(value, dict):
                    clean_dict[key] = {k: v for k, v in value.items() 
                                      if k != 'model' and not isinstance(v, type)}
                else:
                    clean_dict[key] = value
            
            with open(filepath, 'wb') as f:
                pickle.dump(clean_dict, f)
            logger.info(f"Results dictionary saved: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Error saving results dictionary: {str(e)}")
            return None
    
    def save_dataframe(self, df, filename, format='csv'):
        """
        Save DataFrame to file
        
        Args:
            df: Pandas DataFrame
            filename: Output filename
            format: File format ('csv', 'excel', 'json')
        """
        filepath = os.path.join(self.metrics_dir, filename)
        
        try:
            if format == 'csv':
                df.to_csv(filepath, index=False)
            elif format == 'excel':
                df.to_excel(filepath, index=False, engine='openpyxl')
            elif format == 'json':
                df.to_json(filepath, orient='records', indent=2)
            else:
                logger.warning(f"Unknown format: {format}, using CSV")
                filepath = filepath.replace(f'.{format}', '.csv')
                df.to_csv(filepath, index=False)
            
            logger.info(f"DataFrame saved: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Error saving DataFrame: {str(e)}")
            return None
    
    def save_experiment_results(self, results, prefix):
        """
        Save complete experiment results
        
        Args:
            results: Results dictionary from experiment
            prefix: Prefix for output files
        """
        saved_files = {}
        
        # Save model
        if 'model' in results and results['model'] is not None:
            model_file = f"{prefix}_model.pkl"
            saved_files['model'] = self.save_model(results['model'], model_file)
        
        # Save selected features
        if 'selected_feature_names' in results:
            features_df = pd.DataFrame({
                'Feature_Index': results['selected_feature_indices'],
                'Feature_Name': results['selected_feature_names']
            })
            features_file = f"{prefix}_selected_features.csv"
            saved_files['features'] = self.save_dataframe(features_df, features_file)
        
        # Save metrics summary
        metrics_data = {
            'Dataset': results.get('dataset', 'Unknown'),
            'Algorithm': results.get('algorithm', 'Unknown'),
            'Classifier': results.get('classifier', 'Unknown'),
            'Features_Selected': results.get('n_features_selected', 0),
            'Features_Total': results.get('n_features_total', 0),
            'Accuracy': results.get('accuracy', 0),
            'Precision': results.get('precision', 0),
            'Recall': results.get('recall', 0),
            'F1_Score': results.get('f1_score', 0),
            'ROC_AUC': results.get('roc_auc', None),
            'MCC': results.get('mcc', 0),
            'Optimization_Time': results.get('optimization_time', 0)
        }
        metrics_df = pd.DataFrame([metrics_data])
        metrics_file = f"{prefix}_metrics_summary.csv"
        saved_files['metrics_summary'] = self.save_dataframe(metrics_df, metrics_file)
        
        # Save convergence curve
        if 'convergence_curve' in results:
            conv_df = pd.DataFrame({
                'Iteration': range(len(results['convergence_curve'])),
                'Fitness': results['convergence_curve']
            })
            conv_file = f"{prefix}_convergence.csv"
            saved_files['convergence'] = self.save_dataframe(conv_df, conv_file)
        
        # Save confusion matrix
        if 'confusion_matrix' in results:
            cm = results['confusion_matrix']
            cm_df = pd.DataFrame(cm, 
                                columns=[f'Pred_{i}' for i in range(cm.shape[1])],
                                index=[f'True_{i}' for i in range(cm.shape[0])])
            cm_file = f"{prefix}_confusion_matrix.csv"
            saved_files['confusion_matrix'] = self.save_dataframe(cm_df, cm_file)
        
        # Save classification report
        if 'classification_report' in results:
            report_df = pd.DataFrame(results['classification_report']).transpose()
            report_file = f"{prefix}_classification_report.csv"
            saved_files['classification_report'] = self.save_dataframe(report_df, report_file)
        
        return saved_files
    
    def save_summary_table(self, summary_df, filename):
        """
        Save summary table with all experiments
        
        Args:
            summary_df: Summary DataFrame
            filename: Output filename
        """
        # Save as CSV
        csv_file = self.save_dataframe(summary_df, filename)
        
        # Also save as Excel for better formatting
        excel_filename = filename.replace('.csv', '.xlsx')
        excel_file = self.save_dataframe(summary_df, excel_filename, format='excel')
        
        return {'csv': csv_file, 'excel': excel_file}
    
    def save_best_results(self, best_results_dict, filename='best_results.csv'):
        """
        Save best results for each dataset/configuration
        
        Args:
            best_results_dict: Dictionary with best results
            filename: Output filename
        """
        best_data = []
        
        for key, result in best_results_dict.items():
            row = {
                'Configuration': key,
                'Dataset': result.get('dataset', 'Unknown'),
                'Algorithm': result.get('algorithm', 'Unknown'),
                'Classifier': result.get('classifier', 'Unknown'),
                'Features_Selected': result.get('n_features_selected', 0),
                'Accuracy': result.get('accuracy', 0),
                'Precision': result.get('precision', 0),
                'Recall': result.get('recall', 0),
                'F1_Score': result.get('f1_score', 0),
                'ROC_AUC': result.get('roc_auc', None),
                'MCC': result.get('mcc', 0)
            }
            best_data.append(row)
        
        best_df = pd.DataFrame(best_data)
        return self.save_dataframe(best_df, filename)
    
    def save_comparison_analysis(self, comparison_df, prefix='comparison'):
        """
        Save comparison analysis with multiple formats
        
        Args:
            comparison_df: Comparison DataFrame
            prefix: Prefix for output files
        """
        saved_files = {}
        
        # Overall comparison
        overall_file = f"{prefix}_overall.csv"
        saved_files['overall_csv'] = self.save_dataframe(comparison_df, overall_file)
        
        # Save by algorithm
        if 'Algorithm' in comparison_df.columns:
            for algo in comparison_df['Algorithm'].unique():
                algo_df = comparison_df[comparison_df['Algorithm'] == algo]
                algo_file = f"{prefix}_algorithm_{algo}.csv"
                saved_files[f'algo_{algo}'] = self.save_dataframe(algo_df, algo_file)
        
        # Save by classifier
        if 'Classifier' in comparison_df.columns:
            for clf in comparison_df['Classifier'].unique():
                clf_df = comparison_df[comparison_df['Classifier'] == clf]
                clf_file = f"{prefix}_classifier_{clf}.csv"
                saved_files[f'clf_{clf}'] = self.save_dataframe(clf_df, clf_file)
        
        # Save by dataset
        if 'Dataset' in comparison_df.columns:
            for dataset in comparison_df['Dataset'].unique():
                dataset_df = comparison_df[comparison_df['Dataset'] == dataset]
                dataset_file = f"{prefix}_dataset_{dataset}.csv"
                saved_files[f'dataset_{dataset}'] = self.save_dataframe(dataset_df, dataset_file)
        
        return saved_files
    
    def create_summary_report(self, all_results, output_file='summary_report.txt'):
        """
        Create text summary report
        
        Args:
            all_results: Dictionary with all results
            output_file: Output filename
        """
        filepath = os.path.join(self.results_dir, output_file)
        
        try:
            with open(filepath, 'w') as f:
                f.write("="*80 + "\n")
                f.write("FEATURE SELECTION OPTIMIZATION - SUMMARY REPORT\n")
                f.write("="*80 + "\n\n")
                
                # Overall statistics
                total_experiments = len(all_results)
                f.write(f"Total Experiments: {total_experiments}\n\n")
                
                # Best results
                f.write("-"*80 + "\n")
                f.write("BEST RESULTS PER METRIC\n")
                f.write("-"*80 + "\n\n")
                
                metrics = ['accuracy', 'f1_score', 'roc_auc', 'mcc']
                for metric in metrics:
                    best_key = None
                    best_value = -np.inf
                    
                    for key, result in all_results.items():
                        if metric in result and result[metric] is not None:
                            if result[metric] > best_value:
                                best_value = result[metric]
                                best_key = key
                    
                    if best_key:
                        result = all_results[best_key]
                        f.write(f"\nBest {metric.upper()}: {best_value:.4f}\n")
                        f.write(f"  Configuration: {best_key}\n")
                        f.write(f"  Dataset: {result.get('dataset', 'Unknown')}\n")
                        f.write(f"  Algorithm: {result.get('algorithm', 'Unknown')}\n")
                        f.write(f"  Classifier: {result.get('classifier', 'Unknown')}\n")
                        f.write(f"  Features: {result.get('n_features_selected', 0)}/{result.get('n_features_total', 0)}\n")
                
                f.write("\n" + "="*80 + "\n")
            
            logger.info(f"Summary report saved: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Error creating summary report: {str(e)}")
            return None
