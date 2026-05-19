"""
Visualization Module
Create comprehensive visualizations for results
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
import config
import os

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class Visualizer:
    """Create visualizations for model results"""
    
    def __init__(self, save_dir=None):
        """
        Initialize visualizer
        
        Args:
            save_dir: Directory to save plots
        """
        self.save_dir = save_dir or config.PLOTS_DIR
        os.makedirs(self.save_dir, exist_ok=True)
    
    def _save_plot(self, filename):
        """Save plot to file"""
        filepath = os.path.join(self.save_dir, filename)
        plt.savefig(filepath, dpi=config.PLOT_SETTINGS['dpi'], bbox_inches='tight')
        plt.close()
        return filepath
    
    def plot_convergence_curve(self, convergence_data, title, filename):
        """
        Plot convergence curve for optimization
        
        Args:
            convergence_data: Array of fitness values over iterations
            title: Plot title
            filename: Output filename
        """
        plt.figure(figsize=(10, 6))
        plt.plot(convergence_data, linewidth=2)
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Fitness Value', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        return self._save_plot(filename)
    
    def plot_confusion_matrix(self, cm, labels, title, filename):
        """
        Plot confusion matrix
        
        Args:
            cm: Confusion matrix
            labels: Class labels
            title: Plot title
            filename: Output filename
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels,
                   cbar_kws={'label': 'Count'})
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        return self._save_plot(filename)
    
    def plot_roc_curve(self, fpr, tpr, roc_auc, title, filename):
        """
        Plot ROC curve
        
        Args:
            fpr: False positive rates
            tpr: True positive rates
            roc_auc: ROC-AUC score
            title: Plot title
            filename: Output filename
        """
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, alpha=0.3)
        return self._save_plot(filename)
    
    def plot_precision_recall_curve(self, precision, recall, pr_auc, title, filename):
        """
        Plot Precision-Recall curve
        
        Args:
            precision: Precision values
            recall: Recall values
            pr_auc: PR-AUC score
            title: Plot title
            filename: Output filename
        """
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, linewidth=2, label=f'PR Curve (AUC = {pr_auc:.4f})')
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        return self._save_plot(filename)
    
    def plot_feature_importance(self, feature_mask, feature_names, title, filename):
        """
        Plot selected features
        
        Args:
            feature_mask: Binary array of selected features
            feature_names: Names of all features
            title: Plot title
            filename: Output filename
        """
        selected_indices = np.where(feature_mask == 1)[0]
        
        if len(selected_indices) == 0:
            return None
        
        selected_names = [feature_names[i] for i in selected_indices]
        
        plt.figure(figsize=(12, max(6, len(selected_names) * 0.3)))
        plt.barh(range(len(selected_names)), [1] * len(selected_names), color='steelblue')
        plt.yticks(range(len(selected_names)), selected_names)
        plt.xlabel('Selected', fontsize=12)
        plt.title(f'{title}\n({len(selected_names)}/{len(feature_names)} features selected)', 
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        return self._save_plot(filename)
    
    def plot_metrics_comparison(self, results_df, metric_cols, title, filename):
        """
        Plot comparison of metrics across experiments
        
        Args:
            results_df: DataFrame with results
            metric_cols: List of metric columns to plot
            title: Plot title
            filename: Output filename
        """
        # Filter valid metrics
        metric_cols = [col for col in metric_cols if col in results_df.columns]
        
        if not metric_cols:
            return None
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        x = np.arange(len(results_df))
        width = 0.8 / len(metric_cols)
        
        for i, metric in enumerate(metric_cols):
            offset = (i - len(metric_cols)/2) * width + width/2
            ax.bar(x + offset, results_df[metric], width, label=metric)
        
        ax.set_xlabel('Experiment', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(results_df['Experiment'], rotation=45, ha='right')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        return self._save_plot(filename)
    
    def plot_algorithm_comparison(self, comparison_df, title, filename):
        """
        Plot heatmap comparing algorithms across metrics
        
        Args:
            comparison_df: DataFrame with algorithms as rows and metrics as columns
            title: Plot title
            filename: Output filename
        """
        plt.figure(figsize=(14, 10))
        
        # Select numeric columns only
        numeric_cols = comparison_df.select_dtypes(include=[np.number]).columns
        data = comparison_df[numeric_cols]
        
        sns.heatmap(data, annot=True, fmt='.4f', cmap='YlGnBu', 
                   cbar_kws={'label': 'Score'}, linewidths=0.5)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Metrics', fontsize=12)
        plt.ylabel('Algorithms', fontsize=12)
        plt.tight_layout()
        return self._save_plot(filename)
    
    def plot_selected_features_count(self, results_dict, title, filename):
        """
        Plot number of selected features for each algorithm
        
        Args:
            results_dict: Dictionary with algorithm names and feature counts
            title: Plot title
            filename: Output filename
        """
        algorithms = list(results_dict.keys())
        counts = list(results_dict.values())
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(algorithms, counts, color='teal', alpha=0.7, edgecolor='black')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=10)
        
        plt.xlabel('Algorithm', fontsize=12)
        plt.ylabel('Number of Selected Features', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        return self._save_plot(filename)
    
    def plot_convergence_comparison(self, convergence_dict, title, filename):
        """
        Plot convergence curves for multiple algorithms
        
        Args:
            convergence_dict: Dictionary with algorithm names and convergence data
            title: Plot title
            filename: Output filename
        """
        plt.figure(figsize=(12, 7))
        
        for algo_name, curve_data in convergence_dict.items():
            plt.plot(curve_data, linewidth=2, label=algo_name, alpha=0.8)
        
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Fitness Value', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=9, ncol=2)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return self._save_plot(filename)
    
    def plot_box_plot_comparison(self, data_dict, metric_name, title, filename):
        """
        Create box plot for comparing distributions
        
        Args:
            data_dict: Dictionary with labels and data arrays
            metric_name: Name of metric being compared
            title: Plot title
            filename: Output filename
        """
        data = []
        labels = []
        
        for label, values in data_dict.items():
            data.append(values)
            labels.append(label)
        
        plt.figure(figsize=(12, 6))
        bp = plt.boxplot(data, labels=labels, patch_artist=True,
                        boxprops=dict(facecolor='lightblue', alpha=0.7),
                        medianprops=dict(color='red', linewidth=2))
        
        plt.xlabel('Method', fontsize=12)
        plt.ylabel(metric_name, fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        return self._save_plot(filename)
    
    def create_results_dashboard(self, results_data, dataset_name, algorithm_name, 
                                 classifier_name, output_prefix):
        """
        Create comprehensive dashboard with multiple plots
        
        Args:
            results_data: Dictionary containing all results
            dataset_name: Name of dataset
            algorithm_name: Name of algorithm
            classifier_name: Name of classifier
            output_prefix: Prefix for output files
        """
        saved_plots = {}
        
        # Convergence curve
        if 'convergence_curve' in results_data:
            title = f'Convergence: {algorithm_name} - {classifier_name} - {dataset_name}'
            filename = f'{output_prefix}_convergence.png'
            saved_plots['convergence'] = self.plot_convergence_curve(
                results_data['convergence_curve'], title, filename
            )
        
        # Confusion matrix
        if 'confusion_matrix' in results_data:
            title = f'Confusion Matrix: {algorithm_name} - {classifier_name} - {dataset_name}'
            filename = f'{output_prefix}_confusion_matrix.png'
            saved_plots['confusion_matrix'] = self.plot_confusion_matrix(
                results_data['confusion_matrix'], ['Class 0', 'Class 1'], title, filename
            )
        
        # ROC curve
        if 'roc_curve' in results_data and results_data['roc_curve'] is not None:
            title = f'ROC Curve: {algorithm_name} - {classifier_name} - {dataset_name}'
            filename = f'{output_prefix}_roc_curve.png'
            saved_plots['roc'] = self.plot_roc_curve(
                results_data['roc_curve']['fpr'],
                results_data['roc_curve']['tpr'],
                results_data.get('roc_auc', 0),
                title, filename
            )
        
        # Precision-Recall curve
        if 'pr_curve' in results_data and results_data['pr_curve'] is not None:
            title = f'PR Curve: {algorithm_name} - {classifier_name} - {dataset_name}'
            filename = f'{output_prefix}_pr_curve.png'
            saved_plots['pr'] = self.plot_precision_recall_curve(
                results_data['pr_curve']['precision'],
                results_data['pr_curve']['recall'],
                results_data.get('pr_auc', 0),
                title, filename
            )
        
        # Feature importance
        if 'selected_features' in results_data and 'feature_names' in results_data:
            title = f'Selected Features: {algorithm_name} - {classifier_name} - {dataset_name}'
            filename = f'{output_prefix}_features.png'
            saved_plots['features'] = self.plot_feature_importance(
                results_data['selected_features'],
                results_data['feature_names'],
                title, filename
            )
        
        return saved_plots
