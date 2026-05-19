"""
Metrics Module
Calculate comprehensive evaluation metrics
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    matthews_corrcoef, cohen_kappa_score, balanced_accuracy_score,
    roc_curve, precision_recall_curve, average_precision_score
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Calculate and store comprehensive metrics"""
    
    def __init__(self, y_true, y_pred, y_pred_proba=None):
        """
        Initialize metrics calculator
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Prediction probabilities (optional)
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_pred_proba = y_pred_proba
        self.metrics = {}
        
    def calculate_all_metrics(self):
        """Calculate all available metrics"""
        # Basic metrics
        self.metrics['accuracy'] = accuracy_score(self.y_true, self.y_pred)
        self.metrics['balanced_accuracy'] = balanced_accuracy_score(self.y_true, self.y_pred)
        
        # Precision, Recall, F1
        self.metrics['precision'] = precision_score(
            self.y_true, self.y_pred, average='binary', zero_division=0
        )
        self.metrics['recall'] = recall_score(
            self.y_true, self.y_pred, average='binary', zero_division=0
        )
        self.metrics['f1_score'] = f1_score(
            self.y_true, self.y_pred, average='binary', zero_division=0
        )
        
        # Macro and Weighted metrics
        self.metrics['precision_macro'] = precision_score(
            self.y_true, self.y_pred, average='macro', zero_division=0
        )
        self.metrics['recall_macro'] = recall_score(
            self.y_true, self.y_pred, average='macro', zero_division=0
        )
        self.metrics['f1_macro'] = f1_score(
            self.y_true, self.y_pred, average='macro', zero_division=0
        )
        
        self.metrics['precision_weighted'] = precision_score(
            self.y_true, self.y_pred, average='weighted', zero_division=0
        )
        self.metrics['recall_weighted'] = recall_score(
            self.y_true, self.y_pred, average='weighted', zero_division=0
        )
        self.metrics['f1_weighted'] = f1_score(
            self.y_true, self.y_pred, average='weighted', zero_division=0
        )
        
        # Confusion Matrix
        cm = confusion_matrix(self.y_true, self.y_pred)
        self.metrics['confusion_matrix'] = cm
        
        # Specificity and Sensitivity
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            self.metrics['true_positives'] = tp
            self.metrics['true_negatives'] = tn
            self.metrics['false_positives'] = fp
            self.metrics['false_negatives'] = fn
            
            # Sensitivity (TPR) and Specificity (TNR)
            self.metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
            self.metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            # False Positive Rate and False Negative Rate
            self.metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0
            self.metrics['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        # Additional metrics
        self.metrics['mcc'] = matthews_corrcoef(self.y_true, self.y_pred)
        self.metrics['kappa'] = cohen_kappa_score(self.y_true, self.y_pred)
        
        # ROC-AUC and PR-AUC (if probabilities available)
        if self.y_pred_proba is not None:
            try:
                # For binary classification
                if len(np.unique(self.y_true)) == 2:
                    if len(self.y_pred_proba.shape) == 2:
                        proba_positive = self.y_pred_proba[:, 1]
                    else:
                        proba_positive = self.y_pred_proba
                    
                    self.metrics['roc_auc'] = roc_auc_score(self.y_true, proba_positive)
                    self.metrics['pr_auc'] = average_precision_score(self.y_true, proba_positive)
                    
                    # Get ROC and PR curves
                    fpr, tpr, _ = roc_curve(self.y_true, proba_positive)
                    self.metrics['roc_curve'] = {'fpr': fpr, 'tpr': tpr}
                    
                    precision, recall, _ = precision_recall_curve(self.y_true, proba_positive)
                    self.metrics['pr_curve'] = {'precision': precision, 'recall': recall}
            except Exception as e:
                logger.warning(f"Could not calculate ROC-AUC: {str(e)}")
                self.metrics['roc_auc'] = None
                self.metrics['pr_auc'] = None
        
        # Classification report
        report = classification_report(self.y_true, self.y_pred, output_dict=True, zero_division=0)
        self.metrics['classification_report'] = report
        
        return self.metrics
    
    def get_summary_dict(self):
        """Get summary metrics as dictionary"""
        if not self.metrics:
            self.calculate_all_metrics()
        
        summary = {
            'accuracy': self.metrics['accuracy'],
            'balanced_accuracy': self.metrics['balanced_accuracy'],
            'precision': self.metrics['precision'],
            'recall': self.metrics['recall'],
            'f1_score': self.metrics['f1_score'],
            'mcc': self.metrics['mcc'],
            'kappa': self.metrics['kappa']
        }
        
        if 'sensitivity' in self.metrics:
            summary['sensitivity'] = self.metrics['sensitivity']
            summary['specificity'] = self.metrics['specificity']
        
        if 'roc_auc' in self.metrics and self.metrics['roc_auc'] is not None:
            summary['roc_auc'] = self.metrics['roc_auc']
            summary['pr_auc'] = self.metrics['pr_auc']
        
        return summary
    
    def to_dataframe(self):
        """Convert summary metrics to DataFrame"""
        summary = self.get_summary_dict()
        return pd.DataFrame([summary])
    
    def print_summary(self):
        """Print metrics summary"""
        if not self.metrics:
            self.calculate_all_metrics()
        
        print("\n" + "="*60)
        print("CLASSIFICATION METRICS SUMMARY")
        print("="*60)
        
        print(f"\nAccuracy: {self.metrics['accuracy']:.4f}")
        print(f"Balanced Accuracy: {self.metrics['balanced_accuracy']:.4f}")
        print(f"Precision: {self.metrics['precision']:.4f}")
        print(f"Recall: {self.metrics['recall']:.4f}")
        print(f"F1-Score: {self.metrics['f1_score']:.4f}")
        print(f"MCC: {self.metrics['mcc']:.4f}")
        print(f"Cohen's Kappa: {self.metrics['kappa']:.4f}")
        
        if 'sensitivity' in self.metrics:
            print(f"\nSensitivity (TPR): {self.metrics['sensitivity']:.4f}")
            print(f"Specificity (TNR): {self.metrics['specificity']:.4f}")
        
        if 'roc_auc' in self.metrics and self.metrics['roc_auc'] is not None:
            print(f"\nROC-AUC: {self.metrics['roc_auc']:.4f}")
            print(f"PR-AUC: {self.metrics['pr_auc']:.4f}")
        
        print("\nConfusion Matrix:")
        print(self.metrics['confusion_matrix'])
        
        print("\n" + "="*60)


def compare_results(results_dict):
    """
    Compare results from multiple experiments
    
    Args:
        results_dict: Dictionary with structure {exp_name: metrics_dict}
        
    Returns:
        Comparison DataFrame
    """
    comparison_data = []
    
    for exp_name, metrics in results_dict.items():
        row = {'Experiment': exp_name}
        row.update(metrics)
        comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    return df


def get_best_result(results_df, metric='f1_score'):
    """
    Get best result based on metric
    
    Args:
        results_df: DataFrame with results
        metric: Metric to use for comparison
        
    Returns:
        Best result row
    """
    if metric not in results_df.columns:
        logger.warning(f"Metric {metric} not found, using accuracy")
        metric = 'accuracy'
    
    best_idx = results_df[metric].idxmax()
    return results_df.loc[best_idx]
