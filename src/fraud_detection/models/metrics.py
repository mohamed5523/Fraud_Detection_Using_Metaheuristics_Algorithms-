"""
Metrics Module
Calculate comprehensive evaluation metrics for fraud detection.
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
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class MetricsCalculator:
    """Calculate and store comprehensive metrics."""
    
    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: Optional[np.ndarray] = None):
        """
        Initialize metrics calculator.
        
        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            y_pred_proba: Prediction probabilities (optional).
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_pred_proba = y_pred_proba
        self.metrics = {}
        
    def calculate_all_metrics(self) -> Dict[str, Any]:
        """Calculate all available metrics."""
        # Basic metrics
        self.metrics['accuracy'] = accuracy_score(self.y_true, self.y_pred)
        self.metrics['balanced_accuracy'] = balanced_accuracy_score(self.y_true, self.y_pred)
        
        # Binary metrics (assuming 1 is positive class)
        self.metrics['precision'] = precision_score(self.y_true, self.y_pred, zero_division=0)
        self.metrics['recall'] = recall_score(self.y_true, self.y_pred, zero_division=0)
        self.metrics['f1_score'] = f1_score(self.y_true, self.y_pred, zero_division=0)
        
        # Confusion Matrix
        cm = confusion_matrix(self.y_true, self.y_pred)
        self.metrics['confusion_matrix'] = cm.tolist() # Convert to list for serialization
        
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            self.metrics.update({
                'true_positives': int(tp),
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'sensitivity': float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
                'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
            })
        
        # Advanced metrics
        self.metrics['mcc'] = matthews_corrcoef(self.y_true, self.y_pred)
        self.metrics['kappa'] = cohen_kappa_score(self.y_true, self.y_pred)
        
        # ROC-AUC
        if self.y_pred_proba is not None:
            try:
                # Handle binary case
                if len(self.y_pred_proba.shape) == 2:
                    proba_positive = self.y_pred_proba[:, 1]
                else:
                    proba_positive = self.y_pred_proba
                    
                self.metrics['roc_auc'] = roc_auc_score(self.y_true, proba_positive)
                self.metrics['pr_auc'] = average_precision_score(self.y_true, proba_positive)
            except Exception as e:
                logger.warning(f"Could not calculate ROC-AUC: {e}")
                
        return self.metrics

    def print_summary(self):
        """Print metrics summary."""
        if not self.metrics:
            self.calculate_all_metrics()
            
        print("\n" + "="*60)
        print("METRICS SUMMARY")
        print("="*60)
        for k, v in self.metrics.items():
            if isinstance(v, float):
                print(f"{k}: {v:.4f}")
            elif isinstance(v, (int, str, list)):
                print(f"{k}: {v}")
        print("="*60 + "\n")
