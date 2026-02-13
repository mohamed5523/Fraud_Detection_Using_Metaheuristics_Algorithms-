"""
Visualization Module
Handles data exploration and model performance visualizations.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
import logging
from ..config import RESULTS_DIR, LOG_LEVEL

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
logging.basicConfig(level=getattr(logging, LOG_LEVEL))
logger = logging.getLogger(__name__)

class Visualizer:
    """Class for all fraud detection visualizations."""
    
    def __init__(self, save_dir: Union[str, Path] = RESULTS_DIR):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def _save_plot(self, filename: str) -> str:
        filepath = self.save_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved plot: {filepath}")
        return str(filepath)

    # ==================== Data Exploration Plots ====================

    def plot_class_distribution(self, y: pd.Series, title: str = "Class Distribution") -> str:
        """Plot class distribution (Count & Percentage)."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Count Plot
        sns.countplot(x=y, ax=axes[0])
        axes[0].set_title(f"{title} (Count)")
        for container in axes[0].containers:
            axes[0].bar_label(container)
            
        # Pie Chart
        y.value_counts().plot.pie(autopct='%1.1f%%', ax=axes[1], ylabel='')
        axes[1].set_title(f"{title} (Percentage)")
        
        return self._save_plot(f"{title.lower().replace(' ', '_')}.png")

    def plot_correlation_matrix(self, df: pd.DataFrame, title: str = "Correlation Matrix") -> Optional[str]:
        """Plot correlation matrix for numerical features."""
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) < 2:
            return None
            
        plt.figure(figsize=(12, 10))
        corr = df[numerical_cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, cmap='coolwarm', center=0, square=True, linewidths=0.5)
        plt.title(title)
        
        return self._save_plot(f"{title.lower().replace(' ', '_')}.png")

    # ==================== Model Performance Plots ====================

    def plot_confusion_matrix(self, cm: List[List[int]], labels: List[str] = ['Normal', 'Fraud'], 
                             title: str = "Confusion Matrix") -> str:
        """Plot confusion matrix."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(title)
        return self._save_plot(f"{title.lower().replace(' ', '_')}.png")

    def plot_convergence(self, convergence: List[float], title: str = "Optimization Convergence") -> str:
        """Plot optimization convergence curve."""
        plt.figure(figsize=(10, 6))
        plt.plot(convergence, linewidth=2, marker='o', markersize=4)
        plt.xlabel('Iteration')
        plt.ylabel('Fitness')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        return self._save_plot(f"{title.lower().replace(' ', '_')}.png")

    def plot_feature_importance(self, selected_features: List[str], all_features_count: int, 
                               title: str = "Selected Features") -> str:
        """Plot selected features as a horizontal bar chart."""
        if not selected_features:
            return ""
            
        plt.figure(figsize=(10, max(6, len(selected_features) * 0.4)))
        plt.barh(range(len(selected_features)), [1]*len(selected_features), color='steelblue')
        plt.yticks(range(len(selected_features)), selected_features)
        plt.xlabel('Selected')
        plt.title(f"{title} ({len(selected_features)}/{all_features_count})")
        return self._save_plot(f"{title.lower().replace(' ', '_')}.png")
