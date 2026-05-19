"""
Data Visualizer Module
Handles all data visualization tasks for fraud detection analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataVisualizer:
    """Class for creating comprehensive visualizations of fraud detection data"""
    
    def __init__(self, style: str = 'seaborn', figsize: Tuple[int, int] = (12, 8),
                 save_path: Optional[Path] = None):
        """
        Initialize DataVisualizer
        
        Args:
            style: Matplotlib style
            figsize: Default figure size
            save_path: Path to save visualizations
        """

        
        plt.style.use('seaborn-v0_8-darkgrid')
        self.figsize = figsize
        self.save_path = Path(save_path) if save_path else None
        
        if self.save_path:
            self.save_path.mkdir(parents=True, exist_ok=True)
        
        # Set color palettes
        self.colors = {
            'fraud': '#e74c3c',
            'normal': '#3498db',
            'primary': '#2ecc71',
            'secondary': '#f39c12',
            'warning': '#e67e22',
            'danger': '#c0392b'
        }
        
        sns.set_palette("husl")
    
    def plot_class_distribution(self, y: pd.Series, dataset_name: str,
                               save: bool = True) -> None:
        """
        Plot class distribution with detailed statistics
        
        Args:
            y: Target variable
            dataset_name: Name of the dataset
            save: Whether to save the plot
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'Class Distribution Analysis - {dataset_name}', fontsize=16, fontweight='bold')
        
        # Count plot
        class_counts = y.value_counts()
        ax1 = axes[0]
        bars = ax1.bar(class_counts.index.astype(str), class_counts.values)
        bars[0].set_color(self.colors['normal'])
        if len(bars) > 1:
            bars[1].set_color(self.colors['fraud'])
        ax1.set_xlabel('Class')
        ax1.set_ylabel('Count')
        ax1.set_title('Class Distribution (Count)')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):,}', ha='center', va='bottom')
        
        # Percentage plot
        ax2 = axes[1]
        class_pct = y.value_counts(normalize=True) * 100
        wedges, texts, autotexts = ax2.pie(class_pct.values, 
                                           labels=['Normal', 'Fraud'],
                                           colors=[self.colors['normal'], self.colors['fraud']],
                                           autopct='%1.2f%%',
                                           startangle=90)
        ax2.set_title('Class Distribution (Percentage)')
        
        # Imbalance ratio visualization
        ax3 = axes[2]
        imbalance_ratio = class_counts.max() / class_counts.min()
        ax3.text(0.5, 0.5, f'Imbalance Ratio\n{imbalance_ratio:.2f}:1',
                horizontalalignment='center', verticalalignment='center',
                transform=ax3.transAxes, fontsize=20, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax3.axis('off')
        ax3.set_title('Class Imbalance')
        
        plt.tight_layout()
        
        if save and self.save_path:
            plt.savefig(self.save_path / f'{dataset_name}_class_distribution.png', 
                       dpi=100, bbox_inches='tight')
        plt.show()
    
    def plot_correlation_matrix(self, df: pd.DataFrame, dataset_name: str,
                               top_features: int = 30, save: bool = True) -> None:
        """
        Plot correlation matrix heatmap
        
        Args:
            df: DataFrame with features
            dataset_name: Name of the dataset
            top_features: Number of top features to show
            save: Whether to save the plot
        """
        # Select numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) == 0:
            logger.warning("No numerical columns found for correlation matrix")
            return
        
        # Calculate correlation matrix
        corr_matrix = df[numerical_cols].corr()
        
        # Select top features based on variance
        if len(numerical_cols) > top_features:
            variances = df[numerical_cols].var().sort_values(ascending=False)
            top_cols = variances.head(top_features).index
            corr_matrix = df[top_cols].corr()
        
        # Create heatmap
        plt.figure(figsize=(14, 12))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm',
                   center=0, square=True, linewidths=0.5,
                   cbar_kws={"shrink": 0.8})
        
        plt.title(f'Correlation Matrix - {dataset_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save and self.save_path:
            plt.savefig(self.save_path / f'{dataset_name}_correlation_matrix.png', 
                       dpi=100, bbox_inches='tight')
        plt.show()
    
    def plot_feature_importance(self, X: pd.DataFrame, y: pd.Series, 
                               dataset_name: str, top_n: int = 20,
                               save: bool = True) -> None:
        """
        Plot feature importance using correlation with target
        
        Args:
            X: Features DataFrame
            y: Target variable
            dataset_name: Name of the dataset
            top_n: Number of top features to show
            save: Whether to save the plot
        """
        # Combine features and target
        data = X.copy()
        data['target'] = y
        
        # Calculate correlations with target
        correlations = data.corr()['target'].abs().sort_values(ascending=False)
        correlations = correlations.drop('target')  # Remove target itself
        
        # Select top features
        top_features = correlations.head(top_n)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = ['#e74c3c' if corr > 0.3 else '#3498db' if corr > 0.1 else '#95a5a6' 
                 for corr in top_features.values]
        
        bars = ax.barh(range(len(top_features)), top_features.values, color=colors)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features.index)
        ax.set_xlabel('Absolute Correlation with Target')
        ax.set_title(f'Top {top_n} Features by Correlation - {dataset_name}', 
                    fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, top_features.values)):
            ax.text(value + 0.005, bar.get_y() + bar.get_height()/2,
                   f'{value:.3f}', va='center')
        
        plt.tight_layout()
        
        if save and self.save_path:
            plt.savefig(self.save_path / f'{dataset_name}_feature_importance.png', 
                       dpi=100, bbox_inches='tight')
        plt.show()
    
    def plot_distribution_comparison(self, df: pd.DataFrame, y: pd.Series,
                                   features: List[str], dataset_name: str,
                                   save: bool = True) -> None:
        """
        Compare feature distributions between fraud and normal classes
        
        Args:
            df: Features DataFrame
            y: Target variable
            features: List of features to compare
            dataset_name: Name of the dataset
            save: Whether to save the plot
        """
        # Limit features if too many
        features = features[:min(len(features), 12)]
        
        n_features = len(features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        fig.suptitle(f'Feature Distribution Comparison - {dataset_name}', 
                    fontsize=16, fontweight='bold')
        
        axes = axes.flatten() if n_features > 1 else [axes]
        
        for idx, feature in enumerate(features):
            ax = axes[idx]
            
            # Create distribution plot
            for class_val in y.unique():
                data = df[y == class_val][feature]
                label = 'Fraud' if class_val == 1 else 'Normal'
                color = self.colors['fraud'] if class_val == 1 else self.colors['normal']
                ax.hist(data, bins=30, alpha=0.5, label=label, color=color, density=True)
            
            ax.set_xlabel(feature)
            ax.set_ylabel('Density')
            ax.set_title(f'{feature} Distribution')
            ax.legend()
            ax.grid(alpha=0.3)
        
        # Remove empty subplots
        for idx in range(n_features, len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        
        if save and self.save_path:
            plt.savefig(self.save_path / f'{dataset_name}_distribution_comparison.png', 
                       dpi=100, bbox_inches='tight')
        plt.show()
    
    def plot_boxplot_comparison(self, df: pd.DataFrame, y: pd.Series,
                               features: List[str], dataset_name: str,
                               save: bool = True) -> None:
        """
        Create boxplots comparing features between classes
        
        Args:
            df: Features DataFrame
            y: Target variable
            features: List of features to compare
            dataset_name: Name of the dataset
            save: Whether to save the plot
        """
        # Limit features
        features = features[:min(len(features), 8)]
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle(f'Feature Boxplot Comparison - {dataset_name}', 
                    fontsize=16, fontweight='bold')
        
        axes = axes.flatten()
        
        for idx, feature in enumerate(features):
            ax = axes[idx]
            
            data_to_plot = []
            labels = []
            colors_list = []
            
            for class_val in sorted(y.unique()):
                data_to_plot.append(df[y == class_val][feature].values)
                labels.append('Fraud' if class_val == 1 else 'Normal')
                colors_list.append(self.colors['fraud'] if class_val == 1 else self.colors['normal'])
            
            bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
            
            for patch, color in zip(bp['boxes'], colors_list):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            
            ax.set_ylabel(feature)
            ax.set_title(f'{feature}')
            ax.grid(alpha=0.3)
        
        # Remove empty subplots
        for idx in range(len(features), len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        
        if save and self.save_path:
            plt.savefig(self.save_path / f'{dataset_name}_boxplot_comparison.png', 
                       dpi=100, bbox_inches='tight')
        plt.show()
    
    def create_interactive_scatter(self, df: pd.DataFrame, y: pd.Series,
                                  feature1: str, feature2: str,
                                  dataset_name: str, save: bool = True) -> None:
        """
        Create interactive scatter plot using plotly
        
        Args:
            df: Features DataFrame
            y: Target variable
            feature1: First feature for x-axis
            feature2: Second feature for y-axis
            dataset_name: Name of the dataset
            save: Whether to save the plot
        """
        # Prepare data
        plot_df = df[[feature1, feature2]].copy()
        plot_df['Class'] = y.apply(lambda x: 'Fraud' if x == 1 else 'Normal')
        
        # Create scatter plot
        fig = px.scatter(plot_df, x=feature1, y=feature2, color='Class',
                        color_discrete_map={'Fraud': self.colors['fraud'],
                                          'Normal': self.colors['normal']},
                        title=f'Interactive Scatter Plot - {dataset_name}',
                        opacity=0.6)
        
        fig.update_layout(
            width=900,
            height=600,
            template='plotly_white',
            hovermode='closest'
        )
        
        if save and self.save_path:
            fig.write_html(self.save_path / f'{dataset_name}_scatter_{feature1}_{feature2}.html')
        
        fig.show()
    
    def plot_missing_values_heatmap(self, df: pd.DataFrame, dataset_name: str,
                                   save: bool = True) -> None:
        """
        Create heatmap showing missing values pattern
        
        Args:
            df: DataFrame to analyze
            dataset_name: Name of the dataset
            save: Whether to save the plot
        """
        # Calculate missing values
        missing_df = pd.DataFrame({
            'column': df.columns,
            'missing_count': df.isnull().sum(),
            'missing_percentage': (df.isnull().sum() / len(df)) * 100
        })
        
        missing_df = missing_df[missing_df['missing_count'] > 0].sort_values(
            'missing_percentage', ascending=False
        )
        
        if len(missing_df) == 0:
            logger.info(f"No missing values found in {dataset_name}")
            return
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f'Missing Values Analysis - {dataset_name}', 
                    fontsize=16, fontweight='bold')
        
        # Bar plot
        ax1 = axes[0]
        bars = ax1.bar(range(len(missing_df)), missing_df['missing_percentage'].values)
        ax1.set_xticks(range(len(missing_df)))
        ax1.set_xticklabels(missing_df['column'].values, rotation=45, ha='right')
        ax1.set_ylabel('Missing Percentage (%)')
        ax1.set_title('Missing Values by Column')
        
        # Color bars based on percentage
        for bar, pct in zip(bars, missing_df['missing_percentage'].values):
            if pct > 50:
                bar.set_color(self.colors['danger'])
            elif pct > 20:
                bar.set_color(self.colors['warning'])
            else:
                bar.set_color(self.colors['primary'])
        
        # Heatmap
        ax2 = axes[1]
        # Sample data for visualization (limit to 1000 rows for performance)
        sample_size = min(1000, len(df))
        sample_df = df[missing_df['column'].values].sample(n=sample_size, random_state=42)
        
        sns.heatmap(sample_df.isnull().T, cbar=True, cmap='RdYlBu',
                   ax=ax2, yticklabels=True, xticklabels=False)
        ax2.set_title(f'Missing Pattern (Sample of {sample_size} rows)')
        ax2.set_xlabel('Samples')
        
        plt.tight_layout()
        
        if save and self.save_path:
            plt.savefig(self.save_path / f'{dataset_name}_missing_values.png', 
                       dpi=100, bbox_inches='tight')
        plt.show()
    
    def create_summary_dashboard(self, datasets_info: Dict, save: bool = True) -> None:
        """
        Create a summary dashboard for all datasets
        
        Args:
            datasets_info: Dictionary with information about all datasets
            save: Whether to save the plot
        """
        n_datasets = len(datasets_info)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Fraud Detection Datasets - Summary Dashboard', 
                    fontsize=16, fontweight='bold')
        
        # Dataset sizes comparison
        ax1 = axes[0, 0]
        dataset_names = list(datasets_info.keys())
        sizes = [info['shape'][0] for info in datasets_info.values()]
        bars = ax1.bar(dataset_names, sizes)
        ax1.set_ylabel('Number of Samples')
        ax1.set_title('Dataset Sizes')
        ax1.tick_params(axis='x', rotation=45)
        
        for bar, size in zip(bars, sizes):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{size:,}', ha='center', va='bottom')
        
        # Feature counts
        ax2 = axes[0, 1]
        feature_counts = [info['shape'][1] for info in datasets_info.values()]
        bars = ax2.bar(dataset_names, feature_counts, color='orange')
        ax2.set_ylabel('Number of Features')
        ax2.set_title('Feature Counts')
        ax2.tick_params(axis='x', rotation=45)
        
        for bar, count in zip(bars, feature_counts):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{count}', ha='center', va='bottom')
        
        # Memory usage
        ax3 = axes[1, 0]
        memory_usage = [info['memory_usage'] for info in datasets_info.values()]
        bars = ax3.bar(dataset_names, memory_usage, color='green')
        ax3.set_ylabel('Memory Usage (MB)')
        ax3.set_title('Memory Consumption')
        ax3.tick_params(axis='x', rotation=45)
        
        for bar, mem in zip(bars, memory_usage):
            ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{mem:.1f}', ha='center', va='bottom')
        
        # Summary statistics table
        ax4 = axes[1, 1]
        ax4.axis('tight')
        ax4.axis('off')
        
        table_data = []
        for name, info in datasets_info.items():
            table_data.append([
                name,
                f"{info['shape'][0]:,}",
                str(info['shape'][1]),
                f"{info['memory_usage']:.1f} MB"
            ])
        
        table = ax4.table(cellText=table_data,
                         colLabels=['Dataset', 'Samples', 'Features', 'Memory'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        ax4.set_title('Summary Statistics')
        
        plt.tight_layout()
        
        if save and self.save_path:
            plt.savefig(self.save_path / 'summary_dashboard.png', 
                       dpi=100, bbox_inches='tight')
        plt.show()