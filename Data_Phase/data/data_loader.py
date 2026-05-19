"""
Data Loader Module
Handles loading and initial exploration of fraud detection datasets
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Tuple, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Class for loading and exploring fraud detection datasets"""
    
    def __init__(self, data_dir: Path):
        """
        Initialize DataLoader
        
        Args:
            data_dir: Path to directory containing datasets
        """
        self.data_dir = Path(data_dir)
        self.datasets = {}
        self.dataset_info = {}
        
    def load_dataset(self, dataset_name: str, file_path: str, 
                     encoding: str = 'utf-8') -> pd.DataFrame:
        """
        Load a single dataset from file
        
        Args:
            dataset_name: Name identifier for the dataset
            file_path: Path to the CSV file
            encoding: File encoding
            
        Returns:
            Loaded DataFrame
        """
        try:
            full_path = self.data_dir / file_path
            logger.info(f"Loading dataset '{dataset_name}' from {full_path}")
            
            # Try different separators if comma doesn't work
            try:
                df = pd.read_csv(full_path, encoding=encoding)
            except:
                try:
                    df = pd.read_csv(full_path, encoding=encoding, sep='\t')
                except:
                    df = pd.read_csv(full_path, encoding=encoding, sep=';')
            
            self.datasets[dataset_name] = df
            logger.info(f"Successfully loaded {dataset_name}: {df.shape[0]} rows, {df.shape[1]} columns")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading dataset '{dataset_name}': {str(e)}")
            raise
    
    def load_all_datasets(self, dataset_configs: Dict) -> Dict[str, pd.DataFrame]:
        """
        Load all datasets specified in configuration
        
        Args:
            dataset_configs: Dictionary with dataset configurations
            
        Returns:
            Dictionary of loaded DataFrames
        """
        for name, config in dataset_configs.items():
            try:
                self.load_dataset(name, config['filename'])
            except:
                logger.warning(f"Could not load dataset '{name}', skipping...")
                
        return self.datasets
    
    def get_dataset_info(self, df: pd.DataFrame, dataset_name: str) -> Dict:
        """
        Get comprehensive information about a dataset
        
        Args:
            df: DataFrame to analyze
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary with dataset information
        """
        info = {
            'name': dataset_name,
            'shape': df.shape,
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,  # MB
            'columns': list(df.columns),
            'dtypes': df.dtypes.value_counts().to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
            'duplicates': df.duplicated().sum(),
            'numerical_columns': list(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(df.select_dtypes(include=['object']).columns),
            'statistics': {}
        }
        
        # Add basic statistics for numerical columns
        if info['numerical_columns']:
            info['statistics']['numerical'] = df[info['numerical_columns']].describe().to_dict()
        
        # Add value counts for categorical columns
        if info['categorical_columns']:
            info['statistics']['categorical'] = {}
            for col in info['categorical_columns'][:5]:  # Limit to first 5 to avoid too much info
                value_counts = df[col].value_counts().head(10)
                info['statistics']['categorical'][col] = value_counts.to_dict()
        
        self.dataset_info[dataset_name] = info
        return info
    
    def explore_all_datasets(self) -> Dict:
        """
        Explore all loaded datasets
        
        Returns:
            Dictionary with information for all datasets
        """
        for name, df in self.datasets.items():
            self.get_dataset_info(df, name)
            
        return self.dataset_info
    
    def check_class_distribution(self, df: pd.DataFrame, target_column: str) -> Dict:
        """
        Check class distribution in the dataset
        
        Args:
            df: DataFrame to analyze
            target_column: Name of the target column
            
        Returns:
            Dictionary with class distribution information
        """
        if target_column not in df.columns:
            logger.error(f"Target column '{target_column}' not found in dataset")
            return {}
        
        distribution = df[target_column].value_counts()
        distribution_pct = df[target_column].value_counts(normalize=True) * 100
        
        imbalance_ratio = distribution.max() / distribution.min()
        
        return {
            'counts': distribution.to_dict(),
            'percentages': distribution_pct.to_dict(),
            'imbalance_ratio': imbalance_ratio,
            'minority_class': distribution.idxmin(),
            'majority_class': distribution.idxmax()
        }
    
    def get_correlation_with_target(self, df: pd.DataFrame, target_column: str, 
                                   top_n: int = 20) -> pd.Series:
        """
        Get correlation of features with target variable
        
        Args:
            df: DataFrame to analyze
            target_column: Name of the target column
            top_n: Number of top correlations to return
            
        Returns:
            Series with top correlations
        """
        if target_column not in df.columns:
            logger.error(f"Target column '{target_column}' not found in dataset")
            return pd.Series()
        
        # Select only numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) == 0:
            logger.warning("No numerical columns found for correlation analysis")
            return pd.Series()
        
        correlations = df[numerical_cols].corr()[target_column].abs().sort_values(ascending=False)
        
        # Remove target column itself
        correlations = correlations.drop(target_column, errors='ignore')
        
        return correlations.head(top_n)
    
    def generate_data_report(self, output_path: Optional[Path] = None) -> str:
        """
        Generate a comprehensive data report
        
        Args:
            output_path: Optional path to save the report
            
        Returns:
            String containing the report
        """
        report = "=" * 80 + "\n"
        report += "FRAUD DETECTION DATASETS - DATA EXPLORATION REPORT\n"
        report += "=" * 80 + "\n\n"
        
        for name, info in self.dataset_info.items():
            report += f"\n{'='*60}\n"
            report += f"Dataset: {name}\n"
            report += f"{'='*60}\n"
            report += f"Shape: {info['shape'][0]} rows × {info['shape'][1]} columns\n"
            report += f"Memory Usage: {info['memory_usage']:.2f} MB\n"
            report += f"Duplicate Rows: {info['duplicates']}\n"
            report += f"Numerical Columns: {len(info['numerical_columns'])}\n"
            report += f"Categorical Columns: {len(info['categorical_columns'])}\n"
            
            # Missing values summary
            missing_cols = [col for col, pct in info['missing_percentage'].items() if pct > 0]
            if missing_cols:
                report += f"\nColumns with Missing Values: {len(missing_cols)}\n"
                for col in missing_cols[:5]:  # Show top 5
                    report += f"  - {col}: {info['missing_percentage'][col]:.2f}%\n"
            else:
                report += "\nNo Missing Values Found\n"
            
            report += "\n"
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            logger.info(f"Report saved to {output_path}")
        
        return report