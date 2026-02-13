"""
Data Loader Module
Handles loading, exploration, and splitting of fraud detection datasets.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Tuple, Optional, Union
from sklearn.model_selection import train_test_split
from ..config import DATASETS, DATA_DIR, RANDOM_STATE, TEST_SIZE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """Class for loading and handling fraud detection datasets."""
    
    def __init__(self, data_dir: Union[str, Path] = DATA_DIR):
        """
        Initialize DataLoader.
        
        Args:
            data_dir: Path to directory containing datasets.
        """
        self.data_dir = Path(data_dir)
        self.datasets = {}
        
    def load_dataset(self, dataset_name: str) -> Optional[pd.DataFrame]:
        """
        Load a single dataset by name using configuration.
        
        Args:
            dataset_name: Name of the dataset defined in config.DATASETS.
            
        Returns:
            Loaded DataFrame or None if failed.
        """
        if dataset_name not in DATASETS:
            logger.error(f"Dataset '{dataset_name}' not found in configuration.")
            return None
            
        config = DATASETS[dataset_name]
        file_path = self.data_dir / config['filename']
        
        logger.info(f"Loading dataset '{dataset_name}' from {file_path}")
        
        try:
            # Try different separators
            try:
                df = pd.read_csv(file_path)
            except:
                try:
                    df = pd.read_csv(file_path, sep='\t')
                except:
                    df = pd.read_csv(file_path, sep=';')
            
            self.datasets[dataset_name] = df
            logger.info(f"Successfully loaded {dataset_name}: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading dataset '{dataset_name}': {str(e)}")
            return None

    def load_all_datasets(self) -> Dict[str, pd.DataFrame]:
        """
        Load all datasets specified in configuration.
        
        Returns:
            Dictionary of loaded DataFrames.
        """
        for name in DATASETS:
            self.load_dataset(name)
        return self.datasets

    def split_data(self, df: pd.DataFrame, target_column: str, 
                  test_size: float = TEST_SIZE, 
                  stratify: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split dataset into training and testing sets.
        
        Args:
            df: DataFrame to split.
            target_column: Name of the target column.
            test_size: Proportion of dataset to include in the test split.
            stratify: If True, data is split in a stratified fashion.
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame.")
            
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        stratify_y = y if stratify else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=RANDOM_STATE,
            stratify=stratify_y
        )
        
        logger.info(f"Split data: Train {X_train.shape}, Test {X_test.shape}")
        return X_train, X_test, y_train, y_test

    def get_dataset_info(self, df: pd.DataFrame) -> Dict:
        """
        Get basic information about the dataset.
        
        Args:
            df: DataFrame to analyze.
            
        Returns:
            Dictionary with shape, memory usage, and column types.
        """
        return {
            'shape': df.shape,
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
            'numerical_columns': list(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(df.select_dtypes(include=['object', 'category']).columns),
            'missing_values': df.isnull().sum().sum()
        }
