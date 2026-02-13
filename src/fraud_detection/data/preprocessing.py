"""
Data Preprocessor Module
Handles data cleaning, normalization, and encoding.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from typing import Dict, List, Optional, Tuple
import logging
from ..config import MISSING_VALUE_THRESHOLD, OUTLIER_METHOD, OUTLIER_THRESHOLD

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Class for preprocessing fraud detection datasets."""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'median') -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        """
        df_processed = df.copy()
        
        # Drop columns with too many missing values
        missing_pct = df_processed.isnull().mean()
        cols_to_drop = missing_pct[missing_pct > MISSING_VALUE_THRESHOLD].index.tolist()
        if cols_to_drop:
            df_processed.drop(columns=cols_to_drop, inplace=True)
            logger.info(f"Dropped columns with >{MISSING_VALUE_THRESHOLD:.0%} missing values: {cols_to_drop}")
            
        # Fill remaining
        numerical_cols = df_processed.select_dtypes(include=[np.number]).columns
        if strategy == 'median':
            df_processed[numerical_cols] = df_processed[numerical_cols].fillna(df_processed[numerical_cols].median())
        elif strategy == 'mean':
            df_processed[numerical_cols] = df_processed[numerical_cols].fillna(df_processed[numerical_cols].mean())
            
        # Mode for categorical
        categorical_cols = df_processed.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df_processed[col].isnull().any():
                df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])
                
        return df_processed

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows."""
        n_dupes = df.duplicated().sum()
        if n_dupes > 0:
            logger.info(f"Removing {n_dupes} duplicate rows.")
            return df.drop_duplicates()
        return df

    def handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers using IQR or Z-score."""
        df_processed = df.copy()
        numerical_cols = df_processed.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            if OUTLIER_METHOD == 'IQR':
                Q1 = df_processed[col].quantile(0.25)
                Q3 = df_processed[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                df_processed[col] = df_processed[col].clip(lower, upper)
            elif OUTLIER_METHOD == 'Z-score':
                mean = df_processed[col].mean()
                std = df_processed[col].std()
                lower = mean - OUTLIER_THRESHOLD * std
                upper = mean + OUTLIER_THRESHOLD * std
                df_processed[col] = df_processed[col].clip(lower, upper)
                
        return df_processed

    def encode_categorical(self, df: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
        """Encode categorical variables."""
        df_processed = df.copy()
        if columns is None:
            columns = df_processed.select_dtypes(include=['object']).columns.tolist()
            
        for col in columns:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
            self.encoders[col] = le
            
        return df_processed

    def scale_features(self, df: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:
        """Scale numerical features."""
        df_processed = df.copy()
        numerical_cols = df_processed.select_dtypes(include=[np.number]).columns
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
            
        df_processed[numerical_cols] = scaler.fit_transform(df_processed[numerical_cols])
        self.scalers['main'] = scaler
        
        return df_processed

    def preprocess(self, df: pd.DataFrame, dataset_config: Dict) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Run full preprocessing pipeline.
        
        Args:
            df: Raw DataFrame.
            dataset_config: Configuration dictionary for the dataset.
            
        Returns:
            Processed DataFrame (features) and Target Series.
        """
        # 1. Drop specific columns
        if dataset_config.get('drop_columns'):
            df = df.drop(columns=[c for c in dataset_config['drop_columns'] if c in df.columns], errors='ignore')
            
        # 2. Extract Target
        target_col = dataset_config.get('target_column')
        y = None
        if target_col and target_col in df.columns:
            y = df[target_col].copy()
            df = df.drop(columns=[target_col])
            
        # 3. Clean
        df = self.remove_duplicates(df)
        df = self.handle_missing_values(df)
        
        # 4. Outliers
        df = self.handle_outliers(df)
        
        # 5. Encode
        df = self.encode_categorical(df)
        
        # 6. Scale
        df = self.scale_features(df)
        
        return df, y
