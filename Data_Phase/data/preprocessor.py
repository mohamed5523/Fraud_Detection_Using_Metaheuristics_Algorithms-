"""
Data Preprocessor Module
Handles all data preprocessing tasks including cleaning, encoding, and scaling
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import logging
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Class for preprocessing fraud detection datasets"""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize DataPreprocessor
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.scalers = {}
        self.encoders = {}
        self.preprocessing_info = {}
        
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'drop',
                            threshold: float = 0.5) -> pd.DataFrame:
        """
        Handle missing values in the dataset
        
        Args:
            df: DataFrame to process
            strategy: Strategy for handling missing values ('drop', 'mean', 'median', 'mode', 'forward_fill')
            threshold: Threshold for dropping columns with missing values
            
        Returns:
            Processed DataFrame
        """
        df_processed = df.copy()
        initial_shape = df_processed.shape
        
        # Drop columns with too many missing values
        missing_pct = df_processed.isnull().sum() / len(df_processed)
        cols_to_drop = missing_pct[missing_pct > threshold].index.tolist()
        
        if cols_to_drop:
            df_processed = df_processed.drop(columns=cols_to_drop)
            logger.info(f"Dropped {len(cols_to_drop)} columns with >{threshold*100}% missing values")
        
        # Handle remaining missing values
        if strategy == 'drop':
            df_processed = df_processed.dropna()
        elif strategy == 'mean':
            numerical_cols = df_processed.select_dtypes(include=[np.number]).columns
            df_processed[numerical_cols] = df_processed[numerical_cols].fillna(
                df_processed[numerical_cols].mean()
            )
        elif strategy == 'median':
            numerical_cols = df_processed.select_dtypes(include=[np.number]).columns
            df_processed[numerical_cols] = df_processed[numerical_cols].fillna(
                df_processed[numerical_cols].median()
            )
        elif strategy == 'mode':
            for col in df_processed.columns:
                if df_processed[col].isnull().any():
                    mode_val = df_processed[col].mode()
                    if len(mode_val) > 0:
                        df_processed[col] = df_processed[col].fillna(mode_val[0])
        elif strategy == 'forward_fill':
            df_processed = df_processed.fillna(method='ffill').fillna(method='bfill')
        
        logger.info(f"Missing values handled: {initial_shape} -> {df_processed.shape}")
        return df_processed
    
    def remove_duplicates(self, df: pd.DataFrame, subset: Optional[List[str]] = None,
                         keep: str = 'first') -> pd.DataFrame:
        """
        Remove duplicate rows from the dataset
        
        Args:
            df: DataFrame to process
            subset: Columns to consider for duplicates
            keep: Which duplicates to keep ('first', 'last', False)
            
        Returns:
            Processed DataFrame
        """
        df_processed = df.copy()
        initial_rows = len(df_processed)
        
        df_processed = df_processed.drop_duplicates(subset=subset, keep=keep)
        
        removed_rows = initial_rows - len(df_processed)
        if removed_rows > 0:
            logger.info(f"Removed {removed_rows} duplicate rows")
        
        return df_processed
    
    def handle_outliers(self, df: pd.DataFrame, columns: Optional[List[str]] = None,
                       method: str = 'IQR', threshold: float = 3) -> pd.DataFrame:
        """
        Handle outliers in numerical columns
        
        Args:
            df: DataFrame to process
            columns: Columns to check for outliers (None for all numerical)
            method: Method to detect outliers ('IQR', 'Z-score')
            threshold: Threshold for outlier detection
            
        Returns:
            Processed DataFrame
        """
        df_processed = df.copy()
        
        if columns is None:
            columns = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        
        outliers_info = {}
        
        for col in columns:
            if col in df_processed.columns:
                if method == 'IQR':
                    Q1 = df_processed[col].quantile(0.25)
                    Q3 = df_processed[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = ((df_processed[col] < lower_bound) | 
                               (df_processed[col] > upper_bound)).sum()
                    
                    # Cap outliers instead of removing
                    df_processed[col] = df_processed[col].clip(lower_bound, upper_bound)
                    
                elif method == 'Z-score':
                    z_scores = np.abs((df_processed[col] - df_processed[col].mean()) / 
                                     df_processed[col].std())
                    outliers = (z_scores > threshold).sum()
                    
                    # Cap outliers
                    mean = df_processed[col].mean()
                    std = df_processed[col].std()
                    df_processed[col] = df_processed[col].clip(
                        mean - threshold * std,
                        mean + threshold * std
                    )
                
                if outliers > 0:
                    outliers_info[col] = outliers
        
        if outliers_info:
            logger.info(f"Outliers handled in {len(outliers_info)} columns using {method} method")
        
        return df_processed
    
    def encode_categorical_features(self, df: pd.DataFrame, 
                                   categorical_columns: List[str],
                                   encoding_type: str = 'label') -> pd.DataFrame:
        """
        Encode categorical features
        
        Args:
            df: DataFrame to process
            categorical_columns: List of categorical columns to encode
            encoding_type: Type of encoding ('label', 'onehot')
            
        Returns:
            Processed DataFrame
        """
        df_processed = df.copy()
        
        for col in categorical_columns:
            if col in df_processed.columns:
                if encoding_type == 'label':
                    le = LabelEncoder()
                    df_processed[col] = le.fit_transform(df_processed[col].astype(str))
                    self.encoders[col] = le
                elif encoding_type == 'onehot':
                    # One-hot encoding
                    dummies = pd.get_dummies(df_processed[col], prefix=col, drop_first=True)
                    df_processed = pd.concat([df_processed.drop(columns=[col]), dummies], axis=1)
        
        logger.info(f"Encoded {len(categorical_columns)} categorical columns using {encoding_type} encoding")
        return df_processed
    
    def scale_features(self, df: pd.DataFrame, columns: Optional[List[str]] = None,
                      method: str = 'standard', exclude_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Scale numerical features
        
        Args:
            df: DataFrame to process
            columns: Columns to scale (None for all numerical)
            method: Scaling method ('standard', 'minmax', 'robust')
            exclude_columns: Columns to exclude from scaling
            
        Returns:
            Processed DataFrame
        """
        df_processed = df.copy()
        
        if columns is None:
            columns = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        
        if exclude_columns:
            columns = [col for col in columns if col not in exclude_columns]
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        if columns:
            df_processed[columns] = scaler.fit_transform(df_processed[columns])
            self.scalers[method] = scaler
            logger.info(f"Scaled {len(columns)} columns using {method} scaling")
        
        return df_processed
    
    def create_features(self, df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
        """
        Create additional features based on dataset type
        
        Args:
            df: DataFrame to process
            dataset_type: Type of dataset for specific feature engineering
            
        Returns:
            DataFrame with additional features
        """
        df_processed = df.copy()
        
        if dataset_type == 'dataset1':
            # Transaction-based features
            if 'amt' in df_processed.columns:
                df_processed['log_amt'] = np.log1p(df_processed['amt'])
                
            if all(col in df_processed.columns for col in ['lat', 'long', 'merch_lat', 'merch_long']):
                # Calculate distance between customer and merchant
                df_processed['distance'] = np.sqrt(
                    (df_processed['lat'] - df_processed['merch_lat'])**2 +
                    (df_processed['long'] - df_processed['merch_long'])**2
                )
            
            if 'unix_time' in df_processed.columns:
                df_processed['hour'] = pd.to_datetime(df_processed['unix_time'], unit='s').dt.hour
                df_processed['day_of_week'] = pd.to_datetime(df_processed['unix_time'], unit='s').dt.dayofweek
                
        elif dataset_type == 'dataset3':
            # Credit card PCA features
            if 'Amount' in df_processed.columns:
                df_processed['log_amount'] = np.log1p(df_processed['Amount'])
                
            if 'Time' in df_processed.columns:
                df_processed['time_hour'] = (df_processed['Time'] / 3600) % 24
                
        elif dataset_type == 'dataset4':
            # Payment fraud features
            if all(col in df_processed.columns for col in ['oldbalanceOrg', 'newbalanceOrig']):
                df_processed['balance_change_orig'] = df_processed['newbalanceOrig'] - df_processed['oldbalanceOrg']
                
            if all(col in df_processed.columns for col in ['oldbalanceDest', 'newbalanceDest']):
                df_processed['balance_change_dest'] = df_processed['newbalanceDest'] - df_processed['oldbalanceDest']
                
            if 'amount' in df_processed.columns:
                df_processed['log_amount'] = np.log1p(df_processed['amount'])
        
        logger.info(f"Created additional features for {dataset_type}")
        return df_processed
    
    def preprocess_dataset(self, df: pd.DataFrame, config: Dict, 
                          dataset_name: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Complete preprocessing pipeline for a dataset
        
        Args:
            df: DataFrame to preprocess
            config: Configuration dictionary for the dataset
            dataset_name: Name of the dataset
            
        Returns:
            Tuple of (processed features DataFrame, target Series)
        """
        logger.info(f"\nStarting preprocessing for {dataset_name}")
        df_processed = df.copy()
        
        # Store initial info
        initial_shape = df_processed.shape
        
        # 1. Drop unnecessary columns
        if config.get('drop_columns'):
            cols_to_drop = [col for col in config['drop_columns'] if col in df_processed.columns]
            if cols_to_drop:
                df_processed = df_processed.drop(columns=cols_to_drop)
                logger.info(f"Dropped {len(cols_to_drop)} columns")
        
        # 2. Handle missing values
        df_processed = self.handle_missing_values(df_processed, strategy='median')
        
        # 3. Remove duplicates
        df_processed = self.remove_duplicates(df_processed)
        
        # 4. Separate target variable
        target_column = config.get('target_column')
        if target_column and target_column in df_processed.columns:
            y = df_processed[target_column].copy()
            df_processed = df_processed.drop(columns=[target_column])
        else:
            y = None
            logger.warning(f"Target column '{target_column}' not found")
        
        # 5. Handle date columns
        if config.get('date_columns'):
            for col in config['date_columns']:
                if col in df_processed.columns:
                    df_processed[col] = pd.to_datetime(df_processed[col], errors='coerce')
                    # Extract date features
                    df_processed[f'{col}_year'] = df_processed[col].dt.year
                    df_processed[f'{col}_month'] = df_processed[col].dt.month
                    df_processed[f'{col}_day'] = df_processed[col].dt.day
                    df_processed = df_processed.drop(columns=[col])
        
        # 6. Encode categorical features
        categorical_cols = [col for col in config.get('categorical_columns', []) 
                           if col in df_processed.columns]
        if categorical_cols:
            df_processed = self.encode_categorical_features(df_processed, categorical_cols)
        
        # 7. Create additional features
        df_processed = self.create_features(df_processed, dataset_name)
        
        # 8. Handle outliers in numerical columns
        numerical_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        if numerical_cols:
            df_processed = self.handle_outliers(df_processed, numerical_cols)
        
        # 9. Scale features
        df_processed = self.scale_features(df_processed, exclude_columns=[])
        
        # Store preprocessing info
        self.preprocessing_info[dataset_name] = {
            'initial_shape': initial_shape,
            'final_shape': df_processed.shape,
            'target_distribution': y.value_counts().to_dict() if y is not None else None
        }
        
        logger.info(f"Preprocessing complete: {initial_shape} -> {df_processed.shape}")
        
        return df_processed, y
    
    def split_data(self, X: pd.DataFrame, y: pd.Series, 
                  test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, 
                                                   pd.Series, pd.Series]:
        """
        Split data into training and testing sets
        
        Args:
            X: Features DataFrame
            y: Target Series
            test_size: Proportion of data for testing
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, 
            stratify=y
        )
        
        logger.info(f"Data split: Train {X_train.shape}, Test {X_test.shape}")
        logger.info(f"Train class distribution: {y_train.value_counts().to_dict()}")
        logger.info(f"Test class distribution: {y_test.value_counts().to_dict()}")
        
        return X_train, X_test, y_train, y_test