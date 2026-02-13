"""
Data Balancing Module
Handles class imbalance using SMOTE and Random Under Sampling.
"""

import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from typing import Tuple, Dict
import logging
from ..config import RANDOM_STATE

logger = logging.getLogger(__name__)

class DataBalancer:
    """Class for handling class imbalance."""
    
    def __init__(self, random_state: int = RANDOM_STATE):
        self.random_state = random_state
        
    def apply_smote(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Apply SMOTE oversampling."""
        try:
            logger.info("Applying SMOTE...")
            smote = SMOTE(random_state=self.random_state)
            X_res, y_res = smote.fit_resample(X, y)
            
            logger.info(f"SMOTE: {X.shape} -> {X_res.shape}")
            return pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=y.name)
        except Exception as e:
            logger.error(f"SMOTE failed: {e}")
            return X, y

    def apply_rus(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Apply Random Under Sampling."""
        try:
            logger.info("Applying Random Under Sampling...")
            rus = RandomUnderSampler(random_state=self.random_state)
            X_res, y_res = rus.fit_resample(X, y)
            
            logger.info(f"RUS: {X.shape} -> {X_res.shape}")
            return pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=y.name)
        except Exception as e:
            logger.error(f"RUS failed: {e}")
            return X, y

    def balance_dataset(self, X: pd.DataFrame, y: pd.Series, method: str = 'smote') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Balance dataset using specified method.
        
        Args:
            X: Features DataFrame.
            y: Target Series.
            method: 'smote', 'rus', or None.
            
        Returns:
            Balanced X and y.
        """
        if method.lower() == 'smote':
            return self.apply_smote(X, y)
        elif method.lower() == 'rus':
            return self.apply_rus(X, y)
        else:
            return X, y
