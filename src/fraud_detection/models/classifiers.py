"""
Classifier Module
Wrappers for classifier training and evaluation.
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
import logging
from typing import Optional, Any
from ..config import CLASSIFIERS, RANDOM_STATE

logger = logging.getLogger(__name__)

class ClassifierWrapper:
    """Wrapper for classifier training and evaluation."""
    
    def __init__(self, classifier_name: str, X_train: np.ndarray, y_train: np.ndarray):
        """
        Initialize classifier wrapper.
        
        Args:
            classifier_name: Name of classifier defined in config.CLASSIFIERS.
            X_train: Training features.
            y_train: Training labels.
        """
        self.classifier_name = classifier_name
        self.X_train = X_train
        self.y_train = y_train
        self.model = self._get_classifier()
        
    def _get_classifier(self) -> Any:
        """Get classifier instance."""
        if self.classifier_name not in CLASSIFIERS:
            raise ValueError(f"Unknown classifier: {self.classifier_name}")
            
        config = CLASSIFIERS[self.classifier_name]
        params = config.get('params', {}).copy()
        
        if self.classifier_name == 'KNN':
            return KNeighborsClassifier(**params)
        elif self.classifier_name == 'DecisionTree':
            return DecisionTreeClassifier(**params)
        elif self.classifier_name == 'XGBoost':
            return XGBClassifier(**params)
        else:
            raise ValueError(f"Classifier implementation not found for: {self.classifier_name}")
            
    def evaluate_features(self, feature_mask: np.ndarray, cv: int = 5) -> float:
        """
        Evaluate classifier with selected features using cross-validation.
        
        Args:
            feature_mask: Binary array indicating selected features.
            cv: Number of cross-validation folds.
            
        Returns:
            Mean accuracy score.
        """
        selected_indices = np.where(feature_mask == 1)[0]
        
        if len(selected_indices) == 0:
            return 0.0
            
        X_selected = self.X_train[:, selected_indices]
        
        try:
            scores = cross_val_score(
                self.model,
                X_selected,
                self.y_train,
                cv=cv,
                scoring='accuracy',
                n_jobs=-1
            )
            return float(np.mean(scores))
        except Exception as e:
            logger.warning(f"CV Error: {e}")
            return 0.0
            
    def train(self, feature_mask: np.ndarray) -> Any:
        """
        Train classifier with selected features.
        
        Args:
            feature_mask: Binary array indicating selected features.
            
        Returns:
            Trained model instance.
        """
        selected_indices = np.where(feature_mask == 1)[0]
        
        if len(selected_indices) == 0:
            logger.warning("No features selected for training.")
            return None
            
        X_selected = self.X_train[:, selected_indices]
        
        # Retrain a fresh model
        self.model = self._get_classifier()
        self.model.fit(X_selected, self.y_train)
        
        return self.model
        
    def predict(self, X_test: np.ndarray, feature_mask: np.ndarray) -> Optional[np.ndarray]:
        """Make predictions on test set."""
        selected_indices = np.where(feature_mask == 1)[0]
        if len(selected_indices) == 0:
            return None
        return self.model.predict(X_test[:, selected_indices])
        
    def predict_proba(self, X_test: np.ndarray, feature_mask: np.ndarray) -> Optional[np.ndarray]:
        """Make prediction probabilities."""
        selected_indices = np.where(feature_mask == 1)[0]
        if len(selected_indices) == 0:
            return None
        return self.model.predict_proba(X_test[:, selected_indices])
