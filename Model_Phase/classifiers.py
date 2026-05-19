"""
Classifier Module
Handles training and evaluation of classifiers
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
import config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClassifierWrapper:
    """Wrapper for classifier training and evaluation"""
    
    def __init__(self, classifier_name, X_train, y_train):
        """
        Initialize classifier wrapper
        
        Args:
            classifier_name: Name of classifier from config.CLASSIFIERS
            X_train: Training features
            y_train: Training labels
        """
        self.classifier_name = classifier_name
        self.X_train = X_train
        self.y_train = y_train
        self.model = self._get_classifier()
        
    def _get_classifier(self):
        """Get classifier instance based on name"""
        params = config.CLASSIFIERS[self.classifier_name]['params']
        
        if self.classifier_name == 'KNN':
            return KNeighborsClassifier(**params)
        elif self.classifier_name == 'DecisionTree':
            return DecisionTreeClassifier(**params)
        elif self.classifier_name == 'XGBoost':
            return XGBClassifier(**params)
        else:
            raise ValueError(f"Unknown classifier: {self.classifier_name}")
    
    def evaluate_features(self, feature_mask):
        """
        Evaluate classifier with selected features using cross-validation
        
        Args:
            feature_mask: Binary array indicating selected features
            
        Returns:
            Mean cross-validation accuracy
        """
        # Get selected features
        selected_indices = np.where(feature_mask == 1)[0]
        
        # Check if any features selected
        if len(selected_indices) == 0:
            return 0.0
        
        # Select features
        X_selected = self.X_train[:, selected_indices]
        
        # Perform cross-validation
        try:
            scores = cross_val_score(
                self.model,
                X_selected,
                self.y_train,
                cv=config.CV_SETTINGS['n_splits'],
                scoring='accuracy',
                n_jobs=-1
            )
            return np.mean(scores)
        except Exception as e:
            logger.warning(f"Error in cross-validation: {str(e)}")
            return 0.0
    
    def train(self, feature_mask):
        """
        Train classifier with selected features
        
        Args:
            feature_mask: Binary array indicating selected features
            
        Returns:
            Trained model
        """
        selected_indices = np.where(feature_mask == 1)[0]
        
        if len(selected_indices) == 0:
            logger.warning("No features selected for training")
            return None
        
        X_selected = self.X_train[:, selected_indices]
        
        # Get fresh model instance
        self.model = self._get_classifier()
        
        # Train model
        self.model.fit(X_selected, self.y_train)
        
        return self.model
    
    def predict(self, X_test, feature_mask):
        """
        Make predictions on test set
        
        Args:
            X_test: Test features
            feature_mask: Binary array indicating selected features
            
        Returns:
            Predictions
        """
        selected_indices = np.where(feature_mask == 1)[0]
        
        if len(selected_indices) == 0:
            logger.warning("No features selected for prediction")
            return None
        
        X_test_selected = X_test[:, selected_indices]
        
        return self.model.predict(X_test_selected)
    
    def predict_proba(self, X_test, feature_mask):
        """
        Get prediction probabilities
        
        Args:
            X_test: Test features
            feature_mask: Binary array indicating selected features
            
        Returns:
            Prediction probabilities
        """
        selected_indices = np.where(feature_mask == 1)[0]
        
        if len(selected_indices) == 0:
            return None
        
        X_test_selected = X_test[:, selected_indices]
        
        return self.model.predict_proba(X_test_selected)


def get_fitness_function(classifier_wrapper):
    """
    Create fitness function for metaheuristic optimization
    
    Args:
        classifier_wrapper: ClassifierWrapper instance
        
    Returns:
        Fitness function
    """
    def fitness_func(feature_mask):
        """Evaluate feature subset"""
        return classifier_wrapper.evaluate_features(feature_mask)
    
    return fitness_func
