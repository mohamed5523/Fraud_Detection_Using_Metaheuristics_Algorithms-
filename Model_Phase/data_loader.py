"""
Data Loading Module
Handles loading and preprocessing of datasets
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Load and preprocess datasets for model training"""
    
    def __init__(self, dataset_name):
        """
        Initialize DataLoader
        
        Args:
            dataset_name: Name of dataset from config.DATASETS
        """
        self.dataset_name = dataset_name
        self.dataset_config = config.DATASETS[dataset_name]
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def load_data(self):
        """Load dataset from CSV file"""
        file_path = f"{config.DATA_DIR}/{self.dataset_config['file']}"
        logger.info(f"Loading {self.dataset_name} from {file_path}")
        
        try:
            self.data = pd.read_csv(file_path)
            logger.info(f"Loaded {self.dataset_name}: {self.data.shape}")
            return True
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            return False
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return False
    
    def prepare_data(self):
        """Split data into features and target"""
        target_col = self.dataset_config['target']
        
        if target_col not in self.data.columns:
            logger.error(f"Target column '{target_col}' not found in dataset")
            return False
        
        # Separate features and target
        X = self.data.drop(columns=[target_col])
        y = self.data[target_col]
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Convert to numpy arrays
        X = X.values
        y = y.values
        
        # Split data
        stratify_y = y if config.SPLIT_SETTINGS['stratify'] else None
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=config.SPLIT_SETTINGS['test_size'],
            random_state=config.SPLIT_SETTINGS['random_state'],
            stratify=stratify_y
        )
        
        logger.info(f"Train set: {self.X_train.shape}, Test set: {self.X_test.shape}")
        return True
    
    def scale_data(self):
        """Standardize features using StandardScaler"""
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        logger.info("Data scaled successfully")
        return True
    
    def get_data(self):
        """Return processed data"""
        return {
            'X_train': self.X_train,
            'X_test': self.X_test,
            'y_train': self.y_train,
            'y_test': self.y_test,
            'feature_names': self.feature_names,
            'n_features': len(self.feature_names)
        }
    
    def get_class_distribution(self):
        """Get class distribution for train and test sets"""
        train_dist = pd.Series(self.y_train).value_counts(normalize=True)
        test_dist = pd.Series(self.y_test).value_counts(normalize=True)
        
        return {
            'train': train_dist.to_dict(),
            'test': test_dist.to_dict()
        }
    
    def run_pipeline(self):
        """Execute complete data loading pipeline"""
        if not self.load_data():
            return None
        if not self.prepare_data():
            return None
        if not self.scale_data():
            return None
        
        logger.info(f"{self.dataset_name} pipeline completed successfully")
        return self.get_data()


def load_all_datasets():
    """Load all datasets defined in config"""
    datasets = {}
    
    for dataset_name in config.DATASETS.keys():
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing {dataset_name}")
        logger.info(f"{'='*50}")
        
        loader = DataLoader(dataset_name)
        data = loader.run_pipeline()
        
        if data is not None:
            datasets[dataset_name] = {
                'data': data,
                'loader': loader
            }
            
            # Log class distribution
            dist = loader.get_class_distribution()
            logger.info(f"Train class distribution: {dist['train']}")
            logger.info(f"Test class distribution: {dist['test']}")
        else:
            logger.warning(f"Failed to load {dataset_name}")
    
    return datasets


if __name__ == "__main__":
    # Test data loading
    datasets = load_all_datasets()
    print(f"\nSuccessfully loaded {len(datasets)} datasets")
