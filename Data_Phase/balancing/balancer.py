"""
Data Balancer Module
Handles class imbalance using various sampling techniques
"""

import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
import logging
from typing import Dict, Tuple, List, Optional
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataBalancer:
    """Class for handling class imbalance in fraud detection datasets"""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize DataBalancer
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.sampling_results = {}
        self.original_distributions = {}
        
    def analyze_class_distribution(self, y: pd.Series, dataset_name: str) -> Dict:
        """
        Analyze class distribution in the dataset
        
        Args:
            y: Target variable
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary with distribution analysis
        """
        distribution = y.value_counts().sort_index()
        distribution_pct = y.value_counts(normalize=True).sort_index() * 100
        
        imbalance_ratio = distribution.max() / distribution.min()
        minority_class = distribution.idxmin()
        majority_class = distribution.idxmax()
        
        analysis = {
            'counts': distribution.to_dict(),
            'percentages': distribution_pct.to_dict(),
            'imbalance_ratio': imbalance_ratio,
            'minority_class': minority_class,
            'majority_class': majority_class,
            'total_samples': len(y)
        }
        
        self.original_distributions[dataset_name] = analysis
        
        logger.info(f"\n{dataset_name} - Class Distribution Analysis:")
        logger.info(f"Total samples: {analysis['total_samples']:,}")
        for class_val, count in analysis['counts'].items():
            pct = analysis['percentages'][class_val]
            class_name = 'Fraud' if class_val == 1 else 'Normal'
            logger.info(f"  {class_name} (Class {class_val}): {count:,} ({pct:.2f}%)")
        logger.info(f"Imbalance ratio: {imbalance_ratio:.2f}:1")
        
        return analysis
    
    def apply_random_undersampling(self, X: pd.DataFrame, y: pd.Series,
                                  sampling_strategy: str = 'auto') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Apply Random Under-sampling to balance the dataset
        
        Args:
            X: Features DataFrame
            y: Target Series
            sampling_strategy: Sampling strategy for under-sampling
            
        Returns:
            Tuple of (balanced X, balanced y)
        """
        logger.info("Applying Random Under-sampling...")
        
        rus = RandomUnderSampler(
            sampling_strategy=sampling_strategy,
            random_state=self.random_state
        )
        
        X_resampled, y_resampled = rus.fit_resample(X, y)
        
        # Convert back to DataFrame and Series
        X_balanced = pd.DataFrame(X_resampled, columns=X.columns)
        y_balanced = pd.Series(y_resampled, name=y.name)
        
        # Log results
        new_distribution = y_balanced.value_counts().sort_index()
        logger.info(f"RUS Results:")
        logger.info(f"  Original shape: {X.shape}")
        logger.info(f"  Balanced shape: {X_balanced.shape}")
        for class_val, count in new_distribution.items():
            class_name = 'Fraud' if class_val == 1 else 'Normal'
            logger.info(f"  {class_name} (Class {class_val}): {count:,}")
        
        return X_balanced, y_balanced
    
    def apply_smote(self, X: pd.DataFrame, y: pd.Series,
                   sampling_strategy: str = 'auto', k_neighbors: int = 5) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Apply SMOTE (Synthetic Minority Oversampling Technique) to balance the dataset
        
        Args:
            X: Features DataFrame
            y: Target Series
            sampling_strategy: Sampling strategy for SMOTE
            k_neighbors: Number of nearest neighbors for SMOTE
            
        Returns:
            Tuple of (balanced X, balanced y)
        """
        logger.info("Applying SMOTE...")
        
        # Check if we have enough samples for SMOTE
        minority_samples = y.value_counts().min()
        if minority_samples <= k_neighbors:
            k_neighbors = max(1, minority_samples - 1)
            logger.warning(f"Adjusted k_neighbors to {k_neighbors} due to limited minority samples")
        
        smote = SMOTE(
            sampling_strategy=sampling_strategy,
            random_state=self.random_state,
            k_neighbors=k_neighbors
        )
        
        try:
            X_resampled, y_resampled = smote.fit_resample(X, y)
            
            # Convert back to DataFrame and Series
            X_balanced = pd.DataFrame(X_resampled, columns=X.columns)
            y_balanced = pd.Series(y_resampled, name=y.name)
            
            # Log results
            new_distribution = y_balanced.value_counts().sort_index()
            logger.info(f"SMOTE Results:")
            logger.info(f"  Original shape: {X.shape}")
            logger.info(f"  Balanced shape: {X_balanced.shape}")
            for class_val, count in new_distribution.items():
                class_name = 'Fraud' if class_val == 1 else 'Normal'
                logger.info(f"  {class_name} (Class {class_val}): {count:,}")
            
            return X_balanced, y_balanced
            
        except Exception as e:
            logger.error(f"SMOTE failed: {str(e)}")
            logger.info("Falling back to Random Under-sampling")
            return self.apply_random_undersampling(X, y, sampling_strategy)
    
    def compare_balancing_methods(self, X: pd.DataFrame, y: pd.Series,
                                dataset_name: str) -> Dict:
        """
        Compare different balancing methods on the same dataset
        
        Args:
            X: Features DataFrame
            y: Target Series
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary with comparison results
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Comparing Balancing Methods - {dataset_name}")
        logger.info(f"{'='*60}")
        
        # Analyze original distribution
        original_analysis = self.analyze_class_distribution(y, dataset_name)
        
        results = {
            'dataset_name': dataset_name,
            'original': {
                'X': X,
                'y': y,
                'distribution': original_analysis
            },
            'balanced_datasets': {}
        }
        
        # Apply Random Under-sampling
        try:
            X_rus, y_rus = self.apply_random_undersampling(X.copy(), y.copy())
            rus_analysis = self.analyze_class_distribution(y_rus, f"{dataset_name}_RUS")
            
            results['balanced_datasets']['RUS'] = {
                'X': X_rus,
                'y': y_rus,
                'distribution': rus_analysis,
                'method': 'Random Under-sampling'
            }
        except Exception as e:
            logger.error(f"RUS failed for {dataset_name}: {str(e)}")
        
        # Apply SMOTE
        try:
            X_smote, y_smote = self.apply_smote(X.copy(), y.copy())
            smote_analysis = self.analyze_class_distribution(y_smote, f"{dataset_name}_SMOTE")
            
            results['balanced_datasets']['SMOTE'] = {
                'X': X_smote,
                'y': y_smote,
                'distribution': smote_analysis,
                'method': 'SMOTE'
            }
        except Exception as e:
            logger.error(f"SMOTE failed for {dataset_name}: {str(e)}")
        
        # Store results
        self.sampling_results[dataset_name] = results
        
        # Print comparison summary
        self._print_comparison_summary(results)
        
        return results
    
    def _print_comparison_summary(self, results: Dict) -> None:
        """
        Print a summary comparison of balancing methods
        
        Args:
            results: Results dictionary from compare_balancing_methods
        """
        dataset_name = results['dataset_name']
        logger.info(f"\n{'='*50}")
        logger.info(f"SUMMARY - {dataset_name}")
        logger.info(f"{'='*50}")
        
        # Original data
        orig_dist = results['original']['distribution']
        logger.info(f"Original Data:")
        logger.info(f"  Shape: {results['original']['X'].shape}")
        logger.info(f"  Imbalance Ratio: {orig_dist['imbalance_ratio']:.2f}:1")
        
        # Balanced datasets
        for method, data in results['balanced_datasets'].items():
            dist = data['distribution']
            logger.info(f"\n{method}:")
            logger.info(f"  Shape: {data['X'].shape}")
            logger.info(f"  Imbalance Ratio: {dist['imbalance_ratio']:.2f}:1")
            for class_val, count in dist['counts'].items():
                class_name = 'Fraud' if class_val == 1 else 'Normal'
                pct = dist['percentages'][class_val]
                logger.info(f"  {class_name}: {count:,} ({pct:.2f}%)")
    
    def get_balanced_datasets(self, dataset_name: str) -> Optional[Dict]:
        """
        Get balanced datasets for a specific dataset
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary with balanced datasets or None if not found
        """
        return self.sampling_results.get(dataset_name)
    
    def get_all_results(self) -> Dict:
        """
        Get all balancing results
        
        Returns:
            Dictionary with all results
        """
        return self.sampling_results
    
    def save_balanced_datasets(self, output_dir: str, dataset_name: str) -> None:
        """
        Save balanced datasets to files
        
        Args:
            output_dir: Directory to save the files
            dataset_name: Name of the dataset
        """
        if dataset_name not in self.sampling_results:
            logger.error(f"No results found for {dataset_name}")
            return
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        results = self.sampling_results[dataset_name]
        
        # Save original dataset
        orig_data = pd.concat([results['original']['X'], results['original']['y']], axis=1)
        orig_path = os.path.join(output_dir, f"{dataset_name}_original.csv")
        orig_data.to_csv(orig_path, index=False)
        logger.info(f"Saved original dataset to {orig_path}")
        
        # Save balanced datasets
        for method, data in results['balanced_datasets'].items():
            balanced_data = pd.concat([data['X'], data['y']], axis=1)
            balanced_path = os.path.join(output_dir, f"{dataset_name}_{method}_balanced.csv")
            balanced_data.to_csv(balanced_path, index=False)
            logger.info(f"Saved {method} balanced dataset to {balanced_path}")
    
    def create_balancing_report(self, output_file: Optional[str] = None) -> str:
        """
        Create a comprehensive report of all balancing results
        
        Args:
            output_file: Optional file path to save the report
            
        Returns:
            String containing the report
        """
        report = "=" * 80 + "\n"
        report += "FRAUD DETECTION DATASETS - CLASS BALANCING REPORT\n"
        report += "=" * 80 + "\n\n"
        
        for dataset_name, results in self.sampling_results.items():
            report += f"\n{'='*60}\n"
            report += f"Dataset: {dataset_name}\n"
            report += f"{'='*60}\n"
            
            # Original distribution
            orig = results['original']
            report += f"\nOriginal Distribution:\n"
            report += f"  Shape: {orig['X'].shape}\n"
            report += f"  Total Samples: {orig['distribution']['total_samples']:,}\n"
            report += f"  Imbalance Ratio: {orig['distribution']['imbalance_ratio']:.2f}:1\n"
            
            for class_val, count in orig['distribution']['counts'].items():
                pct = orig['distribution']['percentages'][class_val]
                class_name = 'Fraud' if class_val == 1 else 'Normal'
                report += f"    {class_name} (Class {class_val}): {count:,} ({pct:.2f}%)\n"
            
            # Balanced distributions
            for method, data in results['balanced_datasets'].items():
                dist = data['distribution']
                report += f"\n{method} Results:\n"
                report += f"  Shape: {data['X'].shape}\n"
                report += f"  Total Samples: {dist['total_samples']:,}\n"
                report += f"  Imbalance Ratio: {dist['imbalance_ratio']:.2f}:1\n"
                
                for class_val, count in dist['counts'].items():
                    pct = dist['percentages'][class_val]
                    class_name = 'Fraud' if class_val == 1 else 'Normal'
                    report += f"    {class_name} (Class {class_val}): {count:,} ({pct:.2f}%)\n"
                
                # Calculate reduction/increase
                orig_samples = orig['distribution']['total_samples']
                new_samples = dist['total_samples']
                change_pct = ((new_samples - orig_samples) / orig_samples) * 100
                report += f"  Sample Change: {change_pct:+.1f}%\n"
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
            logger.info(f"Balancing report saved to {output_file}")
        
        return report