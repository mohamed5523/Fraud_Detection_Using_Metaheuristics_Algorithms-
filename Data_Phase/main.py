"""
Main script for fraud detection data preprocessing, visualization, and balancing
"""

import pandas as pd
import numpy as np
import logging
import warnings
from pathlib import Path
import sys
import os

# Add project modules to path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

# Import project modules
from config.config import *
from data.data_loader import DataLoader
from data.preprocessor import DataPreprocessor
from visualization.visualizer import DataVisualizer
from balancing.balancer import DataBalancer

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(OUTPUT_DIR / 'fraud_detection_pipeline.log')
    ]
)

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


def main():
    """Main function to run the complete fraud detection data pipeline"""
    
    logger.info("="*80)
    logger.info("FRAUD DETECTION DATA PROCESSING PIPELINE")
    logger.info("="*80)
    
    # Initialize components
    data_loader = DataLoader(DATA_DIR)
    preprocessor = DataPreprocessor(random_state=RANDOM_STATE)
    visualizer = DataVisualizer(
        figsize=FIGURE_SIZE,
        save_path=VISUALIZATIONS_DIR if SAVE_PLOTS else None
    )
    balancer = DataBalancer(random_state=RANDOM_STATE)
    
    # Store processed datasets
    processed_datasets = {}
    datasets_info = {}
    
    logger.info("\n" + "="*60)
    logger.info("STEP 1: LOADING DATASETS")
    logger.info("="*60)
    
    # Load all datasets
    loaded_datasets = data_loader.load_all_datasets(DATASETS)
    
    if not loaded_datasets:
        logger.error("No datasets were loaded successfully!")
        return
    
    logger.info(f"Successfully loaded {len(loaded_datasets)} datasets")
    
    # Explore datasets
    data_loader.explore_all_datasets()
    
    logger.info("\n" + "="*60)
    logger.info("STEP 2: INITIAL VISUALIZATION")
    logger.info("="*60)
    
    # Create summary dashboard for raw datasets
    raw_datasets_info = {}
    for name, df in loaded_datasets.items():
        raw_datasets_info[name] = {
            'shape': df.shape,
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2
        }
    
    visualizer.create_summary_dashboard(raw_datasets_info, save=SAVE_PLOTS)
    
    # Visualize each dataset
    for name, df in loaded_datasets.items():
        logger.info(f"\nVisualizing dataset: {name}")
        config = DATASETS[name]
        target_col = config['target_column']
        
        if target_col in df.columns:
            # Class distribution
            visualizer.plot_class_distribution(df[target_col], name, save=SAVE_PLOTS)
            
            # Missing values heatmap
            visualizer.plot_missing_values_heatmap(df, name, save=SAVE_PLOTS)
            
            # Correlation matrix (for numerical columns)
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            if len(numerical_cols) > 5:  # Only if we have enough numerical columns
                visualizer.plot_correlation_matrix(df, name, save=SAVE_PLOTS)
        
        datasets_info[name] = data_loader.get_dataset_info(df, name)
    
    logger.info("\n" + "="*60)
    logger.info("STEP 3: DATA PREPROCESSING")
    logger.info("="*60)
    
    # Process each dataset
    for name, df in loaded_datasets.items():
        logger.info(f"\n{'='*40}")
        logger.info(f"Processing dataset: {name}")
        logger.info(f"{'='*40}")
        
        config = DATASETS[name]
        
        try:
            # Preprocess the dataset
            X_processed, y_processed = preprocessor.preprocess_dataset(df, config, name)
            
            if y_processed is not None:
                processed_datasets[name] = {
                    'X': X_processed,
                    'y': y_processed,
                    'config': config
                }
                
                logger.info(f"Successfully processed {name}: {X_processed.shape}")
                
                # Post-processing visualizations
                if len(X_processed.columns) > 0:
                    # Feature importance (correlation with target)
                    visualizer.plot_feature_importance(X_processed, y_processed, name, save=SAVE_PLOTS)
                    
                    # Distribution comparison for top features
                    top_features = X_processed.corrwith(y_processed).abs().nlargest(6).index.tolist()
                    if top_features:
                        visualizer.plot_distribution_comparison(
                            X_processed, y_processed, top_features, name, save=SAVE_PLOTS
                        )
                        
                        # Boxplot comparison
                        visualizer.plot_boxplot_comparison(
                            X_processed, y_processed, top_features, name, save=SAVE_PLOTS
                        )
            else:
                logger.error(f"Target variable not found for {name}")
                
        except Exception as e:
            logger.error(f"Error processing {name}: {str(e)}")
            continue
    
    logger.info("\n" + "="*60)
    logger.info("STEP 4: CLASS BALANCING")
    logger.info("="*60)
    
    # Apply balancing techniques to each processed dataset
    all_balancing_results = {}
    
    for name, data in processed_datasets.items():
        logger.info(f"\n{'='*40}")
        logger.info(f"Balancing dataset: {name}")
        logger.info(f"{'='*40}")
        
        X = data['X']
        y = data['y']
        
        try:
            # Compare balancing methods
            balancing_results = balancer.compare_balancing_methods(X, y, name)
            all_balancing_results[name] = balancing_results
            
            # Visualize class distributions after balancing
            original_y = balancing_results['original']['y']
            visualizer.plot_class_distribution(original_y, f"{name}_Original", save=SAVE_PLOTS)
            
            for method, balanced_data in balancing_results['balanced_datasets'].items():
                balanced_y = balanced_data['y']
                visualizer.plot_class_distribution(
                    balanced_y, f"{name}_{method}", save=SAVE_PLOTS
                )
            
            # Save balanced datasets
            balancer.save_balanced_datasets(str(BALANCED_DATA_DIR), name)
            
        except Exception as e:
            logger.error(f"Error balancing {name}: {str(e)}")
            continue
    
    logger.info("\n" + "="*60)
    logger.info("STEP 5: GENERATING REPORTS")
    logger.info("="*60)
    
    # Generate data exploration report
    try:
        data_report = data_loader.generate_data_report(OUTPUT_DIR / 'data_exploration_report.txt')
        logger.info("Data exploration report generated")
    except Exception as e:
        logger.error(f"Error generating data report: {str(e)}")
    
    # Generate balancing report
    try:
        balancing_report = balancer.create_balancing_report(OUTPUT_DIR / 'class_balancing_report.txt')
        logger.info("Class balancing report generated")
    except Exception as e:
        logger.error(f"Error generating balancing report: {str(e)}")
    
    logger.info("\n" + "="*60)
    logger.info("STEP 6: PIPELINE SUMMARY")
    logger.info("="*60)
    
    # Print final summary
    logger.info(f"\nPIPELINE COMPLETED SUCCESSFULLY!")
    logger.info(f"Datasets processed: {len(processed_datasets)}")
    
    for name, data in processed_datasets.items():
        logger.info(f"\n{name}:")
        logger.info(f"  Features: {data['X'].shape[1]}")
        logger.info(f"  Samples: {data['X'].shape[0]:,}")
        logger.info(f"  Target distribution: {data['y'].value_counts().to_dict()}")
        
        if name in all_balancing_results:
            balancing_data = all_balancing_results[name]
            logger.info(f"  Balancing methods applied: {list(balancing_data['balanced_datasets'].keys())}")
    
    logger.info(f"\nOutput files saved to: {OUTPUT_DIR}")
    logger.info(f"  - Processed data: {PROCESSED_DATA_DIR}")
    logger.info(f"  - Visualizations: {VISUALIZATIONS_DIR}")
    logger.info(f"  - Balanced datasets: {BALANCED_DATA_DIR}")
    
    return {
        'processed_datasets': processed_datasets,
        'balancing_results': all_balancing_results,
        'datasets_info': datasets_info
    }


if __name__ == "__main__":
    try:
        results = main()
        logger.info("\nPipeline execution completed!")
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}")
        raise