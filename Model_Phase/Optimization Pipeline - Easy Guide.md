# Optimization Pipeline - Easy Guide

## Overview
This file orchestrates the entire feature selection process using metaheuristic algorithms. Think of it as a **conductor coordinating an orchestra** - it brings together algorithms, classifiers, and datasets to find the best features.

---

## What Problem Does This Solve?

Imagine you have a dataset with 100 features, but only 20 are actually useful. This pipeline:
1. Uses smart algorithms (like genetic algorithms) to search for the best features
2. Tests different combinations systematically
3. Measures how well each combination performs
4. Finds the winning combination

---

## Architecture: Two Main Classes

### 1. `OptimizationPipeline` - Single Dataset Manager
### 2. `MultiDatasetPipeline` - Multiple Datasets Manager

---

## Class 1: OptimizationPipeline

### Purpose
Handles all experiments for ONE dataset. Runs different algorithms and classifiers to find the best feature subset.

### Initialization

```python
pipeline = OptimizationPipeline(dataset_name, data_dict)
```

**What you need to provide:**
- `dataset_name`: A name for your dataset (e.g., "heart_disease")
- `data_dict`: Dictionary containing:
  ```python
  {
      'X_train': training_features,
      'X_test': test_features,
      'y_train': training_labels,
      'y_test': test_labels,
      'feature_names': list_of_feature_names,
      'n_features': number_of_features
  }
  ```

**What it stores:**
- All your training and test data
- Feature names and count
- Results from all experiments

---

## Key Method 1: `run_single_experiment()`

### What It Does
Runs ONE complete experiment: picks an algorithm, picks a classifier, finds best features, evaluates performance.

### The Complete Workflow

```
Start → Initialize Classifier → Create Fitness Function → 
Run Metaheuristic → Get Best Features → Train Final Model → 
Make Predictions → Calculate Metrics → Return Results
```

### Step-by-Step Breakdown

#### Step 1: Initialize Classifier
```python
classifier_wrapper = ClassifierWrapper(classifier_name, X_train, y_train)
```
- Creates a wrapper around your ML model (SVM, Random Forest, etc.)
- Prepares it to work with the optimization process

#### Step 2: Get Fitness Function
```python
fitness_func = get_fitness_function(classifier_wrapper)
```
- Creates a scoring function that the algorithm will try to maximize
- Usually combines accuracy with feature reduction

#### Step 3: Initialize Metaheuristic
```python
algorithm = get_algorithm(
    algorithm_name,
    n_features,
    population_size,
    max_iterations
)
```
- Sets up the search algorithm (PSO, GA, GWO, etc.)
- Configures population size and iterations from config

#### Step 4: Run Optimization
```python
best_solution, best_fitness, convergence_curve = algorithm.optimize(fitness_func)
```
- **best_solution**: Binary array (e.g., [1,0,1,1,0]) where 1 = selected feature
- **best_fitness**: The score of this solution
- **convergence_curve**: How fitness improved over iterations

#### Step 5: Extract Selected Features
```python
selected_features = best_solution  # [1,0,1,0,1]
n_selected = np.sum(selected_features)  # Count of 1s
selected_indices = np.where(selected_features == 1)[0]  # Positions of 1s
```

#### Step 6: Train Final Model
```python
model = classifier_wrapper.train(selected_features)
```
- Trains the classifier using ONLY the selected features

#### Step 7: Make Predictions
```python
y_pred = classifier_wrapper.predict(X_test, selected_features)
y_pred_proba = classifier_wrapper.predict_proba(X_test, selected_features)
```
- Tests on unseen data
- Gets both labels and probabilities

#### Step 8: Calculate Metrics
```python
metrics_calc = MetricsCalculator(y_test, y_pred, y_pred_proba)
metrics = metrics_calc.calculate_all_metrics()
```
- Computes accuracy, F1, ROC-AUC, etc.

### What It Returns

A comprehensive dictionary containing:

```python
{
    # Experiment Info
    'dataset': 'heart_disease',
    'algorithm': 'PSO',
    'classifier': 'SVM',
    
    # Feature Selection Results
    'n_features_total': 100,
    'n_features_selected': 25,
    'feature_selection_ratio': 0.25,
    'selected_features': [1,0,1,0,...],  # Binary array
    'selected_feature_indices': [0,2,5,7,...],  # Positions
    'selected_feature_names': ['age', 'blood_pressure', ...],
    
    # Optimization Info
    'best_fitness': 0.95,
    'convergence_curve': [0.7, 0.8, 0.85, ...],
    'optimization_time': 45.3,  # seconds
    
    # Model and Predictions
    'model': trained_model_object,
    'y_pred': predictions,
    'y_pred_proba': probabilities,
    
    # All Metrics (from MetricsCalculator)
    'accuracy': 0.92,
    'f1_score': 0.88,
    'roc_auc': 0.94,
    # ... and many more
}
```

---

## Key Method 2: `run_all_combinations()`

### What It Does
Runs multiple experiments testing all combinations of algorithms and classifiers.

### Example
If you have:
- 3 algorithms: PSO, GA, GWO
- 2 classifiers: SVM, Random Forest

It will run **6 experiments** (3 × 2):
1. PSO + SVM
2. PSO + Random Forest
3. GA + SVM
4. GA + Random Forest
5. GWO + SVM
6. GWO + Random Forest

### Usage

```python
# Run all configured algorithms and classifiers
results = pipeline.run_all_combinations()

# Or specify which to run
results = pipeline.run_all_combinations(
    algorithms=['PSO', 'GA'],
    classifiers=['SVM', 'RandomForest']
)
```

### What Happens

1. **Counts total experiments**: For progress tracking
2. **Loops through all combinations**: Outer loop = classifiers, inner loop = algorithms
3. **Runs each experiment**: Calls `run_single_experiment()`
4. **Stores results**: Saves with key like "PSO_SVM"
5. **Error handling**: Catches and logs failures without crashing

### Progress Logging

```
##########################################################
Starting 6 experiments for heart_disease
##########################################################

Experiment 1/6
======================================================================
Running: PSO - SVM - heart_disease
======================================================================
Starting optimization...
Optimization completed in 45.23 seconds
Selected 25/100 features
...
```

---

## Key Method 3: `get_results_summary()`

### What It Does
Creates a clean DataFrame summarizing all experiment results.

### Output Format

| Experiment | Dataset | Algorithm | Classifier | Features_Selected | Accuracy | F1_Score | ... |
|------------|---------|-----------|------------|-------------------|----------|----------|-----|
| PSO_SVM | heart | PSO | SVM | 25/100 | 0.92 | 0.88 | ... |
| GA_RF | heart | GA | RF | 30/100 | 0.90 | 0.85 | ... |

### Columns Included

**Basic Info:**
- Experiment name
- Dataset name
- Algorithm and Classifier used

**Feature Selection:**
- Features_Selected (count)
- Features_Total (total available)
- Selection_Ratio (percentage selected)
- Best_Fitness (optimization score)

**Performance Metrics:**
- Accuracy, Balanced_Accuracy
- Precision, Recall, F1_Score
- MCC, Kappa
- Sensitivity, Specificity (if binary classification)
- ROC_AUC, PR_AUC (if probabilities available)

**Timing:**
- Optimization_Time (seconds)

---

## Key Method 4: `get_best_result()`

### What It Does
Finds the best experiment based on a specific metric.

### Usage

```python
# Find best by F1 score (default)
best = pipeline.get_best_result()

# Find best by accuracy
best = pipeline.get_best_result(metric='accuracy')

# Find best by ROC-AUC
best = pipeline.get_best_result(metric='roc_auc')
```

### Returns
The complete results dictionary for the winning experiment.

---

## Class 2: MultiDatasetPipeline

### Purpose
Manages experiments across MULTIPLE datasets simultaneously.

### When to Use
- Comparing algorithm performance across different problems
- Running comprehensive benchmarks
- Publishing research with multiple datasets

### Initialization

```python
datasets = {
    'heart_disease': {
        'data': heart_data_dict
    },
    'diabetes': {
        'data': diabetes_data_dict
    },
    'cancer': {
        'data': cancer_data_dict
    }
}

multi_pipeline = MultiDatasetPipeline(datasets)
```

---

## Key Method: `run_all_datasets()`

### What It Does
For each dataset:
1. Creates an OptimizationPipeline
2. Runs all algorithm/classifier combinations
3. Stores all results

### Usage

```python
all_results = multi_pipeline.run_all_datasets(
    algorithms=['PSO', 'GA'],
    classifiers=['SVM', 'RandomForest']
)
```

### Result Structure

```python
{
    'heart_disease': {
        'PSO_SVM': {...},
        'PSO_RandomForest': {...},
        'GA_SVM': {...},
        'GA_RandomForest': {...}
    },
    'diabetes': {
        'PSO_SVM': {...},
        ...
    },
    'cancer': {
        'PSO_SVM': {...},
        ...
    }
}
```

---

## Key Method: `get_combined_summary()`

### What It Does
Combines summaries from all datasets into one large DataFrame.

### Output Example

| Experiment | Dataset | Algorithm | Classifier | Accuracy | F1_Score | ... |
|------------|---------|-----------|------------|----------|----------|-----|
| PSO_SVM | heart | PSO | SVM | 0.92 | 0.88 | ... |
| PSO_SVM | diabetes | PSO | SVM | 0.85 | 0.82 | ... |
| PSO_SVM | cancer | PSO | SVM | 0.95 | 0.93 | ... |
| GA_RF | heart | GA | RF | 0.90 | 0.86 | ... |

### Use Cases
- Create comparison tables for papers
- Identify which algorithm works best across datasets
- Export to CSV for further analysis

---

## Key Method: `get_best_per_dataset()`

### What It Does
Finds the best experiment for each dataset separately.

### Usage

```python
best_results = multi_pipeline.get_best_per_dataset(metric='f1_score')
```

### Returns

```python
{
    'heart_disease': {
        'algorithm': 'PSO',
        'classifier': 'SVM',
        'f1_score': 0.92,
        ...
    },
    'diabetes': {
        'algorithm': 'GA',
        'classifier': 'RandomForest',
        'f1_score': 0.88,
        ...
    }
}
```

---

## Complete Usage Example

### Single Dataset

```python
# 1. Prepare your data
data_dict = {
    'X_train': X_train,
    'X_test': X_test,
    'y_train': y_train,
    'y_test': y_test,
    'feature_names': feature_names,
    'n_features': X_train.shape[1]
}

# 2. Create pipeline
pipeline = OptimizationPipeline('my_dataset', data_dict)

# 3. Run experiments
results = pipeline.run_all_combinations(
    algorithms=['PSO', 'GA', 'GWO'],
    classifiers=['SVM', 'RandomForest']
)

# 4. Get summary
summary_df = pipeline.get_results_summary()
print(summary_df)

# 5. Find best result
best = pipeline.get_best_result(metric='f1_score')
print(f"Best: {best['algorithm']} + {best['classifier']}")
print(f"F1 Score: {best['f1_score']:.4f}")
print(f"Selected {best['n_features_selected']} features")
```

### Multiple Datasets

```python
# 1. Prepare multiple datasets
datasets = {
    'dataset1': {'data': data_dict1},
    'dataset2': {'data': data_dict2},
    'dataset3': {'data': data_dict3}
}

# 2. Create multi-pipeline
multi_pipeline = MultiDatasetPipeline(datasets)

# 3. Run all
all_results = multi_pipeline.run_all_datasets()

# 4. Get combined summary
combined_df = multi_pipeline.get_combined_summary()
combined_df.to_csv('all_results.csv')

# 5. Find best for each dataset
best_per_dataset = multi_pipeline.get_best_per_dataset()
for dataset, result in best_per_dataset.items():
    print(f"{dataset}: {result['algorithm']} + {result['classifier']}")
```

---

## What Gets Logged

The pipeline provides detailed logging at every step:

### Experiment Start
```
======================================================================
Running: PSO - SVM - heart_disease
======================================================================
```

### Optimization Progress
```
Starting optimization...
Optimization completed in 45.23 seconds
Selected 25/100 features
```

### Model Training
```
Training final model...
Calculating metrics...
```

### Results Summary
```
CLASSIFICATION METRICS SUMMARY
Accuracy: 0.9200
F1-Score: 0.8800
...
```

### Completion
```
Experiment completed successfully!
Best Fitness: 0.9500
Test Accuracy: 0.9200
Test F1-Score: 0.8800
```

---

## Error Handling

The pipeline is robust:

```python
try:
    results = self.run_single_experiment(algorithm_name, classifier_name)
    self.results[key] = results
except Exception as e:
    logger.error(f"Error in experiment: {str(e)}")
    traceback.print_exc()
    # Continues to next experiment instead of crashing
```

**Benefits:**
- One failed experiment doesn't stop the entire pipeline
- Full error traceback for debugging
- Remaining experiments still run

---

## Key Design Patterns

### 1. **Separation of Concerns**
- Classifier logic → `ClassifierWrapper`
- Metrics → `MetricsCalculator`
- Algorithms → `get_algorithm()`
- Pipeline → Orchestrates everything

### 2. **Configuration-Driven**
- All parameters come from `config` module
- Easy to modify without changing code

### 3. **Comprehensive Logging**
- Progress tracking at every step
- Helps with debugging and monitoring

### 4. **Flexible Execution**
- Run one experiment or all combinations
- Run one dataset or multiple datasets
- Choose specific algorithms/classifiers

---

## Summary

### OptimizationPipeline
- **Purpose**: Manage experiments for one dataset
- **Key Methods**: 
  - `run_single_experiment()` - One algorithm + classifier
  - `run_all_combinations()` - Test all combinations
  - `get_results_summary()` - Summary table
  - `get_best_result()` - Find winner

### MultiDatasetPipeline
- **Purpose**: Manage experiments across datasets
- **Key Methods**:
  - `run_all_datasets()` - Run everything
  - `get_combined_summary()` - Combined results table
  - `get_best_per_dataset()` - Best for each dataset

### The Flow
```
Data → OptimizationPipeline → 
Algorithm + Classifier → Feature Selection → 
Training → Testing → Metrics → Results
```

This pipeline is your **automated experimentation framework** - set it up once, and it systematically tests everything for you!