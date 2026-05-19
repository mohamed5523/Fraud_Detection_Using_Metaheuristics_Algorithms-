# Fraud Detection System with Metaheuristic Optimization

![Build Status](https://img.shields.io/github/actions/workflow/status/mohamed5523/fraud_detection/main.yml?branch=main)
![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A modular, production-ready framework for fraud detection using advanced metaheuristic algorithms for feature selection and ensemble classifiers.

## Features

- **Advanced Feature Selection**: Implements 11 Binary Metaheuristic Algorithms (BPSO, BAT, BWOA, etc.) to select optimal features.
- **Class Imbalance Handling**: Integrated SMOTE and Random Under Sampling (RUS) to handle highly imbalanced datasets.
- **Modular Architecture**: Clean separation of concerns (Data, Models, Optimization, Visualization).
- **Comprehensive Visualization**: Automated dashboard generation for data exploration and model performance.
- **Production Ready**: Dockerized environment, CI/CD pipelines, and strict typing.

## Installation

### Prerequisites
- Python 3.9+
- Docker (optional)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/mohamed5523/Fraud_Detection.git
   cd Fraud_Detection
   ```

2. Install dependencies:
   ```bash
   pip install -e .[dev]
   ```
3. **Download Datasets**:
   - Download the required dataset files from [**Google Drive**](https://drive.google.com/drive/folders/1Vja3J6X_K07LN-uXsAilmcI11h1aGVYX?usp=sharing).
   - Place them inside the `datasets/` folder in the project root.



## Usage

### Command Line Interface

The system provides a CLI for both data exploration and model training.

**1. Data Exploration**
Generate distribution plots and correlation matrices.
```bash
python src/fraud_detection/main.py explore European
```

**2. Training & Optimization**
Train a model using a specific metaheuristic algorithm and classifier.
```bash
python src/fraud_detection/main.py train European --algorithm BPSO --classifier XGBoost --balance smote
```

### Supported Algorithms
- **Metaheuristics**: BPSO, BAT, BWOA, BGOA, BHHO, etc.
- **Classifiers**: XGBoost, KNN, DecisionTree.

## Project Structure

```
├── datasets/           # Dataset files (European, Paysim, etc.)
├── results/            # Output plots, metrics, and models
├── logs/               # Execution logs
├── src/
│   └── fraud_detection/
│       ├── config.py           # Central configuration
│       ├── main.py             # CLI Entry point
│       ├── data/               # Data loading & preprocessing
│       ├── models/             # Classifiers & Metrics
│       ├── optimization/       # Metaheuristic algorithms
│       └── visualization/      # Plotting utilities
├── tests/                      # Unit tests
├── pyproject.toml              # Dependencies
├── requirements.txt            # Frozen dependencies
├── Dockerfile                  # Container config
└── .github/workflows/          # CI/CD
```



## License

Distributed under the MIT License. See `LICENSE` for more information.
