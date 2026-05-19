# 🛡️ FraudGuard AI: Fraud Detection with Metaheuristic Optimization

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg?style=for-the-badge&logo=python)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Libraries](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

**FraudGuard AI** is a state-of-the-art, modular framework designed to identify and analyze financial fraud patterns. By combining **advanced Machine Learning classifiers** (like XGBoost, KNN, and Decision Trees) with **11 Binary Metaheuristic Algorithms** for optimized feature selection, this project minimizes the feature space while maximizing detection accuracy.

It features a robust training & data pipeline alongside a **premium Streamlit web-based inference dashboard** for real-time transaction analysis.

---

## 🚀 Key Features

*   **🧬 11 Binary Metaheuristic Feature Selectors**: Implements advanced evolutionary and swarm algorithms to extract the most predictive features, reducing noise and computational cost:
    *   *BSSA* (Binary Salp Swarm Algorithm)
    *   *BPSO* (Binary Particle Swarm Optimization)
    *   *BWOA* (Binary Whale Optimization Algorithm)
    *   *BAT* (Binary Bat Algorithm)
    *   *BGOA* (Binary Grasshopper Optimization Algorithm)
    *   *BHHO* (Binary Harris Hawks Optimization)
    *   *BMOA* (Binary Mayfly Optimization Algorithm)
    *   *BSSA*, *BHGSO*, *BAVO*, *BPSO*, and more.
*   **🤖 State-of-the-Art Classifiers**: Integrated with highly tuned models, including **XGBoost**, **KNN**, and **Decision Trees**.
*   **⚖️ Imbalance Data Control**: Native handling of extremely skewed fraud datasets using **SMOTE** (Synthetic Minority Over-sampling Technique) and **RUS** (Random Under-Sampling).
*   **📊 Automatic Visualization Suite**: Generates stunning performance graphs automatically, including:
    *   *Precision-Recall Curves* and *ROC Curves*
    *   *Confusion Matrices*
    *   *Optimization Convergence Curves* (Fitness vs. Iteration)
    *   *Selected Features Distributions*
*   **🛡️ Premium Web Dashboard**: An interactive, real-time transaction scanner built with Streamlit and Plotly to perform live risk-evaluations and display an interactive fraud risk gauge.

---

## 📂 Project Architecture

The system is cleanly split into three unified phases:

```
├── app.py                    # Streamlit Premium Web Dashboard
├── utils.py                  # Streamlit utility functions & ML loader
├── requirements.txt          # Main library dependencies
├── app_data/                 # Production-serialized models & statistical data
│   ├── model.pkl             # Serialized optimal XGBoost (BSSA-optimized) model
│   └── data_info.json        # Categorical mappings and unique features data
│
├── Data_Phase/               # ── PHASE 1: DATA PREPARATION & BALANCING ──
│   ├── main.py               # Data CLI (Exploration, balancing, & pre-processing)
│   ├── config/               # Configuration settings for preprocessing
│   ├── data/                 # Processing engines & CSV loaders
│   ├── balancing/            # SMOTE, RUS, and Undersampling engines
│   └── visualization/        # Distribution and correlation plotters
│
└── Model_Phase/              # ── PHASE 2: OPTIMIZATION & TRAINING ──
    ├── Main_Model_Phase.py   # Training Pipeline CLI Entry point
    ├── config.py             # Fitness functions & training hyperparameters
    ├── classifiers.py        # ML wrapper definitions (XGBoost, DecisionTree, KNN)
    ├── metaheuristics_base.py# Base metaheuristic class definition
    ├── metaheuristics_algorithms.py # Implementations of 11 swarm algorithms
    ├── optimization_pipeline.py # Evaluates algorithms and trains optimal models
    ├── visualizations.py     # Plotting utilities for ROC/PR/Convergence
    └── results/              # Output directory for plots, metrics, & pickle models
```

---

## 🛠️ Installation & Setup

### Prerequisites
*   Python 3.9 or higher installed on your system.

### 1. Clone & Initialize Environment
Clone this repository to your local machine:
```bash
git clone https://github.com/mohamed5523/Fraud_Detection_Using_Metaheuristics_Algorithms-.git
cd Fraud_Detection_Using_Metaheuristics_Algorithms-
```

Create a virtual environment and install the required dependencies:
```bash
# Create venv
python -m venv venv

# Activate venv
# On Windows:
.\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 2. Download Datasets
*   Download the dataset files (e.g., `European.csv`, `paysim.csv`) from [**Google Drive Link**](https://drive.google.com/drive/folders/1Vja3J6X_K07LN-uXsAilmcI11h1aGVYX?usp=sharing).
*   Create a folder at `Data_Phase/datasets/` and place the `.csv` files inside it.

---

## 💻 Usage

### Phase 1: Data Exploration & Balancing CLI
Analyze and balance your dataset to prepare it for model training.

```bash
# Explore your dataset configurations and generate correlation matrices
python Data_Phase/main.py explore European

# Preprocess and balance the dataset using SMOTE
python Data_Phase/main.py train European --balance smote
```

### Phase 2: Swarm Optimization & Model Training
Execute the training pipeline to run metaheuristic feature selection, choose the best features, and train the classifier:

```bash
# Format:
# python Model_Phase/Main_Model_Phase.py --algorithm <ALG> --classifier <CLASS> --dataset <DATA>
# Example: Use Binary Salp Swarm Algorithm (BSSA) and XGBoost on European dataset
python Model_Phase/Main_Model_Phase.py --algorithm BSSA --classifier XGBoost --dataset European
```

Your trained model, feature lists, and generated charts (ROC, PR, Convergence, and Confusion Matrix) will be saved in `Model_Phase/results/`.

---

## 🖥️ Streamlit Web Dashboard (Inference Engine)

The dashboard allows security analysts and administrators to test the trained, serialized model by feeding real-time customer and transaction metadata details.

### How to Run:
Make sure you are in the root directory and your virtual environment is active, then execute:
```bash
streamlit run app.py
```

### Dashboard Core Functionality:
1.  **Customer Profile Input**: Set demographic attributes such as *Gender*, *Job Category*, *Birth Year*, and *City Population*.
2.  **Transaction Details Input**: Key in transaction parameters like *Amount ($)*, *Merchant Category*, *Merchant Zipcode*, and *Transaction Time*.
3.  **Real-Time Assessment**: Press **Run Analysis** to instantly process features through the BSSA-optimized XGBoost model.
4.  **Visual Risk Gauge**: Renders an interactive Plotly dial indicating a color-coded threat level:
    *   🟢 **Low Risk** (< 50%): Safe to proceed.
    *   🟡 **Moderate Risk** (50% - 85%): Review suggested.
    *   🔴 **High Risk** (> 85%): Strong match for fraudulent activity.
5.  **Technical Feature Vector**: Toggle a detailed collapse bar to inspect the exact scaled vector layout sent to the model.

---

## 📜 License
This project is licensed under the MIT License. See the `LICENSE` file for details.
