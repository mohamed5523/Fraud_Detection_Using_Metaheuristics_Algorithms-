import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'src'))

try:
    import fraud_detection
    from fraud_detection.config import DATASETS
    from fraud_detection.data.loader import DataLoader
    from fraud_detection.models.classifiers import ClassifierWrapper
    from fraud_detection.optimization.algorithms import BPSO
    print("SUCCESS: All modules imported correctly.")
except ImportError as e:
    print(f"FAILURE: {e}")
    sys.exit(1)
