"""
Configuration module for NewsLens AI Classifier.
Centralizes all project configurations.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# 1. Data Configuration
DATA_CONFIG = {
    'test_size': 0.2,           # First split: 20% for test
    'val_size': 0.25,            # Second split: 25% of remaining for validation
    'stratify': True,            # CRITICAL: Maintain original distribution
    'random_state': 42           # Reproducibility
}

# 2. Features and Embeddings
FEATURE_CONFIG = {
    'tfidf': {
        'max_features': 20000,
        'ngram_range': (1, 2),
        'storage': 'sparse_npz'
    },
    'bert': {
        'model_name': 'neuralmind/bert-base-portuguese-cased',
        'implementation': 'sentence-transformers',  # Defined library
        'pooling': 'mean',                          # Defined strategy
        'batch_size': 32,
        'storage': 'dense_npy'
    }
}

# 3. Models Configuration
MODELS_CONFIG = {
    'svm': {
        'kernel': 'linear',
        'C': 1.0,  # Will be optimized
        'gamma': 'scale',  # Will be optimized
        'class_weight': 'balanced',
        'probability': True
    },
    'xgboost': {
        'n_estimators': 100,  # Will be optimized
        'max_depth': 6,  # Will be optimized
        'learning_rate': 0.1,  # Will be optimized
        'subsample': 1.0,  # Will be optimized
        'colsample_bytree': 1.0,  # Will be optimized
        'n_jobs': -1
    }
}

# 3.1. Optimized Hyperparameters (loaded from best_hyperparameters.json if available)
# These will be used after running optimization
OPTIMIZED_HYPERPARAMS = None

# 4. LLM API Configuration (Cost Control)
LLM_CONFIG = {
    'provider': 'groq',
    'model': 'llama-3.3-70b-versatile',  # Updated to available model
    'max_examples_differential': 10,  # Hard limit
    'api_key': os.getenv('GROQ_API_KEY')  # Environment variable
}

# 5. Paths Configuration
PATHS = {
    'data_raw': PROJECT_ROOT / 'data' / 'raw',
    'data_processed': PROJECT_ROOT / 'data' / 'processed',
    'data_embeddings': PROJECT_ROOT / 'data' / 'embeddings',
    'data_novos': PROJECT_ROOT / 'data' / 'novos',
    'logs': PROJECT_ROOT / 'logs',
    'models': PROJECT_ROOT / 'models'
}

# Ensure directories exist
for path in PATHS.values():
    path.mkdir(parents=True, exist_ok=True)

