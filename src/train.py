"""
Training module for NewsLens AI Classifier.
Trains SVM and XGBoost models on TF-IDF and BERT embeddings.
"""
import time
import joblib
from pathlib import Path
from typing import Union, Optional
import numpy as np
from scipy import sparse
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src.config import MODELS_CONFIG, PATHS
from src.data_loader import load_embedding, load_labels


def train_svm(
    X_train: Union[sparse.csr_matrix, np.ndarray],
    y_train: np.ndarray,
    save_path: Optional[Path] = None
) -> SVC:
    """
    Train SVM classifier.
    
    Args:
        X_train: Training embeddings
        y_train: Training labels
        save_path: Path to save model
    
    Returns:
        Trained SVM model
    """
    config = MODELS_CONFIG['svm']
    
    print(f"\n🔵 Training SVM (kernel={config['kernel']})...")
    start_time = time.time()
    
    model = SVC(
        kernel=config['kernel'],
        class_weight=config['class_weight'],
        probability=config['probability'],
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    print(f"✅ SVM trained in {training_time:.2f} seconds")
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, save_path)
        print(f"💾 Model saved to {save_path}")
    
    return model


def train_xgboost(
    X_train: Union[sparse.csr_matrix, np.ndarray],
    y_train: np.ndarray,
    save_path: Optional[Path] = None
) -> XGBClassifier:
    """
    Train XGBoost classifier.
    
    Args:
        X_train: Training embeddings
        y_train: Training labels
        save_path: Path to save model
    
    Returns:
        Trained XGBoost model
    """
    config = MODELS_CONFIG['xgboost']
    
    print(f"\n🟢 Training XGBoost (n_estimators={config['n_estimators']})...")
    start_time = time.time()
    
    model = XGBClassifier(
        n_estimators=config['n_estimators'],
        max_depth=config['max_depth'],
        n_jobs=config['n_jobs'],
        random_state=42,
        eval_metric='mlogloss'
    )
    
    model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    print(f"✅ XGBoost trained in {training_time:.2f} seconds")
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, save_path)
        print(f"💾 Model saved to {save_path}")
    
    return model


def train_all_models(
    embeddings: dict,
    labels: dict,
    save: bool = True
) -> dict:
    """
    Train all 4 model combinations.
    
    Args:
        embeddings: Dictionary with 'tfidf' and 'bert' embeddings
        labels: Dictionary with 'train', 'val', 'test' labels
        save: Whether to save models
    
    Returns:
        Dictionary with all trained models
    """
    print("="*60)
    print("🎯 Training All Models")
    print("="*60)
    
    models = {}
    
    # 1. TF-IDF + SVM
    if 'tfidf' in embeddings:
        models['tfidf_svm'] = train_svm(
            embeddings['tfidf']['train'],
            labels['train'],
            save_path=PATHS['models'] / 'tfidf_svm.pkl' if save else None
        )
    
    # 2. TF-IDF + XGBoost
    if 'tfidf' in embeddings:
        models['tfidf_xgb'] = train_xgboost(
            embeddings['tfidf']['train'],
            labels['train'],
            save_path=PATHS['models'] / 'tfidf_xgb.pkl' if save else None
        )
    
    # 3. BERT + SVM
    if 'bert' in embeddings:
        models['bert_svm'] = train_svm(
            embeddings['bert']['train'],
            labels['train'],
            save_path=PATHS['models'] / 'bert_svm.pkl' if save else None
        )
    
    # 4. BERT + XGBoost
    if 'bert' in embeddings:
        models['bert_xgb'] = train_xgboost(
            embeddings['bert']['train'],
            labels['train'],
            save_path=PATHS['models'] / 'bert_xgb.pkl' if save else None
        )
    
    print("\n" + "="*60)
    print("✅ All Models Trained!")
    print("="*60)
    
    return models


def load_trained_models() -> dict:
    """Load all trained models from disk."""
    models = {}
    model_files = {
        'tfidf_svm': PATHS['models'] / 'tfidf_svm.pkl',
        'tfidf_xgb': PATHS['models'] / 'tfidf_xgb.pkl',
        'bert_svm': PATHS['models'] / 'bert_svm.pkl',
        'bert_xgb': PATHS['models'] / 'bert_xgb.pkl'
    }
    
    for name, path in model_files.items():
        if path.exists():
            models[name] = joblib.load(path)
            print(f"✅ Loaded {name}")
        else:
            print(f"⚠️  Model not found: {path}")
    
    return models

