"""
K-fold Cross-Validation module for robust model evaluation.
Implements stratified K-fold CV for all model combinations.
"""
import numpy as np
from typing import Union, Dict, List, Tuple
from scipy import sparse
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, make_scorer
import pandas as pd
import time

from src.config import MODELS_CONFIG


def stratified_kfold_cv(
    X: Union[sparse.csr_matrix, np.ndarray],
    y: np.ndarray,
    model_type: str,
    n_splits: int = 5,
    hyperparams: Dict = None,
    random_state: int = 42
) -> Dict:
    """
    Perform stratified K-fold cross-validation.
    
    Args:
        X: Feature matrix (embeddings)
        y: Labels
        model_type: 'svm' or 'xgboost'
        n_splits: Number of folds (default: 5)
        hyperparams: Optional hyperparameters to override defaults
        random_state: Random seed for reproducibility
    
    Returns:
        Dictionary with CV results (mean, std, scores per fold)
    """
    print(f"\n📊 Performing {n_splits}-fold Cross-Validation for {model_type.upper()}...")
    
    # Create model with hyperparameters
    if model_type == 'svm':
        config = MODELS_CONFIG['svm'].copy()
        if hyperparams:
            config.update(hyperparams)
        model = SVC(
            kernel=config.get('kernel', 'linear'),
            C=config.get('C', 1.0),
            gamma=config.get('gamma', 'scale'),
            class_weight=config.get('class_weight', 'balanced'),
            probability=config.get('probability', True),
            random_state=random_state
        )
    elif model_type == 'xgboost':
        config = MODELS_CONFIG['xgboost'].copy()
        if hyperparams:
            config.update(hyperparams)
        model = XGBClassifier(
            n_estimators=config.get('n_estimators', 100),
            max_depth=config.get('max_depth', 6),
            learning_rate=config.get('learning_rate', 0.1),
            subsample=config.get('subsample', 1.0),
            colsample_bytree=config.get('colsample_bytree', 1.0),
            n_jobs=config.get('n_jobs', -1),
            random_state=random_state,
            eval_metric='mlogloss'
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    # Create stratified K-fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # F1-macro scorer
    f1_macro_scorer = make_scorer(f1_score, average='macro')
    
    # Perform cross-validation
    start_time = time.time()
    cv_scores = cross_val_score(
        model,
        X,
        y,
        cv=skf,
        scoring=f1_macro_scorer,
        n_jobs=-1
    )
    cv_time = time.time() - start_time
    
    results = {
        'mean_f1': float(np.mean(cv_scores)),
        'std_f1': float(np.std(cv_scores)),
        'scores_per_fold': cv_scores.tolist(),
        'n_splits': n_splits,
        'cv_time': cv_time,
        'hyperparams': hyperparams or config
    }
    
    print(f"✅ CV completed in {cv_time:.2f}s")
    print(f"   Mean F1-Macro: {results['mean_f1']:.4f} (+/- {results['std_f1']:.4f})")
    
    return results


def cv_all_models(
    embeddings: Dict[str, Union[sparse.csr_matrix, np.ndarray]],
    labels: np.ndarray,
    n_splits: int = 5,
    hyperparams: Dict[str, Dict] = None,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Perform K-fold CV for all model combinations.
    
    Args:
        embeddings: Dictionary with 'tfidf' and/or 'bert' embeddings
        labels: Labels array
        n_splits: Number of folds
        hyperparams: Optional hyperparameters per model (e.g., {'tfidf_svm': {'C': 10}})
        random_state: Random seed
    
    Returns:
        DataFrame with CV results for all models
    """
    print("="*60)
    print("🔄 K-Fold Cross-Validation for All Models")
    print("="*60)
    
    results = []
    hyperparams = hyperparams or {}
    
    # 1. TF-IDF + SVM
    if 'tfidf' in embeddings:
        cv_result = stratified_kfold_cv(
            embeddings['tfidf'],
            labels,
            model_type='svm',
            n_splits=n_splits,
            hyperparams=hyperparams.get('tfidf_svm'),
            random_state=random_state
        )
        results.append({
            'Model': 'TF-IDF + SVM',
            'Mean F1-Macro': cv_result['mean_f1'],
            'Std F1-Macro': cv_result['std_f1'],
            'CV Time (s)': cv_result['cv_time'],
            'N Folds': n_splits
        })
    
    # 2. TF-IDF + XGBoost
    if 'tfidf' in embeddings:
        cv_result = stratified_kfold_cv(
            embeddings['tfidf'],
            labels,
            model_type='xgboost',
            n_splits=n_splits,
            hyperparams=hyperparams.get('tfidf_xgb'),
            random_state=random_state
        )
        results.append({
            'Model': 'TF-IDF + XGBoost',
            'Mean F1-Macro': cv_result['mean_f1'],
            'Std F1-Macro': cv_result['std_f1'],
            'CV Time (s)': cv_result['cv_time'],
            'N Folds': n_splits
        })
    
    # 3. BERT + SVM
    if 'bert' in embeddings:
        cv_result = stratified_kfold_cv(
            embeddings['bert'],
            labels,
            model_type='svm',
            n_splits=n_splits,
            hyperparams=hyperparams.get('bert_svm'),
            random_state=random_state
        )
        results.append({
            'Model': 'BERT + SVM',
            'Mean F1-Macro': cv_result['mean_f1'],
            'Std F1-Macro': cv_result['std_f1'],
            'CV Time (s)': cv_result['cv_time'],
            'N Folds': n_splits
        })
    
    # 4. BERT + XGBoost
    if 'bert' in embeddings:
        cv_result = stratified_kfold_cv(
            embeddings['bert'],
            labels,
            model_type='xgboost',
            n_splits=n_splits,
            hyperparams=hyperparams.get('bert_xgb'),
            random_state=random_state
        )
        results.append({
            'Model': 'BERT + XGBoost',
            'Mean F1-Macro': cv_result['mean_f1'],
            'Std F1-Macro': cv_result['std_f1'],
            'CV Time (s)': cv_result['cv_time'],
            'N Folds': n_splits
        })
    
    df_results = pd.DataFrame(results)
    
    print("\n" + "="*60)
    print("📊 Cross-Validation Results Summary")
    print("="*60)
    print(df_results.to_string(index=False))
    print("="*60)
    
    return df_results

