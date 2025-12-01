"""
Hyperparameter Optimization using Optuna (Bayesian Optimization).
Optimizes SVM and XGBoost hyperparameters for all embeddings.
"""
import optuna
import numpy as np
from typing import Union, Dict, Optional
from scipy import sparse
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score, make_scorer
import joblib
from pathlib import Path

from src.config import MODELS_CONFIG, PATHS


def optimize_svm_hyperparameters(
    X: Union[sparse.csr_matrix, np.ndarray],
    y: np.ndarray,
    n_trials: int = 50,
    n_splits: int = 5,
    random_state: int = 42,
    timeout: Optional[int] = None
) -> Dict:
    """
    Optimize SVM hyperparameters using Optuna (Bayesian Optimization).
    
    Args:
        X: Feature matrix
        y: Labels
        n_trials: Number of optimization trials
        n_splits: Number of CV folds
        random_state: Random seed
        timeout: Maximum time in seconds (None for no limit)
    
    Returns:
        Dictionary with best hyperparameters and optimization results
    """
    print(f"\n🔍 Optimizing SVM hyperparameters (n_trials={n_trials})...")
    
    # Create stratified K-fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    f1_macro_scorer = make_scorer(f1_score, average='macro')
    
    def objective(trial):
        # Suggest hyperparameters
        C = trial.suggest_float('C', 0.1, 100.0, log=True)
        kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly'])
        
        if kernel == 'rbf':
            gamma = trial.suggest_categorical('gamma', ['scale', 'auto']) or trial.suggest_float('gamma', 1e-5, 1e-1, log=True)
        elif kernel == 'poly':
            gamma = trial.suggest_categorical('gamma', ['scale', 'auto']) or trial.suggest_float('gamma', 1e-5, 1e-1, log=True)
            degree = trial.suggest_int('degree', 2, 5)
        else:
            gamma = 'scale'
            degree = 3
        
        # Create model
        if kernel == 'poly':
            model = SVC(
                kernel=kernel,
                C=C,
                gamma=gamma,
                degree=degree,
                class_weight='balanced',
                probability=True,
                random_state=random_state
            )
        else:
            model = SVC(
                kernel=kernel,
                C=C,
                gamma=gamma if isinstance(gamma, str) else gamma,
                class_weight='balanced',
                probability=True,
                random_state=random_state
            )
        
        # Cross-validation score
        scores = cross_val_score(
            model,
            X,
            y,
            cv=skf,
            scoring=f1_macro_scorer,
            n_jobs=-1
        )
        
        return np.mean(scores)
    
    # Create study
    study = optuna.create_study(
        direction='maximize',
        study_name='svm_optimization',
        sampler=optuna.samplers.TPESampler(seed=random_state)
    )
    
    # Optimize
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True
    )
    
    best_params = study.best_params.copy()
    best_value = study.best_value
    
    print(f"✅ Best F1-Macro: {best_value:.4f}")
    print(f"   Best params: {best_params}")
    
    return {
        'best_params': best_params,
        'best_score': best_value,
        'n_trials': len(study.trials),
        'study': study
    }


def optimize_xgboost_hyperparameters(
    X: Union[sparse.csr_matrix, np.ndarray],
    y: np.ndarray,
    n_trials: int = 50,
    n_splits: int = 5,
    random_state: int = 42,
    timeout: Optional[int] = None
) -> Dict:
    """
    Optimize XGBoost hyperparameters using Optuna.
    
    Args:
        X: Feature matrix
        y: Labels
        n_trials: Number of optimization trials
        n_splits: Number of CV folds
        random_state: Random seed
        timeout: Maximum time in seconds
    
    Returns:
        Dictionary with best hyperparameters and optimization results
    """
    print(f"\n🔍 Optimizing XGBoost hyperparameters (n_trials={n_trials})...")
    
    # Create stratified K-fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    f1_macro_scorer = make_scorer(f1_score, average='macro')
    
    def objective(trial):
        # Suggest hyperparameters
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
            'gamma': trial.suggest_float('gamma', 0.0, 0.5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0)
        }
        
        model = XGBClassifier(
            **params,
            n_jobs=-1,
            random_state=random_state,
            eval_metric='mlogloss'
        )
        
        # Cross-validation score
        scores = cross_val_score(
            model,
            X,
            y,
            cv=skf,
            scoring=f1_macro_scorer,
            n_jobs=-1
        )
        
        return np.mean(scores)
    
    # Create study
    study = optuna.create_study(
        direction='maximize',
        study_name='xgboost_optimization',
        sampler=optuna.samplers.TPESampler(seed=random_state)
    )
    
    # Optimize
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True
    )
    
    best_params = study.best_params.copy()
    best_value = study.best_value
    
    print(f"✅ Best F1-Macro: {best_value:.4f}")
    print(f"   Best params: {best_params}")
    
    return {
        'best_params': best_params,
        'best_score': best_value,
        'n_trials': len(study.trials),
        'study': study
    }


def optimize_all_models(
    embeddings: Dict[str, Union[sparse.csr_matrix, np.ndarray]],
    labels: np.ndarray,
    n_trials: int = 50,
    n_splits: int = 5,
    random_state: int = 42,
    save_studies: bool = True
) -> Dict:
    """
    Optimize hyperparameters for all model combinations.
    
    Args:
        embeddings: Dictionary with 'tfidf' and/or 'bert' embeddings
        labels: Labels array
        n_trials: Number of trials per optimization
        n_splits: Number of CV folds
        random_state: Random seed
        save_studies: Whether to save Optuna studies
    
    Returns:
        Dictionary with best hyperparameters for all models
    """
    print("="*60)
    print("🎯 Hyperparameter Optimization for All Models")
    print("="*60)
    
    results = {}
    
    # 1. TF-IDF + SVM
    if 'tfidf' in embeddings:
        opt_result = optimize_svm_hyperparameters(
            embeddings['tfidf'],
            labels,
            n_trials=n_trials,
            n_splits=n_splits,
            random_state=random_state
        )
        results['tfidf_svm'] = opt_result
        if save_studies:
            study_path = PATHS['models'] / 'optuna_tfidf_svm.pkl'
            joblib.dump(opt_result['study'], study_path)
            print(f"💾 Study saved to {study_path}")
    
    # 2. TF-IDF + XGBoost
    if 'tfidf' in embeddings:
        opt_result = optimize_xgboost_hyperparameters(
            embeddings['tfidf'],
            labels,
            n_trials=n_trials,
            n_splits=n_splits,
            random_state=random_state
        )
        results['tfidf_xgb'] = opt_result
        if save_studies:
            study_path = PATHS['models'] / 'optuna_tfidf_xgb.pkl'
            joblib.dump(opt_result['study'], study_path)
            print(f"💾 Study saved to {study_path}")
    
    # 3. BERT + SVM
    if 'bert' in embeddings:
        opt_result = optimize_svm_hyperparameters(
            embeddings['bert'],
            labels,
            n_trials=n_trials,
            n_splits=n_splits,
            random_state=random_state
        )
        results['bert_svm'] = opt_result
        if save_studies:
            study_path = PATHS['models'] / 'optuna_bert_svm.pkl'
            joblib.dump(opt_result['study'], study_path)
            print(f"💾 Study saved to {study_path}")
    
    # 4. BERT + XGBoost
    if 'bert' in embeddings:
        opt_result = optimize_xgboost_hyperparameters(
            embeddings['bert'],
            labels,
            n_trials=n_trials,
            n_splits=n_splits,
            random_state=random_state
        )
        results['bert_xgb'] = opt_result
        if save_studies:
            study_path = PATHS['models'] / 'optuna_bert_xgb.pkl'
            joblib.dump(opt_result['study'], study_path)
            print(f"💾 Study saved to {study_path}")
    
    print("\n" + "="*60)
    print("✅ Hyperparameter Optimization Complete!")
    print("="*60)
    
    return results

