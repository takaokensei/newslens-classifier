"""
Evaluation module for NewsLens AI Classifier.
Computes metrics: Accuracy, F1-Macro, F1 per class, Confusion Matrix.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, Dict, Tuple, Optional
from scipy import sparse
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import PATHS


def evaluate_model(
    model,
    X: Union[sparse.csr_matrix, np.ndarray],
    y_true: np.ndarray,
    model_name: str = "Model"
) -> Dict:
    """
    Evaluate a model and return comprehensive metrics.
    
    Args:
        model: Trained classifier
        X: Test embeddings
        y_true: True labels
        model_name: Name for reporting
    
    Returns:
        Dictionary with metrics
    """
    print(f"\n📊 Evaluating {model_name}...")
    
    # Predictions
    y_pred = model.predict(X)
    
    # Probabilities (if available)
    try:
        y_proba = model.predict_proba(X)
        has_proba = True
    except:
        y_proba = None
        has_proba = False
    
    # Metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_per_class = f1_score(y_true, y_pred, average=None)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Classification report
    report = classification_report(y_true, y_pred, output_dict=True)
    
    results = {
        'model_name': model_name,
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_per_class': f1_per_class,
        'confusion_matrix': cm,
        'classification_report': report,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'has_proba': has_proba
    }
    
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1-Macro: {f1_macro:.4f}")
    print(f"  F1 per class: {f1_per_class}")
    
    return results


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: np.ndarray,
    model_name: str,
    save_path: Optional[Path] = None
):
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix
        class_names: Array of class names
        model_name: Name for title
        save_path: Path to save figure
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved confusion matrix to {save_path}")
    
    plt.close()


def evaluate_all_models(
    models: Dict,
    embeddings: Dict,
    labels: Dict,
    class_names: Optional[np.ndarray] = None,
    split: str = 'test',
    save_plots: bool = True
) -> Dict:
    """
    Evaluate all models on specified split.
    
    Args:
        models: Dictionary of trained models
        embeddings: Dictionary of embeddings
        labels: Dictionary of labels
        class_names: Array of class names
        split: 'train', 'val', or 'test'
        save_plots: Whether to save confusion matrices
    
    Returns:
        Dictionary with all evaluation results
    """
    print("="*60)
    print(f"📊 Evaluating All Models on {split.upper()} Set")
    print("="*60)
    
    results = {}
    
    # Get class names if not provided
    if class_names is None:
        unique_labels = np.unique(labels[split])
        class_names = unique_labels
    
    # Evaluate each model
    if 'tfidf_svm' in models and 'tfidf' in embeddings:
        results['tfidf_svm'] = evaluate_model(
            models['tfidf_svm'],
            embeddings['tfidf'][split],
            labels[split],
            f"TF-IDF + SVM ({split})"
        )
        if save_plots:
            plot_confusion_matrix(
                results['tfidf_svm']['confusion_matrix'],
                class_names,
                f"TF-IDF + SVM ({split})",
                PATHS['models'] / f'cm_tfidf_svm_{split}.png'
            )
    
    if 'tfidf_xgb' in models and 'tfidf' in embeddings:
        results['tfidf_xgb'] = evaluate_model(
            models['tfidf_xgb'],
            embeddings['tfidf'][split],
            labels[split],
            f"TF-IDF + XGBoost ({split})"
        )
        if save_plots:
            plot_confusion_matrix(
                results['tfidf_xgb']['confusion_matrix'],
                class_names,
                f"TF-IDF + XGBoost ({split})",
                PATHS['models'] / f'cm_tfidf_xgb_{split}.png'
            )
    
    if 'bert_svm' in models and 'bert' in embeddings:
        results['bert_svm'] = evaluate_model(
            models['bert_svm'],
            embeddings['bert'][split],
            labels[split],
            f"BERT + SVM ({split})"
        )
        if save_plots:
            plot_confusion_matrix(
                results['bert_svm']['confusion_matrix'],
                class_names,
                f"BERT + SVM ({split})",
                PATHS['models'] / f'cm_bert_svm_{split}.png'
            )
    
    if 'bert_xgb' in models and 'bert' in embeddings:
        results['bert_xgb'] = evaluate_model(
            models['bert_xgb'],
            embeddings['bert'][split],
            labels[split],
            f"BERT + XGBoost ({split})"
        )
        if save_plots:
            plot_confusion_matrix(
                results['bert_xgb']['confusion_matrix'],
                class_names,
                f"BERT + XGBoost ({split})",
                PATHS['models'] / f'cm_bert_xgb_{split}.png'
            )
    
    print("\n" + "="*60)
    print("✅ All Models Evaluated!")
    print("="*60)
    
    return results


def generate_comparison_table(results: Dict, split: str = 'test') -> pd.DataFrame:
    """
    Generate comparison table from evaluation results.
    
    Args:
        results: Dictionary of evaluation results
        split: Split name
    
    Returns:
        DataFrame with comparison metrics
    """
    data = []
    for model_name, result in results.items():
        data.append({
            'Model': model_name,
            'Accuracy': result['accuracy'],
            'F1-Macro': result['f1_macro'],
            'Split': split
        })
    
    df = pd.DataFrame(data)
    return df

