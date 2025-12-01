"""
Script to run hyperparameter optimization and cross-validation.
This script:
1. Loads embeddings and labels
2. Runs Optuna optimization for all models
3. Performs K-fold CV with optimized hyperparameters
4. Saves results and best hyperparameters
"""
import sys
from pathlib import Path
import pandas as pd
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import PATHS
from src.data_loader import load_embedding, load_labels
from src.hyperparameter_optimization import optimize_all_models
from src.cross_validation import cv_all_models


def main():
    print("="*60)
    print("🚀 Hyperparameter Optimization & Cross-Validation")
    print("="*60)
    
    # Load embeddings and labels
    print("\n📂 Loading embeddings and labels...")
    embeddings = {
        'tfidf': load_embedding(PATHS['data_embeddings'] / 'tfidf_train.npz', 'tfidf'),
        'bert': load_embedding(PATHS['data_embeddings'] / 'bert_train.npy', 'bert')
    }
    
    # Combine train + val for CV (more data = better CV)
    labels_train = load_labels(PATHS['data_processed'] / 'labels_train.npy')
    labels_val = load_labels(PATHS['data_processed'] / 'labels_val.npy')
    
    import numpy as np
    labels_combined = np.concatenate([labels_train, labels_val])
    
    # Combine embeddings too
    if isinstance(embeddings['tfidf'], type(embeddings['tfidf'])):
        from scipy import sparse
        if sparse.issparse(embeddings['tfidf']):
            embeddings['tfidf'] = sparse.vstack([
                embeddings['tfidf'],
                load_embedding(PATHS['data_embeddings'] / 'tfidf_val.npz', 'tfidf')
            ])
        else:
            embeddings['tfidf'] = np.vstack([
                embeddings['tfidf'],
                load_embedding(PATHS['data_embeddings'] / 'tfidf_val.npz', 'tfidf')
            ])
    
    embeddings['bert'] = np.vstack([
        embeddings['bert'],
        load_embedding(PATHS['data_embeddings'] / 'bert_val.npy', 'bert')
    ])
    
    print(f"✅ Loaded embeddings: TF-IDF {embeddings['tfidf'].shape}, BERT {embeddings['bert'].shape}")
    print(f"✅ Combined labels: {len(labels_combined)} samples")
    
    # Step 1: Hyperparameter Optimization
    print("\n" + "="*60)
    print("STEP 1: Hyperparameter Optimization (Optuna)")
    print("="*60)
    
    optimization_results = optimize_all_models(
        embeddings=embeddings,
        labels=labels_combined,
        n_trials=50,  # Adjust based on time available
        n_splits=5,
        random_state=42,
        save_studies=True
    )
    
    # Extract best hyperparameters
    best_hyperparams = {}
    for model_key, opt_result in optimization_results.items():
        best_hyperparams[model_key] = opt_result['best_params']
        print(f"\n{model_key}:")
        print(f"  Best F1-Macro: {opt_result['best_score']:.4f}")
        print(f"  Best params: {opt_result['best_params']}")
    
    # Save best hyperparameters
    hyperparams_path = PATHS['models'] / 'best_hyperparameters.json'
    with open(hyperparams_path, 'w') as f:
        json.dump(best_hyperparams, f, indent=2)
    print(f"\n💾 Best hyperparameters saved to {hyperparams_path}")
    
    # Step 2: Cross-Validation with Optimized Hyperparameters
    print("\n" + "="*60)
    print("STEP 2: K-Fold Cross-Validation (with optimized hyperparameters)")
    print("="*60)
    
    cv_results = cv_all_models(
        embeddings=embeddings,
        labels=labels_combined,
        n_splits=5,
        hyperparams=best_hyperparams,
        random_state=42
    )
    
    # Save CV results
    cv_results_path = PATHS['models'] / 'cv_results_optimized.csv'
    cv_results.to_csv(cv_results_path, index=False)
    print(f"\n💾 CV results saved to {cv_results_path}")
    
    # Step 3: Comparison with default hyperparameters
    print("\n" + "="*60)
    print("STEP 3: K-Fold Cross-Validation (with default hyperparameters)")
    print("="*60)
    
    cv_results_default = cv_all_models(
        embeddings=embeddings,
        labels=labels_combined,
        n_splits=5,
        hyperparams=None,  # Use defaults
        random_state=42
    )
    
    # Save default CV results
    cv_results_default_path = PATHS['models'] / 'cv_results_default.csv'
    cv_results_default.to_csv(cv_results_default_path, index=False)
    print(f"\n💾 Default CV results saved to {cv_results_default_path}")
    
    # Comparison
    print("\n" + "="*60)
    print("📊 Comparison: Optimized vs Default")
    print("="*60)
    
    comparison = pd.merge(
        cv_results[['Model', 'Mean F1-Macro']].rename(columns={'Mean F1-Macro': 'F1-Optimized'}),
        cv_results_default[['Model', 'Mean F1-Macro']].rename(columns={'Mean F1-Macro': 'F1-Default'}),
        on='Model'
    )
    comparison['Improvement'] = comparison['F1-Optimized'] - comparison['F1-Default']
    comparison['Improvement %'] = (comparison['Improvement'] / comparison['F1-Default'] * 100).round(2)
    
    print(comparison.to_string(index=False))
    
    comparison_path = PATHS['models'] / 'optimization_comparison.csv'
    comparison.to_csv(comparison_path, index=False)
    print(f"\n💾 Comparison saved to {comparison_path}")
    
    print("\n" + "="*60)
    print("✅ Optimization Complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Review best_hyperparameters.json")
    print("2. Update src/config.py with optimized hyperparameters")
    print("3. Retrain models with optimized hyperparameters")
    print("4. Run Phase 2 evaluation with optimized models")


if __name__ == "__main__":
    main()

