"""
Retrain all models with optimized hyperparameters.
This script loads the best hyperparameters and retrains all models.
"""
import sys
from pathlib import Path
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import PATHS, OPTIMIZED_HYPERPARAMS
from src.data_loader import load_embedding, load_labels
from src.train import train_svm, train_xgboost


def main():
    print("="*60)
    print("🔄 Retraining Models with Optimized Hyperparameters")
    print("="*60)
    
    # Check if optimized hyperparameters exist
    if OPTIMIZED_HYPERPARAMS is None:
        print("❌ No optimized hyperparameters found!")
        print("   Please run scripts/run_optimization.py first")
        return
    
    print("\n📋 Using optimized hyperparameters:")
    for model_key, params in OPTIMIZED_HYPERPARAMS.items():
        print(f"   {model_key}: {params}")
    
    # Load embeddings and labels
    print("\n📂 Loading embeddings and labels...")
    embeddings = {
        'tfidf': {
            'train': load_embedding(PATHS['data_embeddings'] / 'tfidf_train.npz', 'tfidf')
        },
        'bert': {
            'train': load_embedding(PATHS['data_embeddings'] / 'bert_train.npy', 'bert')
        }
    }
    labels_train = load_labels(PATHS['data_processed'] / 'labels_train.npy')
    
    print(f"✅ Loaded: TF-IDF {embeddings['tfidf']['train'].shape}, BERT {embeddings['bert']['train'].shape}")
    print(f"✅ Labels: {len(labels_train)} samples")
    
    # Retrain all models with optimized hyperparameters
    print("\n" + "="*60)
    print("🎯 Retraining Models")
    print("="*60)
    
    # 1. TF-IDF + SVM
    if 'tfidf_svm' in OPTIMIZED_HYPERPARAMS:
        print("\n🔵 Retraining TF-IDF + SVM...")
        params = OPTIMIZED_HYPERPARAMS['tfidf_svm']
        from sklearn.svm import SVC
        model = SVC(
            kernel=params.get('kernel', 'linear'),
            C=params.get('C', 1.0),
            gamma=params.get('gamma', 'scale'),
            class_weight='balanced',
            probability=True,
            random_state=42
        )
        model.fit(embeddings['tfidf']['train'], labels_train)
        import joblib
        joblib.dump(model, PATHS['models'] / 'tfidf_svm_optimized.pkl')
        print(f"✅ Saved to {PATHS['models'] / 'tfidf_svm_optimized.pkl'}")
    
    # 2. TF-IDF + XGBoost
    if 'tfidf_xgb' in OPTIMIZED_HYPERPARAMS:
        print("\n🟢 Retraining TF-IDF + XGBoost...")
        params = OPTIMIZED_HYPERPARAMS['tfidf_xgb']
        from xgboost import XGBClassifier
        model = XGBClassifier(
            n_estimators=params.get('n_estimators', 100),
            max_depth=params.get('max_depth', 6),
            learning_rate=params.get('learning_rate', 0.1),
            subsample=params.get('subsample', 1.0),
            colsample_bytree=params.get('colsample_bytree', 1.0),
            min_child_weight=params.get('min_child_weight', 1),
            gamma=params.get('gamma', 0.0),
            reg_alpha=params.get('reg_alpha', 0.0),
            reg_lambda=params.get('reg_lambda', 0.0),
            n_jobs=-1,
            random_state=42,
            eval_metric='mlogloss'
        )
        model.fit(embeddings['tfidf']['train'], labels_train)
        import joblib
        joblib.dump(model, PATHS['models'] / 'tfidf_xgb_optimized.pkl')
        print(f"✅ Saved to {PATHS['models'] / 'tfidf_xgb_optimized.pkl'}")
    
    # 3. BERT + SVM
    if 'bert_svm' in OPTIMIZED_HYPERPARAMS:
        print("\n🔵 Retraining BERT + SVM...")
        params = OPTIMIZED_HYPERPARAMS['bert_svm']
        from sklearn.svm import SVC
        model = SVC(
            kernel=params.get('kernel', 'linear'),
            C=params.get('C', 1.0),
            gamma=params.get('gamma', 'scale'),
            class_weight='balanced',
            probability=True,
            random_state=42
        )
        model.fit(embeddings['bert']['train'], labels_train)
        import joblib
        joblib.dump(model, PATHS['models'] / 'bert_svm_optimized.pkl')
        print(f"✅ Saved to {PATHS['models'] / 'bert_svm_optimized.pkl'}")
    
    # 4. BERT + XGBoost
    if 'bert_xgb' in OPTIMIZED_HYPERPARAMS:
        print("\n🟢 Retraining BERT + XGBoost...")
        params = OPTIMIZED_HYPERPARAMS['bert_xgb']
        from xgboost import XGBClassifier
        model = XGBClassifier(
            n_estimators=params.get('n_estimators', 100),
            max_depth=params.get('max_depth', 6),
            learning_rate=params.get('learning_rate', 0.1),
            subsample=params.get('subsample', 1.0),
            colsample_bytree=params.get('colsample_bytree', 1.0),
            min_child_weight=params.get('min_child_weight', 1),
            gamma=params.get('gamma', 0.0),
            reg_alpha=params.get('reg_alpha', 0.0),
            reg_lambda=params.get('reg_lambda', 0.0),
            n_jobs=-1,
            random_state=42,
            eval_metric='mlogloss'
        )
        model.fit(embeddings['bert']['train'], labels_train)
        import joblib
        joblib.dump(model, PATHS['models'] / 'bert_xgb_optimized.pkl')
        print(f"✅ Saved to {PATHS['models'] / 'bert_xgb_optimized.pkl'}")
    
    print("\n" + "="*60)
    print("✅ All Models Retrained with Optimized Hyperparameters!")
    print("="*60)
    print("\nNext steps:")
    print("1. Run Phase 2 evaluation with optimized models")
    print("2. Compare results with default models")


if __name__ == "__main__":
    main()

