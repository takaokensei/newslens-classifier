"""Quick test of Phase 2 pipeline with small subset."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.prepare_data import load_raw_data, split_data, prepare_embeddings
from src.train import train_all_models
from src.evaluate import evaluate_all_models
import numpy as np
import pandas as pd

print("="*60)
print("🧪 Quick Test - Phase 2 Pipeline")
print("="*60)

# Load data
print("\n1. Loading data...")
df, labels = load_raw_data()

# Get text column
text_cols = [col for col in df.columns if 'texto' in col.lower()]
if 'Texto Expandido' in text_cols:
    text_col = 'Texto Expandido'
else:
    text_col = text_cols[0] if text_cols else df.columns[0]

texts = df[text_col].astype(str).values
texts_series = pd.Series(texts)
mask = (texts_series != '') & (texts_series != 'nan') & (~texts_series.isna())
texts = texts[mask.values]
labels = labels[mask.values]

print(f"✅ Loaded {len(texts)} samples")

# Split (small test)
print("\n2. Splitting data...")
X_train, X_val, X_test, y_train, y_val, y_test = split_data(texts, labels)
print(f"✅ Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# Test TF-IDF only (faster)
print("\n3. Generating TF-IDF embeddings (quick test)...")
embeddings_tfidf = prepare_embeddings(
    X_train[:50], X_val[:20], X_test[:20],  # Small subset for quick test
    embedding_type='tfidf',
    save=False
)

print("\n4. Training models (TF-IDF only)...")
models = {}
if 'tfidf' in embeddings_tfidf:
    from src.train import train_svm, train_xgboost
    models['tfidf_svm'] = train_svm(
        embeddings_tfidf['tfidf']['train'],
        y_train[:50],
        save_path=None
    )
    models['tfidf_xgb'] = train_xgboost(
        embeddings_tfidf['tfidf']['train'],
        y_train[:50],
        save_path=None
    )

print("\n5. Evaluating...")
if models:
    from src.evaluate import evaluate_model
    result_svm = evaluate_model(
        models['tfidf_svm'],
        embeddings_tfidf['tfidf']['test'],
        y_test[:20],
        "TF-IDF + SVM (Quick Test)"
    )
    result_xgb = evaluate_model(
        models['tfidf_xgb'],
        embeddings_tfidf['tfidf']['test'],
        y_test[:20],
        "TF-IDF + XGBoost (Quick Test)"
    )

print("\n" + "="*60)
print("✅ Quick Test Complete!")
print("="*60)

