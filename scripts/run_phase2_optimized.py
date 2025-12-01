"""
Script to evaluate optimized models on test set.
Similar to run_phase2.py but uses optimized models.
"""
import sys
from pathlib import Path
import joblib

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import PATHS
from src.data_loader import load_embedding, load_labels
from src.evaluate import evaluate_model, plot_confusion_matrix
from src.benchmark import benchmark_all_models
from src.class_mapping import CLASS_TO_CATEGORY
import pandas as pd
import numpy as np


def load_optimized_models():
    """Load optimized models from disk."""
    models = {}
    model_files = {
        'tfidf_svm': PATHS['models'] / 'tfidf_svm_optimized.pkl',
        'tfidf_xgb': PATHS['models'] / 'tfidf_xgb_optimized.pkl',
        'bert_svm': PATHS['models'] / 'bert_svm_optimized.pkl',
        'bert_xgb': PATHS['models'] / 'bert_xgb_optimized.pkl'
    }
    
    for name, path in model_files.items():
        if path.exists():
            models[name] = joblib.load(path)
            print(f"✅ Loaded {name}")
        else:
            print(f"⚠️  Model not found: {path}")
    
    return models


def main():
    """Main execution function."""
    print("\n" + "="*60)
    print("PHASE 2: Evaluation with Optimized Models")
    print("="*60 + "\n")
    
    # Load optimized models
    print("Loading optimized models...")
    models = load_optimized_models()
    
    if not models:
        print("❌ No optimized models found! Run scripts/retrain_with_optimized.py first.")
        return
    
    # Load embeddings and labels
    print("\nLoading embeddings and labels...")
    embeddings = {
        'tfidf': {
            'test': load_embedding(PATHS['data_embeddings'] / 'tfidf_test.npz', 'tfidf')
        },
        'bert': {
            'test': load_embedding(PATHS['data_embeddings'] / 'bert_test.npy', 'bert')
        }
    }
    labels_test = load_labels(PATHS['data_processed'] / 'labels_test.npy')
    
    print(f"✅ Loaded test set: {len(labels_test)} samples")
    
    # Evaluate on test set
    print("\n" + "="*60)
    print("Evaluation on Test Set (Optimized Models)")
    print("="*60)
    
    results = {}
    class_names = np.array([CLASS_TO_CATEGORY[i] for i in range(len(CLASS_TO_CATEGORY))])
    
    # Evaluate each model
    model_configs = [
        ('tfidf_svm', 'tfidf', 'TF-IDF + SVM (Optimized)'),
        ('tfidf_xgb', 'tfidf', 'TF-IDF + XGBoost (Optimized)'),
        ('bert_svm', 'bert', 'BERT + SVM (Optimized)'),
        ('bert_xgb', 'bert', 'BERT + XGBoost (Optimized)')
    ]
    
    for model_key, emb_key, model_name in model_configs:
        if model_key in models:
            print(f"\n📊 Evaluating {model_name}...")
            result = evaluate_model(
                models[model_key],
                embeddings[emb_key]['test'],
                labels_test,
                model_name=model_name
            )
            results[model_key] = result
            
            # Plot confusion matrix
            plot_confusion_matrix(
                result['confusion_matrix'],
                class_names,
                model_name=model_key.replace('_', '_').replace('optimized', 'opt'),
                save_path=PATHS['models'] / f'cm_{model_key}_optimized_test.png'
            )
    
    # Generate comparison table
    print("\n" + "="*60)
    print("📊 Results Summary")
    print("="*60)
    
    comparison_data = []
    for model_key, _, model_name in model_configs:
        if model_key in results:
            r = results[model_key]
            comparison_data.append({
                'Model': model_name,
                'F1-Macro': f"{r['f1_macro']:.3f}",
                'Accuracy': f"{r['accuracy']:.3f}",
                'F1-Economia': f"{r['f1_per_class'][0]:.3f}",
                'F1-Esportes': f"{r['f1_per_class'][1]:.3f}",
                'F1-Polícia': f"{r['f1_per_class'][2]:.3f}",
                'F1-Política': f"{r['f1_per_class'][3]:.3f}",
                'F1-Turismo': f"{r['f1_per_class'][4]:.3f}",
                'F1-Variedades': f"{r['f1_per_class'][5]:.3f}"
            })
    
    df_comparison = pd.DataFrame(comparison_data)
    print(df_comparison.to_string(index=False))
    
    # Save results
    results_path = PATHS['models'] / 'results_optimized_test.csv'
    df_comparison.to_csv(results_path, index=False)
    print(f"\n💾 Results saved to {results_path}")
    
    # Benchmark
    print("\n" + "="*60)
    print("⚡ Benchmarking Optimized Models")
    print("="*60)
    
    benchmark_results = benchmark_all_models(
        models,
        embeddings,
        {'test': labels_test},
        split='test'
    )
    
    # Save benchmark results
    benchmark_path = PATHS['models'] / 'benchmark_optimized.csv'
    pd.DataFrame(benchmark_results).to_csv(benchmark_path, index=False)
    print(f"💾 Benchmark results saved to {benchmark_path}")
    
    print("\n" + "="*60)
    print("✅ Evaluation Complete!")
    print("="*60)


if __name__ == "__main__":
    main()

