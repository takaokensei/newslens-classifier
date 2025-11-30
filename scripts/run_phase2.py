"""
Main script for Phase 2: Training & Benchmarking.
Orchestrates the complete pipeline: data preparation, training, evaluation, and benchmarking.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.prepare_data import prepare_full_pipeline
from src.train import train_all_models
from src.evaluate import evaluate_all_models, generate_comparison_table
from src.benchmark import benchmark_all_models
from src.config import PATHS
import pandas as pd
import numpy as np


def main():
    """Main execution function."""
    print("\n" + "="*60)
    print("🎯 PHASE 2: Training & Benchmarking")
    print("="*60 + "\n")
    
    # Step 1: Prepare data
    print("📍 Step 1: Data Preparation")
    print("-" * 60)
    data = prepare_full_pipeline(
        data_path=None,  # Will look in data/raw/
        embedding_type='both',  # Generate both TF-IDF and BERT
        save=True
    )
    
    # Step 2: Train models
    print("\n📍 Step 2: Training Models")
    print("-" * 60)
    models = train_all_models(
        data['embeddings'],
        data['labels'],
        save=True
    )
    
    # Step 3: Evaluate on validation set
    print("\n📍 Step 3: Evaluation on Validation Set")
    print("-" * 60)
    val_results = evaluate_all_models(
        models,
        data['embeddings'],
        data['labels'],
        split='val',
        save_plots=True
    )
    
    # Step 4: Evaluate on test set
    print("\n📍 Step 4: Evaluation on Test Set")
    print("-" * 60)
    test_results = evaluate_all_models(
        models,
        data['embeddings'],
        data['labels'],
        split='test',
        save_plots=True
    )
    
    # Step 5: Benchmark
    print("\n📍 Step 5: Benchmarking")
    print("-" * 60)
    benchmark_results = benchmark_all_models(
        models,
        data['embeddings'],
        data['labels'],
        split='test'
    )
    
    # Step 6: Generate comparison tables
    print("\n📍 Step 6: Generating Comparison Tables")
    print("-" * 60)
    
    # Table A: Efficiency & Performance Global
    table_a_data = []
    for model_key in ['tfidf_svm', 'tfidf_xgb', 'bert_svm', 'bert_xgb']:
        if model_key in test_results and model_key in benchmark_results:
            result = test_results[model_key]
            bench = benchmark_results[model_key]
            
            table_a_data.append({
                'Setup': bench['model_name'],
                'F1-Macro': result['f1_macro'],
                'Accuracy': result['accuracy'],
                'Latency (ms/doc)': bench['latency']['mean_ms'],
                'Cold Start (s)': bench['cold_start']['mean_s'],
                'Tamanho (MB)': bench['model_size_mb']
            })
    
    table_a = pd.DataFrame(table_a_data)
    table_a_path = PATHS['models'] / 'table_a_efficiency.csv'
    table_a.to_csv(table_a_path, index=False)
    print(f"\n✅ Table A saved to {table_a_path}")
    print(table_a.to_string(index=False))
    
    # Table B: Granularity by Class
    class_names = np.unique(data['labels']['test'])
    table_b_data = []
    
    for class_name in class_names:
        row = {'Class': class_name}
        for model_key in ['tfidf_svm', 'tfidf_xgb', 'bert_svm', 'bert_xgb']:
            if model_key in test_results:
                f1_per_class = test_results[model_key]['f1_per_class']
                class_idx = np.where(class_names == class_name)[0][0]
                if class_idx < len(f1_per_class):
                    row[model_key.replace('_', '+').upper()] = f1_per_class[class_idx]
        table_b_data.append(row)
    
    table_b = pd.DataFrame(table_b_data)
    table_b_path = PATHS['models'] / 'table_b_classes.csv'
    table_b.to_csv(table_b_path, index=False)
    print(f"\n✅ Table B saved to {table_b_path}")
    print(table_b.to_string(index=False))
    
    print("\n" + "="*60)
    print("✅ PHASE 2 COMPLETE!")
    print("="*60)
    print("\n📊 Results saved to:")
    print(f"  - Models: {PATHS['models']}")
    print(f"  - Embeddings: {PATHS['data_embeddings']}")
    print(f"  - Tables: {table_a_path}, {table_b_path}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

