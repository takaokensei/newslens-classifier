"""
Benchmark module for measuring inference latency and cold start time.
"""
import time
import numpy as np
from pathlib import Path
from typing import Union, Dict
from scipy import sparse
import joblib
import os

from src.config import PATHS
from src.embeddings import load_tfidf_vectorizer, load_bert_model
from src.preprocessing import preprocess_text


def measure_inference_latency(
    model,
    X: Union[sparse.csr_matrix, np.ndarray],
    batch_size: int = 1,
    n_iterations: int = 100
) -> Dict:
    """
    Measure inference latency (ms per document).
    
    Args:
        model: Trained classifier
        X: Test embeddings
        batch_size: Batch size for inference (1 for real-time simulation)
        n_iterations: Number of iterations to average
    
    Returns:
        Dictionary with latency statistics
    """
    latencies = []
    
    # Warm-up
    _ = model.predict(X[:min(10, len(X))])
    
    # Measure latency
    for _ in range(n_iterations):
        # Select random sample(s)
        indices = np.random.choice(len(X), size=batch_size, replace=False)
        X_batch = X[indices]
        
        start = time.perf_counter()
        _ = model.predict(X_batch)
        end = time.perf_counter()
        
        latency_ms = (end - start) * 1000 / batch_size  # ms per document
        latencies.append(latency_ms)
    
    return {
        'mean_ms': np.mean(latencies),
        'std_ms': np.std(latencies),
        'min_ms': np.min(latencies),
        'max_ms': np.max(latencies),
        'p50_ms': np.percentile(latencies, 50),
        'p95_ms': np.percentile(latencies, 95),
        'p99_ms': np.percentile(latencies, 99)
    }


def measure_cold_start(
    model_path: Path,
    embedding_type: str,
    sample_text: str = "Este é um texto de exemplo para teste."
) -> Dict:
    """
    Measure cold start time (time to load model and make first prediction).
    
    Args:
        model_path: Path to saved model
        embedding_type: 'tfidf' or 'bert'
        sample_text: Sample text for prediction
    
    Returns:
        Dictionary with cold start statistics
    """
    times = []
    
    for _ in range(5):  # Multiple runs to average
        # Clear any cached models
        import gc
        gc.collect()
        
        start = time.perf_counter()
        
        # Load model
        model = joblib.load(model_path)
        
        # Load/prepare embeddings
        if embedding_type == 'tfidf':
            vectorizer_path = PATHS['data_embeddings'] / 'tfidf_vectorizer.pkl'
            if vectorizer_path.exists():
                vectorizer = load_tfidf_vectorizer(vectorizer_path)
                processed = preprocess_text(sample_text)
                embedding = vectorizer.transform([processed])
            else:
                raise FileNotFoundError(f"Vectorizer not found: {vectorizer_path}")
        elif embedding_type == 'bert':
            model_bert = load_bert_model()
            processed = preprocess_text(sample_text)
            embedding = model_bert.encode([processed], convert_to_numpy=True)
        else:
            raise ValueError(f"Unknown embedding type: {embedding_type}")
        
        # First prediction
        _ = model.predict(embedding)
        
        end = time.perf_counter()
        times.append(end - start)
    
    return {
        'mean_s': np.mean(times),
        'std_s': np.std(times),
        'min_s': np.min(times),
        'max_s': np.max(times)
    }


def get_model_size(model_path: Path) -> float:
    """
    Get model file size in MB.
    
    Args:
        model_path: Path to model file
    
    Returns:
        Size in MB
    """
    if model_path.exists():
        size_bytes = model_path.stat().st_size
        return size_bytes / (1024 * 1024)  # Convert to MB
    return 0.0


def benchmark_all_models(
    models: Dict,
    embeddings: Dict,
    labels: Dict,
    split: str = 'test'
) -> Dict:
    """
    Benchmark all models: latency, cold start, model size.
    
    Args:
        models: Dictionary of trained models
        embeddings: Dictionary of embeddings
        labels: Dictionary of labels
        split: Split to use for benchmarking
    
    Returns:
        Dictionary with benchmark results
    """
    print("="*60)
    print("⚡ Benchmarking All Models")
    print("="*60)
    
    results = {}
    
    # Benchmark each model
    model_configs = [
        ('tfidf_svm', 'tfidf', 'TF-IDF + SVM'),
        ('tfidf_xgb', 'tfidf', 'TF-IDF + XGBoost'),
        ('bert_svm', 'bert', 'BERT + SVM'),
        ('bert_xgb', 'bert', 'BERT + XGBoost')
    ]
    
    for model_key, emb_key, model_name in model_configs:
        if model_key in models and emb_key in embeddings:
            print(f"\n🔍 Benchmarking {model_name}...")
            
            X = embeddings[emb_key][split]
            
            # Inference latency
            latency = measure_inference_latency(models[model_key], X, batch_size=1)
            print(f"  Latency: {latency['mean_ms']:.2f} ± {latency['std_ms']:.2f} ms/doc")
            
            # Cold start
            model_path = PATHS['models'] / f'{model_key}.pkl'
            if model_path.exists():
                cold_start = measure_cold_start(model_path, emb_key)
                print(f"  Cold Start: {cold_start['mean_s']:.2f} ± {cold_start['std_s']:.2f} s")
            else:
                cold_start = {'mean_s': 0, 'std_s': 0}
                print(f"  ⚠️  Model file not found for cold start measurement")
            
            # Model size
            model_size = get_model_size(model_path)
            print(f"  Model Size: {model_size:.2f} MB")
            
            results[model_key] = {
                'model_name': model_name,
                'latency': latency,
                'cold_start': cold_start,
                'model_size_mb': model_size
            }
    
    print("\n" + "="*60)
    print("✅ Benchmarking Complete!")
    print("="*60)
    
    return results

