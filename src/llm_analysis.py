"""
LLM Analysis module for NewsLens AI.
Implements class profiling (Chi-Squared + Centroides) and differential error analysis.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy import sparse
from sklearn.feature_selection import chi2
from sklearn.metrics.pairwise import cosine_similarity
from typing import Optional
import json

from src.config import LLM_CONFIG, PATHS, FEATURE_CONFIG
from src.class_mapping import CLASS_TO_CATEGORY
from src.embeddings import load_tfidf_vectorizer, load_bert_model
from src.data_loader import load_embedding, load_labels


# Try to import Groq client
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    print("Warning: groq package not installed. LLM features will be disabled.")


def compute_class_centroids(
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_classes: int
) -> Dict[int, np.ndarray]:
    """
    Compute centroids for each class using BERT embeddings.
    
    Args:
        embeddings: BERT embeddings (n_samples, embedding_dim)
        labels: Class labels (n_samples,)
        n_classes: Number of classes
    
    Returns:
        Dictionary mapping class_idx -> centroid vector
    """
    centroids = {}
    for class_idx in range(n_classes):
        class_mask = labels == class_idx
        if np.sum(class_mask) > 0:
            class_embeddings = embeddings[class_mask]
            centroids[class_idx] = np.mean(class_embeddings, axis=0)
    return centroids


def find_nearest_neighbors(
    centroid: np.ndarray,
    embeddings: np.ndarray,
    labels: np.ndarray,
    target_class: int,
    k: int = 5
) -> List[int]:
    """
    Find k nearest neighbors to centroid in the same class.
    
    Args:
        centroid: Class centroid vector
        embeddings: All embeddings
        labels: All labels
        target_class: Target class index
        k: Number of neighbors to return
    
    Returns:
        List of indices of nearest neighbors
    """
    # Filter embeddings for target class
    class_mask = labels == target_class
    class_embeddings = embeddings[class_mask]
    class_indices = np.where(class_mask)[0]
    
    if len(class_embeddings) == 0:
        return []
    
    # Compute cosine similarities
    similarities = cosine_similarity(
        centroid.reshape(1, -1),
        class_embeddings
    )[0]
    
    # Get top k indices
    top_k_idx = np.argsort(similarities)[::-1][:k]
    return class_indices[top_k_idx].tolist()


def chi_squared_feature_selection(
    X: sparse.csr_matrix,
    y: np.ndarray,
    vectorizer,
    top_k: int = 20
) -> Dict[int, List[Tuple[str, float]]]:
    """
    Perform Chi-Squared feature selection to identify top tokens per class.
    
    Args:
        X: TF-IDF sparse matrix
        y: Class labels
        top_k: Number of top features per class
    
    Returns:
        Dictionary mapping class_idx -> List of (token, chi2_score) tuples
    """
    feature_names = vectorizer.get_feature_names_out()
    class_features = {}
    
    for class_idx in np.unique(y):
        # Create binary target: 1 if class_idx, 0 otherwise
        y_binary = (y == class_idx).astype(int)
        
        # Compute chi-squared scores
        chi2_scores, _ = chi2(X, y_binary)
        
        # Get top k features
        top_indices = np.argsort(chi2_scores)[::-1][:top_k]
        top_features = [
            (feature_names[idx], float(chi2_scores[idx]))
            for idx in top_indices
            if chi2_scores[idx] > 0
        ]
        
        class_features[class_idx] = top_features
    
    return class_features


def profile_classes_hybrid(
    embeddings_tfidf: Dict[str, sparse.csr_matrix],
    embeddings_bert: Dict[str, np.ndarray],
    labels: Dict[str, np.ndarray],
    vectorizer_path: Optional[Path] = None,
    n_neighbors: int = 5,
    top_tokens: int = 20
) -> Dict:
    """
    Create hybrid class profiles using Chi-Squared (TF-IDF) and Centroids (BERT).
    
    Args:
        embeddings_tfidf: Dictionary with 'train' key containing TF-IDF embeddings
        embeddings_bert: Dictionary with 'train' key containing BERT embeddings
        labels: Dictionary with 'train' key containing labels
        vectorizer_path: Path to TF-IDF vectorizer
        n_neighbors: Number of nearest neighbors to find
        top_tokens: Number of top tokens per class
    
    Returns:
        Dictionary with class profiles (archetypes)
    """
    print("="*60)
    print("Creating Hybrid Class Profiles")
    print("="*60)
    
    X_tfidf = embeddings_tfidf['train']
    X_bert = embeddings_bert['train']
    y = labels['train']
    
    n_classes = len(np.unique(y))
    profiles = {}
    
    # Load vectorizer for feature names
    if vectorizer_path is None:
        vectorizer_path = PATHS['data_embeddings'] / 'tfidf_vectorizer.pkl'
    
    vectorizer = load_tfidf_vectorizer(vectorizer_path)
    
    # 1. Chi-Squared feature selection (TF-IDF)
    print("\n📊 Computing Chi-Squared features (TF-IDF)...")
    tfidf_features = chi_squared_feature_selection(
        X_tfidf, y, vectorizer, top_k=top_tokens
    )
    
    # 2. Compute centroids (BERT)
    print("🧠 Computing class centroids (BERT)...")
    centroids = compute_class_centroids(X_bert, y, n_classes)
    
    # 3. Find nearest neighbors for each class
    print("🔍 Finding nearest neighbors...")
    for class_idx in range(n_classes):
        category_name = CLASS_TO_CATEGORY.get(class_idx, f"Class_{class_idx}")
        
        # Get centroid
        centroid = centroids.get(class_idx)
        if centroid is None:
            continue
        
        # Find nearest neighbors
        neighbor_indices = find_nearest_neighbors(
            centroid, X_bert, y, class_idx, k=n_neighbors
        )
        
        # Get top tokens
        top_tokens_list = tfidf_features.get(class_idx, [])
        
        profiles[class_idx] = {
            'category': category_name,
            'centroid': centroid.tolist(),
            'top_tokens_tfidf': [
                {'token': token, 'chi2_score': score}
                for token, score in top_tokens_list
            ],
            'n_samples': int(np.sum(y == class_idx)),
            'neighbor_indices': neighbor_indices[:n_neighbors]
        }
        
        print(f"  ✅ Class {class_idx} ({category_name}): {len(top_tokens_list)} tokens, {len(neighbor_indices)} neighbors")
    
    return profiles


def save_class_profiles(profiles: Dict, save_path: Optional[Path] = None):
    """Save class profiles to JSON file."""
    if save_path is None:
        save_path = PATHS['models'] / 'class_profiles.json'
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(profiles, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Class profiles saved to {save_path}")


def load_class_profiles(profile_path: Optional[Path] = None) -> Dict:
    """Load class profiles from JSON file."""
    if profile_path is None:
        profile_path = PATHS['models'] / 'class_profiles.json'
    
    profile_path = Path(profile_path)
    if not profile_path.exists():
        raise FileNotFoundError(f"Class profiles not found: {profile_path}")
    
    with open(profile_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_groq_client():
    """Initialize and return Groq client."""
    if not GROQ_AVAILABLE:
        raise ImportError("groq package not installed. Install with: pip install groq")
    
    api_key = LLM_CONFIG.get('api_key')
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable not set")
    
    return Groq(api_key=api_key)


def analyze_differential_errors(
    texts: np.ndarray,
    y_true: np.ndarray,
    y_pred_bert: np.ndarray,
    y_pred_tfidf: np.ndarray,
    y_proba_bert: np.ndarray,
    y_proba_tfidf: np.ndarray,
    max_examples: int = 10
) -> List[Dict]:
    """
    Find cases where BERT is correct but TF-IDF is wrong.
    Prioritize by confidence delta.
    
    Args:
        texts: Original texts
        y_true: True labels
        y_pred_bert: BERT predictions
        y_pred_tfidf: TF-IDF predictions
        y_proba_bert: BERT prediction probabilities
        y_proba_tfidf: TF-IDF prediction probabilities
        max_examples: Maximum number of examples to analyze
    
    Returns:
        List of error cases with metadata
    """
    # Filter: BERT correct AND TF-IDF wrong
    bert_correct = (y_pred_bert == y_true)
    tfidf_wrong = (y_pred_tfidf != y_true)
    filter_mask = bert_correct & tfidf_wrong
    
    if not np.any(filter_mask):
        print("No differential errors found (BERT correct, TF-IDF wrong)")
        return []
    
    # Get indices
    error_indices = np.where(filter_mask)[0]
    
    # Compute confidence deltas
    error_cases = []
    for idx in error_indices:
        true_class = int(y_true[idx])
        bert_conf = float(y_proba_bert[idx, true_class])
        tfidf_conf = float(y_proba_tfidf[idx, y_pred_tfidf[idx]])
        delta = bert_conf - tfidf_conf
        
        error_cases.append({
            'index': int(idx),
            'text': str(texts[idx]),
            'true_class': true_class,
            'true_category': CLASS_TO_CATEGORY.get(true_class, f"Class_{true_class}"),
            'bert_pred': int(y_pred_bert[idx]),
            'bert_category': CLASS_TO_CATEGORY.get(int(y_pred_bert[idx]), f"Class_{y_pred_bert[idx]}"),
            'tfidf_pred': int(y_pred_tfidf[idx]),
            'tfidf_category': CLASS_TO_CATEGORY.get(int(y_pred_tfidf[idx]), f"Class_{y_pred_tfidf[idx]}"),
            'bert_confidence': bert_conf,
            'tfidf_confidence': tfidf_conf,
            'confidence_delta': delta
        })
    
    # Sort by confidence delta (descending)
    error_cases.sort(key=lambda x: x['confidence_delta'], reverse=True)
    
    # Return top max_examples
    return error_cases[:max_examples]


def call_groq_llm(
    prompt: str,
    max_tokens: int = 500,
    temperature: float = 0.7
) -> str:
    """
    Call Groq API for LLM analysis.
    
    Args:
        prompt: Input prompt
        max_tokens: Maximum tokens in response
        temperature: Sampling temperature
    
    Returns:
        LLM response text
    """
    if not GROQ_AVAILABLE:
        raise ImportError("groq package not installed")
    
    client = get_groq_client()
    model = LLM_CONFIG.get('model', 'llama-3.1-70b-versatile')
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Você é um especialista em análise de linguagem e classificação de textos em português."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling Groq API: {e}")
        return f"Error: {str(e)}"


def analyze_errors_with_llm(
    error_cases: List[Dict],
    max_calls: int = 10
) -> List[Dict]:
    """
    Analyze differential errors using Groq LLM.
    
    Args:
        error_cases: List of error cases from analyze_differential_errors
        max_calls: Maximum number of LLM calls (cost control)
    
    Returns:
        List of error cases with LLM explanations
    """
    if not GROQ_AVAILABLE:
        print("Warning: Groq not available. Skipping LLM analysis.")
        return error_cases
    
    print(f"\n🤖 Analyzing {min(len(error_cases), max_calls)} errors with LLM...")
    
    analyzed = []
    for i, case in enumerate(error_cases[:max_calls]):
        print(f"  Analyzing case {i+1}/{min(len(error_cases), max_calls)}...")
        
        prompt = f"""O modelo semântico (BERT) classificou corretamente o texto abaixo como "{case['true_category']}", mas o modelo léxico (TF-IDF) classificou incorretamente como "{case['tfidf_category']}".

Texto:
{case['text'][:500]}

Classe verdadeira: {case['true_category']}
Predição BERT: {case['bert_category']} (confiança: {case['bert_confidence']:.3f})
Predição TF-IDF: {case['tfidf_category']} (confiança: {case['tfidf_confidence']:.3f})

Explique qual nuance linguística ou contexto semântico o BERT capturou que o TF-IDF perdeu. Por que o modelo semântico teve sucesso onde o léxico falhou?"""
        
        explanation = call_groq_llm(prompt, max_tokens=300)
        case['llm_explanation'] = explanation
        analyzed.append(case)
    
    return analyzed

