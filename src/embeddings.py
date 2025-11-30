"""
Embedding generation module.
Handles TF-IDF (sparse) and BERT (dense) embeddings via sentence-transformers.
"""
import numpy as np
from pathlib import Path
from typing import List, Union
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Lazy import for sentence-transformers (optional dependency)
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None

from src.config import FEATURE_CONFIG, PATHS
from src.preprocessing import preprocess_batch


def generate_tfidf_embeddings(
    texts: List[str],
    save_path: Union[str, Path] = None,
    fit: bool = True,
    vectorizer: TfidfVectorizer = None
) -> tuple[sparse.csr_matrix, TfidfVectorizer]:
    """
    Generate TF-IDF sparse embeddings.
    
    Args:
        texts: List of preprocessed texts
        save_path: Path to save embeddings (.npz) and vectorizer (.pkl)
        fit: Whether to fit the vectorizer (True for training, False for inference)
        vectorizer: Pre-fitted vectorizer (required if fit=False)
    
    Returns:
        Tuple of (sparse matrix, vectorizer)
    """
    config = FEATURE_CONFIG['tfidf']
    
    if fit:
        if vectorizer is not None:
            raise ValueError("Cannot fit vectorizer when vectorizer is provided")
        
        vectorizer = TfidfVectorizer(
            max_features=config['max_features'],
            ngram_range=config['ngram_range'],
            lowercase=False  # Already lowercase from preprocessing
        )
        embeddings = vectorizer.fit_transform(texts)
    else:
        if vectorizer is None:
            raise ValueError("Vectorizer required when fit=False")
        embeddings = vectorizer.transform(texts)
    
    # Save embeddings and vectorizer if path provided
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save sparse matrix
        np.savez_compressed(
            save_path,
            data=embeddings.data,
            indices=embeddings.indices,
            indptr=embeddings.indptr,
            shape=embeddings.shape
        )
        
        # Save vectorizer
        if fit and vectorizer is not None:
            vectorizer_path = save_path.with_suffix('.pkl')
            joblib.dump(vectorizer, vectorizer_path)
    
    return embeddings, vectorizer


def generate_bert_embeddings(
    texts: List[str],
    save_path: Union[str, Path] = None,
    model = None,
    show_progress: bool = True
) -> tuple[np.ndarray, object]:
    """
    Generate BERT dense embeddings using sentence-transformers.
    
    Args:
        texts: List of preprocessed texts
        save_path: Path to save embeddings (.npy)
        model: Pre-loaded SentenceTransformer model (optional)
        show_progress: Whether to show progress bar
    
    Returns:
        Tuple of (dense array, model)
    """
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        raise ImportError(
            "sentence-transformers is required for BERT embeddings. "
            "Install it with: pip install sentence-transformers"
        )
    
    config = FEATURE_CONFIG['bert']
    
    # Load model if not provided
    if model is None:
        model = SentenceTransformer(
            config['model_name'],
            device='cpu'  # Can be changed to 'cuda' if GPU available
        )
    
    # Generate embeddings with mean pooling (automatic in sentence-transformers)
    embeddings = model.encode(
        texts,
        batch_size=config['batch_size'],
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=False  # Keep raw embeddings
    )
    
    # Ensure float32 for memory efficiency
    embeddings = embeddings.astype(np.float32)
    
    # Save embeddings if path provided
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(save_path, embeddings)
    
    return embeddings, model


def load_tfidf_vectorizer(vectorizer_path: Union[str, Path]) -> TfidfVectorizer:
    """Load a saved TF-IDF vectorizer."""
    return joblib.load(vectorizer_path)


def load_bert_model(model_name: str = None):
    """Load BERT model (sentence-transformer)."""
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        raise ImportError(
            "sentence-transformers is required for BERT embeddings. "
            "Install it with: pip install sentence-transformers"
        )
    
    if model_name is None:
        model_name = FEATURE_CONFIG['bert']['model_name']
    return SentenceTransformer(model_name, device='cpu')

