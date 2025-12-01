"""
Embedding generation module.
Handles TF-IDF (sparse) and BERT (dense) embeddings via sentence-transformers.
"""
import numpy as np
from pathlib import Path
from typing import List, Union
from scipy import sparse

# Lazy imports to avoid multiprocessing issues in Streamlit Cloud
# These will be imported inside functions when needed
TfidfVectorizer = None
joblib = None
SentenceTransformer = None
SENTENCE_TRANSFORMERS_AVAILABLE = False

def _lazy_import_sklearn():
    """Lazy import sklearn to avoid atexit issues."""
    global TfidfVectorizer
    if TfidfVectorizer is None:
        from sklearn.feature_extraction.text import TfidfVectorizer
    return TfidfVectorizer

def _lazy_import_joblib():
    """Lazy import joblib to avoid atexit issues."""
    global joblib
    if joblib is None:
        import joblib
    return joblib

def _lazy_import_sentence_transformers():
    """Lazy import sentence-transformers."""
    global SentenceTransformer, SENTENCE_TRANSFORMERS_AVAILABLE
    if SentenceTransformer is None:
        try:
            from sentence_transformers import SentenceTransformer
            SENTENCE_TRANSFORMERS_AVAILABLE = True
        except ImportError:
            SENTENCE_TRANSFORMERS_AVAILABLE = False
            SentenceTransformer = None
    return SentenceTransformer, SENTENCE_TRANSFORMERS_AVAILABLE

from src.config import FEATURE_CONFIG, PATHS
from src.preprocessing import preprocess_batch


def generate_tfidf_embeddings(
    texts: List[str],
    save_path: Union[str, Path] = None,
    fit: bool = True,
    vectorizer = None
) -> tuple:
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
    
    # Lazy import TfidfVectorizer
    TfidfVectorizer = _lazy_import_sklearn()
    
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
            joblib = _lazy_import_joblib()
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
    # Lazy import SentenceTransformer
    SentenceTransformer, SENTENCE_TRANSFORMERS_AVAILABLE = _lazy_import_sentence_transformers()
    
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


def load_tfidf_vectorizer(vectorizer_path: Union[str, Path]):
    """Load a saved TF-IDF vectorizer."""
    joblib = _lazy_import_joblib()
    return joblib.load(vectorizer_path)


def load_bert_model(model_name: str = None):
    """Load BERT model (sentence-transformer)."""
    # Lazy import SentenceTransformer first
    SentenceTransformer, SENTENCE_TRANSFORMERS_AVAILABLE = _lazy_import_sentence_transformers()
    
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        raise ImportError(
            "sentence-transformers is required for BERT embeddings. "
            "Install it with: pip install sentence-transformers"
        )
    
    if model_name is None:
        model_name = FEATURE_CONFIG['bert']['model_name']
    return SentenceTransformer(model_name, device='cpu')

