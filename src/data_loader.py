"""
Polymorphic data loader module.
Handles loading of sparse (.npz) and dense (.npy) embeddings.
"""
import numpy as np
from pathlib import Path
from typing import Union, Tuple
from scipy import sparse
import pandas as pd

from src.config import PATHS


def load_sparse_embedding(filepath: Union[str, Path]) -> sparse.csr_matrix:
    """
    Load sparse embedding matrix from .npz file (TF-IDF).
    
    Args:
        filepath: Path to .npz file
    
    Returns:
        Sparse CSR matrix
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Embedding file not found: {filepath}")
    
    loaded = np.load(filepath, allow_pickle=True)
    
    # Handle different .npz formats
    if 'data' in loaded and 'indices' in loaded and 'indptr' in loaded and 'shape' in loaded:
        # Direct sparse matrix components
        return sparse.csr_matrix(
            (loaded['data'], loaded['indices'], loaded['indptr']),
            shape=loaded['shape']
        )
    elif len(loaded.files) == 1:
        # Single array in file
        key = loaded.files[0]
        matrix = loaded[key]
        if sparse.issparse(matrix):
            return matrix
        else:
            return sparse.csr_matrix(matrix)
    else:
        raise ValueError(f"Unexpected .npz format in {filepath}")


def load_dense_embedding(filepath: Union[str, Path]) -> np.ndarray:
    """
    Load dense embedding matrix from .npy file (BERT).
    
    Args:
        filepath: Path to .npy file
    
    Returns:
        Dense numpy array (float32)
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Embedding file not found: {filepath}")
    
    embedding = np.load(filepath, allow_pickle=True)
    
    # Ensure float32 for memory efficiency
    if embedding.dtype != np.float32:
        embedding = embedding.astype(np.float32)
    
    return embedding


def load_embedding(filepath: Union[str, Path], embedding_type: str = 'auto') -> Union[sparse.csr_matrix, np.ndarray]:
    """
    Polymorphic loader: automatically detects or loads specified embedding type.
    
    Args:
        filepath: Path to embedding file
        embedding_type: 'auto', 'sparse', 'dense', 'tfidf', 'bert'
    
    Returns:
        Sparse matrix (TF-IDF) or dense array (BERT)
    """
    filepath = Path(filepath)
    
    # Auto-detect from extension if not specified
    if embedding_type == 'auto':
        if filepath.suffix == '.npz':
            embedding_type = 'sparse'
        elif filepath.suffix == '.npy':
            embedding_type = 'dense'
        else:
            raise ValueError(f"Cannot auto-detect embedding type for {filepath}")
    
    # Normalize type names
    if embedding_type.lower() in ['sparse', 'tfidf']:
        return load_sparse_embedding(filepath)
    elif embedding_type.lower() in ['dense', 'bert']:
        return load_dense_embedding(filepath)
    else:
        raise ValueError(f"Unknown embedding type: {embedding_type}")


def load_labels(filepath: Union[str, Path]) -> np.ndarray:
    """
    Load labels from CSV or numpy file.
    
    Args:
        filepath: Path to labels file
    
    Returns:
        Numpy array of labels
    """
    filepath = Path(filepath)
    
    if filepath.suffix == '.csv':
        df = pd.read_csv(filepath)
        # Assume first column or 'label' column
        if 'label' in df.columns:
            return df['label'].values
        else:
            return df.iloc[:, 0].values
    elif filepath.suffix in ['.npy', '.npz']:
        return np.load(filepath, allow_pickle=True)
    else:
        raise ValueError(f"Unsupported label file format: {filepath.suffix}")


def load_data_split(
    embedding_file: Union[str, Path],
    labels_file: Union[str, Path],
    embedding_type: str = 'auto'
) -> Tuple[Union[sparse.csr_matrix, np.ndarray], np.ndarray]:
    """
    Load embeddings and labels for a data split (train/val/test).
    
    Args:
        embedding_file: Path to embedding file
        labels_file: Path to labels file
        embedding_type: 'auto', 'sparse', 'dense', 'tfidf', 'bert'
    
    Returns:
        Tuple of (embeddings, labels)
    """
    embeddings = load_embedding(embedding_file, embedding_type)
    labels = load_labels(labels_file)
    
    # Sanity check: shapes must match
    if len(embeddings) != len(labels):
        raise ValueError(
            f"Shape mismatch: embeddings has {len(embeddings)} samples, "
            f"labels has {len(labels)} samples"
        )
    
    return embeddings, labels

