"""
Sanity check module for data validation.
Verifies shapes, NaNs, and class distribution after data splits.
"""
import numpy as np
from pathlib import Path
from typing import Union, Tuple
from scipy import sparse
import pandas as pd

from src.data_loader import load_embedding, load_labels


def check_shapes(
    embeddings: Union[sparse.csr_matrix, np.ndarray],
    labels: np.ndarray
) -> bool:
    """
    Check if embeddings and labels have matching shapes.
    
    Args:
        embeddings: Embedding matrix (sparse or dense)
        labels: Label array
    
    Returns:
        True if shapes match, False otherwise
    """
    n_samples_emb = embeddings.shape[0]
    n_samples_labels = len(labels)
    
    if n_samples_emb != n_samples_labels:
        print(f"❌ Shape mismatch: {n_samples_emb} embeddings vs {n_samples_labels} labels")
        return False
    
    print(f"✅ Shapes match: {n_samples_emb} samples")
    return True


def check_nans(embeddings: Union[sparse.csr_matrix, np.ndarray]) -> bool:
    """
    Check for NaN values in embeddings.
    
    Args:
        embeddings: Embedding matrix
    
    Returns:
        True if no NaNs found, False otherwise
    """
    if sparse.issparse(embeddings):
        has_nan = np.isnan(embeddings.data).any()
        nan_count = np.isnan(embeddings.data).sum()
    else:
        has_nan = np.isnan(embeddings).any()
        nan_count = np.isnan(embeddings).sum()
    
    if has_nan:
        print(f"❌ Found {nan_count} NaN values in embeddings")
        return False
    
    print("✅ No NaN values found")
    return True


def check_inf(embeddings: Union[sparse.csr_matrix, np.ndarray]) -> bool:
    """
    Check for infinite values in embeddings.
    
    Args:
        embeddings: Embedding matrix
    
    Returns:
        True if no inf found, False otherwise
    """
    if sparse.issparse(embeddings):
        has_inf = np.isinf(embeddings.data).any()
        inf_count = np.isinf(embeddings.data).sum()
    else:
        has_inf = np.isinf(embeddings).any()
        inf_count = np.isinf(embeddings).sum()
    
    if has_inf:
        print(f"❌ Found {inf_count} infinite values in embeddings")
        return False
    
    print("✅ No infinite values found")
    return True


def check_class_distribution(
    labels: np.ndarray,
    split_name: str = "dataset"
) -> dict:
    """
    Check class distribution in labels.
    
    Args:
        labels: Label array
        split_name: Name of the split (for reporting)
    
    Returns:
        Dictionary with class distribution statistics
    """
    unique, counts = np.unique(labels, return_counts=True)
    n_classes = len(unique)
    n_samples = len(labels)
    
    distribution = {}
    for cls, count in zip(unique, counts):
        pct = (count / n_samples) * 100
        distribution[cls] = {'count': int(count), 'percentage': pct}
        print(f"  Class '{cls}': {count} samples ({pct:.2f}%)")
    
    print(f"✅ {split_name}: {n_samples} samples, {n_classes} classes")
    
    return {
        'n_samples': n_samples,
        'n_classes': n_classes,
        'distribution': distribution,
        'is_balanced': _check_balance(counts)
    }


def _check_balance(counts: np.ndarray, threshold: float = 0.1) -> bool:
    """Check if classes are reasonably balanced."""
    if len(counts) < 2:
        return True
    
    min_count = counts.min()
    max_count = counts.max()
    ratio = min_count / max_count
    
    return ratio >= threshold


def check_embedding_stats(embeddings: Union[sparse.csr_matrix, np.ndarray]) -> dict:
    """
    Get statistics about embeddings.
    
    Args:
        embeddings: Embedding matrix
    
    Returns:
        Dictionary with statistics
    """
    if sparse.issparse(embeddings):
        n_samples, n_features = embeddings.shape
        density = embeddings.nnz / (n_samples * n_features)
        stats = {
            'shape': embeddings.shape,
            'dtype': str(embeddings.dtype),
            'sparse': True,
            'density': density,
            'nnz': embeddings.nnz
        }
        print(f"✅ Sparse matrix: {n_samples}x{n_features}, density={density:.4f}")
    else:
        n_samples, n_features = embeddings.shape
        stats = {
            'shape': embeddings.shape,
            'dtype': str(embeddings.dtype),
            'sparse': False,
            'mean': float(embeddings.mean()),
            'std': float(embeddings.std()),
            'min': float(embeddings.min()),
            'max': float(embeddings.max())
        }
        print(f"✅ Dense matrix: {n_samples}x{n_features}, dtype={embeddings.dtype}")
    
    return stats


def full_sanity_check(
    embeddings: Union[sparse.csr_matrix, np.ndarray],
    labels: np.ndarray,
    split_name: str = "dataset"
) -> dict:
    """
    Run complete sanity check on a data split.
    
    Args:
        embeddings: Embedding matrix
        labels: Label array
        split_name: Name of the split
    
    Returns:
        Dictionary with all check results
    """
    print(f"\n{'='*60}")
    print(f"🔍 Sanity Check: {split_name}")
    print(f"{'='*60}\n")
    
    results = {
        'split_name': split_name,
        'shapes_ok': check_shapes(embeddings, labels),
        'no_nans': check_nans(embeddings),
        'no_inf': check_inf(embeddings),
        'embedding_stats': check_embedding_stats(embeddings),
        'class_distribution': check_class_distribution(labels, split_name)
    }
    
    all_ok = all([
        results['shapes_ok'],
        results['no_nans'],
        results['no_inf']
    ])
    
    results['all_checks_passed'] = all_ok
    
    print(f"\n{'='*60}")
    if all_ok:
        print(f"✅ All sanity checks passed for {split_name}")
    else:
        print(f"❌ Some sanity checks failed for {split_name}")
    print(f"{'='*60}\n")
    
    return results


def sanity_check_from_files(
    embedding_file: Union[str, Path],
    labels_file: Union[str, Path],
    embedding_type: str = 'auto',
    split_name: str = "dataset"
) -> dict:
    """
    Run sanity check loading from files.
    
    Args:
        embedding_file: Path to embedding file
        labels_file: Path to labels file
        embedding_type: Type of embedding ('auto', 'sparse', 'dense')
        split_name: Name of the split
    
    Returns:
        Dictionary with all check results
    """
    print(f"Loading embeddings from: {embedding_file}")
    print(f"Loading labels from: {labels_file}\n")
    
    embeddings = load_embedding(embedding_file, embedding_type)
    labels = load_labels(labels_file)
    
    return full_sanity_check(embeddings, labels, split_name)

