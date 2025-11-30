"""
Test sanity check module.
"""
import sys
from pathlib import Path
import numpy as np
from scipy import sparse

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sanity_check import (
    check_shapes,
    check_nans,
    check_inf,
    check_class_distribution,
    check_embedding_stats,
    full_sanity_check
)


def test_sanity_check_basic():
    """Test basic sanity check functions."""
    print("\n" + "="*60)
    print("🧪 Testing Sanity Check Module")
    print("="*60 + "\n")
    
    # Create test data
    n_samples = 100
    n_features = 50
    n_classes = 3
    
    # Test sparse embeddings
    sparse_emb = sparse.csr_matrix(np.random.rand(n_samples, n_features))
    labels = np.random.randint(0, n_classes, n_samples)
    
    print("Testing with sparse embeddings:")
    assert check_shapes(sparse_emb, labels)
    assert check_nans(sparse_emb)
    assert check_inf(sparse_emb)
    stats = check_embedding_stats(sparse_emb)
    assert stats['sparse'] == True
    print()
    
    # Test dense embeddings
    dense_emb = np.random.rand(n_samples, n_features).astype(np.float32)
    
    print("Testing with dense embeddings:")
    assert check_shapes(dense_emb, labels)
    assert check_nans(dense_emb)
    assert check_inf(dense_emb)
    stats = check_embedding_stats(dense_emb)
    assert stats['sparse'] == False
    print()
    
    # Test class distribution
    dist = check_class_distribution(labels, "test_split")
    assert dist['n_samples'] == n_samples
    assert dist['n_classes'] == n_classes
    print()
    
    # Test full sanity check
    results = full_sanity_check(dense_emb, labels, "test_split")
    assert results['all_checks_passed'] == True
    
    print("✅ All sanity check tests passed!\n")


if __name__ == "__main__":
    test_sanity_check_basic()

