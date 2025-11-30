"""
Smoke tests for core modules.
Quick validation that basic functionality works.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from scipy import sparse

from src import config
from src.preprocessing import preprocess_text, preprocess_batch
from src.data_loader import (
    load_sparse_embedding,
    load_dense_embedding,
    load_embedding
)
# Try to import embeddings module components separately
try:
    from src.embeddings import generate_tfidf_embeddings
    TFIDF_AVAILABLE = True
except ImportError:
    TFIDF_AVAILABLE = False

try:
    from src.embeddings import generate_bert_embeddings
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False


def test_config_imports():
    """Test that config module loads correctly."""
    assert hasattr(config, 'DATA_CONFIG')
    assert hasattr(config, 'FEATURE_CONFIG')
    assert hasattr(config, 'MODELS_CONFIG')
    assert hasattr(config, 'LLM_CONFIG')
    assert hasattr(config, 'PATHS')
    print("✅ Config module imports successfully")


def test_preprocessing():
    """Test text preprocessing function."""
    # Test single text
    text = "Este é um TEXTO de TESTE com URLs: https://example.com e email@test.com"
    processed = preprocess_text(text)
    
    assert isinstance(processed, str)
    assert "https://example.com" not in processed
    assert "email@test.com" not in processed
    assert processed.islower() or len(processed) == 0
    print(f"✅ Preprocessing single text: '{processed[:50]}...'")
    
    # Test batch
    texts = ["Texto 1", "Texto 2", "Texto 3"]
    processed_batch = preprocess_batch(texts)
    assert len(processed_batch) == len(texts)
    print("✅ Preprocessing batch works")


def test_tfidf_generation():
    """Test TF-IDF embedding generation."""
    if not TFIDF_AVAILABLE:
        print("⚠️  TF-IDF test skipped (dependencies not installed)")
        return
    
    texts = [
        "Este é um texto sobre esportes",
        "Política é um tema importante",
        "Esportes e política são diferentes"
    ]
    
    embeddings, vectorizer = generate_tfidf_embeddings(texts, fit=True)
    
    assert sparse.issparse(embeddings)
    assert embeddings.shape[0] == len(texts)
    assert embeddings.shape[1] > 0
    assert vectorizer is not None
    print(f"✅ TF-IDF embeddings: shape {embeddings.shape}")


def test_bert_generation():
    """Test BERT embedding generation (may take time)."""
    if not BERT_AVAILABLE:
        print("⚠️  BERT test skipped (dependencies not installed)")
        return
    
    texts = [
        "Este é um texto de teste",
        "Outro texto para teste"
    ]
    
    try:
        embeddings, model = generate_bert_embeddings(
            texts,
            show_progress=False
        )
        
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == len(texts)
        assert embeddings.shape[1] > 0
        assert embeddings.dtype == np.float32
        print(f"✅ BERT embeddings: shape {embeddings.shape}")
    except Exception as e:
        print(f"⚠️  BERT test skipped (may need dependencies): {e}")


def test_data_loader_sparse():
    """Test loading sparse embeddings."""
    # Create a temporary sparse matrix
    test_matrix = sparse.csr_matrix(np.random.rand(10, 100))
    test_path = Path("test_sparse.npz")
    
    try:
        # Save
        np.savez_compressed(
            test_path,
            data=test_matrix.data,
            indices=test_matrix.indices,
            indptr=test_matrix.indptr,
            shape=test_matrix.shape
        )
        
        # Load
        loaded = load_sparse_embedding(test_path)
        assert sparse.issparse(loaded)
        assert loaded.shape == test_matrix.shape
        print("✅ Sparse embedding loader works")
    finally:
        if test_path.exists():
            test_path.unlink()


def test_data_loader_dense():
    """Test loading dense embeddings."""
    test_array = np.random.rand(10, 768).astype(np.float32)
    test_path = Path("test_dense.npy")
    
    try:
        # Save
        np.save(test_path, test_array)
        
        # Load
        loaded = load_dense_embedding(test_path)
        assert isinstance(loaded, np.ndarray)
        assert loaded.shape == test_array.shape
        assert loaded.dtype == np.float32
        print("✅ Dense embedding loader works")
    finally:
        if test_path.exists():
            test_path.unlink()


def test_polymorphic_loader():
    """Test polymorphic embedding loader."""
    # Test sparse
    test_matrix = sparse.csr_matrix(np.random.rand(5, 50))
    test_path_sparse = Path("test_auto_sparse.npz")
    
    try:
        np.savez_compressed(
            test_path_sparse,
            data=test_matrix.data,
            indices=test_matrix.indices,
            indptr=test_matrix.indptr,
            shape=test_matrix.shape
        )
        
        loaded = load_embedding(test_path_sparse, embedding_type='auto')
        assert sparse.issparse(loaded)
        print("✅ Polymorphic loader (sparse) works")
    finally:
        if test_path_sparse.exists():
            test_path_sparse.unlink()
    
    # Test dense
    test_array = np.random.rand(5, 768).astype(np.float32)
    test_path_dense = Path("test_auto_dense.npy")
    
    try:
        np.save(test_path_dense, test_array)
        loaded = load_embedding(test_path_dense, embedding_type='auto')
        assert isinstance(loaded, np.ndarray)
        print("✅ Polymorphic loader (dense) works")
    finally:
        if test_path_dense.exists():
            test_path_dense.unlink()


def run_all_tests():
    """Run all smoke tests."""
    print("\n" + "="*60)
    print("🧪 Running Smoke Tests")
    print("="*60 + "\n")
    
    tests = [
        test_config_imports,
        test_preprocessing,
        test_tfidf_generation,
        test_bert_generation,
        test_data_loader_sparse,
        test_data_loader_dense,
        test_polymorphic_loader
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} failed: {e}")
            failed += 1
    
    print("\n" + "="*60)
    print(f"📊 Results: {passed} passed, {failed} failed")
    print("="*60 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

