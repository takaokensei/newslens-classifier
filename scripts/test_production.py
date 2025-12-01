"""
Script de teste para validar o ambiente de produção.
Testa todas as funcionalidades principais do sistema.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import PATHS
from src.preprocessing import preprocess_text
from src.embeddings import load_tfidf_vectorizer, load_bert_model
from src.train import load_trained_models
from src.logging_system import log_prediction, load_prediction_logs, get_log_statistics
from src.class_mapping import CLASS_TO_CATEGORY


def test_model_loading():
    """Test if models can be loaded."""
    print("="*60)
    print("Test 1: Model Loading")
    print("="*60)
    try:
        models = load_trained_models()
        print("✓ Models loaded successfully")
        print(f"  Available models: {list(models.keys())}")
        return True
    except Exception as e:
        print(f"✗ Error loading models: {e}")
        return False


def test_embeddings_loading():
    """Test if embeddings can be loaded."""
    print("\n" + "="*60)
    print("Test 2: Embeddings Loading")
    print("="*60)
    try:
        vectorizer = load_tfidf_vectorizer(PATHS['data_embeddings'] / 'tfidf_vectorizer.pkl')
        print("✓ TF-IDF vectorizer loaded")
        
        bert_model = load_bert_model()
        print("✓ BERT model loaded")
        return True
    except Exception as e:
        print(f"✗ Error loading embeddings: {e}")
        return False


def test_classification():
    """Test text classification."""
    print("\n" + "="*60)
    print("Test 3: Text Classification")
    print("="*60)
    
    test_text = "O Brasil registrou crescimento econômico de 2,5% no último trimestre."
    
    try:
        # Load models
        models = load_trained_models()
        vectorizer = load_tfidf_vectorizer(PATHS['data_embeddings'] / 'tfidf_vectorizer.pkl')
        bert_model = load_bert_model()
        
        # Preprocess
        processed = preprocess_text(test_text)
        print(f"✓ Text preprocessed: {processed[:50]}...")
        
        # TF-IDF + SVM
        tfidf_emb = vectorizer.transform([processed])
        pred_tfidf = models['tfidf_svm'].predict(tfidf_emb)[0]
        proba_tfidf = models['tfidf_svm'].predict_proba(tfidf_emb)[0]
        print(f"✓ TF-IDF + SVM: {CLASS_TO_CATEGORY[pred_tfidf]} (score: {proba_tfidf[pred_tfidf]:.3f})")
        
        # BERT + SVM
        bert_emb = bert_model.encode([processed], convert_to_numpy=True, show_progress_bar=False)
        pred_bert = models['bert_svm'].predict(bert_emb)[0]
        proba_bert = models['bert_svm'].predict_proba(bert_emb)[0]
        print(f"✓ BERT + SVM: {CLASS_TO_CATEGORY[pred_bert]} (score: {proba_bert[pred_bert]:.3f})")
        
        return True
    except Exception as e:
        print(f"✗ Error in classification: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_logging():
    """Test logging system."""
    print("\n" + "="*60)
    print("Test 4: Logging System")
    print("="*60)
    
    try:
        # Log a prediction
        log_prediction(
            texto="Teste de log do sistema de produção",
            classe_predita=0,
            score=0.95,
            embedding_usado="TF-IDF",
            modelo_usado="SVM",
            fonte="test_script",
            categoria_predita="Economia"
        )
        print("✓ Prediction logged successfully")
        
        # Load logs
        logs = load_prediction_logs()
        print(f"✓ Logs loaded: {len(logs)} total predictions")
        
        # Get statistics
        stats = get_log_statistics()
        print(f"✓ Statistics computed")
        if stats:
            print(f"  Total: {stats.get('total', len(logs))}")
            print(f"  Average score: {stats.get('avg_score', logs['score'].mean() if 'score' in logs.columns else 0):.3f}")
        else:
            print(f"  Total: {len(logs)}")
            print(f"  Average score: {logs['score'].mean():.3f if 'score' in logs.columns else 0:.3f}")
        
        return True
    except Exception as e:
        print(f"✗ Error in logging: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_production_script():
    """Test production script."""
    print("\n" + "="*60)
    print("Test 5: Production Script")
    print("="*60)
    
    try:
        from scripts.processar_novos import process_new_texts
        
        # Check if test file exists
        test_file = PATHS['data_novos'] / 'test_sample.txt'
        if test_file.exists():
            print(f"✓ Test file found: {test_file}")
            
            # Process texts
            results = process_new_texts(
                directory=PATHS['data_novos'],
                use_model='best',
                log_predictions=True
            )
            
            if not results.empty:
                print(f"✓ Processed {len(results)} texts")
                print(f"  Results: {results['categoria_predita'].value_counts().to_dict()}")
                return True
            else:
                print("⚠ No results returned (directory might be empty)")
                return True
        else:
            print("⚠ Test file not found, skipping production script test")
            return True
    except Exception as e:
        print(f"✗ Error in production script: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("PRODUCTION ENVIRONMENT VALIDATION")
    print("="*60)
    print()
    
    tests = [
        test_model_loading,
        test_embeddings_loading,
        test_classification,
        test_logging,
        test_production_script
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            results.append(False)
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed! Production environment is ready.")
        return 0
    else:
        print("✗ Some tests failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    exit(main())

