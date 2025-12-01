"""
Test script for SQLite database functionality.
Validates that database logging works correctly.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database import init_database, log_prediction_db, load_predictions_db, get_db_statistics
from src.config import PATHS


def test_database():
    """Test SQLite database functionality."""
    print("="*60)
    print("SQLite Database Test")
    print("="*60)
    print()
    
    # Test 1: Initialize database
    print("Test 1: Initializing database...")
    try:
        conn = init_database()
        print("✅ Database initialized successfully")
        conn.close()
    except Exception as e:
        print(f"❌ Error initializing database: {e}")
        return False
    
    # Test 2: Log predictions
    print("\nTest 2: Logging predictions...")
    try:
        log_prediction_db(
            texto="Teste de predição no banco SQLite",
            classe_predita=0,
            score=0.95,
            embedding_usado="TF-IDF",
            modelo_usado="SVM",
            fonte="test_script",
            categoria_predita="Economia"
        )
        log_prediction_db(
            texto="Segundo teste de predição",
            classe_predita=1,
            score=0.88,
            embedding_usado="BERT",
            modelo_usado="XGBoost",
            fonte="test_script",
            categoria_predita="Esportes"
        )
        print("✅ Predictions logged successfully")
    except Exception as e:
        print(f"❌ Error logging predictions: {e}")
        return False
    
    # Test 3: Load predictions
    print("\nTest 3: Loading predictions...")
    try:
        df = load_predictions_db()
        print(f"✅ Loaded {len(df)} predictions from database")
        if not df.empty:
            print(f"   Columns: {list(df.columns)}")
            print(f"   Sample data:")
            print(df[['timestamp', 'categoria_predita', 'score', 'embedding_usado']].head())
    except Exception as e:
        print(f"❌ Error loading predictions: {e}")
        return False
    
    # Test 4: Get statistics
    print("\nTest 4: Getting statistics...")
    try:
        stats = get_db_statistics()
        print("✅ Statistics retrieved successfully")
        print(f"   Total predictions: {stats['total_predictions']}")
        print(f"   By class: {stats['by_class']}")
        print(f"   By model: {stats['by_model']}")
        print(f"   By embedding: {stats['by_embedding']}")
        print(f"   Average score: {stats['avg_score']:.3f}")
    except Exception as e:
        print(f"❌ Error getting statistics: {e}")
        return False
    
    print("\n" + "="*60)
    print("✅ All database tests passed!")
    print("="*60)
    return True


if __name__ == "__main__":
    success = test_database()
    exit(0 if success else 1)

