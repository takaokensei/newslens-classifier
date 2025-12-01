"""
Automatic model training script.
Checks if models exist and trains them if necessary.
This is useful for Streamlit Cloud deployment where models aren't committed.
"""
import sys
from pathlib import Path
import traceback

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import PATHS
from src.prepare_data import prepare_full_pipeline
from src.train import train_all_models


def check_models_exist() -> bool:
    """Check if all required models exist."""
    required_models = [
        'tfidf_svm.pkl',
        'tfidf_xgb.pkl',
        'bert_svm.pkl',
        'bert_xgb.pkl'
    ]
    
    # Also check for optimized versions
    optimized_models = [
        'tfidf_svm_optimized.pkl',
        'tfidf_xgb_optimized.pkl',
        'bert_svm_optimized.pkl',
        'bert_xgb_optimized.pkl'
    ]
    
    # Check if at least regular models exist
    regular_exist = all((PATHS['models'] / model).exists() for model in required_models)
    
    # Check if optimized models exist
    optimized_exist = all((PATHS['models'] / model).exists() for model in optimized_models)
    
    return regular_exist or optimized_exist


def check_data_exists() -> bool:
    """Check if training data exists."""
    # Check for CSV in data/raw/
    raw_dir = PATHS['data_raw']
    csv_files = list(raw_dir.glob('*.csv'))
    return len(csv_files) > 0


def check_embeddings_exist() -> bool:
    """Check if embeddings already exist (to avoid regenerating)."""
    required_embeddings = [
        'tfidf_train.npz',
        'tfidf_val.npz',
        'tfidf_test.npz',
        'bert_train.npy',
        'bert_val.npy',
        'bert_test.npy',
        'tfidf_vectorizer.pkl'
    ]
    
    embeddings_dir = PATHS['data_embeddings']
    return all((embeddings_dir / emb).exists() for emb in required_embeddings)


def train_models_automatically(force: bool = False) -> dict:
    """
    Automatically train models if they don't exist.
    
    Args:
        force: If True, retrain even if models exist
    
    Returns:
        Dictionary with status and message
    """
    print("="*60)
    print("🤖 Automatic Model Training")
    print("="*60)
    
    # Check if models already exist
    if not force and check_models_exist():
        print("✅ Models already exist. Skipping training.")
        return {
            'success': True,
            'message': 'Models already exist',
            'models_trained': False
        }
    
    # Check if data exists
    if not check_data_exists():
        error_msg = (
            f"❌ Training data not found in {PATHS['data_raw']}\n"
            f"Please add a CSV file with columns: 'Texto', 'Classe', 'Categoria'"
        )
        print(error_msg)
        return {
            'success': False,
            'message': error_msg,
            'models_trained': False
        }
    
    try:
        # Step 1: Prepare data and generate embeddings
        print("\n📊 Step 1: Preparing data and generating embeddings...")
        print("-" * 60)
        
        # Check if embeddings already exist (to save time)
        if check_embeddings_exist():
            print("✅ Embeddings already exist. Loading from disk...")
            from src.data_loader import load_embedding, load_labels
            
            data = {
                'embeddings': {
                    'tfidf': {
                        'train': load_embedding(PATHS['data_embeddings'] / 'tfidf_train.npz', 'tfidf'),
                        'val': load_embedding(PATHS['data_embeddings'] / 'tfidf_val.npz', 'tfidf'),
                        'test': load_embedding(PATHS['data_embeddings'] / 'tfidf_test.npz', 'tfidf')
                    },
                    'bert': {
                        'train': load_embedding(PATHS['data_embeddings'] / 'bert_train.npy', 'bert'),
                        'val': load_embedding(PATHS['data_embeddings'] / 'bert_val.npy', 'bert'),
                        'test': load_embedding(PATHS['data_embeddings'] / 'bert_test.npy', 'bert')
                    }
                },
                'labels': {
                    'train': load_labels(PATHS['data_processed'] / 'labels_train.npy'),
                    'val': load_labels(PATHS['data_processed'] / 'labels_val.npy'),
                    'test': load_labels(PATHS['data_processed'] / 'labels_test.npy')
                }
            }
        else:
            print("🔄 Generating embeddings (this may take a few minutes)...")
            data = prepare_full_pipeline(
                data_path=None,  # Will look in data/raw/
                embedding_type='both',  # Generate both TF-IDF and BERT
                save=True
            )
        
        # Step 2: Train models
        print("\n🎯 Step 2: Training models...")
        print("-" * 60)
        models = train_all_models(
            data['embeddings'],
            data['labels'],
            save=True
        )
        
        print("\n" + "="*60)
        print("✅ Automatic training completed successfully!")
        print("="*60)
        
        return {
            'success': True,
            'message': 'Models trained successfully',
            'models_trained': True,
            'models': models
        }
        
    except Exception as e:
        error_msg = f"❌ Error during automatic training: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        return {
            'success': False,
            'message': error_msg,
            'models_trained': False,
            'error': str(e)
        }


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Automatically train models if they don\'t exist')
    parser.add_argument('--force', action='store_true', help='Force retraining even if models exist')
    args = parser.parse_args()
    
    result = train_models_automatically(force=args.force)
    
    if result['success']:
        print("\n✅ Success!")
        sys.exit(0)
    else:
        print("\n❌ Failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()

