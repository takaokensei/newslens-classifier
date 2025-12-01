"""
Script to classify new texts from data/novos/ directory.
Simulates production environment by processing all texts and logging predictions.

Usage Examples:
    # Process all texts in data/novos/ using best model (BERT+SVM)
    python scripts/processar_novos.py
    
    # Use specific model
    python scripts/processar_novos.py --model tfidf_svm
    
    # Process texts from custom directory
    python scripts/processar_novos.py --directory /path/to/texts
    
    # Process without logging predictions
    python scripts/processar_novos.py --no-log
    
    # Available models: best, tfidf_svm, tfidf_xgb, bert_svm, bert_xgb
"""
import sys
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import PATHS
from src.preprocessing import preprocess_text
from src.embeddings import generate_tfidf_embeddings, generate_bert_embeddings, load_tfidf_vectorizer, load_bert_model
from src.train import load_trained_models
from src.logging_system import log_prediction
from src.class_mapping import CLASS_TO_CATEGORY


def load_texts_from_directory(directory: Path) -> list:
    """
    Load all text files from directory.
    
    Args:
        directory: Path to directory containing text files
    
    Returns:
        List of (filename, text) tuples
    """
    texts = []
    
    # Support .txt files
    txt_files = list(directory.glob('*.txt'))
    for txt_file in txt_files:
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                text = f.read().strip()
                if text:
                    texts.append((txt_file.name, text))
        except Exception as e:
            print(f"Error reading {txt_file}: {e}")
    
    # Support CSV files with text column
    csv_files = list(directory.glob('*.csv'))
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            # Try to find text column
            text_cols = [col for col in df.columns if 'texto' in col.lower() or 'text' in col.lower()]
            if text_cols:
                text_col = text_cols[0]
                for idx, row in df.iterrows():
                    text = str(row[text_col]).strip()
                    if text and text != 'nan':
                        texts.append((f"{csv_file.name}_row_{idx}", text))
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
    
    return texts


def classify_text(
    text: str,
    model_tfidf_svm,
    model_tfidf_xgb,
    model_bert_svm,
    model_bert_xgb,
    vectorizer,
    bert_model,
    use_model: str = "best"  # "best", "tfidf_svm", "tfidf_xgb", "bert_svm", "bert_xgb"
) -> dict:
    """
    Classify a single text using specified model.
    
    Args:
        text: Input text
        model_*: Trained models
        vectorizer: TF-IDF vectorizer
        bert_model: BERT model
        use_model: Which model to use
    
    Returns:
        Dictionary with prediction results
    """
    # Preprocess
    processed_text = preprocess_text(text)
    
    # Generate embeddings
    tfidf_emb = vectorizer.transform([processed_text])
    bert_emb = bert_model.encode([processed_text], convert_to_numpy=True, show_progress_bar=False)
    
    # Select model
    if use_model == "best" or use_model == "bert_svm":
        model = model_bert_svm
        embedding = bert_emb
        embedding_name = "BERT"
        model_name = "SVM"
    elif use_model == "tfidf_svm":
        model = model_tfidf_svm
        embedding = tfidf_emb
        embedding_name = "TF-IDF"
        model_name = "SVM"
    elif use_model == "bert_xgb":
        model = model_bert_xgb
        embedding = bert_emb
        embedding_name = "BERT"
        model_name = "XGBoost"
    elif use_model == "tfidf_xgb":
        model = model_tfidf_xgb
        embedding = tfidf_emb
        embedding_name = "TF-IDF"
        model_name = "XGBoost"
    else:
        raise ValueError(f"Unknown model: {use_model}")
    
    # Predict
    pred_class = model.predict(embedding)[0]
    
    # Get probability
    try:
        proba = model.predict_proba(embedding)[0]
        score = float(proba[pred_class])
    except:
        score = 1.0
    
    return {
        'classe_predita': int(pred_class),
        'categoria_predita': CLASS_TO_CATEGORY.get(int(pred_class), f"Class_{pred_class}"),
        'score': score,
        'embedding_usado': embedding_name,
        'modelo_usado': model_name
    }


def process_new_texts(
    directory: Optional[Path] = None,
    use_model: str = "best",
    log_predictions: bool = True
) -> pd.DataFrame:
    """
    Process all texts in data/novos/ directory.
    
    Args:
        directory: Directory containing new texts (default: data/novos/)
        use_model: Which model to use for classification
        log_predictions: Whether to log predictions
    
    Returns:
        DataFrame with results
    """
    if directory is None:
        directory = PATHS['data_novos']
    
    directory = Path(directory)
    if not directory.exists():
        print(f"Directory not found: {directory}")
        print(f"Creating directory...")
        directory.mkdir(parents=True, exist_ok=True)
        print(f"Please add text files (.txt) or CSV files to {directory}")
        return pd.DataFrame()
    
    print("="*60)
    print("Processing New Texts")
    print("="*60)
    print(f"Directory: {directory}")
    print(f"Model: {use_model}")
    print()
    
    # Load texts
    texts = load_texts_from_directory(directory)
    
    if not texts:
        print("No text files found in directory.")
        return pd.DataFrame()
    
    print(f"Found {len(texts)} texts to process.\n")
    
    # Load models
    print("Loading models...")
    models = load_trained_models()
    
    # Load vectorizer and BERT model
    vectorizer_path = PATHS['data_embeddings'] / 'tfidf_vectorizer.pkl'
    vectorizer = load_tfidf_vectorizer(vectorizer_path)
    
    from src.embeddings import load_bert_model
    bert_model = load_bert_model()
    
    print("Models loaded.\n")
    
    # Process each text
    results = []
    for i, (filename, text) in enumerate(texts, 1):
        print(f"Processing {i}/{len(texts)}: {filename}")
        
        try:
            result = classify_text(
                text,
                models['tfidf_svm'],
                models['tfidf_xgb'],
                models['bert_svm'],
                models['bert_xgb'],
                vectorizer,
                bert_model,
                use_model=use_model
            )
            
            result['filename'] = filename
            result['text'] = text[:200] + "..." if len(text) > 200 else text
            results.append(result)
            
            # Log prediction
            if log_predictions:
                log_prediction(
                    texto=text,
                    classe_predita=result['classe_predita'],
                    score=result['score'],
                    embedding_usado=result['embedding_usado'],
                    modelo_usado=result['modelo_usado'],
                    fonte="script_producao",
                    categoria_predita=result['categoria_predita']
                )
            
            print(f"  → {result['categoria_predita']} (score: {result['score']:.3f})")
        
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results.append({
                'filename': filename,
                'text': text[:200] + "..." if len(text) > 200 else text,
                'error': str(e)
            })
    
    # Create summary
    df_results = pd.DataFrame(results)
    
    if not df_results.empty and 'categoria_predita' in df_results.columns:
        print("\n" + "="*60)
        print("Summary")
        print("="*60)
        print(f"Total processed: {len(df_results)}")
        print(f"\nPredictions by class:")
        print(df_results['categoria_predita'].value_counts())
        print(f"\nAverage score: {df_results['score'].mean():.3f}")
    
    return df_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process new texts from data/novos/")
    parser.add_argument(
        '--model',
        choices=['best', 'tfidf_svm', 'tfidf_xgb', 'bert_svm', 'bert_xgb'],
        default='best',
        help='Model to use for classification'
    )
    parser.add_argument(
        '--directory',
        type=str,
        default=None,
        help='Directory containing texts (default: data/novos/)'
    )
    parser.add_argument(
        '--no-log',
        action='store_true',
        help='Do not log predictions'
    )
    
    args = parser.parse_args()
    
    results = process_new_texts(
        directory=Path(args.directory) if args.directory else None,
        use_model=args.model,
        log_predictions=not args.no_log
    )
    
    if not results.empty:
        output_path = PATHS['data_novos'] / 'results.csv'
        results.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")

