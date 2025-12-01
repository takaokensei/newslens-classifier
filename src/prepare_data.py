"""
Data preparation module.
Handles data loading, splitting, preprocessing, and embedding generation.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split
import joblib

from src.config import DATA_CONFIG, FEATURE_CONFIG, PATHS
from src.preprocessing import preprocess_batch
from src.embeddings import generate_tfidf_embeddings, generate_bert_embeddings
from src.sanity_check import full_sanity_check


def load_raw_data(data_path: Optional[Path] = None) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Load raw data from CSV file.
    
    Args:
        data_path: Path to CSV file. If None, looks in data/raw/
    
    Returns:
        Tuple of (dataframe, labels array)
    """
    if data_path is None:
        # Look for CSV files in data/raw
        raw_dir = PATHS['data_raw']
        csv_files = list(raw_dir.glob('*.csv'))
        if not csv_files:
            raise FileNotFoundError(
                f"No CSV files found in {raw_dir}. "
                "Please place your dataset CSV file in data/raw/"
            )
        data_path = csv_files[0]
        print(f"📂 Loading data from: {data_path}")
    
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # Try common column names for text and labels
    df = pd.read_csv(data_path)
    
    # Detect text column (common names: text, texto, content, noticia, etc.)
    # Priority: "Texto Expandido" > "Texto Original" > other text columns
    text_cols = [col for col in df.columns if col.lower() in 
                 ['texto expandido', 'texto original', 'text', 'texto', 'content', 'noticia', 'news', 'article']]
    if not text_cols:
        # Try partial match
        text_cols = [col for col in df.columns if 'texto' in col.lower() or 'text' in col.lower()]
    
    if not text_cols:
        # Assume first column is text
        text_col = df.columns[0]
        print(f"⚠️  No text column found, using first column: {text_col}")
    else:
        # Prefer "Texto Expandido" if available
        if 'Texto Expandido' in text_cols:
            text_col = 'Texto Expandido'
        elif 'Texto Original' in text_cols:
            text_col = 'Texto Original'
        else:
            text_col = text_cols[0]
        print(f"✅ Using text column: {text_col}")
    
    # Detect label column (common names: label, classe, category, etc.)
    # Priority: "Categoria" > "Classe" > other label columns
    # Note: We prefer "Categoria" for interpretability, but will convert to numeric if needed
    label_cols = [col for col in df.columns if col.lower() in 
                  ['classe', 'categoria', 'label', 'class', 'category']]
    if not label_cols:
        # Try partial match
        label_cols = [col for col in df.columns if 'classe' in col.lower() or 'categoria' in col.lower()]
    
    if not label_cols:
        # Assume last column is label
        label_col = df.columns[-1]
        print(f"⚠️  No label column found, using last column: {label_col}")
    else:
        # Prefer "Categoria" if available (more interpretable)
        if 'Categoria' in label_cols:
            label_col = 'Categoria'
            # Check if we also have "Classe" for numeric mapping
            if 'Classe' in df.columns:
                # Use Categoria but store mapping for later
                print(f"✅ Using label column: {label_col} (with Classe mapping available)")
            else:
                print(f"✅ Using label column: {label_col}")
        elif 'Classe' in label_cols:
            label_col = 'Classe'
            print(f"✅ Using label column: {label_col}")
        else:
            label_col = label_cols[0]
            print(f"✅ Using label column: {label_col}")
    
    texts = df[text_col].astype(str).values
    labels_raw = df[label_col].values
    
    # Convert categorical labels to numeric if needed
    # If label_col is "Categoria", we need to map to numeric classes
    if label_col == 'Categoria' and 'Classe' in df.columns:
        # Use the numeric Classe column for training
        labels = df['Classe'].values
        print(f"📊 Using numeric 'Classe' for training (mapped from 'Categoria')")
    elif label_col == 'Categoria':
        # Need to create numeric mapping
        from src.class_mapping import map_categories_to_classes
        labels = map_categories_to_classes(labels_raw)
        print(f"📊 Converted 'Categoria' to numeric classes for training")
    else:
        labels = labels_raw
    
    # Remove rows with empty texts or NaN
    texts_series = pd.Series(texts)
    mask = (texts_series != '') & (texts_series != 'nan') & (~texts_series.isna())
    
    texts = texts[mask.values]
    labels = labels[mask.values]
    
    print(f"📊 Loaded {len(texts)} samples with {len(np.unique(labels))} classes")
    print(f"   Classes: {sorted(np.unique(labels))}")
    
    return df, labels


def split_data(
    texts: np.ndarray,
    labels: np.ndarray,
    test_size: float = None,
    val_size: float = None,
    random_state: int = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train/validation/test sets with stratification.
    
    Args:
        texts: Array of texts
        labels: Array of labels
        test_size: Proportion for test set (default from config)
        val_size: Proportion for validation from remaining (default from config)
        random_state: Random state (default from config)
    
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    config = DATA_CONFIG
    test_size = test_size or config['test_size']
    val_size = val_size or config['val_size']
    random_state = random_state or config['random_state']
    
    # First split: train+val (80%) vs test (20%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        texts,
        labels,
        test_size=test_size,
        stratify=labels if config['stratify'] else None,
        random_state=random_state
    )
    
    # Second split: train (75% of 80% = 60%) vs val (25% of 80% = 20%)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_size,
        stratify=y_temp if config['stratify'] else None,
        random_state=random_state
    )
    
    print(f"\n📊 Data Split:")
    print(f"  Train: {len(X_train)} samples ({len(X_train)/len(texts)*100:.1f}%)")
    print(f"  Validation: {len(X_val)} samples ({len(X_val)/len(texts)*100:.1f}%)")
    print(f"  Test: {len(X_test)} samples ({len(X_test)/len(texts)*100:.1f}%)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def prepare_embeddings(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    embedding_type: str = 'both',
    save: bool = True
) -> dict:
    """
    Generate embeddings for all splits.
    
    Args:
        X_train: Training texts
        X_val: Validation texts
        X_test: Test texts
        embedding_type: 'tfidf', 'bert', or 'both'
        save: Whether to save embeddings to disk
    
    Returns:
        Dictionary with embeddings and vectorizers/models
    """
    results = {}
    
    # Preprocess all texts
    print("\n🔄 Preprocessing texts...")
    X_train_processed = preprocess_batch(X_train.tolist())
    X_val_processed = preprocess_batch(X_val.tolist())
    X_test_processed = preprocess_batch(X_test.tolist())
    
    if embedding_type in ['tfidf', 'both']:
        print("\n📊 Generating TF-IDF embeddings...")
        
        # Train TF-IDF on training data
        X_train_tfidf, vectorizer = generate_tfidf_embeddings(
            X_train_processed,
            save_path=PATHS['data_embeddings'] / 'tfidf_train.npz' if save else None,
            fit=True
        )
        
        # Transform validation and test
        X_val_tfidf, _ = generate_tfidf_embeddings(
            X_val_processed,
            save_path=PATHS['data_embeddings'] / 'tfidf_val.npz' if save else None,
            fit=False,
            vectorizer=vectorizer
        )
        
        X_test_tfidf, _ = generate_tfidf_embeddings(
            X_test_processed,
            save_path=PATHS['data_embeddings'] / 'tfidf_test.npz' if save else None,
            fit=False,
            vectorizer=vectorizer
        )
        
        results['tfidf'] = {
            'train': X_train_tfidf,
            'val': X_val_tfidf,
            'test': X_test_tfidf,
            'vectorizer': vectorizer
        }
        
        # Save vectorizer
        if save:
            vectorizer_path = PATHS['data_embeddings'] / 'tfidf_vectorizer.pkl'
            joblib.dump(vectorizer, vectorizer_path)
            print(f"✅ Saved vectorizer to {vectorizer_path}")
    
    if embedding_type in ['bert', 'both']:
        print("\n🧠 Generating BERT embeddings...")
        print("   (This may take a while...)")
        
        # Generate BERT embeddings
        X_train_bert, model = generate_bert_embeddings(
            X_train_processed,
            save_path=PATHS['data_embeddings'] / 'bert_train.npy' if save else None,
            show_progress=True
        )
        
        X_val_bert, _ = generate_bert_embeddings(
            X_val_processed,
            save_path=PATHS['data_embeddings'] / 'bert_val.npy' if save else None,
            model=model,
            show_progress=True
        )
        
        X_test_bert, _ = generate_bert_embeddings(
            X_test_processed,
            save_path=PATHS['data_embeddings'] / 'bert_test.npy' if save else None,
            model=model,
            show_progress=True
        )
        
        results['bert'] = {
            'train': X_train_bert,
            'val': X_val_bert,
            'test': X_test_bert,
            'model': model
        }
    
    return results


def save_labels(y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray, save: bool = True):
    """Save labels to disk."""
    if save:
        np.save(PATHS['data_processed'] / 'labels_train.npy', y_train)
        np.save(PATHS['data_processed'] / 'labels_val.npy', y_val)
        np.save(PATHS['data_processed'] / 'labels_test.npy', y_test)
        print(f"\n✅ Saved labels to {PATHS['data_processed']}")


def prepare_full_pipeline(
    data_path: Optional[Path] = None,
    embedding_type: str = 'both',
    save: bool = True
) -> dict:
    """
    Complete data preparation pipeline.
    
    Args:
        data_path: Path to raw data CSV
        embedding_type: 'tfidf', 'bert', or 'both'
        save: Whether to save intermediate files
    
    Returns:
        Dictionary with all prepared data
    """
    print("="*60)
    print("🚀 Starting Data Preparation Pipeline")
    print("="*60)
    
    # 1. Load raw data (this already filters empty texts)
    df_full, labels_full = load_raw_data(data_path)
    
    # Get text column name
    text_cols = [col for col in df_full.columns if col.lower() in 
                 ['texto expandido', 'texto original', 'text', 'texto', 'content', 'noticia', 'news', 'article']]
    if not text_cols:
        text_cols = [col for col in df_full.columns if 'texto' in col.lower() or 'text' in col.lower()]
    if 'Texto Expandido' in text_cols:
        text_col = 'Texto Expandido'
    elif 'Texto Original' in text_cols:
        text_col = 'Texto Original'
    elif text_cols:
        text_col = text_cols[0]
    else:
        text_col = df_full.columns[0]
    
    # Get label column name
    label_cols = [col for col in df_full.columns if col.lower() in 
                  ['classe', 'categoria', 'label', 'class', 'category']]
    if 'Classe' in label_cols:
        label_col = 'Classe'
    elif 'Categoria' in label_cols:
        label_col = 'Categoria'
    elif label_cols:
        label_col = label_cols[0]
    else:
        label_col = df_full.columns[-1]
    
    # Apply same filtering as in load_raw_data
    texts = df_full[text_col].astype(str).values
    labels = df_full[label_col].values
    
    texts_series = pd.Series(texts)
    mask = (texts_series != '') & (texts_series != 'nan') & (~texts_series.isna())
    texts = texts[mask.values]
    labels = labels[mask.values]
    
    # 2. Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        texts,
        labels
    )
    
    # 3. Generate embeddings
    embeddings = prepare_embeddings(
        X_train, X_val, X_test,
        embedding_type=embedding_type,
        save=save
    )
    
    # 4. Save labels
    save_labels(y_train, y_val, y_test, save=save)
    
    # 5. Sanity checks
    print("\n🔍 Running sanity checks...")
    if 'tfidf' in embeddings:
        full_sanity_check(embeddings['tfidf']['train'], y_train, "TF-IDF Train")
        full_sanity_check(embeddings['tfidf']['val'], y_val, "TF-IDF Validation")
        full_sanity_check(embeddings['tfidf']['test'], y_test, "TF-IDF Test")
    
    if 'bert' in embeddings:
        full_sanity_check(embeddings['bert']['train'], y_train, "BERT Train")
        full_sanity_check(embeddings['bert']['val'], y_val, "BERT Validation")
        full_sanity_check(embeddings['bert']['test'], y_test, "BERT Test")
    
    print("\n" + "="*60)
    print("✅ Data Preparation Complete!")
    print("="*60)
    
    return {
        'embeddings': embeddings,
        'labels': {
            'train': y_train,
            'val': y_val,
            'test': y_test
        },
        'texts': {
            'train': X_train,
            'val': X_val,
            'test': X_test
        }
    }

