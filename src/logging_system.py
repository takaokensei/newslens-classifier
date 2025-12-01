"""
Logging system for prediction tracking.
Implements log_prediction() and manages logs/predicoes.csv
Also supports SQLite database (bonus feature from Módulo 16).
"""
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import os

from src.config import PATHS

# Try to import database module (optional bonus feature)
try:
    from src.database import log_prediction_db, load_predictions_db, get_db_statistics
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False


def log_prediction(
    texto: str,
    classe_predita: int,
    score: float,
    embedding_usado: str,
    modelo_usado: str,
    fonte: str = "streamlit",
    categoria_predita: Optional[str] = None,
    texto_hash: Optional[str] = None,
    use_db: bool = True  # Bonus: Use SQLite if available
) -> None:
    """
    Log a prediction to logs/predicoes.csv and optionally to SQLite database.
    
    Args:
        texto: Original text (or hash if very long)
        classe_predita: Predicted class index
        score: Prediction confidence/score
        embedding_usado: "TF-IDF" or "BERT"
        modelo_usado: "SVM" or "XGBoost"
        fonte: "streamlit" or "script_producao"
        categoria_predita: Optional category name
        texto_hash: Optional hash of text (if text is too long)
        use_db: Whether to also log to SQLite database (bonus feature)
    """
    log_file = PATHS['logs'] / 'predicoes.csv'
    
    # Create log entry
    entry = {
        'timestamp': datetime.now().isoformat(),
        'texto': texto if len(texto) < 1000 else (texto_hash or texto[:500] + "..."),
        'classe_predita': classe_predita,
        'categoria_predita': categoria_predita or f"Class_{classe_predita}",
        'score': score,
        'embedding_usado': embedding_usado,
        'modelo_usado': modelo_usado,
        'fonte': fonte
    }
    
    # Append to CSV (always, for compatibility)
    df_new = pd.DataFrame([entry])
    
    if log_file.exists():
        df_existing = pd.read_csv(log_file)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new
    
    # Save CSV
    df_combined.to_csv(log_file, index=False)
    
    # Bonus: Also log to SQLite database if available
    if use_db and DB_AVAILABLE:
        try:
            log_prediction_db(
                texto=texto,
                classe_predita=classe_predita,
                score=score,
                embedding_usado=embedding_usado,
                modelo_usado=modelo_usado,
                fonte=fonte,
                categoria_predita=categoria_predita
            )
        except Exception as e:
            # Silently fail if DB logging fails (CSV is primary)
            pass


def load_prediction_logs(
    log_file: Optional[Path] = None,
    limit: Optional[int] = None
) -> pd.DataFrame:
    """
    Load prediction logs from CSV.
    
    Args:
        log_file: Path to log file (default: logs/predicoes.csv)
        limit: Optional limit on number of rows
    
    Returns:
        DataFrame with logs
    """
    if log_file is None:
        log_file = PATHS['logs'] / 'predicoes.csv'
    
    log_file = Path(log_file)
    if not log_file.exists():
        return pd.DataFrame()
    
    df = pd.read_csv(log_file)
    
    # Convert timestamp to datetime
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    if limit:
        df = df.tail(limit)
    
    return df


def get_log_statistics(log_file: Optional[Path] = None) -> Dict[str, Any]:
    """
    Get statistics from prediction logs.
    
    Args:
        log_file: Path to log file
    
    Returns:
        Dictionary with statistics
    """
    df = load_prediction_logs(log_file)
    
    if df.empty:
        return {
            'total_predictions': 0,
            'by_class': {},
            'by_model': {},
            'by_embedding': {},
            'avg_score': 0.0
        }
    
    stats = {
        'total_predictions': len(df),
        'by_class': df['categoria_predita'].value_counts().to_dict() if 'categoria_predita' in df.columns else {},
        'by_model': df['modelo_usado'].value_counts().to_dict() if 'modelo_usado' in df.columns else {},
        'by_embedding': df['embedding_usado'].value_counts().to_dict() if 'embedding_usado' in df.columns else {},
        'avg_score': float(df['score'].mean()) if 'score' in df.columns else 0.0,
        'date_range': {
            'start': df['timestamp'].min().isoformat() if 'timestamp' in df.columns else None,
            'end': df['timestamp'].max().isoformat() if 'timestamp' in df.columns else None
        }
    }
    
    return stats

