"""
SQLite database module for prediction logging.
Implements database initialization and logging functions as bonus feature (Módulo 16).
"""
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import pandas as pd

from src.config import PATHS


def init_database(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """
    Initialize SQLite database for prediction logs.
    
    Args:
        db_path: Path to database file (default: logs/predicoes.db)
    
    Returns:
        Database connection
    """
    if db_path is None:
        db_path = PATHS['logs'] / 'predicoes.db'
    
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # Create table if not exists
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            texto TEXT,
            classe_predita INTEGER NOT NULL,
            categoria_predita TEXT,
            score REAL NOT NULL,
            embedding_usado TEXT NOT NULL,
            modelo_usado TEXT NOT NULL,
            fonte TEXT DEFAULT 'streamlit',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create index for faster queries
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_timestamp ON predictions(timestamp)
    ''')
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_categoria ON predictions(categoria_predita)
    ''')
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_fonte ON predictions(fonte)
    ''')
    
    conn.commit()
    return conn


def log_prediction_db(
    texto: str,
    classe_predita: int,
    score: float,
    embedding_usado: str,
    modelo_usado: str,
    fonte: str = "streamlit",
    categoria_predita: Optional[str] = None,
    db_path: Optional[Path] = None
) -> None:
    """
    Log a prediction to SQLite database.
    
    Args:
        texto: Original text (or hash if very long)
        classe_predita: Predicted class index
        score: Prediction confidence/score
        embedding_usado: "TF-IDF" or "BERT"
        modelo_usado: "SVM" or "XGBoost"
        fonte: "streamlit" or "script_producao"
        categoria_predita: Optional category name
        db_path: Path to database file
    """
    conn = init_database(db_path)
    cursor = conn.cursor()
    
    # Truncate text if too long
    texto_stored = texto if len(texto) < 1000 else texto[:500] + "..."
    
    timestamp = datetime.now().isoformat()
    
    cursor.execute('''
        INSERT INTO predictions 
        (timestamp, texto, classe_predita, categoria_predita, score, 
         embedding_usado, modelo_usado, fonte)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        timestamp,
        texto_stored,
        classe_predita,
        categoria_predita or f"Class_{classe_predita}",
        score,
        embedding_usado,
        modelo_usado,
        fonte
    ))
    
    conn.commit()
    conn.close()


def load_predictions_db(
    db_path: Optional[Path] = None,
    limit: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Load predictions from SQLite database.
    
    Args:
        db_path: Path to database file
        limit: Optional limit on number of rows
        start_date: Optional start date filter (ISO format)
        end_date: Optional end date filter (ISO format)
    
    Returns:
        DataFrame with predictions
    """
    db_path = db_path or PATHS['logs'] / 'predicoes.db'
    db_path = Path(db_path)
    
    if not db_path.exists():
        return pd.DataFrame()
    
    conn = sqlite3.connect(str(db_path))
    
    query = "SELECT * FROM predictions WHERE 1=1"
    params = []
    
    if start_date:
        query += " AND timestamp >= ?"
        params.append(start_date)
    
    if end_date:
        query += " AND timestamp <= ?"
        params.append(end_date)
    
    query += " ORDER BY timestamp DESC"
    
    if limit:
        query += " LIMIT ?"
        params.append(limit)
    
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    
    # Convert timestamp to datetime
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    return df


def get_db_statistics(db_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Get statistics from SQLite database.
    
    Args:
        db_path: Path to database file
    
    Returns:
        Dictionary with statistics
    """
    db_path = db_path or PATHS['logs'] / 'predicoes.db'
    db_path = Path(db_path)
    
    if not db_path.exists():
        return {
            'total_predictions': 0,
            'by_class': {},
            'by_model': {},
            'by_embedding': {},
            'avg_score': 0.0
        }
    
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # Total predictions
    cursor.execute('SELECT COUNT(*) FROM predictions')
    total = cursor.fetchone()[0]
    
    # By category
    cursor.execute('''
        SELECT categoria_predita, COUNT(*) as count 
        FROM predictions 
        GROUP BY categoria_predita
    ''')
    by_class = {row[0]: row[1] for row in cursor.fetchall()}
    
    # By model
    cursor.execute('''
        SELECT modelo_usado, COUNT(*) as count 
        FROM predictions 
        GROUP BY modelo_usado
    ''')
    by_model = {row[0]: row[1] for row in cursor.fetchall()}
    
    # By embedding
    cursor.execute('''
        SELECT embedding_usado, COUNT(*) as count 
        FROM predictions 
        GROUP BY embedding_usado
    ''')
    by_embedding = {row[0]: row[1] for row in cursor.fetchall()}
    
    # Average score
    cursor.execute('SELECT AVG(score) FROM predictions')
    avg_score = cursor.fetchone()[0] or 0.0
    
    # Date range
    cursor.execute('SELECT MIN(timestamp), MAX(timestamp) FROM predictions')
    date_range = cursor.fetchone()
    
    conn.close()
    
    return {
        'total_predictions': total,
        'by_class': by_class,
        'by_model': by_model,
        'by_embedding': by_embedding,
        'avg_score': float(avg_score),
        'date_range': {
            'start': date_range[0] if date_range[0] else None,
            'end': date_range[1] if date_range[1] else None
        }
    }

