"""
Class mapping module.
Maps numeric class indices to category names for better interpretability.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional

from src.config import PATHS


# Class to Category mapping (based on data analysis)
CLASS_TO_CATEGORY = {
    0: 'Economia',
    1: 'Esportes',
    2: 'Polícia e Direitos',
    3: 'Política',
    4: 'Turismo',
    5: 'Variedades e Sociedade'
}

CATEGORY_TO_CLASS = {v: k for k, v in CLASS_TO_CATEGORY.items()}


def get_class_mapping(data_path: Optional[Path] = None) -> Dict[int, str]:
    """
    Load class to category mapping from data file.
    
    Args:
        data_path: Path to CSV file. If None, looks in data/raw/
    
    Returns:
        Dictionary mapping class index to category name
    """
    if data_path is None:
        raw_dir = PATHS['data_raw']
        csv_files = list(raw_dir.glob('*.csv'))
        if not csv_files:
            return CLASS_TO_CATEGORY  # Use default mapping
        data_path = csv_files[0]
    
    try:
        df = pd.read_csv(data_path)
        if 'Classe' in df.columns and 'Categoria' in df.columns:
            mapping = dict(zip(df['Classe'].unique(), df.groupby('Classe')['Categoria'].first()))
            return mapping
    except Exception as e:
        print(f"Warning: Could not load mapping from data file: {e}")
    
    return CLASS_TO_CATEGORY


def map_classes_to_categories(class_indices: np.ndarray, mapping: Optional[Dict] = None) -> np.ndarray:
    """
    Map numeric class indices to category names.
    
    Args:
        class_indices: Array of numeric class indices
        mapping: Optional mapping dictionary. If None, uses default.
    
    Returns:
        Array of category names
    """
    if mapping is None:
        mapping = CLASS_TO_CATEGORY
    
    return np.array([mapping.get(int(c), f'Unknown_{c}') for c in class_indices])


def map_categories_to_classes(categories: np.ndarray, mapping: Optional[Dict] = None) -> np.ndarray:
    """
    Map category names to numeric class indices.
    
    Args:
        categories: Array of category names
        mapping: Optional reverse mapping. If None, uses default.
    
    Returns:
        Array of numeric class indices
    """
    if mapping is None:
        reverse_mapping = CATEGORY_TO_CLASS
    else:
        reverse_mapping = {v: k for k, v in mapping.items()}
    
    return np.array([reverse_mapping.get(c, -1) for c in categories])

