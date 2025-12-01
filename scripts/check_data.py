"""Quick script to check data structure."""
import pandas as pd
from pathlib import Path

df = pd.read_csv('data/raw/Base_dados_textos_6_classes.csv')
print(f"Shape: {df.shape}")
print(f"\nColunas: {list(df.columns)}")
print(f"\nClasses únicas: {sorted(df['Classe'].unique())}")
print(f"\nDistribuição de classes:")
print(df['Classe'].value_counts().sort_index())
print(f"\nPrimeiras linhas:")
print(df[['Texto Expandido', 'Classe', 'Categoria']].head(3))

