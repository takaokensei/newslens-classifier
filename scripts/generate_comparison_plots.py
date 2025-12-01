"""
Script to generate additional comparison plots for the report.
Creates visualizations showing trade-offs and comparisons.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import PATHS

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def plot_f1_by_class_comparison():
    """Generate F1 score comparison by class for all models."""
    table_b_path = PATHS['models'] / 'table_b_classes_with_names.csv'
    
    if not table_b_path.exists():
        print(f"Table B not found: {table_b_path}")
        return
    
    df = pd.read_csv(table_b_path)
    
    # Prepare data for plotting
    categories = df['Category'].tolist()
    models = ['TF-IDF + SVM', 'TF-IDF + XGBoost', 'BERT + SVM', 'BERT + XGBoost']
    columns = ['TFIDF+SVM', 'TFIDF+XGB', 'BERT+SVM', 'BERT+XGB']
    
    x = np.arange(len(categories))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for i, (model, col) in enumerate(zip(models, columns)):
        values = df[col].tolist()
        offset = (i - 1.5) * width
        ax.bar(x + offset, values, width, label=model, alpha=0.8)
    
    ax.set_xlabel('Categoria', fontsize=12, fontweight='bold')
    ax.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
    ax.set_title('F1-Score por Classe: Comparação entre Modelos', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.1])
    
    plt.tight_layout()
    output_path = PATHS['models'] / 'f1_by_class_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()


def plot_performance_efficiency_tradeoff():
    """Generate performance vs efficiency trade-off plot."""
    table_a_path = PATHS['models'] / 'table_a_efficiency.csv'
    
    if not table_a_path.exists():
        print(f"Table A not found: {table_a_path}")
        return
    
    df = pd.read_csv(table_a_path)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['#4A90E2', '#EE4C2C', '#00C853', '#FF6600']
    
    for idx, row in df.iterrows():
        ax.scatter(
            row['Latency (ms/doc)'],
            row['F1-Macro'],
            s=300,
            c=colors[idx % len(colors)],
            alpha=0.7,
            edgecolors='black',
            linewidth=2,
            label=row['Setup']
        )
        ax.annotate(
            row['Setup'],
            (row['Latency (ms/doc)'], row['F1-Macro']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=9,
            fontweight='bold'
        )
    
    ax.set_xlabel('Latência (ms/documento)', fontsize=12, fontweight='bold')
    ax.set_ylabel('F1-Macro', fontsize=12, fontweight='bold')
    ax.set_title('Trade-off: Performance vs Eficiência', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=9)
    
    plt.tight_layout()
    output_path = PATHS['models'] / 'performance_efficiency_tradeoff.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()


def plot_cold_start_comparison():
    """Generate cold start time comparison."""
    table_a_path = PATHS['models'] / 'table_a_efficiency.csv'
    
    if not table_a_path.exists():
        print(f"Table A not found: {table_a_path}")
        return
    
    df = pd.read_csv(table_a_path)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#4A90E2', '#EE4C2C', '#00C853', '#FF6600']
    bars = ax.barh(df['Setup'], df['Cold Start (s)'], color=colors, alpha=0.8, edgecolor='black')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, df['Cold Start (s)'])):
        ax.text(val + 0.05, i, f'{val:.3f}s', va='center', fontweight='bold')
    
    ax.set_xlabel('Tempo de Cold Start (segundos)', fontsize=12, fontweight='bold')
    ax.set_title('Comparação de Cold Start entre Modelos', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    output_path = PATHS['models'] / 'cold_start_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    plt.close()


def main():
    """Generate all comparison plots."""
    print("="*60)
    print("Generating Comparison Plots")
    print("="*60)
    print()
    
    plot_f1_by_class_comparison()
    plot_performance_efficiency_tradeoff()
    plot_cold_start_comparison()
    
    print("\n" + "="*60)
    print("✅ All comparison plots generated!")
    print("="*60)


if __name__ == "__main__":
    main()

