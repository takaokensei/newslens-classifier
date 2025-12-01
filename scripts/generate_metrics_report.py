"""
Generate comprehensive metrics report for the final report.
Consolidates all results from Phase 2 into a structured format.
"""
import sys
from pathlib import Path
import pandas as pd
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import PATHS
from src.class_mapping import CLASS_TO_CATEGORY


def generate_metrics_report():
    """Generate comprehensive metrics report."""
    print("="*60)
    print("Generating Metrics Report")
    print("="*60)
    
    # Load existing tables
    table_a_path = PATHS['models'] / 'table_a_efficiency.csv'
    table_b_path = PATHS['models'] / 'table_b_classes.csv'
    
    if not table_a_path.exists() or not table_b_path.exists():
        print("Error: Tables not found. Please run Phase 2 first.")
        return
    
    table_a = pd.read_csv(table_a_path)
    table_b = pd.read_csv(table_b_path)
    
    # Update Table B with category names
    table_b_updated = table_b.copy()
    table_b_updated['Category'] = table_b_updated['Class'].map(CLASS_TO_CATEGORY)
    
    # Reorder columns
    cols = ['Class', 'Category'] + [c for c in table_b_updated.columns if c not in ['Class', 'Category']]
    table_b_updated = table_b_updated[cols]
    
    # Save updated table B
    table_b_updated_path = PATHS['models'] / 'table_b_classes_with_names.csv'
    table_b_updated.to_csv(table_b_updated_path, index=False)
    print(f"\n✅ Updated Table B saved to {table_b_updated_path}")
    
    # Create comprehensive report
    reports_dir = Path('reports')
    reports_dir.mkdir(exist_ok=True)
    
    # Generate summary
    summary = {
        "experiment_info": {
            "dataset": "Base_dados_textos_6_classes.csv",
            "n_samples": 315,
            "n_classes": 6,
            "train_size": 189,
            "val_size": 63,
            "test_size": 63,
            "split_ratio": "60/20/20",
            "random_state": 42
        },
        "class_mapping": {str(k): v for k, v in CLASS_TO_CATEGORY.items()},
        "table_a_efficiency": table_a.to_dict('records'),
        "table_b_classes": table_b_updated.to_dict('records')
    }
    
    # Save JSON report
    json_path = reports_dir / 'metrics_summary.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"✅ JSON report saved to {json_path}")
    
    # Save markdown report
    md_path = reports_dir / 'metrics_summary.md'
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# Metrics Summary Report\n\n")
        f.write("## Experiment Information\n\n")
        f.write(f"- Dataset: {summary['experiment_info']['dataset']}\n")
        f.write(f"- Total Samples: {summary['experiment_info']['n_samples']}\n")
        f.write(f"- Classes: {summary['experiment_info']['n_classes']}\n")
        f.write(f"- Train/Val/Test Split: {summary['experiment_info']['split_ratio']}\n\n")
        
        f.write("## Class Mapping\n\n")
        for class_idx, category in CLASS_TO_CATEGORY.items():
            f.write(f"- Class {class_idx}: {category}\n")
        f.write("\n")
        
        f.write("## Table A: Efficiency & Performance\n\n")
        f.write(table_a.to_string(index=False))
        f.write("\n\n")
        
        f.write("## Table B: F1-Score by Class\n\n")
        f.write(table_b_updated.to_string(index=False))
        f.write("\n\n")
        
        f.write("## Key Findings\n\n")
        f.write("### Best Performance\n")
        best_perf = table_a.loc[table_a['F1-Macro'].idxmax()]
        f.write(f"- **{best_perf['Setup']}**: F1-Macro={best_perf['F1-Macro']:.4f}, Accuracy={best_perf['Accuracy']:.4f}\n\n")
        
        f.write("### Best Efficiency\n")
        best_eff = table_a.loc[table_a['Latency (ms/doc)'].idxmin()]
        f.write(f"- **{best_eff['Setup']}**: Latency={best_eff['Latency (ms/doc)']:.4f} ms/doc, F1-Macro={best_eff['F1-Macro']:.4f}\n\n")
        
        f.write("### Best Balance\n")
        # Calculate balance score (normalized F1 / normalized latency)
        table_a_norm = table_a.copy()
        table_a_norm['F1_norm'] = (table_a_norm['F1-Macro'] - table_a_norm['F1-Macro'].min()) / (table_a_norm['F1-Macro'].max() - table_a_norm['F1-Macro'].min())
        table_a_norm['Latency_norm'] = 1 - (table_a_norm['Latency (ms/doc)'] - table_a_norm['Latency (ms/doc)'].min()) / (table_a_norm['Latency (ms/doc)'].max() - table_a_norm['Latency (ms/doc)'].min())
        table_a_norm['Balance'] = table_a_norm['F1_norm'] + table_a_norm['Latency_norm']
        best_balance = table_a_norm.loc[table_a_norm['Balance'].idxmax()]
        f.write(f"- **{best_balance['Setup']}**: F1-Macro={best_balance['F1-Macro']:.4f}, Latency={best_balance['Latency (ms/doc)']:.4f} ms/doc\n\n")
    
    print(f"✅ Markdown report saved to {md_path}")
    
    print("\n" + "="*60)
    print("Metrics Report Generated Successfully!")
    print("="*60)
    print(f"\nFiles created:")
    print(f"  - {table_b_updated_path}")
    print(f"  - {json_path}")
    print(f"  - {md_path}")


if __name__ == "__main__":
    generate_metrics_report()

