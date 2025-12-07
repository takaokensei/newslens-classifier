"""
Aplicação Streamlit para NewsLens AI Classifier.
Interface principal para classificação e monitoramento.
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import json
import base64

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Lazy imports to avoid multiprocessing issues in Streamlit Cloud
# Import ONLY config and class mapping (no heavy dependencies)
from src.config import PATHS
from src.class_mapping import CLASS_TO_CATEGORY

# Delay ALL other imports until needed (inside functions)
# This prevents "can't register atexit after shutdown" errors
def _lazy_imports():
    """Lazy import of heavy dependencies to avoid multiprocessing issues."""
    from src.preprocessing import preprocess_text
    from src.embeddings import load_tfidf_vectorizer, load_bert_model
    from src.train import load_trained_models
    from src.logging_system import log_prediction, load_prediction_logs, get_log_statistics
    from src.llm_analysis import call_groq_llm, load_class_profiles
    from src.prepare_data import load_raw_data, split_data
    return {
        'preprocess_text': preprocess_text,
        'load_tfidf_vectorizer': load_tfidf_vectorizer,
        'load_bert_model': load_bert_model,
        'load_trained_models': load_trained_models,
        'log_prediction': log_prediction,
        'load_prediction_logs': load_prediction_logs,
        'get_log_statistics': get_log_statistics,
        'call_groq_llm': call_groq_llm,
        'load_class_profiles': load_class_profiles,
        'load_raw_data': load_raw_data,
        'split_data': split_data
    }


def test_entire_validation_set(models, vectorizer, bert_model, embedding_type, model_type):
    """Test all validation set samples and return metrics."""
    try:
        imports = _lazy_imports()
        
        # Load raw data
        df, labels_filtered = imports['load_raw_data']()
        
        # Get text column
        text_cols = [col for col in df.columns if col.lower() in 
                    ['texto expandido', 'texto original', 'text', 'texto', 'content', 'noticia', 'news', 'article']]
        if not text_cols:
            text_cols = [col for col in df.columns if 'texto' in col.lower() or 'text' in col.lower()]
        if 'Texto Expandido' in text_cols:
            text_col = 'Texto Expandido'
        elif 'Texto Original' in text_cols:
            text_col = 'Texto Original'
        elif text_cols:
            text_col = text_cols[0]
        else:
            text_col = df.columns[0]
        
        # Apply same filtering as load_raw_data
        texts_all = df[text_col].astype(str).values
        texts_series = pd.Series(texts_all)
        mask = (texts_series != '') & (texts_series != 'nan') & (~texts_series.isna())
        texts = texts_all[mask.values]
        
        if len(texts) != len(labels_filtered):
            min_len = min(len(texts), len(labels_filtered))
            texts = texts[:min_len]
            labels_array = labels_filtered[:min_len]
        else:
            labels_array = labels_filtered
        
        # Split data using same random_state as training
        from src.config import DATA_CONFIG
        X_train, X_val, X_test, y_train, y_val, y_test = imports['split_data'](
            texts, labels_array,
            random_state=DATA_CONFIG['random_state']
        )
        
        # Classify all validation samples
        correct = 0
        total = len(X_val)
        predictions = []
        
        for i, text in enumerate(X_val):
            try:
                result = classify_text_streamlit(
                    text,
                    embedding_type,
                    model_type,
                    models,
                    vectorizer,
                    bert_model
                )
                predicted_class = result['classe_predita']
                true_label = int(y_val[i])
                is_correct = (predicted_class == true_label)
                if is_correct:
                    correct += 1
                predictions.append({
                    'text': text,  # Full text, not truncated
                    'predicted': predicted_class,
                    'predicted_label': CLASS_TO_CATEGORY.get(predicted_class, f"Classe {predicted_class}"),
                    'true': true_label,
                    'true_label': CLASS_TO_CATEGORY.get(true_label, f"Classe {true_label}"),
                    'correct': is_correct,
                    'score': result['score'],
                    'embedding_usado': embedding_type,
                    'modelo_usado': model_type
                })
            except Exception as e:
                print(f"Error classifying sample {i}: {e}")
                continue
        
        accuracy = correct / total if total > 0 else 0
        
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'predictions': predictions
        }
    except Exception as e:
        print(f"❌ Error testing validation set: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_validation_sample():
    """Load a random text from validation set (not seen during training).
    
    Returns:
        Tuple of (text, true_label) or None if error
    """
    try:
        imports = _lazy_imports()
        
        # Check if raw data file exists
        raw_dir = PATHS['data_raw']
        csv_files = list(raw_dir.glob('*.csv'))
        if not csv_files:
            print(f"❌ No CSV files found in {raw_dir}")
            return None
        
        print(f"✅ Found CSV file: {csv_files[0]}")
        
        # Load raw data - this already filters empty texts and returns filtered labels
        df, labels_filtered = imports['load_raw_data']()
        print(f"✅ Loaded {len(df)} rows from CSV, {len(labels_filtered)} after filtering")
        
        # Get text column (same logic as load_raw_data)
        text_cols = [col for col in df.columns if col.lower() in 
                    ['texto expandido', 'texto original', 'text', 'texto', 'content', 'noticia', 'news', 'article']]
        if not text_cols:
            text_cols = [col for col in df.columns if 'texto' in col.lower() or 'text' in col.lower()]
        if 'Texto Expandido' in text_cols:
            text_col = 'Texto Expandido'
        elif 'Texto Original' in text_cols:
            text_col = 'Texto Original'
        elif text_cols:
            text_col = text_cols[0]
        else:
            text_col = df.columns[0]
        
        print(f"✅ Using text column: {text_col}")
        
        # Apply same filtering as load_raw_data to get texts
        texts_all = df[text_col].astype(str).values
        texts_series = pd.Series(texts_all)
        mask = (texts_series != '') & (texts_series != 'nan') & (~texts_series.isna())
        texts = texts_all[mask.values]
        
        # labels_filtered is already filtered, so it should match texts length
        if len(texts) != len(labels_filtered):
            print(f"⚠️  Warning: texts ({len(texts)}) and labels ({len(labels_filtered)}) length mismatch, using labels length")
            # Use the shorter length to ensure consistency
            min_len = min(len(texts), len(labels_filtered))
            texts = texts[:min_len]
            labels_array = labels_filtered[:min_len]
        else:
            labels_array = labels_filtered
        
        print(f"✅ After filtering: {len(texts)} texts, {len(labels_array)} labels")
        
        # Split data using same random_state as training
        from src.config import DATA_CONFIG
        X_train, X_val, X_test, y_train, y_val, y_test = imports['split_data'](
            texts, labels_array,
            random_state=DATA_CONFIG['random_state']
        )
        
        print(f"✅ Validation set size: {len(X_val)}")
        
        # Select random sample from validation set
        import numpy as np
        if len(X_val) == 0:
            print("❌ Validation set is empty")
            return None
        
        random_idx = np.random.randint(0, len(X_val))
        sample_text = X_val[random_idx]
        true_label = int(y_val[random_idx])  # Get the true label for this sample
        
        print(f"✅ Selected sample at index {random_idx}, length: {len(sample_text)}, true_label: {true_label}")
        return (sample_text, true_label)
    except Exception as e:
        print(f"❌ Error loading validation sample: {e}")
        import traceback
        traceback.print_exc()
        return None


# Translations
TRANSLATIONS = {
    'pt': {
        'title': 'NewsLens AI Classifier',
        'subtitle': 'Projeto Final ELE 606 - UFRN',
        'prof': 'Prof. José Alfredo F. Costa',
        'config': 'Configuração',
        'data_cleanup': 'Limpeza de Dados',
        'tests': 'Testes',
        'embedding_type': 'Tipo de Embedding',
        'embedding_help': 'Escolha entre BERT (denso) ou TF-IDF (esparso)',
        'model_type': 'Tipo de Modelo',
        'model_help': 'Escolha entre SVM ou XGBoost',
        'performance': 'Desempenho dos Modelos',
        'best_perf': 'Melhor Performance:',
        'best_eff': 'Melhor Eficiência:',
        'about': 'Sobre',
        'classification': 'Classificação',
        'monitoring': 'Monitoramento',
        'text_classification': 'Classificação de Texto',
        'enter_text': 'Digite o texto para classificar:',
        'text_placeholder': 'Cole ou digite uma notícia aqui...',
        'classify': 'Classificar',
        'save_log': 'Salvar predição no log',
        'loading_models': 'Carregando modelos...',
        'models_loaded': 'Modelos carregados com sucesso!',
        'models_error': 'Falha ao carregar modelos. Verifique se os modelos foram treinados.',
        'classifying': 'Classificando...',
        'predicted_class': 'Classe Predita',
        'confidence': 'Confiança',
        'model': 'Modelo',
        'prob_dist': 'Distribuição de Probabilidades',
        'ai_explanation': 'Explicação por IA',
        'generate_explanation': 'Gerar Explicação',
        'generating': 'Gerando explicação...',
        'explanation_error': 'Não foi possível gerar explicação:',
        'explanation_info': 'A explicação por LLM requer a variável de ambiente GROQ_API_KEY.',
        'saved_log': 'Predição salva no log!',
        'monitoring_dashboard': 'Dashboard de Monitoramento',
        'no_predictions': 'Nenhuma predição registrada ainda. Comece a classificar textos para ver estatísticas aqui.',
        'total_predictions': 'Total de Predições',
        'avg_score': 'Score Médio',
        'most_common': 'Classe Mais Comum',
        'date_range': 'Período',
        'by_class': 'Predições por Classe',
        'by_model': 'Predições por Modelo',
        'temporal_evolution': 'Evolução Temporal',
        'recent_predictions': 'Predições Recentes',
        'dist_by_category': 'Distribuição por Categoria',
        'usage_by_model': 'Uso por Modelo',
        'predictions_over_time': 'Predições ao Longo do Tempo'
    },
    'en': {
        'title': 'NewsLens AI Classifier',
        'subtitle': 'ELE 606 Final Project - UFRN',
        'prof': 'Prof. José Alfredo F. Costa',
        'config': 'Configuration',
        'data_cleanup': 'Data Cleanup',
        'tests': 'Tests',
        'embedding_type': 'Embedding Type',
        'embedding_help': 'Choose between BERT (dense) or TF-IDF (sparse) embeddings',
        'model_type': 'Model Type',
        'model_help': 'Choose between SVM or XGBoost classifier',
        'performance': 'Model Performance',
        'best_perf': 'Best Performance:',
        'best_eff': 'Best Efficiency:',
        'about': 'About',
        'classification': 'Classification',
        'monitoring': 'Monitoring',
        'text_classification': 'Text Classification',
        'enter_text': 'Enter text to classify:',
        'text_placeholder': 'Paste or type a news article here...',
        'classify': 'Classify',
        'save_log': 'Save prediction to log',
        'loading_models': 'Loading models...',
        'models_loaded': 'Models loaded successfully!',
        'models_error': 'Failed to load models. Please check if models are trained.',
        'classifying': 'Classifying...',
        'predicted_class': 'Predicted Class',
        'confidence': 'Confidence',
        'model': 'Model',
        'prob_dist': 'Probability Distribution',
        'ai_explanation': 'AI Explanation',
        'generate_explanation': 'Generate Explanation',
        'generating': 'Generating explanation...',
        'explanation_error': 'Could not generate explanation:',
        'explanation_info': 'LLM explanation requires GROQ_API_KEY environment variable.',
        'saved_log': 'Prediction saved to log!',
        'monitoring_dashboard': 'Monitoring Dashboard',
        'no_predictions': 'No predictions logged yet. Start classifying texts to see statistics here.',
        'total_predictions': 'Total Predictions',
        'avg_score': 'Average Score',
        'most_common': 'Most Common Class',
        'date_range': 'Date Range',
        'by_class': 'Predictions by Class',
        'by_model': 'Predictions by Model',
        'temporal_evolution': 'Temporal Evolution',
        'recent_predictions': 'Recent Predictions',
        'dist_by_category': 'Distribution by Category',
        'usage_by_model': 'Usage by Model',
        'predictions_over_time': 'Predictions Over Time'
    }

}


def get_text(key: str, lang: str = 'pt') -> str:
    """Get translated text."""
    return TRANSLATIONS.get(lang, TRANSLATIONS['pt']).get(key, key)


# SVG Icons
ICONS = {
    'logo': '''<svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M4 4H20C21.1 4 22 4.9 22 6V18C22 19.1 21.1 20 20 20H4C2.9 20 2 19.1 2 18V6C2 4.9 2.9 4 4 4Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><path d="M22 6L12 13L2 6" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>''',
    'settings': '''<svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M12 15C13.6569 15 15 13.6569 15 12C15 10.3431 13.6569 9 12 9C10.3431 9 9 10.3431 9 12C9 13.6569 10.3431 15 12 15Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><path d="M19.4 15C20.0627 14.3373 20.0627 13.2627 19.4 12.6L18 11.2C17.3373 10.5373 17.3373 9.46274 18 8.8L19.4 7.4C20.0627 6.73726 20.0627 5.66274 19.4 5C18.7373 4.33726 17.6627 4.33726 17 5L15.6 6.4C14.9373 7.06274 13.8627 7.06274 13.2 6.4L11.8 5C11.1373 4.33726 10.0627 4.33726 9.4 5L8 6.4C7.33726 7.06274 6.26274 7.06274 5.6 6.4L4.2 5C3.53726 4.33726 2.46274 4.33726 1.8 5C1.13726 5.66274 1.13726 6.73726 1.8 7.4L3.2 8.8C3.86274 9.46274 3.86274 10.5373 3.2 11.2L1.8 12.6C1.13726 13.2627 1.13726 14.3373 1.8 15C2.46274 15.6627 3.53726 15.6627 4.2 15L5.6 13.6C6.26274 12.9373 7.33726 12.9373 8 13.6L9.4 15C10.0627 15.6627 11.1373 15.6627 11.8 15L13.2 13.6C13.8627 12.9373 14.9373 12.9373 15.6 13.6L17 15C17.6627 15.6627 18.7373 15.6627 19.4 15Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>''',
    'search': '''<svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="11" cy="11" r="8" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><path d="M21 21L16.65 16.65" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>''',
    'chart': '''<svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M18 20V10" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><path d="M12 20V4" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><path d="M6 20V14" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>''',
    'check': '''<svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M20 6L9 17L4 12" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>''',
    'alert': '''<svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><path d="M12 8V12" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><path d="M12 16H12.01" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>''',
    'trash': '''<svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M3 6H5H21" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><path d="M8 6V4C8 3.46957 8.21071 2.96086 8.58579 2.58579C8.96086 2.21071 9.46957 2 10 2H14C14.5304 2 15.0391 2.21071 15.4142 2.58579C15.7893 2.96086 16 3.46957 16 4V6M19 6V20C19 20.5304 18.7893 21.0391 18.4142 21.4142C18.0391 21.7893 17.5304 22 17 22H7C6.46957 22 5.96086 21.7893 5.58579 21.4142C5.21071 21.0391 5 20.5304 5 20V6H19Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>''',
    'info': '''<svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><path d="M12 16V12" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><path d="M12 8H12.01" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>''',
    'robot': '''<svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M12 2V6" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><path d="M12 18V22" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><path d="M4.93 4.93L7.76 7.76" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><path d="M16.24 16.24L19.07 19.07" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><path d="M2 12H6" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><path d="M18 12H22" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><path d="M4.93 19.07L7.76 16.24" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><path d="M16.24 7.76L19.07 4.93" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>''',
    'sun': '''<svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="12" cy="12" r="5" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><path d="M12 1V3" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><path d="M12 21V23" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><path d="M4.22 4.22L5.64 5.64" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><path d="M18.36 18.36L19.78 19.78" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><path d="M1 12H3" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><path d="M21 12H23" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><path d="M4.22 19.78L5.64 18.36" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><path d="M18.36 5.64L19.78 4.22" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>''',
    'moon': '''<svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>'''
}

def render_svg(icon_name, size=24, color="currentColor"):
    """Render SVG icon with specified size and color."""
    if icon_name not in ICONS:
        return ""
    svg = ICONS[icon_name]
    # Replace width/height/stroke if needed, but simple string replacement is risky.
    # Instead, we wrap it in a div that controls size and color.
    return f'<div style="display: inline-flex; align-items: center; justify-content: center; width: {size}px; height: {size}px; color: {color};">{svg}</div>'


def _apply_custom_css(theme='dark'):
    """Apply Swiss Design inspired CSS with Dynamic Theme."""
    
    # Theme Colors
    colors = {
        'light': {
            'bg': '#ffffff',
            'text': '#1a1a1a',
            'sidebar_bg': '#f0f0f0',
            'sidebar_text': '#000000',
            'border': '#e0e0e0',
            'accent': '#000000',
            'secondary_text': '#666666',
            'success_bg': '#f0fff4',
            'success_border': '#4CD964',
            'card_bg': '#ffffff'
        },
        'dark': {
            'bg': '#0e1117',
            'text': '#fafafa',
            'sidebar_bg': '#262730',
            'sidebar_text': '#ffffff',
            'border': '#464b5f',
            'accent': '#ffffff',
            'secondary_text': '#b0b0b0',
            'success_bg': 'rgba(76, 217, 100, 0.1)',
            'success_border': '#4CD964',
            'card_bg': '#1e2130'
        }
    }
    
    c = colors[theme]
    
    st.markdown(f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
        @import url('https://fonts.googleapis.com/icon?family=Material+Icons');

        /* Global Typography & Theme */
        html, body, [class*="css"], [data-testid="stAppViewContainer"] {{
            font-family: 'Inter', sans-serif;
            color: {c['text']};
            background-color: {c['bg']};
            transition: background-color 0.3s ease, color 0.3s ease;
        }}
        
        /* Main Container */
        .stApp {{
            background-color: {c['bg']};
            transition: background-color 0.3s ease;
        }}

        /* Headers */
        h1, h2, h3 {{
            font-family: 'Inter', sans-serif;
            font-weight: 700;
            letter-spacing: -0.02em;
            color: {c['accent']};
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        
        h1 {{ font-size: 2.5rem; margin-bottom: 2rem; }}
        h2 {{ font-size: 1.75rem; margin-top: 2.5rem; margin-bottom: 1.25rem; border-bottom: 1px solid {c['accent']}; padding-bottom: 0.5rem; }}
        h3 {{ font-size: 1.25rem; margin-top: 1.5rem; margin-bottom: 0.75rem; }}

        /* Sidebar */
        [data-testid="stSidebar"] {{
            background-color: {c['sidebar_bg']};
            border-right: 1px solid {c['border']};
            transition: background-color 0.3s ease;
        }}
        
        [data-testid="stSidebar"] .stMarkdown, 
        [data-testid="stSidebar"] p, 
        [data-testid="stSidebar"] span, 
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] div {{
            color: {c['sidebar_text']} !important;
            font-family: 'Inter', sans-serif;
        }}
        
        /* Sidebar Headers */
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {{
            color: {c['sidebar_text']} !important;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-size: 1rem;
        }}

        /* Buttons */
        .stButton > button {{
            border-radius: 0px;
            font-weight: 600;
            border: 1px solid {c['accent']} !important;
            background-color: {c['bg']} !important;
            color: {c['accent']} !important;
            transition: all 0.2s ease;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-size: 0.85rem;
            padding: 0.5rem 1rem;
        }}
        
        .stButton > button:hover {{
            background-color: {c['accent']} !important;
            color: {c['bg']} !important;
            border-color: {c['accent']} !important;
        }}
        
        /* Primary Button */
        .stButton > button[kind="primary"] {{
            background-color: {c['accent']} !important;
            color: {c['bg']} !important;
            border: 1px solid {c['accent']} !important;
        }}
        
        .stButton > button[kind="primary"]:hover {{
            background-color: {c['secondary_text']} !important;
            border-color: {c['secondary_text']} !important;
        }}

        /* Metrics */
        [data-testid="stMetricValue"] {{
            font-family: 'Inter', sans-serif;
            font-weight: 700;
            font-size: 2rem;
            color: {c['accent']};
        }}
        
        [data-testid="stMetricLabel"] {{
            font-family: 'Inter', sans-serif;
            font-weight: 500;
            font-size: 0.9rem;
            color: {c['secondary_text']};
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}

        /* Dividers */
        hr {{
            margin-top: 3rem;
            margin-bottom: 3rem;
            border-top: 1px solid {c['accent']};
            opacity: 0.1;
        }}
        
        /* Expander */
        .streamlit-expanderHeader {{
            font-family: 'Inter', sans-serif;
            font-weight: 600;
            background-color: {c['card_bg']} !important;
            border: 1px solid {c['border']} !important;
            border-radius: 0px;
            color: {c['accent']} !important;
        }}
        
        /* Fix for Material Icon text rendering */
        .material-icons {{
            font-family: 'Material Icons';
            font-weight: normal;
            font-style: normal;
            font-size: 24px;
            display: inline-block;
            line-height: 1;
            text-transform: none;
            letter-spacing: normal;
            word-wrap: normal;
            white-space: nowrap;
            direction: ltr;
            -webkit-font-smoothing: antialiased;
            text-rendering: optimizeLegibility;
            -moz-osx-font-smoothing: grayscale;
            font-feature-settings: 'liga';
        }}
        
        /* Dataframes and Tables */
        .stDataFrame,
        [data-testid="stDataFrame"],
        .dataframe {{
            background-color: {c['card_bg']} !important;
            border: 1px solid {c['border']} !important;
        }}
        
        .stDataFrame table,
        [data-testid="stDataFrame"] table,
        .dataframe table {{
            background-color: {c['card_bg']}  !important;
            color: {c['text']} !important;
        }}
        
        .stDataFrame th,
        [data-testid="stDataFrame"] th,
        .dataframe th {{
            background-color: {c['sidebar_bg']} !important;
            color: {c['text']} !important;
            border-bottom: 2px solid {c['accent']} !important;
            font-weight: 600 !important;
        }}
        
        .stDataFrame td,
        [data-testid="stDataFrame"] td,
        .dataframe td {{
            background-color: {c['card_bg']} !important;
            color: {c['text']} !important;
            border-bottom: 1px solid {c['border']} !important;
        }}
        
        /* Markdown in cards */
        .stMarkdown {{
            color: {c['text']} !important;
        }}
        
        /* Checkboxes */
        .stCheckbox {{
            color: {c['text']} !important;
        }}
        
        .stCheckbox label {{
            color: {c['text']} !important;
        }}
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 2rem;
            border-bottom: 1px solid {c['border']};
            padding-bottom: 0px;
        }}
        
        .stTabs [data-baseweb="tab"] {{
            height: 50px;
            white-space: pre-wrap;
            background-color: transparent;
            border-radius: 0px;
            gap: 1px;
            padding-top: 10px;
            padding-bottom: 10px;
            font-family: 'Inter', sans-serif;
            font-weight: 500;
            color: {c['secondary_text']};
            border-bottom: 3px solid transparent;
        }}
        
        .stTabs [aria-selected="true"] {{
            border-bottom: 3px solid {c['accent']};
            color: {c['accent']} !important;
            font-weight: 700;
        }}
        
        /* Inputs */
        .stTextInput > div > div > input,
        .stSelectbox > div > div > div,
        .stTextArea > div > div > textarea {{
            border-radius: 0px;
            border: 1px solid {c['border']};
            color: {c['text']};
            background-color: {c['bg']};
        }}
        
        .stTextInput > div > div > input:focus,
        .stSelectbox > div > div > div:focus,
        .stTextArea > div > div > textarea:focus {{
            border-color: {c['accent']};
            box-shadow: none;
        }}
        
        /* Custom Success Box */
        .custom-success {{
            padding: 1rem;
            border: 1px solid {c['success_border']};
            background-color: {c['success_bg']};
            color: {c['text']};
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin-bottom: 1rem;
        }}

        </style>
    """, unsafe_allow_html=True)



def update_chart_layout(fig):
    """Apply Swiss Design theme to Plotly charts for DARK MODE."""
    fig.update_layout(
        font=dict(family="Inter, sans-serif", size=12, color="#fafafa"),  # Light text for dark mode
        title_font=dict(family="Inter, sans-serif", size=16, weight=700, color="#fafafa"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(
            showgrid=False,
            showline=True,
            linewidth=1,
            linecolor="#464b5f",  # Dark mode border
            tickfont=dict(family="Inter, sans-serif", size=10, color="#fafafa")
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor="#262730",  # Dark mode grid
            showline=False,
            tickfont=dict(family="Inter, sans-serif", size=10, color="#fafafa")
        ),
        colorway=["#FF3B30", "#007AFF", "#4CD964", "#5856D6", "#FF9500", "#FFCC00"], # Brighter colors for dark mode
        hoverlabel=dict(
            bgcolor="#1a1a1a",  # Dark background
            font_size=13,
            font_color="#ffffff",  # White text
            font_family="Inter, sans-serif",
            bordercolor="#ffffff"
        )
    )
    return fig



# Page configuration
st.set_page_config(
    page_title="NewsLens AI Classifier",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'vectorizer' not in st.session_state:
    st.session_state.vectorizer = None
if 'bert_model' not in st.session_state:
    st.session_state.bert_model = None
if 'language' not in st.session_state:
    st.session_state.language = 'pt'  # Default: Portuguese
if 'last_classification_result' not in st.session_state:
    st.session_state.last_classification_result = None
if 'last_text_input' not in st.session_state:
    st.session_state.last_text_input = None
if 'explanation_generated' not in st.session_state:
    st.session_state.explanation_generated = False
if 'llm_explanation' not in st.session_state:
    st.session_state.llm_explanation = None
# Cookie-based prediction logs (persists across page refreshes)
# Initialize from cookies if available, otherwise empty list
if 'session_predictions' not in st.session_state:
    st.session_state.session_predictions = []


def save_predictions_to_cookie(predictions: list) -> None:
    """Save predictions to browser cookie using JavaScript."""
    if not predictions:
        cookie_data = ""
    else:
        # Encode predictions as base64 JSON to avoid cookie size limits
        json_str = json.dumps(predictions)
        cookie_data = base64.b64encode(json_str.encode('utf-8')).decode('utf-8')
        # Escape quotes for JavaScript
        cookie_data = cookie_data.replace('"', '\\"').replace("'", "\\'")
    
    # JavaScript to set cookie (expires in 30 days)
    js_code = f"""
    <script>
    (function() {{
        function setCookie(name, value, days) {{
            var expires = "";
            if (days) {{
                var date = new Date();
                date.setTime(date.getTime() + (days * 24 * 60 * 60 * 1000));
                expires = "; expires=" + date.toUTCString();
            }}
            document.cookie = name + "=" + encodeURIComponent(value || "") + expires + "; path=/; SameSite=Lax";
        }}
        setCookie("newslens_predictions", "{cookie_data}", 30);
    }})();
    </script>
    """
    st.components.v1.html(js_code, height=0)


def load_and_sync_cookie_predictions():
    """Load predictions from cookie and sync with session_state on page load."""
    # Initialize cookie_loaded flag if not exists
    if 'cookie_loaded' not in st.session_state:
        st.session_state.cookie_loaded = False
    
    # Only try to load if we don't have predictions in session_state and haven't loaded yet
    if not st.session_state.session_predictions and not st.session_state.cookie_loaded:
        # Check query params first (set by JavaScript component on first load)
        query_params = st.query_params
        cookie_data_param = query_params.get("cookie_data", None)
        
        if cookie_data_param:
            # Decode from base64
            try:
                decoded = base64.b64decode(cookie_data_param).decode('utf-8')
                predictions = json.loads(decoded)
                if predictions and isinstance(predictions, list):
                    st.session_state.session_predictions = predictions
                    st.session_state.cookie_loaded = True
                    # Clear query param to avoid reloading
                    st.query_params.clear()
                    # Note: Don't call st.rerun() here - let the natural rerun from query param change handle it
            except Exception as e:
                print(f"Error loading predictions from cookie: {e}")
                st.session_state.cookie_loaded = True  # Mark as loaded even on error to avoid retry loop
        else:
            # JavaScript component to read cookie and update query params (only if no data in session_state)
            # This triggers a rerun with the cookie data
            js_code = """
            <script>
            (function() {
                function getCookie(name) {
                    var nameEQ = name + "=";
                    var ca = document.cookie.split(';');
                    for(var i = 0; i < ca.length; i++) {
                        var c = ca[i];
                        while (c.charAt(0) == ' ') c = c.substring(1, c.length);
                        if (c.indexOf(nameEQ) == 0) {
                            return decodeURIComponent(c.substring(nameEQ.length, c.length));
                        }
                    }
                    return null;
                }
                
                // Access parent window (main Streamlit app window, not iframe)
                try {
                    var parentWindow = window.parent;
                    var parentUrl = new URL(parentWindow.location.href);
                    
                    // Only try to load if query param doesn't already have data
                    if (!parentUrl.searchParams.has("cookie_data")) {
                        var cookieValue = getCookie("newslens_predictions");
                        if (cookieValue && cookieValue.length > 0) {
                            // Update parent window URL with cookie data to trigger rerun
                            parentUrl.searchParams.set("cookie_data", cookieValue);
                            // Use parent window's history.replaceState (works in main window)
                            parentWindow.history.replaceState({}, '', parentUrl.toString());
                            // Trigger Streamlit rerun by reloading parent window
                            parentWindow.location.reload();
                        }
                    }
                } catch(e) {
                    // If we can't access parent (security restriction), try direct reload with query param
                    console.log("Cannot access parent window, using direct reload");
                    var cookieValue = getCookie("newslens_predictions");
                    if (cookieValue && cookieValue.length > 0) {
                        var currentUrl = new URL(window.location.href);
                        if (!currentUrl.searchParams.has("cookie_data")) {
                            currentUrl.searchParams.set("cookie_data", cookieValue);
                            window.location.href = currentUrl.toString();
                        }
                    }
                }
            })();
            </script>
            """
            st.components.v1.html(js_code, height=0)
    else:
        # If we already have predictions or have loaded, mark as loaded
        if st.session_state.session_predictions:
            st.session_state.cookie_loaded = True


@st.cache_resource
def load_all_models():
    """Carrega todos os modelos e embeddings (em cache)."""
    try:
        imports = _lazy_imports()
        models = imports['load_trained_models']()
        
        # Check if models were loaded
        if not models:
            # Try automatic training (only if not in Streamlit Cloud to avoid blocking)
            # In Streamlit Cloud, we'll show a message instead
            try:
                import sys
                # Check if we're in a Streamlit environment
                is_streamlit = 'streamlit' in sys.modules
                
                if is_streamlit:
                    # In Streamlit, we'll let the UI handle the training message
                    # Don't block here, return None and let UI show message
                    print("⚠️  Models not found. UI will show training option.")
                else:
                    # In CLI/script mode, try automatic training
                    from scripts.auto_train_models import train_models_automatically
                    print("🔄 Models not found. Attempting automatic training...")
                    result = train_models_automatically(force=False)
                    
                    if result['success'] and result.get('models_trained', False):
                        # Reload models after training
                        models = imports['load_trained_models']()
                        print("✅ Models trained and loaded successfully!")
                    else:
                        print(f"⚠️  Automatic training failed or skipped: {result.get('message', 'Unknown error')}")
                        return None, None, None
            except Exception as e:
                print(f"⚠️  Error during automatic training: {e}")
                import traceback
                traceback.print_exc()
                return None, None, None
        
        # Try to load vectorizer
        vectorizer_path = PATHS['data_embeddings'] / 'tfidf_vectorizer.pkl'
        vectorizer = None
        if vectorizer_path.exists():
            vectorizer = imports['load_tfidf_vectorizer'](vectorizer_path)
        else:
            print(f"⚠️  Vectorizer not found: {vectorizer_path}")
        
        # Try to load BERT model
        bert_model = None
        try:
            bert_model = imports['load_bert_model']()
        except Exception as e:
            print(f"⚠️  Error loading BERT model: {e}")
        
        return models, vectorizer, bert_model
    except Exception as e:
        import traceback
        print(f"❌ Error loading models: {e}")
        traceback.print_exc()
        return None, None, None


def classify_text_streamlit(
    text: str,
    embedding_type: str,
    model_type: str,
    models,
    vectorizer,
    bert_model
) -> dict:
    """Classify text using specified model."""
    # Lazy import
    imports = _lazy_imports()
    # Preprocess
    processed_text = imports['preprocess_text'](text)
    
    # Generate embeddings
    if embedding_type == "TF-IDF":
        embedding = vectorizer.transform([processed_text])
        if model_type == "SVM":
            model = models['tfidf_svm']
        else:
            model = models['tfidf_xgb']
    else:  # BERT
        embedding = bert_model.encode([processed_text], convert_to_numpy=True, show_progress_bar=False)
        if model_type == "SVM":
            model = models['bert_svm']
        else:
            model = models['bert_xgb']
    
    # Predict
    pred_class = model.predict(embedding)[0]
    
    # Get probability
    try:
        proba = model.predict_proba(embedding)[0]
        score = float(proba[pred_class])
        all_probas = {CLASS_TO_CATEGORY.get(i, f"Class_{i}"): float(proba[i]) for i in range(len(proba))}
    except:
        score = 1.0
        all_probas = {}
    
    return {
        'classe_predita': int(pred_class),
        'categoria_predita': CLASS_TO_CATEGORY.get(int(pred_class), f"Class_{pred_class}"),
        'score': score,
        'all_probas': all_probas,
        'embedding_usado': embedding_type,
        'modelo_usado': model_type
    }


def main():
    """Aplicação principal do Streamlit."""
    # Initialize theme
    if 'theme' not in st.session_state:
        st.session_state.theme = 'dark'

    # Apply Swiss Design CSS with current theme
    _apply_custom_css(st.session_state.theme)

    # Load predictions from cookie on page load (survives F5)
    load_and_sync_cookie_predictions()
    
    # Sidebar - Theme Toggle & Language
    with st.sidebar:
        col_theme, col_lang = st.columns([1, 1])
        with col_theme:
            # Theme Toggle Button
            is_dark = st.session_state.theme == 'dark'
            toggle_label = "Light Mode" if is_dark else "Dark Mode"
            
            # Use a primary button for the toggle to make it stand out
            if st.button(toggle_label, key='theme_toggle_btn', use_container_width=True):
                st.session_state.theme = 'light' if is_dark else 'dark'
                st.rerun()
        
        with col_lang:
            # Language selector (simplified)
            lang = st.selectbox(
                "Idioma / Language",
                ["Português", "English"],
                index=0,
                key="lang_selector",
                label_visibility="collapsed"
            )
    current_lang = 'pt' if lang == "Português" else 'en'
    st.session_state.language = current_lang
    
    t = lambda key: get_text(key, current_lang)
    
    # Determine current icon color based on theme
    icon_color = '#ffffff' if st.session_state.theme == 'dark' else '#000000'
    
    # Header with Logo - Better spacing
    col_logo, col_title = st.columns([1, 20])  # More space for title
    with col_logo:
        st.markdown(render_svg('logo', 48, icon_color), unsafe_allow_html=True)
    with col_title:
        st.title(t('title'))
        st.markdown(f"**{t('subtitle')}**")
        st.caption(t('prof'))
    
    # Sidebar Header
    with st.sidebar:
        st.divider()
        # Use markdown for header to support SVG
        st.markdown(f"### {render_svg('settings', 20, icon_color)} {t('config')}", unsafe_allow_html=True)
        
        # Model selection
        embedding_type = st.selectbox(
            t('embedding_type'),
            ["BERT", "TF-IDF"],
            index=0,
            help=t('embedding_help')
        )
        
        model_type = st.selectbox(
            t('model_type'),
            ["SVM", "XGBoost"],
            index=0,
            help=t('model_help')
        )
        
        st.divider()
        
        st.markdown(f"### {t('performance')}")
        if current_lang == 'pt':
            st.info(f"""
            **{t('best_perf')}** BERT + SVM (F1=1.0)
            
            **{t('best_eff')}** TF-IDF + SVM (F1=0.97, 0.14ms/doc)
            """)
        else:
            st.info(f"""
            **{t('best_perf')}** BERT + SVM (F1=1.0)
            
            **{t('best_eff')}** TF-IDF + SVM (F1=0.97, 0.14ms/doc)
            """)
        
        st.divider()
        
        st.divider()
        
        # Test entire validation set button
        st.markdown(f"### {render_svg('check', 20)} {t('tests')}", unsafe_allow_html=True)
        if st.button(
            "Testar todo o conjunto de validação" if current_lang == 'pt' else "Test entire validation set",
            width='stretch',
            help="Testa todos os exemplos do conjunto de validação e mostra métricas de desempenho" if current_lang == 'pt' else "Tests all validation set examples and shows performance metrics"
        ):
            st.session_state.test_validation_set = True
            st.rerun()
        
        st.divider()
        
        st.divider()
        
        # Clear metrics button with confirmation
        st.markdown(f"### {render_svg('trash', 20)} {t('data_cleanup')}", unsafe_allow_html=True)
        if st.button(
            "Apagar métricas de desempenho" if current_lang == 'pt' else "Clear performance metrics",
            width='stretch',
            help="Remove todas as métricas de desempenho (cookies e dados locais)" if current_lang == 'pt' else "Removes all performance metrics (cookies and local data)"
        ):
            st.session_state.show_clear_confirmation = True
            st.rerun()
        
        # Confirmation dialog
        if st.session_state.get('show_clear_confirmation', False):
            st.warning(
                "⚠️ **Tem certeza que deseja apagar todas as métricas de desempenho?**\n\n"
                "Esta ação irá remover:\n"
                "- Todas as predições salvas nos cookies\n"
                "- Todos os dados da sessão atual\n\n"
                "Esta ação não pode ser desfeita." if current_lang == 'pt' else
                "**Are you sure you want to clear all performance metrics?**\n\n"
                "This action will remove:\n"
                "- All predictions saved in cookies\n"
                "- All current session data\n\n"
                "This action cannot be undone."
            )
            col_yes, col_no = st.columns(2)
            with col_yes:
                if st.button("Sim, apagar tudo" if current_lang == 'pt' else "Yes, clear all", type="primary", width='stretch'):
                    # Clear session state predictions
                    st.session_state.session_predictions = []
                    # Clear cookies using JavaScript
                    clear_cookies_js = """
                    <script>
                    (function() {
                        // Clear the cookie
                        document.cookie = "newslens_predictions=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/;";
                        // Also try to clear from parent window if in iframe
                        try {
                            if (window.parent && window.parent !== window) {
                                window.parent.document.cookie = "newslens_predictions=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/;";
                            }
                        } catch(e) {
                            console.log("Cannot access parent window");
                        }
                    })();
                    </script>
                    """
                    st.components.v1.html(clear_cookies_js, height=0)
                    st.session_state.show_clear_confirmation = False
                    st.success("Métricas apagadas com sucesso!" if current_lang == 'pt' else "Metrics cleared successfully!")
                    st.rerun()
            with col_no:
                if st.button("Cancelar" if current_lang == 'pt' else "Cancel", width='stretch'):
                    st.session_state.show_clear_confirmation = False
                    st.rerun()
        
        st.divider()
        
        st.divider()
        
        st.markdown(f"### {render_svg('info', 20)} {t('about')}", unsafe_allow_html=True)
        st.caption("""
        NewsLens AI - Projeto Final ELE 606
        
        UFRN - Prof. José Alfredo F. Costa
        """)
    
    # Main tabs
    tab1, tab2 = st.tabs([t('classification'), t('monitoring')])
    
    # Tab 1: Classification
    with tab1:
        st.header(t('text_classification'))
        
        # Load models
        if not st.session_state.models_loaded:
            loading_text = t('loading_models')
            with st.spinner(loading_text):
                models, vectorizer, bert_model = load_all_models()
                
                # Check if automatic training is needed
                if models is None:
                    # Check if data exists for training
                    raw_dir = PATHS['data_raw']
                    csv_files = list(raw_dir.glob('*.csv'))
                    
                    if csv_files:
                        # Data exists, try to train automatically
                        st.info("🔄 **Treinando modelos automaticamente...** Isso pode levar alguns minutos na primeira vez." if current_lang == 'pt' else "🔄 **Training models automatically...** This may take a few minutes on first run.")
                        
                        # Try to train in background (non-blocking)
                        try:
                            # Import training function
                            import sys
                            sys.path.insert(0, str(Path(__file__).parent.parent))
                            from scripts.auto_train_models import train_models_automatically
                            
                            # Show progress
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            status_text.text("📊 Preparando dados..." if current_lang == 'pt' else "📊 Preparing data...")
                            progress_bar.progress(20)
                            
                            # Train models
                            result = train_models_automatically(force=False)
                            
                            progress_bar.progress(80)
                            
                            if result['success'] and result.get('models_trained', False):
                                status_text.text("✅ Modelos treinados! Carregando..." if current_lang == 'pt' else "✅ Models trained! Loading...")
                                progress_bar.progress(100)
                                
                                # Reload models
                                imports = _lazy_imports()
                                models = imports['load_trained_models']()
                                
                                if models and len(models) > 0:
                                    st.session_state.models = models
                                    st.session_state.vectorizer = imports['load_tfidf_vectorizer'](PATHS['data_embeddings'] / 'tfidf_vectorizer.pkl') if (PATHS['data_embeddings'] / 'tfidf_vectorizer.pkl').exists() else None
                                    st.session_state.bert_model = imports['load_bert_model']()
                                    st.session_state.models_loaded = True
                                    st.success(t('models_loaded'))
                                    progress_bar.empty()
                                    status_text.empty()
                                    st.rerun()
                                else:
                                    progress_bar.empty()
                                    status_text.empty()
                                    st.error("Erro ao carregar modelos após treinamento." if current_lang == 'pt' else "Error loading models after training.")
                                    st.stop()
                            else:
                                progress_bar.empty()
                                status_text.empty()
                                st.warning(f"{result.get('message', 'Training failed')}")
                                st.info("Tente recarregar a página ou execute `python scripts/auto_train_models.py` manualmente." if current_lang == 'pt' else "Try reloading the page or run `python scripts/auto_train_models.py` manually.")
                                st.stop()
                        except Exception as e:
                            st.error(f"Erro durante treinamento: {str(e)}" if current_lang == 'pt' else f"Error during training: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())
                            st.info("Execute `python scripts/auto_train_models.py` manualmente para treinar os modelos." if current_lang == 'pt' else "Run `python scripts/auto_train_models.py` manually to train models.")
                            st.stop()
                    else:
                        st.error(t('models_error'))
                        st.info("**Dica**: Adicione um arquivo CSV em `data/raw/` com colunas: 'Texto', 'Classe', 'Categoria'. Os modelos serão treinados automaticamente." if current_lang == 'pt' else "**Tip**: Add a CSV file in `data/raw/` with columns: 'Texto', 'Classe', 'Categoria'. Models will be trained automatically.")
                        st.stop()
                elif models and len(models) > 0:
                    st.session_state.models = models
                    st.session_state.vectorizer = vectorizer
                    st.session_state.bert_model = bert_model
                    st.session_state.models_loaded = True
                    st.success(t('models_loaded'))
                    if not vectorizer:
                        st.warning("TF-IDF vectorizer não encontrado. Apenas modelos BERT estarão disponíveis." if current_lang == 'pt' else "TF-IDF vectorizer not found. Only BERT models will be available.")
                    if not bert_model:
                        st.warning("Modelo BERT não encontrado. Apenas modelos TF-IDF estarão disponíveis." if current_lang == 'pt' else "BERT model not found. Only TF-IDF models will be available.")
                else:
                    st.error(t('models_error'))
                    st.info("**Dica**: Os modelos precisam ser treinados primeiro. Execute `python scripts/auto_train_models.py` para treinar automaticamente." if current_lang == 'pt' else "**Tip**: Models need to be trained first. Run `python scripts/auto_train_models.py` to train automatically.")
                    st.stop()
        else:
            # Models already loaded, get from session state
            models = st.session_state.models
            vectorizer = st.session_state.vectorizer
            bert_model = st.session_state.bert_model
        
        
        # --- Class Profiles Analysis (Requirement C4) ---
        st.divider()
        # --- Class Profiles Analysis (Requirement C4) ---
        st.divider()
        # Fix: Use st.markdown with unsafe_allow_html for SVG in header
        st.markdown(f"### {render_svg('robot', 24, icon_color)} {t('class_profiles') if 'class_profiles' in locals() else 'Análise de Perfis de Classe (LLM)'}", unsafe_allow_html=True)
        
        with st.expander("🧠 Explorar Arquétipos das Classes", expanded=False):
            try:
                # Load profiles
                imports = _lazy_imports()
                profiles = imports['load_class_profiles']()
                
                # Select class to analyze
                class_options = {int(k): v['category'] for k, v in profiles.items()}
                selected_class_idx = st.selectbox(
                    "Selecione uma Classe para Analisar",
                    options=list(class_options.keys()),
                    format_func=lambda x: f"{x}: {class_options[x]}"
                )
                
                if selected_class_idx is not None:
                    profile = profiles[str(selected_class_idx)]
                    
                    col_p1, col_p2 = st.columns(2)
                    
                    with col_p1:
                        st.markdown("#### 🔑 Top Tokens (TF-IDF)")
                        # Create a nice dataframe for tokens
                        tokens_df = pd.DataFrame(profile['top_tokens_tfidf'])
                        st.dataframe(
                            tokens_df[['token', 'chi2_score']].head(10).style.format({'chi2_score': '{:.2f}'}),
                            use_container_width=True,
                            hide_index=True
                        )
                        
                    with col_p2:
                        st.markdown("#### 👥 Vizinhos do Centróide")
                        st.write(f"Amostras mais representativas da classe **{profile['category']}**:")
                        
                        # Show top 5 representative texts instead of just IDs
                        imports = _lazy_imports()
                        df, _ = imports['load_raw_data']()
                        text_col = 'Texto Expandido' if 'Texto Expandido' in df.columns else 'Texto Original' if 'Texto Original' in df.columns else df.columns[0]
                        
                        for idx in profile['neighbor_indices'][:5]:  # Top 5
                            if idx < len(df):
                                sample_text = df.iloc[idx][text_col]
                                # Show more text - 300 chars
                                display_text = str(sample_text)[:300] + '...' if len(str(sample_text)) > 300 else str(sample_text)
                                st.text_area(
                                    f"ID {idx}",
                                    display_text,
                                    height=150,
                                    disabled=True,
                                    key=f"neighbor_{selected_class_idx}_{idx}"
                                )
                            
                    # LLM Generation for Profile Description
                    st.markdown("#### 📝 Descrição do Perfil (Gerada por IA)")
                    
                    profile_key = f"profile_desc_{selected_class_idx}"
                    if profile_key in st.session_state:
                        st.info(st.session_state[profile_key])
                    
                    if st.button(f"✨ Gerar Perfil Típico para '{profile['category']}'"):
                        with st.spinner("Analisando perfil com LLM..."):
                            # Construct prompt
                            top_tokens_str = ", ".join([t['token'] for t in profile['top_tokens_tfidf'][:15]])
                            prompt = f"""Atue como um linguista computacional. Analise os dados abaixo referentes à classe de notícias "{profile['category']}".
                            
                            Top Tokens (mais distintivos): {top_tokens_str}
                            
                            Com base nesses tokens, descreva em um parágrafo o "Perfil Típico" desta categoria. O que caracteriza as notícias desta classe? Que tipo de vocabulário é predominante?"""
                            
                            try:
                                description = imports['call_groq_llm'](prompt)
                                st.session_state[profile_key] = description
                                st.rerun()
                            except Exception as e:
                                st.error(f"Erro ao chamar LLM: {e}")
            
            except FileNotFoundError:
                st.warning("⚠️ Perfis de classe não encontrados. Execute o treinamento/análise primeiro.")
            except Exception as e:
                st.error(f"Erro ao carregar perfis: {e}")

        # Handle validation set testing
        # Check if we should run a new test or show existing results
        if st.session_state.get('test_validation_set', False):
            if st.session_state.models_loaded and models and vectorizer and bert_model:
                st.divider()
                st.subheader("🧪 Teste do Conjunto de Validação" if current_lang == 'pt' else "🧪 Validation Set Test")
                
                with st.spinner("Testando todos os exemplos do conjunto de validação..." if current_lang == 'pt' else "Testing all validation set examples..."):
                    results = test_entire_validation_set(
                        models, vectorizer, bert_model, embedding_type, model_type
                    )
                    
                    if results:
                        # Save results to session_state for persistence
                        st.session_state.validation_test_results = results
                        st.session_state.validation_test_embedding = embedding_type
                        st.session_state.validation_test_model = model_type
                        # Reset saved flag for new test
                        st.session_state.validation_test_saved = False
                        # Don't clear test_validation_set yet - keep it to show results
                    
        # Show validation test results if available (either from new test or previous)
        if st.session_state.get('validation_test_results'):
            results = st.session_state.validation_test_results
            embedding_type_display = st.session_state.get('validation_test_embedding', embedding_type)
            model_type_display = st.session_state.get('validation_test_model', model_type)
            
            # Only show "Teste concluído" if this is a new test
            if st.session_state.get('test_validation_set', False):
                st.divider()
                st.subheader("🧪 Teste do Conjunto de Validação" if current_lang == 'pt' else "🧪 Validation Set Test")
                st.markdown(f"### {render_svg('check', 24, icon_color)} {'Teste concluído!' if current_lang == 'pt' else 'Test completed!'}", unsafe_allow_html=True)
                # Clear the flag after showing success message
                st.session_state.test_validation_set = False
                
                # Save predictions to dashboard (only once, when test is first completed)
                if 'validation_test_saved' not in st.session_state or not st.session_state.validation_test_saved:
                    for pred in results['predictions']:
                        prediction_entry = {
                            'timestamp': datetime.now().isoformat(),
                            'texto': pred['text'][:200] + '...' if len(pred['text']) > 200 else pred['text'],
                            'classe_predita': pred['predicted'],
                            'categoria_predita': pred['predicted_label'],
                            'score': pred['score'],
                            'embedding_usado': pred['embedding_usado'],
                            'modelo_usado': pred['modelo_usado'],
                            'fonte': 'validation_test'
                        }
                        st.session_state.session_predictions.append(prediction_entry)
                    
                    # Save to cookie
                    save_predictions_to_cookie(st.session_state.session_predictions)
                    st.session_state.validation_test_saved = True
            else:
                # Show existing results (from previous test)
                st.divider()
                st.subheader("🧪 Teste do Conjunto de Validação" if current_lang == 'pt' else "🧪 Validation Set Test")
            
            if results:
                # Display metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Precisão" if current_lang == 'pt' else "Accuracy",
                        f"{results['accuracy']:.2%}"
                    )
                with col2:
                    st.metric(
                        "Corretos" if current_lang == 'pt' else "Correct",
                        f"{results['correct']}/{results['total']}"
                    )
                with col3:
                    st.metric(
                        "Total" if current_lang == 'pt' else "Total",
                        results['total']
                    )
                
                # Separate correct and incorrect predictions
                correct_preds = [p for p in results['predictions'] if p['correct']]
                incorrect_preds = [p for p in results['predictions'] if not p['correct']]
                
                # Show all predictions: errors first, then correct ones
                if results['predictions']:
                    st.subheader("📊 Todas as Predições do Conjunto de Validação" if current_lang == 'pt' else "📊 All Validation Set Predictions")
                    
                    # Show errors first if any
                    if incorrect_preds:
                        st.markdown(f"### {render_svg('alert', 24, icon_color)} {'Predições Incorretas' if current_lang == 'pt' else 'Incorrect Predictions'}", unsafe_allow_html=True)
                        error_df = pd.DataFrame([{
                            'Texto': p['text'][:150] + '...' if len(p['text']) > 150 else p['text'],
                            'Classe Real': p['true_label'],
                            'Classe Predita': p['predicted_label'],
                            'Correto': 'Não' if not p['correct'] else 'Sim'
                        } for p in incorrect_preds])
                        st.dataframe(error_df, use_container_width=True, hide_index=True)
                        
                        # AI Explanation for errors (generalized for multiple errors)
                        st.markdown("### 🤖 Análise de Erros por IA" if current_lang == 'pt' else "### 🤖 AI Error Analysis")
                        
                        # Check if error analysis already exists
                        error_analysis_key = f'validation_error_analysis_{embedding_type_display}_{model_type_display}'
                        if error_analysis_key in st.session_state and st.session_state[error_analysis_key]:
                            st.info(st.session_state[error_analysis_key])
                            if st.button("🔄 Gerar Nova Análise" if current_lang == 'pt' else "🔄 Generate New Analysis", key="regenerate_error_analysis"):
                                st.session_state[error_analysis_key] = None
                                st.rerun()
                        else:
                            if st.button("🔍 Analisar Erros com IA" if current_lang == 'pt' else "🔍 Analyze Errors with AI", key="analyze_errors_ai"):
                                try:
                                    # Create a generalized prompt for multiple errors
                                    num_errors = len(incorrect_preds)
                                    error_examples = incorrect_preds[:5]  # Use first 5 errors as examples
                                    
                                    examples_text = "\n\n".join([
                                        f"Exemplo {i+1}:\n"
                                        f"Texto: {p['text'][:200]}...\n"
                                        f"Classe Real: {p['true_label']}\n"
                                        f"Classe Predita: {p['predicted_label']}"
                                        for i, p in enumerate(error_examples)
                                    ])
                                    
                                    if current_lang == 'pt':
                                        prompt = f"""O classificador de texto cometeu {num_errors} erros ao classificar um conjunto de {results['total']} notícias.

Exemplos de erros:
{examples_text}

Analise os padrões de erro e explique brevemente (2-4 razões principais) por que o classificador pode estar errando. Seja conciso e focado nos motivos mais prováveis."""
                                    else:
                                        prompt = f"""The text classifier made {num_errors} errors when classifying a set of {results['total']} news articles.

Error examples:
{examples_text}

Analyze the error patterns and briefly explain (2-4 main reasons) why the classifier may be making mistakes. Be concise and focus on the most likely reasons."""
                                    
                                    with st.spinner("Analisando erros com IA..." if current_lang == 'pt' else "Analyzing errors with AI..."):
                                        imports = _lazy_imports()
                                        error_analysis = imports['call_groq_llm'](prompt, max_tokens=600)
                                        # Save to session_state for persistence
                                        st.session_state[error_analysis_key] = error_analysis
                                        st.rerun()  # Rerun to show the analysis
                                except Exception as e:
                                    st.warning(f"❌ Erro ao analisar com IA: {e}" if current_lang == 'pt' else f"❌ Error analyzing with AI: {e}")
                                    st.info(t('explanation_info'))
                        
                        st.divider()
                    
                    # Show all predictions in a table
                    st.markdown("### 📋 Todas as Predições" if current_lang == 'pt' else "### 📋 All Predictions")
                    all_preds_df = pd.DataFrame([{
                        'Texto': p['text'][:150] + '...' if len(p['text']) > 150 else p['text'],
                        'Classe Real': p['true_label'],
                        'Classe Predita': p['predicted_label'],
                        'Confiança': f"{p['score']:.2%}",
                        'Correto': 'Sim' if p['correct'] else 'Não'
                    } for p in (incorrect_preds + correct_preds)])  # Errors first
                    st.dataframe(all_preds_df, use_container_width=True, hide_index=True)
            else:
                # Error case - clear results
                st.error("❌ Erro ao testar conjunto de validação. Verifique os logs." if current_lang == 'pt' else "❌ Error testing validation set. Check logs.")
                if 'validation_test_results' in st.session_state:
                    del st.session_state.validation_test_results
                st.session_state.test_validation_set = False
        elif st.session_state.get('test_validation_set', False):
            # Test was requested but models not loaded
            st.warning("⚠️ Modelos não carregados. Aguarde o carregamento dos modelos." if current_lang == 'pt' else "⚠️ Models not loaded. Please wait for models to load.")
            st.session_state.test_validation_set = False
        
        # Text input
        # Text input with sample button
        col_text, col_btn = st.columns([4, 1])
        
        with col_text:
            # Initialize text_input_area in session state if not exists
            if 'text_input_area' not in st.session_state:
                st.session_state.text_input_area = ''
            
            # If we have a new sample, update the text area
            if 'sample_text' in st.session_state and st.session_state.sample_text:
                # sample_text can be either a string (old format) or tuple (text, true_label)
                if isinstance(st.session_state.sample_text, tuple):
                    sample_text, true_label = st.session_state.sample_text
                    st.session_state.text_input_area = sample_text
                    st.session_state.true_label = true_label
                    # Store original text to detect modifications
                    st.session_state.original_sample_text = sample_text
                else:
                    # Old format: just text
                    st.session_state.text_input_area = st.session_state.sample_text
                    # Clear true_label if it exists
                    if 'true_label' in st.session_state:
                        del st.session_state.true_label
                    if 'original_sample_text' in st.session_state:
                        del st.session_state.original_sample_text
                # Clear sample_text after using it
                st.session_state.sample_text = ''
            
            # Use key to bind to session_state (don't use value parameter when using key)
            text_input = st.text_area(
                t('enter_text'),
                height=200,
                placeholder=t('text_placeholder'),
                key="text_input_area"
            )
            
            # Track original sample text to detect user modifications
            if 'sample_text' in st.session_state and isinstance(st.session_state.sample_text, tuple):
                original_sample_text = st.session_state.sample_text[0]
            elif 'original_sample_text' in st.session_state:
                original_sample_text = st.session_state.original_sample_text
            else:
                original_sample_text = None
            
            # Detect if user modified the text (and clear true_label if so)
            if original_sample_text and text_input != original_sample_text:
                # User modified the text - mark for removal (will be removed on next rerun)
                if 'true_label' in st.session_state:
                    # Set flag to trigger fade-out animation
                    st.session_state.remove_true_label = True
                if 'original_sample_text' in st.session_state:
                    del st.session_state.original_sample_text
            
            # Display true label if available (from validation sample)
            if 'true_label' in st.session_state:
                true_label = st.session_state.true_label
                true_category = CLASS_TO_CATEGORY.get(int(true_label), f"Classe {true_label}")
                
                # Add CSS and JavaScript for fade-out animation
                animation_code = """
                <style>
                @keyframes fadeOut {
                    from { 
                        opacity: 1; 
                        transform: translateY(0);
                    }
                    to { 
                        opacity: 0; 
                        transform: translateY(-10px);
                    }
                }
                .ground-truth-fade-out {
                    animation: fadeOut 0.5s ease-out forwards;
                }
                </style>
                <script>
                (function() {
                    // Check if we should trigger fade-out
                    var shouldFadeOut = """ + str(st.session_state.get('remove_true_label', False)).lower() + """;
                    if (shouldFadeOut) {
                        // Find the ground truth label element
                        var elements = window.parent.document.querySelectorAll('[data-testid="stInfo"]');
                        if (elements.length > 0) {
                            var lastElement = elements[elements.length - 1];
                            // Check if it contains ground truth text
                            if (lastElement.textContent.includes('Classe Real') || lastElement.textContent.includes('True Label')) {
                                lastElement.classList.add('ground-truth-fade-out');
                                // Remove after animation completes
                                setTimeout(function() {
                                    lastElement.style.display = 'none';
                                }, 500);
                            }
                        }
                    }
                })();
                </script>
                """
                st.components.v1.html(animation_code, height=0)
                
                # Clear the flag after applying animation
                if st.session_state.get('remove_true_label', False):
                    del st.session_state.remove_true_label
                    del st.session_state.true_label
                
                st.info(
                    f"**Classe Real (Ground Truth):** {true_category}" if current_lang == 'pt' 
                    else f"**True Label (Ground Truth):** {true_category}",
                    icon=None
                )
        
        with col_btn:
            st.write("")  # Spacing
            st.write("")  # Spacing
            
            if st.button(
                "Exemplo do Conjunto de Validação" if current_lang == 'pt' else "Validation Set Sample",
                width='stretch',
                help="Carrega um texto aleatório do conjunto de validação (não visto durante o treinamento)" if current_lang == 'pt' else "Load a random text from validation set (not seen during training)"
            ):
                with st.spinner("Carregando exemplo..." if current_lang == 'pt' else "Loading sample..."):
                    result = get_validation_sample()
                    if result:
                        # result is now a tuple (text, true_label)
                        # Store sample in session_state - will be applied on next rerun
                        st.session_state.sample_text = result
                        st.rerun()
                    else:
                        st.error("Não foi possível carregar exemplo. Verifique os logs do console para mais detalhes." if current_lang == 'pt' else "Could not load sample. Check console logs for details.")
        
        # Clear sample_text after use to avoid persistence
        if 'sample_text' in st.session_state and st.session_state.sample_text:
            # Check if text was modified (compare with text from sample)
            sample_text_value = st.session_state.sample_text
            if isinstance(sample_text_value, tuple):
                sample_text_value = sample_text_value[0]
            if text_input != sample_text_value:
                st.session_state.sample_text = ''
                # Also clear true_label if user modified the text
                if 'true_label' in st.session_state:
                    del st.session_state.true_label
        
        col1, col2 = st.columns([1, 4])
        with col1:
            classify_button = st.button(t('classify'), type="primary", width='stretch')
        with col2:
            save_prediction = st.checkbox(t('save_log'), value=True)
        
        # Handle classification
        if classify_button and text_input:
            with st.spinner(t('classifying')):
                result = classify_text_streamlit(
                    text_input,
                    embedding_type,
                    model_type,
                    models,
                    vectorizer,
                    bert_model
                )
            
            # Store result in session state
            st.session_state.last_classification_result = result
            st.session_state.last_text_input = text_input
            st.session_state.last_embedding_type = embedding_type
            st.session_state.last_model_type = model_type
            st.session_state.explanation_generated = False
            st.session_state.llm_explanation = None
            st.session_state.log_saved_for_current = False
        
        # Display results if available (from current classification or previous)
        result = st.session_state.last_classification_result
        text_input_for_display = st.session_state.get('last_text_input', '')
        
        # Show results if we have a result (even if text_input_for_display is empty, we can still show explanation)
        if result is not None:
            # Display results
            st.divider()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                # Check if prediction is correct (if we have true_label from validation sample)
                is_correct = False
                if 'true_label' in st.session_state:
                    predicted_class = result['classe_predita']
                    true_label = int(st.session_state.true_label)
                    is_correct = (predicted_class == true_label)
                
                # Display predicted class with checkmark if correct
                if is_correct:
                    # Show success indicator
                    st.markdown(f"### {t('predicted_class')}")
                    st.markdown(f"**{result['categoria_predita']}**")
                    # SVG checkmark
                    checkmark_svg = """
                    <svg width="60" height="60" viewBox="0 0 60 60" xmlns="http://www.w3.org/2000/svg">
                        <circle cx="30" cy="30" r="28" fill="#10b981" opacity="0.2"/>
                        <path d="M 20 30 L 27 37 L 40 24" stroke="#10b981" stroke-width="4" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
                    </svg>
                    """
                    st.markdown(checkmark_svg, unsafe_allow_html=True)
                else:
                    st.metric(t('predicted_class'), result['categoria_predita'])
            with col2:
                st.metric(t('confidence'), f"{result['score']:.2%}")
            with col3:
                st.metric(t('model'), f"{result['embedding_usado']} + {result['modelo_usado']}")
            
            # Probability distribution
            if result['all_probas']:
                st.subheader(t('prob_dist'))
                prob_df = pd.DataFrame({
                    'Categoria' if current_lang == 'pt' else 'Category': list(result['all_probas'].keys()),
                    'Probabilidade' if current_lang == 'pt' else 'Probability': list(result['all_probas'].values())
                }).sort_values('Probabilidade' if current_lang == 'pt' else 'Probability', ascending=False)
                
                fig = px.bar(
                    prob_df,
                    x='Categoria' if current_lang == 'pt' else 'Category',
                    y='Probabilidade' if current_lang == 'pt' else 'Probability',
                    color='Probabilidade' if current_lang == 'pt' else 'Probability',
                    color_continuous_scale='Blues',
                    title=t('prob_dist')
                )
                fig.update_layout(showlegend=False, height=400)
                fig = update_chart_layout(fig)
                st.plotly_chart(fig, width='stretch')
            
            # LLM Explanation section (always visible when result exists)
            st.markdown(f"### {render_svg('robot', 24)} {t('ai_explanation')}", unsafe_allow_html=True)
            
            # Check if prediction is incorrect (if we have true_label)
            is_incorrect = False
            true_category = None
            if 'true_label' in st.session_state:
                predicted_class = result['classe_predita']
                true_label = int(st.session_state.true_label)
                is_incorrect = (predicted_class != true_label)
                if is_incorrect:
                    true_category = CLASS_TO_CATEGORY.get(true_label, f"Classe {true_label}")
            
            # Show explanation if already generated
            if st.session_state.explanation_generated and st.session_state.llm_explanation:
                st.info(st.session_state.llm_explanation)
                if st.button("🔄 Gerar Nova Explicação" if current_lang == 'pt' else "🔄 Generate New Explanation", key="regenerate_explain"):
                    st.session_state.explanation_generated = False
                    st.session_state.llm_explanation = None
                    st.rerun()
            else:
                explain_button = st.button(t('generate_explanation'), key="explain")
                
                if explain_button:
                    try:
                        # Use text if available, otherwise use a generic message
                        text_snippet = text_input_for_display[:500] if text_input_for_display else "Texto classificado anteriormente"
                        
                        if is_incorrect and true_category:
                            # Error explanation: explain why the classifier made a mistake
                            if current_lang == 'pt':
                                prompt = f"""O classificador de texto cometeu um erro ao classificar a seguinte notícia.

Texto:
{text_snippet}

Classe Real (correta): {true_category}
Classe Predita (incorreta): {result['categoria_predita']}
Confiança da predição: {result['score']:.2%}

Explique brevemente por que o classificador pode ter errado. Liste possíveis motivos de forma concisa (2-3 razões principais)."""
                            else:
                                prompt = f"""The text classifier made an error when classifying the following news.

Text:
{text_snippet}

True Label (correct): {true_category}
Predicted Label (incorrect): {result['categoria_predita']}
Prediction confidence: {result['score']:.2%}

Briefly explain why the classifier may have made this mistake. List possible reasons concisely (2-3 main reasons)."""
                        else:
                            # Correct prediction: explain why it's correct
                            if current_lang == 'pt':
                                prompt = f"""Classifique o seguinte texto de notícia e explique por que ele foi categorizado como "{result['categoria_predita']}".

Texto:
{text_snippet}

Categoria predita: {result['categoria_predita']}
Confiança: {result['score']:.2%}

Explique de forma clara e concisa por que este texto pertence a esta categoria."""
                            else:
                                prompt = f"""Classify the following news text and explain why it was categorized as "{result['categoria_predita']}".

Text:
{text_snippet}

Predicted category: {result['categoria_predita']}
Confidence: {result['score']:.2%}

Explain clearly and concisely why this text belongs to this category."""
                        
                        with st.spinner(t('generating')):
                            imports = _lazy_imports()
                            explanation = imports['call_groq_llm'](prompt, max_tokens=600)
                            st.session_state.llm_explanation = explanation
                            st.session_state.explanation_generated = True
                            st.rerun()
                    except Exception as e:
                        st.warning(f"{t('explanation_error')} {e}")
                        st.info(t('explanation_info'))
            
            # Save to cookie-based log (persists across page refreshes)
            if classify_button and save_prediction and text_input:
                # Add to cookie-based predictions (persists across F5)
                prediction_entry = {
                    'timestamp': datetime.now().isoformat(),
                    'texto': text_input[:200] + "..." if len(text_input) > 200 else text_input,  # Truncate for display
                    'classe_predita': result['classe_predita'],
                    'categoria_predita': result['categoria_predita'],
                    'score': result['score'],
                    'embedding_usado': result['embedding_usado'],
                    'modelo_usado': result['modelo_usado'],
                    'fonte': 'streamlit'
                }
                st.session_state.session_predictions.append(prediction_entry)
                # Save to cookie (persists across F5 and browser restarts for 30 days)
                save_predictions_to_cookie(st.session_state.session_predictions)
                
                if not st.session_state.get('log_saved_for_current', False):
                    # Success message with dynamic styling
                    success_bg = 'rgba(76, 217, 100, 0.1)' if st.session_state.theme == 'dark' else '#f0fff4'
                    success_text = '#fafafa' if st.session_state.theme == 'dark' else '#1a1a1a'
                    
                    st.markdown(f'''
                        <div style="padding: 1rem; border: 1px solid #4CD964; background-color: {success_bg}; color: {success_text}; display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1rem;">
                            {render_svg('check', 20, '#4CD964')}
                            <span style="font-weight: 500;">{t('saved_log')}</span>
                        </div>
                    ''', unsafe_allow_html=True)
                    st.session_state.log_saved_for_current = True
    
    # Tab 2: Monitoring
    with tab2:
        st.header(t('monitoring_dashboard'))
        
        # Advanced filters (bonus feature inspired by Módulo 16)
        with st.expander("Filtros Avançados" if current_lang == 'pt' else "Advanced Filters"):
            col1, col2, col3 = st.columns(3)
            with col1:
                filter_category = st.multiselect(
                    "Categoria" if current_lang == 'pt' else "Category",
                    options=["Economia", "Esportes", "Polícia e Direitos", "Política", "Turismo", "Variedades e Sociedade"],
                    default=[]
                )
            with col2:
                filter_embedding = st.multiselect(
                    "Embedding" if current_lang == 'pt' else "Embedding",
                    options=["TF-IDF", "BERT"],
                    default=[]
                )
            with col3:
                filter_model = st.multiselect(
                    "Modelo" if current_lang == 'pt' else "Model",
                    options=["SVM", "XGBoost"],
                    default=[]
                )
        
        # Use cookie-based predictions (persists across page refreshes)
        # Load from session_state (which is synced with cookies on save)
        session_predictions = st.session_state.get('session_predictions', [])
        
        if not session_predictions:
            st.info(t('no_predictions'))
            st.caption("**Dica**: As predições são salvas em cookies e persistem mesmo após atualizar a página (F5). Cada navegador/computador tem seu próprio histórico." if current_lang == 'pt' else "**Tip**: Predictions are saved in cookies and persist even after refreshing the page (F5). Each browser/computer has its own history.")
        else:
            # Convert to DataFrame for easier manipulation
            logs_df = pd.DataFrame(session_predictions)
            
            # Apply filters
            if filter_category:
                logs_df = logs_df[logs_df['categoria_predita'].isin(filter_category)]
            if filter_embedding:
                logs_df = logs_df[logs_df['embedding_usado'].isin(filter_embedding)]
            if filter_model:
                logs_df = logs_df[logs_df['modelo_usado'].isin(filter_model)]
            
            if logs_df.empty:
                st.info("Nenhuma predição encontrada com os filtros selecionados." if current_lang == 'pt' else "No predictions found with selected filters.")
            else:
                # Calculate statistics from session data
                stats = {
                    'total_predictions': len(logs_df),
                    'avg_score': float(logs_df['score'].mean()) if 'score' in logs_df.columns else 0.0,
                    'by_class': logs_df['categoria_predita'].value_counts().to_dict() if 'categoria_predita' in logs_df.columns else {},
                    'by_model': logs_df['modelo_usado'].value_counts().to_dict() if 'modelo_usado' in logs_df.columns else {},
                    'by_embedding': logs_df['embedding_usado'].value_counts().to_dict() if 'embedding_usado' in logs_df.columns else {},
                    'date_range': {
                        'start': logs_df['timestamp'].min() if 'timestamp' in logs_df.columns and not logs_df.empty else None,
                        'end': logs_df['timestamp'].max() if 'timestamp' in logs_df.columns and not logs_df.empty else None
                    }
                }
                
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric(t('total_predictions'), stats['total_predictions'])
                with col2:
                    st.metric(t('avg_score'), f"{stats['avg_score']:.2%}")
                with col3:
                    most_common = max(stats['by_class'].items(), key=lambda x: x[1]) if stats['by_class'] else ("N/A", 0)
                    st.metric(t('most_common'), f"{most_common[0]} ({most_common[1]})")
                with col4:
                    if stats['date_range']['start']:
                        st.metric(t('date_range'), f"{stats['date_range']['start'][:10]} a {stats['date_range']['end'][:10]}" if current_lang == 'pt' else f"{stats['date_range']['start'][:10]} to {stats['date_range']['end'][:10]}")
                
                st.divider()
                
                # Charts
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader(t('by_class'))
                    if stats['by_class']:
                        class_df = pd.DataFrame({
                            'Categoria' if current_lang == 'pt' else 'Category': list(stats['by_class'].keys()),
                            'Quantidade' if current_lang == 'pt' else 'Count': list(stats['by_class'].values())
                        })
                        fig = px.pie(
                            class_df,
                            values='Quantidade' if current_lang == 'pt' else 'Count',
                            names='Categoria' if current_lang == 'pt' else 'Category',
                            title=t('dist_by_category')
                        )
                        fig = update_chart_layout(fig)
                        st.plotly_chart(fig, width='stretch')
                
                with col2:
                    st.subheader(t('by_model'))
                    if stats['by_model']:
                        model_df = pd.DataFrame({
                            'Modelo' if current_lang == 'pt' else 'Model': list(stats['by_model'].keys()),
                            'Quantidade' if current_lang == 'pt' else 'Count': list(stats['by_model'].values())
                        })
                        fig = px.bar(
                            model_df,
                            x='Modelo' if current_lang == 'pt' else 'Model',
                            y='Quantidade' if current_lang == 'pt' else 'Count',
                            title=t('usage_by_model'),
                            color='Quantidade' if current_lang == 'pt' else 'Count',
                            color_continuous_scale='Blues'
                        )
                        fig = update_chart_layout(fig)
                        st.plotly_chart(fig, width='stretch')
                
                # Temporal evolution
                if 'timestamp' in logs_df.columns:
                    st.subheader(t('temporal_evolution'))
                    logs_df['date'] = pd.to_datetime(logs_df['timestamp']).dt.date
                    daily_counts = logs_df.groupby('date').size().reset_index(name='count')
                    daily_counts = daily_counts.sort_values('date')
                    
                    fig = px.line(
                        daily_counts,
                        x='date',
                        y='count',
                        title=t('predictions_over_time'),
                        markers=True
                    )
                    fig = update_chart_layout(fig)
                    st.plotly_chart(fig, width='stretch')
            
            # Additional visualizations
            st.divider()
            st.subheader("📊 Análise Comparativa Avançada" if current_lang == 'pt' else "📊 Advanced Comparative Analysis")
            
            # Optimization Comparison Section
            st.divider()
            st.subheader("🎯 Comparação: Antes vs Depois da Otimização (Optuna)" if current_lang == 'pt' else "🎯 Comparison: Before vs After Optimization (Optuna)")
            
            try:
                opt_comparison_path = Path(__file__).parent.parent / 'models' / 'optimization_comparison.csv'
                if opt_comparison_path.exists():
                    opt_df = pd.read_csv(opt_comparison_path)
                    
                    # Create comparison chart
                    fig = go.Figure()
                    
                    models_list = opt_df['Model'].tolist()
                    f1_optimized = opt_df['F1-Optimized'].tolist()
                    f1_default = opt_df['F1-Default'].tolist()
                    improvements = opt_df['Improvement %'].tolist()
                    
                    fig.add_trace(go.Bar(
                        name='F1-Default (Padrão)' if current_lang == 'pt' else 'F1-Default',
                        x=models_list,
                        y=f1_default,
                        marker_color='lightblue',
                        text=[f'{val:.3f}' for val in f1_default],
                        textposition='auto'
                    ))
                    
                    fig.add_trace(go.Bar(
                        name='F1-Optimized (Otimizado)' if current_lang == 'pt' else 'F1-Optimized',
                        x=models_list,
                        y=f1_optimized,
                        marker_color='darkblue',
                        text=[f'{val:.3f}' for val in f1_optimized],
                        textposition='auto'
                    ))
                    
                    fig.update_layout(
                        title='Comparação F1-Macro: Padrão vs Otimizado (K-fold CV)' if current_lang == 'pt' else 'F1-Macro Comparison: Default vs Optimized (K-fold CV)',
                        xaxis_title='Modelo' if current_lang == 'pt' else 'Model',
                        yaxis_title='F1-Macro',
                        barmode='group',
                        height=500,
                        showlegend=True
                    )
                    fig = update_chart_layout(fig)
                    
                    st.plotly_chart(fig, width='stretch')
                    
                    # Improvement metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Melhorias por Modelo**" if current_lang == 'pt' else "**Improvements by Model**")
                        improvement_df = pd.DataFrame({
                            'Modelo' if current_lang == 'pt' else 'Model': models_list,
                            'Melhoria (%)' if current_lang == 'pt' else 'Improvement (%)': [f"+{imp:.2f}%" for imp in improvements]
                        })
                        st.dataframe(improvement_df, width='stretch', hide_index=True)
                    
                    with col2:
                        st.markdown("**Tabela Comparativa Completa**" if current_lang == 'pt' else "**Complete Comparison Table**")
                        display_df = opt_df.copy()
                        display_df['F1-Optimized'] = display_df['F1-Optimized'].round(4)
                        display_df['F1-Default'] = display_df['F1-Default'].round(4)
                        display_df['Improvement'] = display_df['Improvement'].round(4)
                        display_df['Improvement %'] = display_df['Improvement %'].round(2)
                        if current_lang == 'pt':
                            display_df.columns = ['Modelo', 'F1-Otimizado', 'F1-Padrão', 'Melhoria', 'Melhoria (%)']
                        st.dataframe(display_df, width='stretch', hide_index=True)
                    
                    # Key insights
                    st.info("""
                    **💡 Principais Descobertas:**
                    - **BERT + XGBoost**: Maior ganho (+3.96%) - otimização muito benéfica
                    - **TF-IDF + XGBoost**: Ganho significativo (+2.32%) - hiperparâmetros padrão não eram ideais
                    - **BERT + SVM**: Pequeno ganho (+0.37%) - mudou para kernel RBF (importante!)
                    - **TF-IDF + SVM**: Ganho marginal (+0.02%) - já estava bem otimizado
                    """ if current_lang == 'pt' else """
                    **💡 Key Findings:**
                    - **BERT + XGBoost**: Largest gain (+3.96%) - optimization very beneficial
                    - **TF-IDF + XGBoost**: Significant gain (+2.32%) - default hyperparameters not ideal
                    - **BERT + SVM**: Small gain (+0.37%) - changed to RBF kernel (important!)
                    - **TF-IDF + SVM**: Marginal gain (+0.02%) - already well optimized
                    """)
                else:
                    st.warning("Arquivo de comparação de otimização não encontrado. Execute scripts/run_optimization.py primeiro." if current_lang == 'pt' else "Optimization comparison file not found. Run scripts/run_optimization.py first.")
            except Exception as e:
                st.warning(f"Erro ao carregar comparação de otimização: {e}" if current_lang == 'pt' else f"Error loading optimization comparison: {e}")
            
            st.divider()
            st.subheader("📈 Visualizações Adicionais" if current_lang == 'pt' else "📈 Additional Visualizations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Score distribution by embedding
                if 'embedding_usado' in logs_df.columns and 'score' in logs_df.columns:
                    st.markdown("**Distribuição de Scores por Embedding**" if current_lang == 'pt' else "**Score Distribution by Embedding**")
                    fig = px.box(
                        logs_df,
                        x='embedding_usado',
                        y='score',
                        color='embedding_usado',
                        title="Score Distribution" if current_lang == 'en' else "Distribuição de Scores",
                        labels={'embedding_usado': 'Embedding' if current_lang == 'en' else 'Embedding', 'score': 'Score'}
                    )
                    fig = update_chart_layout(fig)
                    st.plotly_chart(fig, width='stretch')
            
            with col2:
                # Score distribution by model
                if 'modelo_usado' in logs_df.columns and 'score' in logs_df.columns:
                    st.markdown("**Distribuição de Scores por Modelo**" if current_lang == 'pt' else "**Score Distribution by Model**")
                    fig = px.box(
                        logs_df,
                        x='modelo_usado',
                        y='score',
                        color='modelo_usado',
                        title="Score Distribution" if current_lang == 'en' else "Distribuição de Scores",
                        labels={'modelo_usado': 'Model' if current_lang == 'en' else 'Modelo', 'score': 'Score'}
                    )
                    fig = update_chart_layout(fig)
                    st.plotly_chart(fig, width='stretch')
            
            # Performance vs Efficiency Trade-off (if we have model performance data)
            try:
                efficiency_path = Path(__file__).parent.parent / 'models' / 'table_a_efficiency.csv'
                if efficiency_path.exists():
                    st.divider()
                    st.subheader("⚖️ Trade-off: Performance vs Eficiência" if current_lang == 'pt' else "⚖️ Trade-off: Performance vs Efficiency")
                    
                    eff_df = pd.read_csv(efficiency_path)
                    
                    fig = go.Figure()
                    
                    for idx, row in eff_df.iterrows():
                        fig.add_trace(go.Scatter(
                            x=[row['Latency (ms/doc)']],
                            y=[row['F1-Macro']],
                            mode='markers+text',
                            name=row['Setup'],
                            text=[row['Setup']],
                            textposition="top center",
                            marker=dict(
                                size=15,
                                color=['#4A90E2', '#EE4C2C', '#00C853', '#FF6600'][idx % 4]
                            ),
                            hovertemplate=f"<b>{row['Setup']}</b><br>" +
                                        f"F1-Macro: {row['F1-Macro']:.3f}<br>" +
                                        f"Latency: {row['Latency (ms/doc)']:.3f} ms/doc<br>" +
                                        f"Cold Start: {row['Cold Start (s)']:.3f} s<extra></extra>"
                        ))
                    
                    fig.update_layout(
                        title="Performance vs Latency Trade-off" if current_lang == 'en' else "Trade-off: Performance vs Latência",
                        xaxis_title="Latência (ms/doc)" if current_lang == 'pt' else "Latency (ms/doc)",
                        yaxis_title="F1-Macro",
                        height=500,
                        showlegend=False
                    )
                    fig = update_chart_layout(fig)
                    
                    st.plotly_chart(fig, width='stretch')
                    
                    # Add efficiency comparison table
                    st.markdown("**Comparação de Eficiência**" if current_lang == 'pt' else "**Efficiency Comparison**")
                    display_eff = eff_df[['Setup', 'F1-Macro', 'Accuracy', 'Latency (ms/doc)', 'Cold Start (s)', 'Tamanho (MB)']].copy()
                    display_eff = display_eff.round(3)
                    st.dataframe(display_eff, width='stretch', hide_index=True)
            except Exception as e:
                pass  # Silently fail if efficiency data not available
            
                # Recent predictions table (session-based, private per user)
                st.divider()
                st.subheader(t('recent_predictions'))
                if not logs_df.empty and 'timestamp' in logs_df.columns:
                    display_df = logs_df[['timestamp', 'categoria_predita', 'score', 'embedding_usado', 'modelo_usado']].tail(20).copy()
                    display_df = display_df.sort_values('timestamp', ascending=False)
                    # Format timestamp for better readability
                    display_df['timestamp'] = pd.to_datetime(display_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
                    st.dataframe(display_df, width='stretch', hide_index=True)
                    st.caption("📊 **Histórico da sua sessão atual** - Os dados são privados e não são compartilhados com outros usuários." if current_lang == 'pt' else "📊 **Your current session history** - Data is private and not shared with other users.")
                else:
                    st.info("Nenhuma predição recente na sua sessão." if current_lang == 'pt' else "No recent predictions in your session.")
                
                # Export data and clear session (bonus feature - Módulo 16)
                st.divider()
                col1, col2, col3 = st.columns(3)
                with col1:
                    if not logs_df.empty:
                        csv_export = logs_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Exportar CSV" if current_lang == 'pt' else "📥 Export CSV",
                            data=csv_export,
                            file_name=f"predicoes_sessao_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            width='stretch'
                        )
                with col2:
                    if not logs_df.empty and 'timestamp' in logs_df.columns:
                        st.info(f"📊 {len(logs_df)} predições na sua sessão" if current_lang == 'pt' else f"📊 {len(logs_df)} predictions in your session")
                with col3:
                    if st.session_state.get('session_predictions'):
                        if st.button("🗑️ Limpar Histórico" if current_lang == 'pt' else "🗑️ Clear History", width='stretch'):
                            st.session_state.session_predictions = []
                            # Clear cookie
                            save_predictions_to_cookie([])
                            st.success("Histórico limpo!" if current_lang == 'pt' else "History cleared!")
                            st.rerun()


if __name__ == "__main__":
    main()

