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

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import PATHS
from src.preprocessing import preprocess_text
from src.embeddings import load_tfidf_vectorizer, load_bert_model
from src.train import load_trained_models
from src.logging_system import log_prediction, load_prediction_logs, get_log_statistics
from src.class_mapping import CLASS_TO_CATEGORY
from src.llm_analysis import call_groq_llm, load_class_profiles


# Translations
TRANSLATIONS = {
    'pt': {
        'title': '📰 NewsLens AI Classifier',
        'subtitle': 'Análise Comparativa de Representações Esparsas vs. Densas',
        'config': '⚙️ Configuração',
        'embedding_type': 'Tipo de Embedding',
        'embedding_help': 'Escolha entre BERT (denso) ou TF-IDF (esparso)',
        'model_type': 'Tipo de Modelo',
        'model_help': 'Escolha entre SVM ou XGBoost',
        'performance': '📊 Desempenho dos Modelos',
        'best_perf': 'Melhor Performance:',
        'best_eff': 'Melhor Eficiência:',
        'about': 'ℹ️ Sobre',
        'classification': '🔍 Classificação',
        'monitoring': '📊 Monitoramento',
        'text_classification': 'Classificação de Texto',
        'enter_text': 'Digite o texto para classificar:',
        'text_placeholder': 'Cole ou digite uma notícia aqui...',
        'classify': '🔍 Classificar',
        'save_log': 'Salvar predição no log',
        'loading_models': 'Carregando modelos...',
        'models_loaded': 'Modelos carregados com sucesso!',
        'models_error': 'Falha ao carregar modelos. Verifique se os modelos foram treinados.',
        'classifying': 'Classificando...',
        'predicted_class': 'Classe Predita',
        'confidence': 'Confiança',
        'model': 'Modelo',
        'prob_dist': 'Distribuição de Probabilidades',
        'ai_explanation': '🤖 Explicação por IA',
        'generate_explanation': 'Gerar Explicação',
        'generating': 'Gerando explicação...',
        'explanation_error': 'Não foi possível gerar explicação:',
        'explanation_info': 'A explicação por LLM requer a variável de ambiente GROQ_API_KEY.',
        'saved_log': '✅ Predição salva no log!',
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
        'title': '📰 NewsLens AI Classifier',
        'subtitle': 'Comparative Analysis of Sparse vs. Dense Representations',
        'config': '⚙️ Configuration',
        'embedding_type': 'Embedding Type',
        'embedding_help': 'Choose between BERT (dense) or TF-IDF (sparse) embeddings',
        'model_type': 'Model Type',
        'model_help': 'Choose between SVM or XGBoost classifier',
        'performance': '📊 Model Performance',
        'best_perf': 'Best Performance:',
        'best_eff': 'Best Efficiency:',
        'about': 'ℹ️ About',
        'classification': '🔍 Classification',
        'monitoring': '📊 Monitoring',
        'text_classification': 'Text Classification',
        'enter_text': 'Enter text to classify:',
        'text_placeholder': 'Paste or type a news article here...',
        'classify': '🔍 Classify',
        'save_log': 'Save prediction to log',
        'loading_models': 'Loading models...',
        'models_loaded': 'Models loaded successfully!',
        'models_error': 'Failed to load models. Please check if models are trained.',
        'classifying': 'Classifying...',
        'predicted_class': 'Predicted Class',
        'confidence': 'Confidence',
        'model': 'Model',
        'prob_dist': 'Probability Distribution',
        'ai_explanation': '🤖 AI Explanation',
        'generate_explanation': 'Generate Explanation',
        'generating': 'Generating explanation...',
        'explanation_error': 'Could not generate explanation:',
        'explanation_info': 'LLM explanation requires GROQ_API_KEY environment variable.',
        'saved_log': '✅ Prediction saved to log!',
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


# Page configuration
st.set_page_config(
    page_title="NewsLens AI Classifier",
    page_icon="📰",
    layout="wide"
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


@st.cache_resource
def load_all_models():
    """Carrega todos os modelos e embeddings (em cache)."""
    try:
        models = load_trained_models()
        vectorizer = load_tfidf_vectorizer(PATHS['data_embeddings'] / 'tfidf_vectorizer.pkl')
        bert_model = load_bert_model()
        return models, vectorizer, bert_model
    except Exception as e:
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
    # Preprocess
    processed_text = preprocess_text(text)
    
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
    # Language selector
    lang = st.sidebar.selectbox(
        "🌐 Idioma / Language",
        ["Português", "English"],
        index=0,
        key="lang_selector"
    )
    current_lang = 'pt' if lang == "Português" else 'en'
    st.session_state.language = current_lang
    
    t = lambda key: get_text(key, current_lang)
    
    st.title(t('title'))
    st.markdown(f"**{t('subtitle')}**")
    
    # Sidebar
    with st.sidebar:
        st.header(t('config'))
        
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
        
        st.markdown(f"### {t('about')}")
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
            with st.spinner(t('loading_models')):
                models, vectorizer, bert_model = load_all_models()
                if models is not None:
                    st.session_state.models = models
                    st.session_state.vectorizer = vectorizer
                    st.session_state.bert_model = bert_model
                    st.session_state.models_loaded = True
                    st.success(t('models_loaded'))
                else:
                    st.error(t('models_error'))
                    st.stop()
        else:
            models = st.session_state.models
            vectorizer = st.session_state.vectorizer
            bert_model = st.session_state.bert_model
        
        # Text input
        text_input = st.text_area(
            t('enter_text'),
            height=200,
            placeholder=t('text_placeholder')
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            classify_button = st.button(t('classify'), type="primary", use_container_width=True)
        with col2:
            save_prediction = st.checkbox(t('save_log'), value=True)
        
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
            
            # Display results
            st.divider()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(t('predicted_class'), result['categoria_predita'])
            with col2:
                st.metric(t('confidence'), f"{result['score']:.2%}")
            with col3:
                st.metric(t('model'), f"{result['embedding_usado']} + {result['modelo_usado']}")
            
            # Probability distribution
            if result['all_probas']:
                st.subheader(t('prob_dist'))
                prob_df = pd.DataFrame({
                    'Category' if current_lang == 'en' else 'Categoria': list(result['all_probas'].keys()),
                    'Probability' if current_lang == 'en' else 'Probabilidade': list(result['all_probas'].values())
                }).sort_values('Probability' if current_lang == 'en' else 'Probabilidade', ascending=False)
                
                fig = px.bar(
                    prob_df,
                    x='Category' if current_lang == 'en' else 'Categoria',
                    y='Probability' if current_lang == 'en' else 'Probabilidade',
                    color='Probability' if current_lang == 'en' else 'Probabilidade',
                    color_continuous_scale='Blues',
                    title=t('prob_dist')
                )
                fig.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # LLM Explanation
            st.subheader(t('ai_explanation'))
            explain_button = st.button(t('generate_explanation'), key="explain")
            
            if explain_button:
                try:
                    if current_lang == 'pt':
                        prompt = f"""Classifique o seguinte texto de notícia e explique por que ele foi categorizado como "{result['categoria_predita']}".

Texto:
{text_input[:500]}

Categoria predita: {result['categoria_predita']}
Confiança: {result['score']:.2%}

Explique de forma clara e concisa por que este texto pertence a esta categoria."""
                    else:
                        prompt = f"""Classify the following news text and explain why it was categorized as "{result['categoria_predita']}".

Text:
{text_input[:500]}

Predicted category: {result['categoria_predita']}
Confidence: {result['score']:.2%}

Explain clearly and concisely why this text belongs to this category."""
                    
                    with st.spinner(t('generating')):
                        explanation = call_groq_llm(prompt, max_tokens=200)
                        st.info(explanation)
                except Exception as e:
                    st.warning(f"{t('explanation_error')} {e}")
                    st.info(t('explanation_info'))
            
            # Save to log
            if save_prediction:
                log_prediction(
                    texto=text_input,
                    classe_predita=result['classe_predita'],
                    score=result['score'],
                    embedding_usado=result['embedding_usado'],
                    modelo_usado=result['modelo_usado'],
                    fonte="streamlit",
                    categoria_predita=result['categoria_predita']
                )
                st.success(t('saved_log'))
    
    # Tab 2: Monitoring
    with tab2:
        st.header(t('monitoring_dashboard'))
        
        # Load logs
        logs_df = load_prediction_logs()
        
        if logs_df.empty:
            st.info(t('no_predictions'))
        else:
            stats = get_log_statistics()
            
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
                    st.plotly_chart(fig, use_container_width=True)
            
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
                    st.plotly_chart(fig, use_container_width=True)
            
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
                st.plotly_chart(fig, use_container_width=True)
            
            # Recent predictions table
            st.subheader(t('recent_predictions'))
            display_df = logs_df[['timestamp', 'categoria_predita', 'score', 'embedding_usado', 'modelo_usado']].tail(20)
            display_df = display_df.sort_values('timestamp', ascending=False)
            st.dataframe(display_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()

