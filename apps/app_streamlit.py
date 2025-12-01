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
    # Load predictions from cookie on page load (survives F5)
    load_and_sync_cookie_predictions()
    
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
                                    st.error("❌ Erro ao carregar modelos após treinamento." if current_lang == 'pt' else "❌ Error loading models after training.")
                                    st.stop()
                            else:
                                progress_bar.empty()
                                status_text.empty()
                                st.warning(f"⚠️ {result.get('message', 'Training failed')}")
                                st.info("💡 Tente recarregar a página ou execute `python scripts/auto_train_models.py` manualmente." if current_lang == 'pt' else "💡 Try reloading the page or run `python scripts/auto_train_models.py` manually.")
                                st.stop()
                        except Exception as e:
                            st.error(f"❌ Erro durante treinamento: {str(e)}" if current_lang == 'pt' else f"❌ Error during training: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())
                            st.info("💡 Execute `python scripts/auto_train_models.py` manualmente para treinar os modelos." if current_lang == 'pt' else "💡 Run `python scripts/auto_train_models.py` manually to train models.")
                            st.stop()
                    else:
                        st.error(t('models_error'))
                        st.info("💡 **Dica**: Adicione um arquivo CSV em `data/raw/` com colunas: 'Texto', 'Classe', 'Categoria'. Os modelos serão treinados automaticamente." if current_lang == 'pt' else "💡 **Tip**: Add a CSV file in `data/raw/` with columns: 'Texto', 'Classe', 'Categoria'. Models will be trained automatically.")
                        st.stop()
                elif models and len(models) > 0:
                    st.session_state.models = models
                    st.session_state.vectorizer = vectorizer
                    st.session_state.bert_model = bert_model
                    st.session_state.models_loaded = True
                    st.success(t('models_loaded'))
                    if not vectorizer:
                        st.warning("⚠️ TF-IDF vectorizer não encontrado. Apenas modelos BERT estarão disponíveis." if current_lang == 'pt' else "⚠️ TF-IDF vectorizer not found. Only BERT models will be available.")
                    if not bert_model:
                        st.warning("⚠️ Modelo BERT não encontrado. Apenas modelos TF-IDF estarão disponíveis." if current_lang == 'pt' else "⚠️ BERT model not found. Only TF-IDF models will be available.")
                else:
                    st.error(t('models_error'))
                    st.info("💡 **Dica**: Os modelos precisam ser treinados primeiro. Execute `python scripts/auto_train_models.py` para treinar automaticamente." if current_lang == 'pt' else "💡 **Tip**: Models need to be trained first. Run `python scripts/auto_train_models.py` to train automatically.")
                    st.stop()
        else:
            models = st.session_state.models
            vectorizer = st.session_state.vectorizer
            bert_model = st.session_state.bert_model
        
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
                else:
                    # Old format: just text
                    st.session_state.text_input_area = st.session_state.sample_text
                    # Clear true_label if it exists
                    if 'true_label' in st.session_state:
                        del st.session_state.true_label
                # Clear sample_text after using it
                st.session_state.sample_text = ''
            
            # Use key to bind to session_state (don't use value parameter when using key)
            text_input = st.text_area(
                t('enter_text'),
                height=200,
                placeholder=t('text_placeholder'),
                key="text_input_area"
            )
            
            # Display true label if available (from validation sample)
            if 'true_label' in st.session_state:
                true_label = st.session_state.true_label
                true_category = CLASS_TO_CATEGORY.get(int(true_label), f"Classe {true_label}")
                st.info(
                    f"🏷️ **Classe Real (Ground Truth):** {true_category}" if current_lang == 'pt' 
                    else f"🏷️ **True Label (Ground Truth):** {true_category}",
                    icon="ℹ️"
                )
        
        with col_btn:
            st.write("")  # Spacing
            st.write("")  # Spacing
            
            # Check if data is available before showing button
            raw_dir = PATHS['data_raw']
            csv_files = list(raw_dir.glob('*.csv'))
            data_available = len(csv_files) > 0
            
            if data_available:
                if st.button(
                    "📄 Exemplo do Conjunto de Validação" if current_lang == 'pt' else "📄 Validation Set Sample",
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
                            st.error("❌ Não foi possível carregar exemplo. Verifique os logs do console para mais detalhes." if current_lang == 'pt' else "❌ Could not load sample. Check console logs for details.")
            else:
                # Disable button if data not available
                st.button(
                    "📄 Exemplo do Conjunto de Validação" if current_lang == 'pt' else "📄 Validation Set Sample",
                    width='stretch',
                    disabled=True,
                    help="Dados não disponíveis no deploy. Disponível apenas localmente." if current_lang == 'pt' else "Data not available in deployment. Only available locally."
                )
        
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
        
        if result is not None and text_input_for_display:
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
                st.plotly_chart(fig, width='stretch')
            
            # LLM Explanation section (always visible when result exists)
            st.subheader(t('ai_explanation'))
            
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
                        if current_lang == 'pt':
                            prompt = f"""Classifique o seguinte texto de notícia e explique por que ele foi categorizado como "{result['categoria_predita']}".

Texto:
{text_input_for_display[:500]}

Categoria predita: {result['categoria_predita']}
Confiança: {result['score']:.2%}

Explique de forma clara e concisa por que este texto pertence a esta categoria."""
                        else:
                            prompt = f"""Classify the following news text and explain why it was categorized as "{result['categoria_predita']}".

Text:
{text_input_for_display[:500]}

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
                    st.success(t('saved_log'))
                    st.session_state.log_saved_for_current = True
    
    # Tab 2: Monitoring
    with tab2:
        st.header(t('monitoring_dashboard'))
        
        # Advanced filters (bonus feature inspired by Módulo 16)
        with st.expander("🔍 Filtros Avançados" if current_lang == 'pt' else "🔍 Advanced Filters"):
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
            st.caption("💡 **Dica**: As predições são salvas em cookies e persistem mesmo após atualizar a página (F5). Cada navegador/computador tem seu próprio histórico." if current_lang == 'pt' else "💡 **Tip**: Predictions are saved in cookies and persist even after refreshing the page (F5). Each browser/computer has its own history.")
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

