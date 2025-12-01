"""
Streamlit application for NewsLens AI Classifier.
Main interface for classification and monitoring.
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


@st.cache_resource
def load_all_models():
    """Load all models and embeddings (cached)."""
    try:
        models = load_trained_models()
        vectorizer = load_tfidf_vectorizer(PATHS['data_embeddings'] / 'tfidf_vectorizer.pkl')
        bert_model = load_bert_model()
        return models, vectorizer, bert_model
    except Exception as e:
        st.error(f"Error loading models: {e}")
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
    """Main Streamlit application."""
    st.title("📰 NewsLens AI Classifier")
    st.markdown("**Comparative Analysis of Sparse vs. Dense Representations**")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        embedding_type = st.selectbox(
            "Embedding Type",
            ["BERT", "TF-IDF"],
            index=0,
            help="Choose between BERT (dense) or TF-IDF (sparse) embeddings"
        )
        
        model_type = st.selectbox(
            "Model Type",
            ["SVM", "XGBoost"],
            index=0,
            help="Choose between SVM or XGBoost classifier"
        )
        
        st.divider()
        
        st.markdown("### 📊 Model Performance")
        st.info("""
        **Best Performance:** BERT + SVM (F1=1.0)
        
        **Best Efficiency:** TF-IDF + SVM (F1=0.97, 0.14ms/doc)
        """)
        
        st.divider()
        
        st.markdown("### ℹ️ About")
        st.caption("""
        NewsLens AI - ELE 606 Final Project
        
        UFRN - Prof. José Alfredo F. Costa
        """)
    
    # Main tabs
    tab1, tab2 = st.tabs(["🔍 Classification", "📊 Monitoring"])
    
    # Tab 1: Classification
    with tab1:
        st.header("Text Classification")
        
        # Load models
        if not st.session_state.models_loaded:
            with st.spinner("Loading models..."):
                models, vectorizer, bert_model = load_all_models()
                if models is not None:
                    st.session_state.models = models
                    st.session_state.vectorizer = vectorizer
                    st.session_state.bert_model = bert_model
                    st.session_state.models_loaded = True
                    st.success("Models loaded successfully!")
                else:
                    st.error("Failed to load models. Please check if models are trained.")
                    st.stop()
        else:
            models = st.session_state.models
            vectorizer = st.session_state.vectorizer
            bert_model = st.session_state.bert_model
        
        # Text input
        text_input = st.text_area(
            "Enter text to classify:",
            height=200,
            placeholder="Paste or type a news article here..."
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            classify_button = st.button("🔍 Classify", type="primary", use_container_width=True)
        with col2:
            save_prediction = st.checkbox("Save prediction to log", value=True)
        
        if classify_button and text_input:
            with st.spinner("Classifying..."):
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
                st.metric("Predicted Class", result['categoria_predita'])
            with col2:
                st.metric("Confidence", f"{result['score']:.2%}")
            with col3:
                st.metric("Model", f"{result['embedding_usado']} + {result['modelo_usado']}")
            
            # Probability distribution
            if result['all_probas']:
                st.subheader("Probability Distribution")
                prob_df = pd.DataFrame({
                    'Category': list(result['all_probas'].keys()),
                    'Probability': list(result['all_probas'].values())
                }).sort_values('Probability', ascending=False)
                
                fig = px.bar(
                    prob_df,
                    x='Category',
                    y='Probability',
                    color='Probability',
                    color_continuous_scale='Blues',
                    title="Prediction Probabilities"
                )
                fig.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # LLM Explanation
            st.subheader("🤖 AI Explanation")
            explain_button = st.button("Generate Explanation", key="explain")
            
            if explain_button:
                try:
                    prompt = f"""Classifique o seguinte texto de notícia e explique por que ele foi categorizado como "{result['categoria_predita']}".

Texto:
{text_input[:500]}

Categoria predita: {result['categoria_predita']}
Confiança: {result['score']:.2%}

Explique de forma clara e concisa por que este texto pertence a esta categoria."""
                    
                    with st.spinner("Generating explanation..."):
                        explanation = call_groq_llm(prompt, max_tokens=200)
                        st.info(explanation)
                except Exception as e:
                    st.warning(f"Could not generate explanation: {e}")
                    st.info("LLM explanation requires GROQ_API_KEY environment variable.")
            
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
                st.success("✅ Prediction saved to log!")
    
    # Tab 2: Monitoring
    with tab2:
        st.header("📊 Monitoring Dashboard")
        
        # Load logs
        logs_df = load_prediction_logs()
        
        if logs_df.empty:
            st.info("No predictions logged yet. Start classifying texts to see statistics here.")
        else:
            stats = get_log_statistics()
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Predictions", stats['total_predictions'])
            with col2:
                st.metric("Average Score", f"{stats['avg_score']:.2%}")
            with col3:
                most_common = max(stats['by_class'].items(), key=lambda x: x[1]) if stats['by_class'] else ("N/A", 0)
                st.metric("Most Common Class", f"{most_common[0]} ({most_common[1]})")
            with col4:
                if stats['date_range']['start']:
                    st.metric("Date Range", f"{stats['date_range']['start'][:10]} to {stats['date_range']['end'][:10]}")
            
            st.divider()
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Predictions by Class")
                if stats['by_class']:
                    class_df = pd.DataFrame({
                        'Category': list(stats['by_class'].keys()),
                        'Count': list(stats['by_class'].values())
                    })
                    fig = px.pie(
                        class_df,
                        values='Count',
                        names='Category',
                        title="Distribution by Category"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Predictions by Model")
                if stats['by_model']:
                    model_df = pd.DataFrame({
                        'Model': list(stats['by_model'].keys()),
                        'Count': list(stats['by_model'].values())
                    })
                    fig = px.bar(
                        model_df,
                        x='Model',
                        y='Count',
                        title="Usage by Model",
                        color='Count',
                        color_continuous_scale='Blues'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Temporal evolution
            if 'timestamp' in logs_df.columns:
                st.subheader("Temporal Evolution")
                logs_df['date'] = pd.to_datetime(logs_df['timestamp']).dt.date
                daily_counts = logs_df.groupby('date').size().reset_index(name='count')
                daily_counts = daily_counts.sort_values('date')
                
                fig = px.line(
                    daily_counts,
                    x='date',
                    y='count',
                    title="Predictions Over Time",
                    markers=True
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Recent predictions table
            st.subheader("Recent Predictions")
            display_df = logs_df[['timestamp', 'categoria_predita', 'score', 'embedding_usado', 'modelo_usado']].tail(20)
            display_df = display_df.sort_values('timestamp', ascending=False)
            st.dataframe(display_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()

