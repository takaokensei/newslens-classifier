# NewsLens AI Classifier

Production-grade text classifier benchmarking **Sparse (TF-IDF)** vs. **Dense (BERT)** embeddings. Focus on **Cold Start**, **Latency** & **Semantic Performance** trade-offs using **SVM/XGBoost**.

## 🎯 Project Overview

**NewsLens AI** is a comparative analysis system for text classification that evaluates the trade-off between semantic performance (BERT) and computational efficiency (TF-IDF) for news classification tasks.

### Key Features

- **Dual Embedding Strategy**: TF-IDF (sparse) + BERT (dense) via sentence-transformers
- **Multiple Classifiers**: SVM (linear) and XGBoost
- **Production-Ready**: Logging, monitoring, and Streamlit interface
- **LLM Integration**: Groq API for class profiling and error analysis
- **Comprehensive Evaluation**: Accuracy, F1-macro, F1 per class, confusion matrices

## 📋 Requirements

- Python 3.8+
- See `requirements.txt` for full dependency list

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/takaokensei/newslens-classifier.git
cd newslens-classifier

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Set up environment variables:

```bash
export GROQ_API_KEY=your_api_key_here
```

### Running the Streamlit App

```bash
streamlit run apps/app_streamlit.py
```

## 📁 Project Structure

```
newslens-classifier/
├── data/
│   ├── raw/              # Original news dataset (6 classes)
│   ├── processed/        # Preprocessed data
│   ├── embeddings/       # Saved embeddings (.npz for TF-IDF, .npy for BERT)
│   └── novos/            # New texts for production simulation
├── logs/
│   └── predicoes.csv     # Prediction logs
├── models/               # Trained models (.pkl or .joblib)
├── src/
│   ├── config.py         # Centralized configurations
│   ├── preprocessing.py  # Unified preprocessing function
│   ├── data_loader.py    # Polymorphic data loading
│   ├── embeddings.py    # Embedding generation (TF-IDF and BERT)
│   ├── train.py          # Training scripts
│   ├── evaluate.py       # Evaluation and metrics
│   └── llm_analysis.py   # Groq API integration
├── scripts/
│   └── processar_novos.py # Script to classify texts in data/novos/
├── apps/
│   └── app_streamlit.py  # Main Streamlit application
└── requirements.txt      # Project dependencies
```

## 🔬 Technical Details

### Embeddings

- **E1 - TF-IDF**: 20k features, unigrams + bigrams, sparse matrix (.npz)
- **E2 - BERT**: `neuralmind/bert-base-portuguese-cased`, mean pooling, dense array (.npy)

### Models

- **M1 - SVM**: Linear kernel, balanced class weights, probability=True
- **M2 - XGBoost**: 100 estimators, max_depth=6, parallel processing

### Data Split

- **Stratified split**: Train (60%) / Validation (20%) / Test (20%)
- Random state: 42 for reproducibility

## 📊 Evaluation Metrics

- Accuracy
- F1-Macro
- F1 per class
- Confusion matrices (4 combinations)
- Latency (ms/document)
- Cold start time
- Model size (MB)

## 📝 License

MIT License

## 👤 Author

**Cauã Vitor Figueredo Silva** - ELE 606 (UFRN) - Final Project

## 🙏 Acknowledgments

- Prof. José Alfredo F. Costa (UFRN)
- NeuralMind for Portuguese BERT model
- Groq for LLM API access
