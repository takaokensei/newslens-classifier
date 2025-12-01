# NewsLens AI Classifier

Production-grade text classifier benchmarking **Sparse (TF-IDF)** vs. **Dense (BERT)** embeddings. Focus on **Cold Start**, **Latency** & **Semantic Performance** trade-offs using **SVM/XGBoost**.

## 🎯 Project Overview

**NewsLens AI** is a comparative analysis system for text classification that evaluates the trade-off between semantic performance (BERT) and computational efficiency (TF-IDF) for news classification tasks in Portuguese.

### Key Features

- **Dual Embedding Strategy**: TF-IDF (sparse, 20k features) + BERT (dense, 768 dims) via sentence-transformers
- **Multiple Classifiers**: SVM (linear) and XGBoost with comprehensive evaluation
- **Production-Ready**: Complete logging system, monitoring dashboard, and Streamlit interface
- **LLM Integration**: Groq API (llama-3.3-70b-versatile) for class profiling and differential error analysis
- **Comprehensive Evaluation**: Accuracy, F1-macro, F1 per class, confusion matrices, latency, cold start
- **6 News Categories**: Economia, Esportes, Polícia e Direitos, Política, Turismo, Variedades e Sociedade

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

Create a `.env` file in the project root with your Groq API key:

```bash
# .env
GROQ_API_KEY=your_api_key_here
```

Or set it as an environment variable:

```bash
# Linux/Mac
export GROQ_API_KEY=your_api_key_here

# Windows PowerShell
$env:GROQ_API_KEY="your_api_key_here"
```

**Note:** A `.env.example` file is provided as a template. Copy it to `.env` and add your API key.

### Running the Streamlit App

```bash
streamlit run apps/app_streamlit.py
```

**⚠️ Important Tip for Windows Users:**

Always use the virtual environment activated before running Streamlit:

```powershell
# Activate virtual environment first
.venv\Scripts\Activate.ps1
streamlit run apps/app_streamlit.py
```

Or use the full path to ensure the correct Python environment:

```powershell
.venv\Scripts\streamlit.exe run apps/app_streamlit.py
```

This ensures that Streamlit uses the correct Python environment with all dependencies installed.

## 📁 Project Structure

```
newslens-classifier/
├── data/
│   ├── raw/              # Original news dataset (6 classes, 315 samples)
│   ├── processed/        # Preprocessed data and labels
│   ├── embeddings/       # Saved embeddings (.npz for TF-IDF, .npy for BERT)
│   └── novos/            # New texts for production simulation
├── logs/
│   └── predicoes.csv     # Prediction logs (timestamp, text, class, score, model)
├── models/               # Trained models (.pkl), metrics, confusion matrices
│   ├── *.pkl             # Trained models (tfidf_svm, tfidf_xgb, bert_svm, bert_xgb)
│   ├── table_*.csv        # Performance tables
│   ├── cm_*.png          # Confusion matrices
│   ├── class_profiles.json    # LLM-generated class archetypes
│   └── differential_errors.json  # LLM analysis of model differences
├── reports/
│   ├── relatorio.tex     # LaTeX report (10-20 pages)
│   └── prompt_gamma_ai.md # Presentation prompt for Gamma AI
├── src/
│   ├── config.py         # Centralized configurations
│   ├── preprocessing.py  # Unified preprocessing function
│   ├── data_loader.py    # Polymorphic data loading
│   ├── embeddings.py     # Embedding generation (TF-IDF and BERT)
│   ├── train.py          # Training scripts
│   ├── evaluate.py       # Evaluation and metrics
│   ├── llm_analysis.py   # Groq API integration
│   ├── logging_system.py # Prediction logging system
│   └── class_mapping.py  # Class to category mapping
├── scripts/
│   ├── run_phase2.py     # Phase 2: Training and evaluation
│   ├── run_phase3.py     # Phase 3: LLM analysis and profiling
│   ├── processar_novos.py # Production script for batch classification
│   └── test_production.py # Production environment validation
├── apps/
│   └── app_streamlit.py  # Main Streamlit application (classification + monitoring)
├── .env                  # Environment variables (not in git)
├── .env.example          # Environment variables template
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

## 📊 Results Summary

### Best Performance
- **BERT + SVM**: F1=1.0, Accuracy=1.0 (Perfect classification on test set)
- **TF-IDF + SVM**: F1=0.968, Accuracy=0.968 (96.8% of BERT performance)

### Efficiency Comparison
- **TF-IDF + SVM**: 0.14ms/doc, Cold Start: 0.08s, Size: 0.18 MB
- **BERT + SVM**: 0.12ms/doc, Cold Start: 2.23s, Size: 0.88 MB

### Key Findings
- BERT achieves perfect performance but has 28x longer cold start
- TF-IDF offers excellent efficiency with competitive performance (96.8%)
- SVM outperforms XGBoost in both embedding types
- BERT is essential for semantically ambiguous cases

## 🚀 Usage

### Training Models

```bash
# Phase 2: Train and evaluate models
python scripts/run_phase2.py

# Phase 3: Generate class profiles and error analysis
python scripts/run_phase3.py
```

### Production Classification

```bash
# Classify texts in data/novos/ directory
python scripts/processar_novos.py --model best

# Available models: best, tfidf_svm, tfidf_xgb, bert_svm, bert_xgb
```

### Streamlit Interface

The Streamlit app provides:
- **Classification Tab**: Real-time text classification with model selection
- **Monitoring Tab**: Dashboard with statistics, charts, and prediction logs
- **LLM Explanations**: AI-generated explanations for classifications (requires GROQ_API_KEY)

### Production Validation

```bash
# Run production environment tests
python scripts/test_production.py
```

## 📊 Evaluation Metrics

- **Accuracy**: Overall classification accuracy
- **F1-Macro**: Macro-averaged F1 score across all classes
- **F1 per class**: Individual F1 scores for each category
- **Confusion matrices**: Visual representation for all 4 model combinations
- **Latency**: Inference time per document (ms)
- **Cold start**: Model loading time (s)
- **Model size**: Disk space usage (MB)

## 📚 Documentation

- **Report**: LaTeX report available in `reports/relatorio.tex` (compile with pdflatex)
- **Presentation**: Prompt for Gamma AI in `reports/prompt_gamma_ai.md`
- **Project Plan**: See `prompt_master.md` for complete project roadmap

## 🔧 Development

### Project Phases

- ✅ **Phase 1**: Data preprocessing and embedding generation
- ✅ **Phase 2**: Model training and evaluation
- ✅ **Phase 3**: LLM analysis (class profiling, error analysis)
- ✅ **Phase 4**: Report writing, presentation, and final validation

### Testing

```bash
# Validate production environment
python scripts/test_production.py

# Quick model loading test
python -c "from src.train import load_trained_models; load_trained_models()"
```

## 📝 License

MIT License

## 👤 Author

**Cauã Vitor Figueredo Silva**  
ELE 606 - Final Project  
UFRN - Prof. José Alfredo F. Costa

## 🙏 Acknowledgments

- **Prof. José Alfredo F. Costa** (UFRN) - Project advisor
- **NeuralMind** - Portuguese BERT model (`neuralmind/bert-base-portuguese-cased`)
- **Groq** - LLM API access (llama-3.3-70b-versatile)
- **Streamlit** - Web application framework
