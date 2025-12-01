# 🎓 PROJECT MASTER PLAN: NewsLens AI (Final Gold Version - C4)

**Role:** Senior ML Engineer & Data Scientist.
**Contexto:** Entrega Final ELE 606 (UFRN).
**Constraint:** Rigorosa aderência ao **Conjunto C4** + Padrões de Produção (Cold Start, Latência, Robustez).

-----

### 1\. Visão do Produto & Hipótese Científica

**Título:** NewsLens AI - Comparative Analysis of Sparse vs. Dense Representations.
**Hipótese Central:** "O ganho semântico do BERT (Dense) justifica o aumento de latência e custo computacional em comparação a um TF-IDF (Sparse) bem ajustado para classificação de notícias?"
**Objetivo:** Um sistema de produção que classifica notícias e quantifica o trade-off entre **Performance (F1/Recall)** e **Eficiência (Inferência/Memória/Cold Start)**.

-----

### ⚙️ 2. Arquitetura e Engenharia de Dados

#### **A. Pipeline de Embeddings (Definição Técnica)**

  * **E1 - Representação Esparsa (Baseline):**
      * **Método:** TF-IDF (`scipy.sparse`).
      * **Config:** Top 20k features, unigramas + bigramas.
      * **Armazenamento:** `.npz` (Matriz esparsa comprimida).
  * **E2 - Representação Densa (SOTA):**
      * **Modelo:** `neuralmind/bert-base-portuguese-cased`.
      * **Implementação:** Via biblioteca `sentence-transformers` (para garantir pooling otimizado e facilidade de uso).
      * **Estratégia:** **Mean Pooling** automático da biblioteca.
      * **Armazenamento:** `.npy` (Matriz densa `float32`).

#### **B. Pré-processamento de Textos (Função Única)**

  * **Função:** `preprocess_text()` - usada em todo o pipeline (treino, validação, teste, produção).
  * **Etapas:**
      * Lowercase.
      * Remoção de caracteres especiais (manter acentos para português).
      * Normalização de espaços em branco.
      * Remoção de URLs e emails (opcional, conforme necessidade).
  * **Implementação:** `src/preprocessing.py` com função reutilizável.

#### **C. Estratégia de Validação (Requisito Obrigatório)**

  * **Split Obrigatório:** **Divisão Estratificada em Treino / Validação / Teste** (conforme requisito do professor).
  * **Proporção Recomendada:** 60/20/20 (treino/validação/teste) ou 70/15/15, dependendo do tamanho da base.
  * **Implementação:** 
      * Primeiro split: treino+validação (80%) vs teste (20%) - estratificado
      * Segundo split: treino (75% do 80%) vs validação (25% do 80%) - estratificado
      * Resultado final: ~60% treino, ~20% validação, ~20% teste
  * **Uso dos Splits:**
      * **Treino:** Treinar os modelos
      * **Validação:** Ajuste fino de hiperparâmetros e seleção de modelo (opcional)
      * **Teste:** Avaliação final e relatório (não tocar após escolha do modelo)
  * **Safety:** `stratify=y` e `random_state=42` obrigatórios em ambos os splits.

#### **D. Os Modelos**

  * **M1: SVM (Support Vector Machine):** Kernel **Linear** (padrão para alta dimensão), `class_weight='balanced'`, `probability=True`.
  * **M2: XGBoost:** `n_estimators=100`, `max_depth=6`, `n_jobs=-1` (paralelismo total).

-----

### 🧠 3. Módulo de Inteligência & LLM (Groq API)

#### **Task 3.1: Perfilamento de Classes (Híbrido)**

  * **Via BERT:** Calcular Centroide dos embeddings -\> Buscar vizinhos mais próximos.
  * **Via TF-IDF:** **Chi-Squared Feature Selection**. Identificar os tokens mais correlacionados com a classe (superior à média simples).
  * **Output:** JSON "Arquétipos".

#### **Task 3.2: Análise Diferencial de Erros**

  * **Filtro:** `(Pred_BERT == Correto) AND (Pred_TFIDF == Incorreto)`.
  * **Priorização:** Selecionar os **Top-10 casos** com maior delta de confiança.
  * **Prompt:** "O modelo semântico (BERT) capturou o contexto, mas o léxico (TF-IDF) falhou. Explique qual nuance linguística causou isso."

-----

### 📊 4. O "Scoreboard" de Engenharia (Expandido)

O relatório deve conter **duas** tabelas principais:

**Tabela A: Eficiência & Performance Global**
| Setup | F1-Macro | Acurácia | Tempo Treino (s) | **Latência (ms/doc)** | **Cold Start (s)** | Tamanho (MB) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| TF-IDF + SVM | ... | ... | ... | ... | ... | ... |
| BERT + XGB | ... | ... | ... | ... | ... | ... |

**Tabela B: Granularidade por Classe (Requisito C4)**

  * Linhas: Classes (Esporte, Política, etc.)
  * Colunas: F1-Score (TF-IDF+SVM), F1-Score (TF-IDF+XGB), F1-Score (BERT+SVM), F1-Score (BERT+XGB).
  * *Meta:* Identificar em quais tópicos a semântica do BERT é indispensável.

**Visualizações Obrigatórias:**

  * **Matriz de Confusão:** Uma por combinação (4 matrizes no total) ou matriz comparativa.
  * **Gráficos de Comparação:** F1 por classe (barras agrupadas), Accuracy comparativa, Latência vs Performance.

-----

### 5\. Roadmap Tático (10 Dias)

**📍 FASE 1: Data Engineering (Dias 1-3)**

  * [x] **Task 1.1:** Setup do `config.py` e estrutura completa de pastas (conforme seção 7).
  * [x] **Task 1.2:** `src/preprocessing.py` com função única `preprocess_text()`.
  * [x] **Task 1.3:** `data_loader.py` polimórfico (`.npz`/`.npy`).
  * [x] **Task 1.4:** Gerar embeddings BERT via `sentence-transformers` e salvar.
  * [x] **Task 1.5:** **Sanity Check:** Verificar shapes, NaNs e contagem de classes pós-split.

**📍 FASE 2: Training & Benchmarking (Dias 4-5)**

  * [x] **Task 2.1:** Treinar os 4 pares de modelos (TF-IDF+SVM, TF-IDF+XGB, BERT+SVM, BERT+XGB) usando conjunto de TREINO.
  * [x] **Task 2.2:** Avaliação no conjunto de VALIDAÇÃO para ajuste fino (opcional) e comparação inicial.
  * [x] **Task 2.3:** Avaliação final no conjunto de TESTE: Accuracy, F1-Macro, F1 por classe, Matriz de Confusão (4 matrizes).
  * [x] **Task 2.4:** Script de benchmark: medir inferência com `batch_size=1` (simulação real).
  * [x] **Task 2.5:** Gerar Tabela A (Eficiência) e Tabela B (Classes) + visualizações.

**📍 FASE 3: AI Analysis & Dashboard (Dias 6-8)**

  * [x] **Task 3.1:** Pipeline de Protótipos (Chi-Squared + Centroides) para perfilamento de classes.
  * [x] **Task 3.2:** Pipeline LLM Diferencial (max 10 calls) para análise de erros.
  * [x] **Task 3.3:** Sistema de Logs: implementar `log_prediction()` e `logs/predicoes.csv`.
  * [x] **Task 3.4:** Script de Produção: `scripts/processar_novos.py` para classificar textos em `data/novos/`.
  * [x] **Task 3.5:** Streamlit com 2 páginas principais:
      * **Tab 1 - "Classificação":** Entrada de texto → Classe predita, Score, Explicação (via LLM).
      * **Tab 2 - "Monitoramento":** Dashboard com gráficos de logs (contagem por classe, evolução temporal, estatísticas).

**📍 FASE 4: Consolidação (Dias 9-10)**

  * [x] **Task 4.1:** Escrita do relatório (10-20 páginas) com estrutura completa (seção 8).
      * ✅ Template LaTeX criado (`reports/relatorio.tex`)
      * ✅ Todas as seções estruturadas conforme seção 8
      * ✅ Dados reais preenchidos nas tabelas
      * ✅ Análises detalhadas incluídas
      * ⚠️ **Pendente:** Compilação final do PDF (usuário fará via Streamlit ou Overleaf)
  * [x] **Task 4.2:** Preparação da apresentação PPT (10-15 minutos).
      * ✅ Prompt completo criado para Gamma AI (`reports/prompt_gamma_ai.md`)
      * ✅ 20 slides estruturados com todo o conteúdo
      * ✅ Dados reais incluídos
      * ⚠️ **Pendente:** Geração no Gamma AI (usuário fará)
  * [x] **Task 4.3:** Testes finais do Streamlit e validação do ambiente de produção.
      * ✅ Script de validação criado (`scripts/test_production.py`)
      * ✅ Todos os testes passando
      * ✅ Bug de truncamento de explicação LLM corrigido
      * ✅ Sistema funcional e validado
  * [x] **Task 4.4:** Documentação final: `README.md` com instruções de instalação e execução.
      * ✅ README completo e profissional
      * ✅ Instruções detalhadas de instalação
      * ✅ Documentação de uso e estrutura
      * ✅ Métricas e resultados atualizados

-----

### 6\. Estrutura de Pastas (Organização do Projeto)

```
newslens-classifier/
├── data/
│   ├── raw/              # Base original de notícias (6 classes)
│   ├── processed/        # Dados pré-processados
│   ├── embeddings/       # Embeddings salvos (.npz para TF-IDF, .npy para BERT)
│   └── novos/            # Novos textos para simulação de produção (OBRIGATÓRIO)
├── logs/
│   └── predicoes.csv     # Log de todas as predições (OBRIGATÓRIO)
├── models/               # Modelos treinados salvos (.pkl ou .joblib)
├── src/
│   ├── config.py         # Configurações centralizadas
│   ├── preprocessing.py  # Função única de pré-processamento
│   ├── data_loader.py    # Carregamento polimórfico de dados
│   ├── embeddings.py     # Geração de embeddings (TF-IDF e BERT)
│   ├── train.py          # Script de treinamento
│   ├── evaluate.py       # Script de avaliação e métricas
│   └── llm_analysis.py   # Integração com Groq API
├── scripts/
│   └── processar_novos.py # Script para classificar textos em data/novos/
├── apps/
│   └── app_streamlit.py  # Aplicação Streamlit principal
├── tools/                # Scripts auxiliares (se necessário)
├── notebooks/            # Jupyter notebooks para análise exploratória
├── requirements.txt      # Dependências do projeto
└── README.md             # Documentação e instruções
```

### 7\. Sistema de Logs e Monitoramento

#### **A. Formato do Log (`logs/predicoes.csv`)**

Colunas obrigatórias:
  * `timestamp`: Data e hora da predição
  * `texto`: Texto original (ou hash se muito longo)
  * `classe_predita`: Classe retornada pelo modelo
  * `score`: Score/confiança da predição
  * `embedding_usado`: "TF-IDF" ou "BERT"
  * `modelo_usado`: "SVM" ou "XGBoost"
  * `fonte`: "streamlit" ou "script_producao"

#### **B. Função de Log**

```python
def log_prediction(texto, classe_predita, score, embedding_usado, modelo_usado, fonte="streamlit"):
    """Registra predição no arquivo logs/predicoes.csv"""
    # Implementação com pandas.to_csv(mode='a') ou similar
```

#### **C. Script de Produção (`scripts/processar_novos.py`)**

  * **Objetivo:** Classificar todos os textos em `data/novos/` e registrar em logs.
  * **Funcionalidades:**
      * Ler todos os arquivos de texto em `data/novos/`.
      * Aplicar pré-processamento.
      * Classificar com os 4 modelos (ou modelo escolhido).
      * Registrar cada predição em `logs/predicoes.csv`.
      * Gerar relatório resumido.

### 8\. Interface Streamlit (`apps/app_streamlit.py`)

#### **Estrutura da Aplicação:**

  * **Sidebar:** Instruções de uso e seleção de modelo/embedding.
  * **Tab 1 - "Classificação":**
      * Caixa de texto para entrada.
      * Botão "Classificar".
      * Exibição de resultados:
          * Classe predita (destaque visual).
          * Score/confiança.
          * Explicação via LLM ("Por que este texto foi classificado como X?").
      * Opção de salvar predição (registra em log).
  * **Tab 2 - "Monitoramento":**
      * Leitura de `logs/predicoes.csv`.
      * Gráficos:
          * Contagem de predições por classe (bar chart).
          * Evolução temporal (line chart).
          * Distribuição de scores (histograma).
          * Estatísticas simples (total de predições, classe mais frequente, etc.).
      * Filtros por data, modelo, embedding.

### 9\. Configuração Técnica (`src/config.py`)

```python
import os

# 1. Configuração de Dados
DATA_CONFIG = {
    'test_size': 0.2,           # Primeiro split: 20% para teste
    'val_size': 0.25,           # Segundo split: 25% do restante para validação
    'stratify': True,           # CRÍTICO: Manter distribuição original
    'random_state': 42          # Reproduzibilidade
}

# 2. Features e Embeddings
FEATURE_CONFIG = {
    'tfidf': {
        'max_features': 20000,
        'ngram_range': (1, 2),
        'storage': 'sparse_npz'
    },
    'bert': {
        'model_name': 'neuralmind/bert-base-portuguese-cased',
        'implementation': 'sentence-transformers', # Biblioteca definida
        'pooling': 'mean',                         # Estratégia definida
        'batch_size': 32,
        'storage': 'dense_npy'
    }
}

# 3. Modelos
MODELS_CONFIG = {
    'svm': {
        'kernel': 'linear',
        'class_weight': 'balanced',
        'probability': True
    },
    'xgboost': {
        'n_estimators': 100,
        'max_depth': 6,
        'n_jobs': -1
    }
}

# 4. Limites de API (Controle de Custo)
LLM_CONFIG = {
    'provider': 'groq',
    'model': 'llama-3.1-70b-versatile',
    'max_examples_differential': 10,  # Hard limit
    'api_key': os.getenv('GROQ_API_KEY')  # Variável de ambiente
}

# 5. Caminhos de Pastas
PATHS = {
    'data_raw': 'data/raw',
    'data_processed': 'data/processed',
    'data_embeddings': 'data/embeddings',
    'data_novos': 'data/novos',
    'logs': 'logs',
    'models': 'models'
}
```

-----

### 10\. Estrutura do Relatório (PDF - 10 a 20 páginas)

#### **Seções Obrigatórias:**

1. **Introdução**
   * Objetivo do trabalho
   * Hipótese científica central
   * Contexto e motivação

2. **Descrição da Base de Dados**
   * Características da base (6 classes de notícias)
   * Estatísticas descritivas (distribuição de classes, tamanho médio de textos)
   * Pré-processamento aplicado

3. **Métodos e Pipeline**
   * Embeddings utilizados (TF-IDF e BERT) - detalhamento técnico
   * Modelos de classificação (SVM e XGBoost) - hiperparâmetros
   * Estratégia de validação (divisão treino/validação/teste estratificada)
   * Uso de LLMs (perfilamento de classes e análise de erros)

4. **Experimentos e Resultados**
   * Tabela A: Eficiência & Performance Global
   * Tabela B: Granularidade por Classe
   * Matrizes de Confusão (4 combinações)
   * Gráficos comparativos (F1 por classe, Accuracy, Latência vs Performance)
   * Análise de trade-offs (Performance vs Eficiência)

5. **Uso de LLMs**
   * Perfilamento de classes (exemplos de arquétipos gerados)
   * Análise diferencial de erros (casos analisados)
   * Discussão sobre o valor agregado das explicações

6. **Sistema de Produção e Monitoramento**
   * Arquitetura do sistema Streamlit
   * Sistema de logs implementado
   * Exemplos de uso em produção (textos de `data/novos/`)
   * Dashboard de monitoramento (screenshots e análise)

7. **Discussão**
   * Comparação entre embeddings (TF-IDF vs BERT)
   * Comparação entre modelos (SVM vs XGBoost)
   * Resposta à hipótese científica: quando o BERT justifica o custo?
   * Limitações e trabalhos futuros

8. **Conclusões**
   * Principais achados
   * Contribuições do trabalho
   * Recomendações práticas

9. **Referências**
   * Artigos, bibliotecas e recursos utilizados

### 11\. Entregáveis Finais

#### **Obrigatórios:**

1. **Repositório GitHub ou Google Drive:**
   * Código completo e organizado (src/, apps/, tools/, data/, logs/)
   * `requirements.txt` atualizado
   * `README.md` com instruções de instalação e execução
   * Estrutura de pastas conforme seção 6

2. **Relatório PDF (10-20 páginas):**
   * Estrutura conforme seção 10
   * Tabelas e gráficos de qualidade
   * Análise crítica dos resultados

3. **Apresentação PPT (10-15 minutos):**
   * Objetivo e hipótese científica
   * Arquitetura da solução
   * Principais resultados (tabelas e gráficos)
   * Demonstração do Streamlit (screenshots ou vídeo)
   * Comentários sobre comportamento em produção

4. **Sistema Funcional:**
   * Streamlit rodando localmente (`streamlit run apps/app_streamlit.py`)
   * Script de produção funcionando
   * Logs sendo gerados corretamente

-----

### Próximo Passo

Como Engenheiro Sênior, recomendo iniciarmos pela **Task 1.1, 1.2 e 1.3**:
1. Criar estrutura completa de pastas
2. Implementar `src/preprocessing.py` com função única
3. Implementar `src/config.py` com todas as configurações
4. Implementar `src/data_loader.py` com carregamento polimórfico