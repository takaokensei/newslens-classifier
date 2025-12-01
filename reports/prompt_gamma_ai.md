# NewsLens AI: Análise Comparativa de Representações Esparsas vs. Densas

## Classificação de Notícias em Português com TF-IDF e BERT

---

**Autor:** Cauã Vitor Figueredo Silva | **Orientador:** Prof. Dr. José Alfredo F. Costa

**UFRN - Engenharia Elétrica - ELE 606 | Dezembro 2024**

---

# A Hipótese Central

**TF-IDF vs BERT: O ganho semântico justifica o custo?**

* **Contexto:** Classificação de notícias em português (6 categorias)

* **Desafio:** Balancear performance e eficiência computacional

* **A Pergunta:** O ganho semântico do BERT compensa o aumento de latência e custo em comparação a um TF-IDF bem ajustado?

> **Métricas de Decisão:** Performance (F1/Accuracy) vs Eficiência (Latência/Cold Start/Memória)

---

# Base de Dados e Distribuição

**Dataset: 315 Amostras Válidas**

Classes (distribuição estratificada):

* **Economia** | **Esportes** | **Polícia e Direitos**

* **Política** | **Turismo** | **Variedades e Sociedade**

**Split Estratificado:**
* 60% Treino | 20% Validação | 20% Teste

> **Característica:** Distribuição relativamente balanceada entre classes, reduzindo viés de amostragem

![Distribuição de F1-Score por Classe](models/f1_by_class_comparison.png)

---

# Arquitetura: Representações Esparsas (TF-IDF)

**Vetorização Clássica com Otimização**

* **Features:** 20.000 termos (unigramas + bigramas)

* **Matriz Esparsa:** Densidade ~1% (formato `.npz`)

* **Eficiência:**
    * Latência: **0.14ms/documento**
    * Cold Start: **0.08s**
    * Tamanho: **0.182MB** (SVM)

**Vantagens:** Alta escalabilidade, interpretabilidade, baixo custo computacional

---

# Arquitetura: Representações Densas (BERT)

**Embeddings Contextuais Multilíngues**

* **Modelo:** `neuralmind/bert-base-portuguese-cased`

* **Dimensões:** 768 (mean pooling)

* **Matriz Densa:** Formato `.npy`

* **Eficiência:**
    * Latência: **0.12ms/documento** (pós-embedding)
    * Cold Start: **2.23s** → **0.62s** (otimizado)
    * Tamanho: **0.875MB** (SVM)

**Vantagens:** Captura contexto semântico, relações sintáticas, ambiguidade lexical

---

# Pipeline de Modelos

**Classificadores Testados**

| Modelo | Configuração | Adequação |
| :--- | :--- | :--- |
| **SVM** | Linear/RBF Kernel | Alta dimensionalidade |
| | `class_weight='balanced'` | Classes desbalanceadas |
| | `probability=True` | Scores calibrados |
| **XGBoost** | Ensemble de árvores | Features não-lineares |
| | `n_estimators`, `max_depth` | Otimizados via Optuna |
| | Paralelismo total | Treinamento eficiente |

---

# Validação Robusta: K-Fold Cross-Validation

**Garantindo Confiabilidade Estatística**

* **Estratégia:** 5-Fold Estratificado

* **Objetivo:** Reduzir variância e evitar overfitting

* **Resultados:** Desvio padrão < 0.06 para todos os modelos

**Prevenção de Leakage:**
1. Fit de vetorizadores/embeddings apenas no fold de treino
2. Transform aplicado em validação/teste

---

# Otimização Bayesiana (Optuna)

**Buscando Hiperparâmetros Ótimos**

* **Algoritmo:** TPE Sampler (Tree-structured Parzen Estimator)

* **Trials:** 50 por modelo

* **Espaço de Busca:**
    * SVM: `C`, `kernel`, `gamma`
    * XGBoost: `n_estimators`, `max_depth`, `learning_rate`, `subsample`

* **Ganhos Observados:**
    * TF-IDF + XGBoost: **+2.32%** F1-Macro
    * BERT + XGBoost: **+3.96%** F1-Macro ⭐⭐

![Trade-off Performance vs Eficiência](models/performance_efficiency_tradeoff.png)

*Nota: Gráfico de trade-off mostra a relação entre latência e F1-Macro para todos os modelos otimizados.*

---

# Impacto da Otimização

**Antes vs Depois (F1-Macro)**

| Modelo | F1-Padrão | F1-Otimizado | Melhoria |
| :--- | :--- | :--- | :--- |
| TF-IDF + SVM | 0.9680 | 0.9682 | +0.02% |
| TF-IDF + XGBoost | 0.8478 | 0.8675 | **+2.32%** ⭐ |
| BERT + SVM | 0.9881 | 0.9918 | +0.37% |
| BERT + XGBoost | 0.9277 | 0.9645 | **+3.96%** ⭐⭐ |

**Descoberta-chave:** XGBoost se beneficia massivamente da otimização, enquanto SVM já estava próximo do ótimo

---

# Performance Global: Modelos Otimizados

**Trade-off Performance vs Eficiência**

| Setup | F1-Macro | Accuracy | Latência | Cold Start | Tamanho |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **TF-IDF + SVM** | 0.968 | 0.968 | 0.14ms | 0.04s | 0.182MB |
| TF-IDF + XGBoost | 0.697 | 0.714 | 0.37ms | 0.06s | 0.489MB |
| **BERT + SVM** | **1.000** | **1.000** | 0.16ms | 0.62s | 0.875MB |
| BERT + XGBoost | 0.967 | 0.968 | 0.39ms | 0.55s | 0.428MB |

> **Insight:** BERT+SVM atinge performance perfeita (F1=1.0), mas com cold start **28x maior** que TF-IDF+SVM

---

# Granularidade por Classe (F1-Score)

**Performance Detalhada por Categoria**

| Categoria | TF-IDF+SVM | TF-IDF+XGB | BERT+SVM | BERT+XGB |
| :--- | :--- | :--- | :--- | :--- |
| **Economia** | 0.952 | 0.571 | **1.000** | **1.000** |
| **Esportes** | 0.952 | 0.783 | **1.000** | 0.900 |
| **Polícia e Direitos** | **1.000** | 0.870 | **1.000** | 0.957 |
| **Política** | **1.000** | 0.870 | **1.000** | **1.000** |
| **Turismo** | 0.960 | 0.421 | **1.000** | **1.000** |
| **Variedades** | 0.941 | 0.667 | **1.000** | 0.947 |

**Observação:** BERT+SVM alcança 100% em todas as classes; TF-IDF+SVM mantém performance competitiva

---

# Matrizes de Confusão

**Análise de Erros no Conjunto de Teste**

* **BERT + SVM:** Zero erros (matriz diagonal perfeita)

* **TF-IDF + SVM:** 2 classificações incorretas (Accuracy 96.8%)

* **Padrão de Erros (TF-IDF):**
    * Confusão entre classes semanticamente próximas
    * Ambiguidade lexical não capturada por tokens

![Matriz de Confusão - TF-IDF+SVM](models/cm_tfidf_svm_optimized_test.png)

*TF-IDF+SVM: 2 erros (Accuracy 96.8%)*

![Matriz de Confusão - BERT+SVM](models/cm_bert_svm_optimized_test.png)

*BERT+SVM: Perfeito - 0 erros (Accuracy 100%)*

---

# Análise de Trade-offs

**Quando Usar Cada Abordagem?**

✅ **Use BERT quando:**
* Performance crítica (F1=1.0 necessário)
* Classes com alta ambiguidade semântica
* Ganho de 3.2% é crucial para o negócio

✅ **Use TF-IDF quando:**
* Alta escala / baixa latência
* Recursos computacionais limitados
* 96.8% de performance é suficiente
* Interpretabilidade é prioridade

> **Conclusão Prática:** TF-IDF+SVM oferece excelente equilíbrio custo-benefício para a maioria dos casos

---

# LLMs para Perfilamento de Classes

**Metodologia Híbrida de Caracterização**

**Entrada (por classe):**
1. **Chi-Squared (TF-IDF):** Top 20 tokens discriminativos
2. **Centroides BERT:** 5 exemplos representativos

**Processamento:**
* LLM analisa padrões e gera arquétipos JSON

**Output:** Perfis descritivos de cada categoria

> **Valor:** Entender *o que* distingue cada classe além das métricas

*Exemplo de perfil JSON disponível em: `models/class_profiles.json`*

---

# LLMs para Análise Diferencial de Erros

**Explicando a Vantagem Semântica do BERT**

**Método:**
1. Identificar casos: BERT ✓ correto, TF-IDF ✗ incorreto
2. Top-10 casos enviados para Groq API (`llama-3.3-70b-versatile`)
3. LLM explica *por que* BERT acertou

**Insights Obtidos:**
* Contexto sintático capturado pelo BERT
* Ambiguidade lexical (mesma palavra, sentidos diferentes)
* Relações semânticas implícitas

*Análise diferencial de erros disponível em: `models/differential_errors.json`*

![Comparação de Cold Start](models/cold_start_comparison.png)

---

# Sistema de Produção: Streamlit

**Interface Web Completa**

**Tab 1: Classificação em Tempo Real**
* Seleção de embedding (TF-IDF/BERT) e modelo (SVM/XGBoost)
* Input de texto livre
* Output: Classe predita + probabilidades
* Explicação via LLM (opcional)

**Tab 2: Dashboard de Monitoramento**
* Distribuição de classes (pie chart)
* Uso por modelo (bar chart)
* Evolução temporal (line chart)

**Features:** Multilíngue (PT/EN), logging automático, suporte a batch processing

---

# Logging e Monitoramento

**Pipeline de Observabilidade**

**Logs (CSV):**
* Timestamp | Texto | Classe | Score | Modelo | Embedding

**Dashboard Analítico:**
* Métricas agregadas (total de predições, confiança média)
* Visualizações interativas (Plotly)
* Filtragem temporal

**Script de Produção:**
* Processar textos em lote
* Export de resultados
* Integração com pipelines MLOps

*Screenshots do Streamlit disponíveis no repositório ou podem ser gerados a partir da aplicação em: `apps/app_streamlit.py`*

**Interface disponível em:** [Streamlit Cloud](https://newslens-classifier.streamlit.app) ou localmente via `streamlit run apps/app_streamlit.py`

---

# Respondendo à Hipótese Central

**"O ganho semântico do BERT justifica o custo?"**

**Resposta: Depende do Contexto de Aplicação**

| Critério | BERT+SVM | TF-IDF+SVM |
| :--- | :--- | :--- |
| **Performance** | F1=1.000 (100%) | F1=0.968 (96.8%) |
| **Cold Start** | 0.62s (28x maior) | 0.04s |
| **Latência** | 0.16ms | 0.14ms |
| **Tamanho** | 0.875MB (4.8x maior) | 0.182MB |
| **Interpretabilidade** | Baixa (caixa-preta) | Alta (pesos TF-IDF) |

**Decisão:** Para a maioria dos casos, **TF-IDF+SVM** oferece o melhor ROI

---

# Principais Achados

**Contribuições Científicas e Práticas**

1. **BERT+SVM:** Performance perfeita (F1=1.0) no conjunto de teste

2. **TF-IDF+SVM:** 96.8% da performance com eficiência superior

3. **SVM > XGBoost:** Supera em ambos os embeddings após otimização

4. **BERT é indispensável** para ambiguidade semântica complexa

5. **LLMs agregam valor** na explicação e análise de erros

---

# Contribuições do Projeto

**Entregáveis e Inovações**

* Sistema completo de produção (Streamlit, logs, monitoramento)

* Análise quantitativa rigorosa do trade-off performance-eficiência

* Metodologia híbrida de perfilamento (Chi-Squared + Centroides)

* Framework para análise diferencial usando LLMs

* Base de código reutilizável e documentada (`github.com/takaokensei/newslens-classifier`)

---

# Limitações e Trabalhos Futuros

**Restrições do Estudo**

* Base pequena (315 amostras) - risco de overfitting
* F1=1.0 pode indicar vazamento de dados (requer auditoria)
* Apenas um modelo BERT testado

**Próximos Passos**

* Expansão da base de dados (10x+ amostras)
* Ensemble methods (TF-IDF + BERT híbrido)
* Fine-tuning do BERT em domínio específico
* Deploy em produção real com A/B testing
* Benchmark com modelos multilíngues (mBERT, XLM-R)

---

# Conclusões

**Síntese dos Resultados**

1. BERT oferece ganho semântico significativo (**F1=1.0 vs 0.968**)

2. TF-IDF mantém performance competitiva com **eficiência 28x superior**

3. A escolha depende do **contexto de aplicação** (performance crítica vs escala)

4. Sistema de produção completo desenvolvido e **funcional**

5. LLMs agregam valor na **explicação** e **análise de erros**

**Mensagem Final:** Nem sempre o modelo mais complexo é o mais adequado - o contexto é rei.

---

# Demonstração: Streamlit em Ação

**Fluxo de Uso**

1. **Input:** "Bolsa de Valores cai 3% após anúncio do Fed"

2. **Processamento:** Seleção TF-IDF+SVM

3. **Output:**
    * Classe: Economia (Score: 0.94)
    * Probabilidades: Economia (94%), Política (4%), Outros (2%)

4. **Explicação LLM:** "A referência à Bolsa de Valores e Fed indica contexto econômico..."

*Screencast pode ser gerado a partir da aplicação Streamlit em produção*

---

# Obrigado!

**Perguntas?**

**Cauã Vitor Figueredo Silva**

`cauavitorfigueredo@gmail.com`

**Repositório:** `github.com/takaokensei/newslens-classifier`

**Orientador:** Prof. Dr. José Alfredo F. Costa (UFRN - ELE 606)