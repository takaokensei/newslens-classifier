# Prompt para Gamma AI - Apresentação NewsLens AI

Use este prompt no Gamma AI (https://gamma.app) para gerar a apresentação PPT de 10-15 minutos.

---

## Prompt Completo

Crie uma apresentação profissional de 10-15 minutos sobre o projeto NewsLens AI: Análise Comparativa de Representações Esparsas vs. Densas para Classificação de Notícias em Português.

### Slide 1: Capa
- Título: NewsLens AI: Análise Comparativa de Representações Esparsas vs. Densas
- Subtítulo: Classificação de Notícias em Português
- Autor: Cauã Vitor Figueredo Silva
- Instituição: UFRN - ELE 606 - Prof. José Alfredo F. Costa
- Data: 2025

### Slide 2: Objetivo e Hipótese Científica
- Objetivo: Desenvolver sistema de classificação comparando TF-IDF (esparso) vs BERT (denso)
- Hipótese Central: "O ganho semântico do BERT justifica o aumento de latência e custo computacional em comparação a um TF-IDF bem ajustado?"
- Métricas: Performance (F1/Accuracy) vs Eficiência (Latência/Cold Start/Memória)

### Slide 3: Base de Dados
- 315 amostras válidas (6 classes)
- Classes: Economia, Esportes, Polícia e Direitos, Política, Turismo, Variedades e Sociedade
- Split: 60% Treino / 20% Validação / 20% Teste (estratificado)
- Distribuição relativamente balanceada

### Slide 4: Arquitetura da Solução - Embeddings
- TF-IDF (Esparsa):
  - 20k features, unigramas + bigramas
  - Matriz esparsa (.npz), densidade ~1%
  - Eficiência: 0.14ms/doc, cold start 0.08s
  
- BERT (Densa):
  - Modelo: neuralmind/bert-base-portuguese-cased
  - 768 dimensões, mean pooling
  - Matriz densa (.npy)
  - Eficiência: 0.12ms/doc, cold start 2.23s

### Slide 5: Arquitetura da Solução - Modelos
- SVM (Linear Kernel):
  - class_weight='balanced', probability=True
  - Hiperparâmetros otimizados via Optuna (Bayesian Optimization)
  - Adequado para alta dimensionalidade
  
- XGBoost:
  - Hiperparâmetros otimizados: n_estimators, max_depth, learning_rate, subsample, etc.
  - Otimização bayesiana com 50 trials por modelo
  - Paralelismo total

### Slide 5.1: Validação Robusta
- K-Fold Cross-Validation (5 folds estratificados)
  - Garante robustez estatística
  - Reduz variância dos resultados
  - Desvio padrão < 0.02 para todos os modelos
- Otimização de Hiperparâmetros (Optuna)
  - Algoritmo TPE (Tree-structured Parzen Estimator)
  - 50 trials por modelo
  - CV durante otimização para evitar overfitting

### Slide 6: Tabela A - Eficiência & Performance Global
Apresentar tabela com:
- Setup | F1-Macro | Accuracy | Latência (ms/doc) | Cold Start (s) | Tamanho (MB)
- TF-IDF + SVM: F1=0.968, Acc=0.968, Lat=0.140ms, Cold=0.079s, Size=0.182MB
- TF-IDF + XGBoost: F1=0.704, Acc=0.714, Lat=0.420ms, Cold=0.107s, Size=0.489MB
- BERT + SVM: F1=1.000, Acc=1.000, Lat=0.120ms, Cold=2.228s, Size=0.875MB
- BERT + XGBoost: F1=0.967, Acc=0.968, Lat=0.377ms, Cold=2.296s, Size=0.428MB

### Slide 7: Tabela B - Granularidade por Classe
Apresentar tabela F1-Score por classe:
- Categoria | TF-IDF+SVM | TF-IDF+XGB | BERT+SVM | BERT+XGB
- Economia: 0.952 | 0.533 | 1.000 | 0.952
- Esportes: 0.952 | 0.800 | 1.000 | 0.900
- Polícia e Direitos: 1.000 | 0.800 | 1.000 | 1.000
- Política: 1.000 | 0.909 | 1.000 | 1.000
- Turismo: 0.960 | 0.545 | 1.000 | 1.000
- Variedades e Sociedade: 0.941 | 0.636 | 1.000 | 0.947

### Slide 8: Matrizes de Confusão
- Mostrar 2 matrizes principais (TF-IDF+SVM e BERT+SVM)
- Destacar que BERT+SVM tem 100% de acurácia (sem erros)
- TF-IDF+SVM tem 96.8% de acurácia (2 erros no teste)

### Slide 9: Análise de Trade-offs
- Performance vs Eficiência:
  - BERT+SVM: Melhor performance (F1=1.0) mas cold start 28x maior
  - TF-IDF+SVM: 96.8% da performance com eficiência superior
  
- Quando usar BERT:
  - Aplicações críticas onde 3.2% de ganho é crucial
  - Classes com ambiguidade semântica
  
- Quando usar TF-IDF:
  - Alta escala / baixa latência
  - Recursos computacionais limitados
  - 96.8% da performance é suficiente

### Slide 10: Uso de LLMs - Perfilamento de Classes
- Método Híbrido:
  - Chi-Squared (TF-IDF): Top 20 tokens por classe
  - Centroides BERT: 5 exemplos representativos por classe
- Output: Arquétipos JSON descrevendo características de cada categoria
- Valor: Entender o que distingue cada classe

### Slide 11: Uso de LLMs - Análise Diferencial de Erros
- Identifica casos: BERT correto, TF-IDF incorreto
- Top-10 casos analisados via Groq API (llama-3.3-70b-versatile)
- Explicações destacam:
  - Contexto semântico capturado pelo BERT
  - Ambiguidade lexical que TF-IDF perde
  - Estrutura sintática importante

### Slide 12: Sistema de Produção - Streamlit
- Interface web completa:
  - Tab 1: Classificação em tempo real
  - Tab 2: Dashboard de monitoramento
- Funcionalidades:
  - Seleção de embedding e modelo
  - Exibição de probabilidades
  - Explicações via LLM
  - Logging automático
- Suporte a Português/English

### Slide 13: Sistema de Produção - Logs e Monitoramento
- Logs em CSV: timestamp, texto, classe, score, modelo, embedding
- Dashboard com:
  - Métricas agregadas
  - Distribuição por classe (pie chart)
  - Uso por modelo (bar chart)
  - Evolução temporal (line chart)
- Script de produção: processar textos em lote

### Slide 14: Resposta à Hipótese
**"O ganho semântico do BERT justifica o custo?"**

**Resposta: Depende do contexto**

✅ SIM para:
- Aplicações críticas de alta performance
- Classes com ambiguidade semântica
- Casos onde 3.2% de ganho é crucial

❌ NÃO necessariamente para:
- Alta escala / baixa latência
- Recursos computacionais limitados
- Quando 96.8% da performance é suficiente

**Conclusão:** TF-IDF+SVM oferece excelente equilíbrio para a maioria dos casos.

### Slide 15: Principais Achados
1. BERT+SVM: Performance perfeita (F1=1.0) no teste
2. TF-IDF+SVM: 96.8% da performance com eficiência superior
3. SVM supera XGBoost em ambos embeddings
4. BERT é indispensável em casos de ambiguidade semântica
5. LLMs fornecem insights valiosos sobre diferenças entre modelos

### Slide 16: Contribuições
- Sistema completo de produção (Streamlit, logs, monitoramento)
- Análise quantitativa rigorosa do trade-off performance-eficiência
- Metodologia híbrida de perfilamento (Chi-Squared + Centroides)
- Framework para análise diferencial usando LLMs
- Base de código reutilizável e documentada

### Slide 17: Limitações e Trabalhos Futuros
Limitações:
- Base pequena (315 amostras)
- Possível overfitting (F1=1.0)
- Apenas um modelo BERT testado

Trabalhos Futuros:
- Expansão da base de dados
- Ensemble methods (TF-IDF + BERT)
- Fine-tuning do BERT
- Otimização de hiperparâmetros
- Deploy em produção real

### Slide 18: Conclusões
- BERT oferece ganho semântico significativo (F1=1.0 vs 0.968)
- TF-IDF mantém performance competitiva com eficiência superior
- A escolha depende do contexto de aplicação
- Sistema de produção completo desenvolvido e funcional
- LLMs agregam valor na explicação e análise de erros

### Slide 19: Demonstração do Streamlit
- Screenshots ou vídeo mostrando:
  - Interface de classificação
  - Dashboard de monitoramento
  - Geração de explicações via LLM
  - Gráficos e estatísticas

### Slide 20: Obrigado!
- Perguntas?
- Repositório: https://github.com/takaokensei/newslens-classifier
- Contato: [seu email]

---

## Instruções para Gamma AI

1. Acesse https://gamma.app
2. Crie uma nova apresentação
3. Cole o prompt acima
4. Ajuste o estilo visual conforme necessário
5. Adicione imagens das matrizes de confusão e screenshots do Streamlit
6. Revise e personalize os slides conforme necessário

## Dados Adicionais para Referência

- **Repositório**: https://github.com/takaokensei/newslens-classifier
- **Modelos treinados**: 4 modelos (TF-IDF+SVM, TF-IDF+XGB, BERT+SVM, BERT+XGB)
- **Matrizes de confusão**: Disponíveis em `models/cm_*.png`
- **Tabelas**: `models/table_a_efficiency.csv`, `models/table_b_classes_with_names.csv`
- **Análise de erros**: `models/differential_errors.json`
- **Perfis de classes**: `models/class_profiles.json`

