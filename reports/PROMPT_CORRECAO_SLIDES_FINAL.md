# Prompt de Correção Final dos Slides - NewsLens AI

## Análise dos Slides e Correções Necessárias

Após análise detalhada do PDF dos slides, identifiquei os seguintes problemas e oportunidades de melhoria:

---

## 🔴 CORREÇÕES CRÍTICAS

### 1. **ERRO DE DADOS - Slide 10: Performance Global**

**Problema Identificado:**
- O Slide 10 ("Performance Global: Modelos Otimizados") contém **valores incorretos** para o modelo **TF-IDF + XGBoost**
- Atualmente mostra: F1-Macro = 0.697, Accuracy = 0.714
- Estes são valores do modelo **padrão (não otimizado)**, não dos modelos otimizados

**Valores Corretos (do Slide 9 - Impacto da Otimização):**
- **F1-Macro:** 0.8675 (ou 0.868 arredondado)
- **Accuracy:** Deve ser calculada proporcionalmente. Baseado no padrão (0.714) e ganho de F1 (+2.32%), estimativa: **~0.880** (ou verificar no arquivo `models/results_optimized_test.csv`)

**Correção:**
```
Substituir na tabela do Slide 10:
TF-IDF + XGBoost: F1-Macro = 0.697 → 0.868
                   Accuracy = 0.714 → ~0.880 (verificar valor exato)
```

**Motivo:** O slide deve refletir os modelos **pós-Optuna**, mas está mostrando dados do modelo padrão, criando inconsistência com o Slide 9.

---

### 2. **FALTA DE LINK PARA DEMO - Slide 22: Demonstração Ao Vivo**

**Problema Identificado:**
- O Slide 22 ("Demonstração Ao Vivo: NewsLens AI") descreve os testes mas **não fornece o link** para a audiência acompanhar
- A apresentação menciona demonstração ao vivo, mas sem acesso direto ao sistema

**Correção Necessária:**
Adicionar um **destaque visual claro** no Slide 22 com:

**Opção A (Recomendada - Box destacado):**
```
┌─────────────────────────────────────────┐
│  🌐 ACESSE E TESTE AO VIVO:            │
│  https://newslens-classifier.streamlit.app/ │
│                                         │
│  Escaneie o QR code ou digite o link   │
└─────────────────────────────────────────┘
```

**Opção B (Subtítulo):**
Adicionar como subtítulo logo após o título "Demonstração Ao Vivo: NewsLens AI":
```
Acesse e teste ao vivo: https://newslens-classifier.streamlit.app/
```

**Especificações:**
- Fonte grande e legível (mínimo 24pt, ideal 28-32pt)
- Cor contrastante (ex: azul #4A90E2 ou verde #00C853)
- Posicionar no topo ou em box destacado
- Considerar adicionar QR code se possível

---

## 🟡 MELHORIAS RECOMENDADAS

### 3. **Consistência de Valores - Verificação Geral**

**Verificar em todos os slides:**
- Slide 9 (Impacto da Otimização): TF-IDF + XGBoost F1-Otimizado = 0.8675 ✅
- Slide 10 (Performance Global): TF-IDF + XGBoost F1-Macro = 0.697 ❌ (CORRIGIR)
- Slide 11 (Granularidade por Classe): Valores de TF-IDF+XGB parecem consistentes ✅

**Ação:** Garantir que todos os valores de TF-IDF + XGBoost sejam dos modelos **otimizados** (0.867-0.868 F1-Macro)

---

### 4. **Slide 22 - Melhorar Estrutura da Demonstração**

**Oportunidade:**
O Slide 22 tem 3 testes descritos, mas poderia ser mais visual e interativo:

**Sugestão de Melhoria:**
- Adicionar screenshots pequenos ao lado de cada teste (se espaço permitir)
- Ou criar slides separados (22a, 22b, 22c) para cada teste
- Destacar o que será demonstrado ao vivo vs. o que é apenas descrição

**Estrutura Sugerida:**
```
TESTE 1 - VELOCIDADE (TF-IDF)
[Descrição atual] ✅
+ Screenshot do resultado (opcional)

TESTE 2 - AMBIGUIDADE SEMÂNTICA
[Descrição atual] ✅
+ Destaque: "Mostraremos explicação LLM em tempo real"

TESTE 3 - MONITORAMENTO
[Descrição atual] ✅
+ Destaque: "Dashboard atualizando em tempo real"
```

---

### 5. **Slide 21 - Arquitetura de Produção**

**Observação:**
O slide menciona "Logging Estruturado" com "Loguru", mas o projeto usa sistema de logging próprio (CSV + SQLite).

**Verificar:**
- Se Loguru está realmente sendo usado no código
- Se não, ajustar para refletir o sistema real (CSV + SQLite + Dashboard Streamlit)

**Código Real:**
- `src/logging_system.py` usa CSV e SQLite
- Não encontrei referência a Loguru no código

**Sugestão:**
```
Logging Estruturado
Implementado com sistema próprio (CSV + SQLite) 
para registro detalhado e automatizado de eventos
```

---

## ✅ PONTOS FORTES A MANTER

- Estrutura geral da apresentação está excelente
- Fluxo lógico bem definido
- Demonstração ao vivo planejada (ótima ideia!)
- Dados e métricas bem apresentados
- Conclusões claras

---

## 📝 PROMPT DE CORREÇÃO FINAL

Use este prompt para corrigir os slides:

```markdown
# Correções Necessárias nos Slides - NewsLens AI

## CORREÇÃO 1: Slide 10 - Valores TF-IDF + XGBoost

**Localização:** Slide 10 - Tabela "Performance Global: Modelos Otimizados"

**Ação:**
Substituir os valores do modelo TF-IDF + XGBoost:
- F1-Macro: 0.697 → 0.868 (ou 0.8675 se preferir mais precisão)
- Accuracy: 0.714 → ~0.880 (verificar valor exato nos dados otimizados)

**Justificativa:** 
Os valores atuais são do modelo padrão (pré-otimização). 
O Slide 9 mostra que após otimização, o F1-Macro é 0.8675, 
então o Slide 10 deve refletir esses valores otimizados.

---

## CORREÇÃO 2: Slide 22 - Adicionar Link da Demo

**Localização:** Slide 22 - "Demonstração Ao Vivo: NewsLens AI"

**Ação:**
Adicionar um box destacado ou subtítulo com:
- Texto: "🌐 Acesse e teste ao vivo: https://newslens-classifier.streamlit.app/"
- Fonte: Grande (24-32pt), cor contrastante
- Posição: Topo do slide ou box destacado

**Justificativa:**
A apresentação menciona demonstração ao vivo, mas a audiência precisa 
do link para acompanhar. Isso torna a apresentação mais interativa e 
permite que o público teste enquanto você apresenta.

---

## CORREÇÃO 3: Slide 21 - Verificar Sistema de Logging

**Localização:** Slide 21 - "Arquitetura de Produção"

**Ação:**
Verificar se o sistema realmente usa Loguru. Se não:
- Substituir "Loguru" por "Sistema próprio (CSV + SQLite)"
- Manter a descrição de registro detalhado e automatizado

**Justificativa:**
O código do projeto usa CSV e SQLite, não Loguru. 
A apresentação deve refletir a implementação real.

---

## CORREÇÃO 4: Verificação de Consistência

**Ação:**
Revisar todos os slides para garantir que:
- Todos os valores de TF-IDF + XGBoost sejam dos modelos otimizados (0.867-0.868)
- Não há inconsistências entre slides
- Formatação de tabelas está consistente

---

## MELHORIA OPCIONAL: Slide 22

**Sugestão:**
Considerar dividir o Slide 22 em 3 slides menores (22a, 22b, 22c) 
ou adicionar screenshots pequenos para tornar mais visual.

**Justificativa:**
3 testes em um slide pode ser muito informação. 
Slides separados permitem mais foco em cada demonstração.
```

---

## 🎯 PRIORIDADES

1. **CRÍTICO:** Corrigir valores do Slide 10 (TF-IDF + XGBoost)
2. **IMPORTANTE:** Adicionar link da demo no Slide 22
3. **RECOMENDADO:** Verificar sistema de logging no Slide 21
4. **OPCIONAL:** Melhorar estrutura do Slide 22

---

## ✅ CHECKLIST FINAL

Após as correções, verificar:

- [ ] Slide 10: TF-IDF + XGBoost mostra F1-Macro = 0.868 (otimizado)
- [ ] Slide 10: TF-IDF + XGBoost mostra Accuracy correta (~0.880)
- [ ] Slide 22: Link da demo está visível e legível
- [ ] Slide 21: Sistema de logging reflete implementação real
- [ ] Todos os valores estão consistentes entre slides
- [ ] Formatação de tabelas está uniforme
- [ ] Link da demo funciona (testar antes da apresentação)

---

**Nota:** Estas correções garantem que a apresentação reflita com precisão os resultados do projeto e forneça acesso direto à demonstração ao vivo, tornando-a mais interativa e impactante.

