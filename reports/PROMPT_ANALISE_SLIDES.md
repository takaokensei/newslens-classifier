# Prompt para Análise de Slides em PDF

## Instruções para a IA Analisadora

Você é um especialista em análise de apresentações acadêmicas e científicas. Sua tarefa é analisar os slides em PDF fornecidos e identificar pontos de melhoria, especialmente relacionados a:

1. **Exemplos placeholder ou genéricos** que poderiam ser substituídos por demonstrações reais
2. **Excesso de descrições teóricas** quando demonstrações práticas seriam mais impactantes
3. **Oportunidades de usar recursos interativos** (como demonstração ao vivo do Streamlit)
4. **Estrutura e fluxo da apresentação**
5. **Clareza e objetividade das informações**

---

## Tarefa Principal

Analise o PDF dos slides fornecido e identifique:

### 1. Análise de Conteúdo Placeholder

- **Exemplos genéricos ou hipotéticos** que poderiam ser substituídos por demonstrações reais
- **Descrições de funcionalidades** que poderiam ser mostradas ao vivo
- **Casos de uso teóricos** que poderiam ser demonstrados na prática

**Exemplo do que procurar:**
- "O Streamlit permite fazer X, Y, Z..." → Pode ser demonstrado ao vivo
- "Exemplo de classificação: 'Texto exemplo' → Classe X" → Pode usar exemplos reais do sistema
- Screenshots ou descrições de interface → Pode mostrar a interface funcionando

### 2. Oportunidades de Demonstração Interativa

- **Funcionalidades do Streamlit** que podem ser demonstradas ao vivo durante a apresentação
- **Recursos interativos** que tornariam a apresentação mais envolvente
- **Casos de uso reais** que podem ser testados na hora

### 3. Estrutura e Fluxo

- **Seções que poderiam ser mais concisas** (se há muita teoria quando prática seria melhor)
- **Ordem lógica** das informações
- **Equilíbrio** entre teoria e prática

### 4. Clareza e Impacto

- **Informações redundantes** ou desnecessárias
- **Pontos que poderiam ser mais diretos**
- **Oportunidades de aumentar o impacto visual ou prático**

---

## Formato da Resposta Esperada

Após a análise, retorne um **prompt de correção estruturado** no seguinte formato:

```markdown
# Análise dos Slides - Pontos de Melhoria

## 🔴 Problemas Identificados

### 1. [Título do Problema]
- **Localização:** [Slide X, Seção Y]
- **Problema:** [Descrição detalhada]
- **Impacto:** [Por que isso é um problema]
- **Sugestão:** [O que fazer ao invés]

### 2. [Título do Problema]
...

## 🟡 Oportunidades de Melhoria

### 1. [Título da Oportunidade]
- **Localização:** [Slide X, Seção Y]
- **Oportunidade:** [O que pode ser melhorado]
- **Sugestão:** [Como melhorar, especialmente com demonstração ao vivo]

### 2. [Título da Oportunidade]
...

## ✅ Pontos Fortes

- [Lista de pontos que estão bons e devem ser mantidos]

## 📝 Prompt de Correção Final

[Um prompt completo e detalhado que o usuário pode usar para corrigir os slides, 
focando especialmente em:
- Substituir exemplos placeholder por demonstrações reais
- Reduzir descrições teóricas quando demonstração prática é possível
- Sugerir momentos específicos para demonstrar o Streamlit ao vivo
- Melhorar estrutura e fluxo]
```

---

## Contexto do Projeto

**Projeto:** NewsLens AI Classifier - Classificação de Notícias em Português  
**Tipo:** Trabalho acadêmico (ELE 606 - UFRN)  
**Apresentação:** 10-15 minutos  
**Recursos disponíveis:**
- Sistema Streamlit funcional e deployado
- Interface web completa e interativa
- Dados reais de classificação
- Dashboard de monitoramento funcional
- Conjunto de validação para testes ao vivo

**Objetivo da Análise:**
Identificar onde os slides usam exemplos placeholder ou descrições genéricas que poderiam ser substituídas por **demonstrações ao vivo do Streamlit** ou exemplos reais do sistema funcionando.

---

## Critérios de Análise

1. **Prioridade Alta:**
   - Exemplos placeholder que podem ser demonstrações ao vivo
   - Descrições de funcionalidades que podem ser mostradas funcionando
   - Screenshots estáticos quando demonstração interativa é possível

2. **Prioridade Média:**
   - Excesso de teoria quando prática seria mais impactante
   - Informações redundantes
   - Estrutura que pode ser otimizada

3. **Prioridade Baixa:**
   - Ajustes de formatação
   - Melhorias de clareza textual

---

## Instruções Finais

1. Analise o PDF dos slides fornecido
2. Identifique todos os pontos de melhoria, especialmente relacionados a exemplos placeholder
3. Foque em oportunidades de demonstração ao vivo do Streamlit
4. Retorne um prompt de correção detalhado e acionável
5. Seja específico: mencione slides, seções e sugestões concretas

**Lembre-se:** O objetivo é transformar uma apresentação com exemplos teóricos em uma apresentação com demonstrações práticas e interativas, especialmente do sistema Streamlit funcionando ao vivo.

---

**Agora, analise o PDF fornecido e retorne o prompt de correção conforme solicitado.**

