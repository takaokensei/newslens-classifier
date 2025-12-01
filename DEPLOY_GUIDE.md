# 🚀 Guia de Deploy no Streamlit Cloud

## Configuração do Deploy

### 1. Informações do Repositório

✅ **Repository**: `takaokensei/newslens-classifier`  
✅ **Branch**: `main`  
✅ **Main file path**: `apps/app_streamlit.py`  
✅ **App URL**: `newslens-classifier` (disponível)  
✅ **Python version**: `3.13` (ou use `3.11` se houver problemas)

### 2. Secrets (Variáveis de Ambiente)

No Streamlit Cloud, adicione as seguintes secrets na seção "Secrets":

```toml
GROQ_API_KEY = "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"
```

**Como adicionar:**
1. No dashboard do Streamlit Cloud, vá em "Settings" → "Secrets"
2. Cole o conteúdo acima no editor TOML
3. Salve

### 3. Arquivos Necessários

Certifique-se de que os seguintes arquivos existem no repositório:

✅ `.streamlit/config.toml` - Configurações do Streamlit (criado)  
✅ `requirements.txt` ou `requirements_streamlit.txt` - Dependências  
✅ `apps/app_streamlit.py` - Arquivo principal  
✅ Modelos treinados em `models/` (ou configure download automático)

### 4. Estrutura de Pastas Esperada

```
newslens-classifier/
├── .streamlit/
│   └── config.toml
├── apps/
│   └── app_streamlit.py
├── src/
│   ├── config.py
│   ├── preprocessing.py
│   ├── embeddings.py
│   ├── train.py
│   └── ...
├── models/
│   ├── tfidf_svm.pkl
│   ├── tfidf_xgb.pkl
│   ├── bert_svm.pkl
│   ├── bert_xgb.pkl
│   └── ...
├── requirements.txt
└── README.md
```

### 5. Modelos no Git

**IMPORTANTE**: Os modelos `.pkl` são grandes. Você tem duas opções:

#### Opção A: Commitar Modelos (Recomendado para deploy rápido)
```bash
git add models/*.pkl
git commit -m "Add trained models for deployment"
git push origin main
```

#### Opção B: Download Automático (Recomendado para repositório limpo)
Crie um script que baixa os modelos na primeira execução:

```python
# Em apps/app_streamlit.py, adicione no início:
import os
from pathlib import Path

MODELS_DIR = Path(__file__).parent.parent / 'models'
if not (MODELS_DIR / 'tfidf_svm.pkl').exists():
    # Download models from cloud storage (Google Drive, S3, etc.)
    st.warning("Downloading models...")
    # Seu código de download aqui
```

### 6. Verificações Finais

Antes de fazer deploy, verifique:

- [ ] `GROQ_API_KEY` está configurada nos Secrets
- [ ] `requirements.txt` está atualizado com todas as dependências
- [ ] Modelos estão disponíveis (committed ou download automático)
- [ ] `.streamlit/config.toml` existe
- [ ] Caminhos no código usam `Path` relativo (não absoluto)

### 7. Deploy

1. Acesse [Streamlit Cloud](https://share.streamlit.io/)
2. Clique em "New app"
3. Conecte seu repositório GitHub
4. Preencha:
   - **Repository**: `takaokensei/newslens-classifier`
   - **Branch**: `main`
   - **Main file path**: `apps/app_streamlit.py`
   - **App URL**: `newslens-classifier`
5. Configure Secrets (GROQ_API_KEY)
6. Clique em "Deploy"

### 8. Troubleshooting

#### Erro: "ModuleNotFoundError"
- Verifique se todas as dependências estão em `requirements.txt`
- Streamlit Cloud instala automaticamente do `requirements.txt`

#### Erro: "FileNotFoundError: models/..."
- Certifique-se de que os modelos estão commitados
- Ou implemente download automático

#### Erro: "GROQ_API_KEY not found"
- Verifique se o secret está configurado corretamente
- Nome deve ser exatamente `GROQ_API_KEY`

#### App lento para carregar
- Primeira execução carrega modelos (cold start)
- BERT leva ~2-3 segundos para carregar
- Considere usar modelos otimizados menores

### 9. Monitoramento

Após deploy:
- Acesse o dashboard de monitoramento no Streamlit
- Verifique logs em "Manage app" → "Logs"
- Monitore uso de recursos

### 10. Atualizações

Para atualizar o app:
1. Faça commit das mudanças
2. Push para `main`
3. Streamlit Cloud faz redeploy automaticamente

---

## ✅ Checklist Final

- [x] Repository: `takaokensei/newslens-classifier`
- [x] Branch: `main`
- [x] Main file: `apps/app_streamlit.py`
- [x] App URL: `newslens-classifier`
- [x] Python: 3.13 (ou 3.11)
- [ ] Secrets: GROQ_API_KEY configurado
- [ ] Modelos: Commitados ou download automático
- [ ] Requirements: Atualizado
- [ ] Config: `.streamlit/config.toml` criado

**Status**: Pronto para deploy! 🚀

