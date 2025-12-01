# 🔧 Correção do Erro de Deploy no Streamlit Cloud

## Problema

O erro `RuntimeError: can't register atexit after shutdown` ocorre quando há imports de módulos pesados (sklearn, joblib) no nível do módulo do Streamlit. Isso causa conflitos com o sistema de multiprocessing do Python em ambientes de deploy.

## Solução Implementada

### Lazy Imports

Todos os imports pesados foram movidos para uma função `_lazy_imports()` que só é chamada quando necessário, dentro das funções que realmente precisam desses módulos.

### Antes (Causava Erro):
```python
from src.preprocessing import preprocess_text
from src.embeddings import load_tfidf_vectorizer, load_bert_model
from src.train import load_trained_models
# ... outros imports pesados
```

### Depois (Corrigido):
```python
# Apenas imports leves no nível do módulo
from src.config import PATHS
from src.class_mapping import CLASS_TO_CATEGORY

# Lazy imports dentro de função
def _lazy_imports():
    """Lazy import of heavy dependencies."""
    from src.preprocessing import preprocess_text
    from src.embeddings import load_tfidf_vectorizer, load_bert_model
    from src.train import load_trained_models
    # ... outros imports
    return { ... }

# Uso dentro das funções
def load_all_models():
    imports = _lazy_imports()
    models = imports['load_trained_models']()
    # ...
```

## Por que Funciona?

1. **Evita Import Circular**: Os imports só acontecem quando a função é chamada, não no carregamento do módulo
2. **Evita Multiprocessing Issues**: O joblib/sklearn não tenta registrar handlers de atexit durante o shutdown
3. **Mantém Performance**: O `@st.cache_resource` ainda funciona normalmente

## Verificação

O código foi testado localmente e a sintaxe está correta. O deploy no Streamlit Cloud deve funcionar agora.

## Arquivos Modificados

- `apps/app_streamlit.py`: Implementado lazy imports em todas as funções que usam módulos pesados

---

**Status**: ✅ Corrigido e commitado

