# 📝 O que são arquivos `.gitkeep`?

## Propósito

Os arquivos `.gitkeep` são uma **convenção do Git** para manter pastas vazias no repositório.

## Por que são necessários?

O Git **não rastreia pastas vazias**. Se você criar uma pasta vazia e fazer commit, ela não será incluída no repositório.

### Problema:
```
data/
├── raw/          # Pasta vazia - NÃO será commitada
├── processed/   # Pasta vazia - NÃO será commitada
└── embeddings/  # Pasta vazia - NÃO será commitada
```

### Solução:
```
data/
├── raw/
│   └── .gitkeep  # Arquivo vazio - FORÇA o Git a rastrear a pasta
├── processed/
│   └── .gitkeep
└── embeddings/
    └── .gitkeep
```

## Como funciona?

1. **`.gitkeep` é apenas um arquivo vazio** (ou com comentário)
2. **O nome não importa** - poderia ser `.keep`, `README.md`, etc.
3. **A convenção `.gitkeep`** é amplamente usada na comunidade
4. **O Git rastreia o arquivo**, então a pasta é incluída no repositório

## No nosso projeto

Temos `.gitkeep` em:
- `data/raw/.gitkeep` - Para manter estrutura mesmo sem dados commitados
- `data/processed/.gitkeep` - Para manter estrutura
- `data/embeddings/.gitkeep` - Para manter estrutura (mas temos vectorizer commitado)
- `data/novos/.gitkeep` - Para manter estrutura
- `models/.gitkeep` - Para manter estrutura (mas temos modelos commitados)
- `logs/.gitkeep` - Para manter estrutura (mas logs não são commitados)

## Quando remover?

Você pode remover `.gitkeep` quando:
- A pasta já tem arquivos commitados (ex: `models/` tem `.pkl` files)
- Não precisa mais da estrutura vazia

## Exemplo prático

```bash
# Sem .gitkeep - pasta vazia não é commitada
mkdir data/raw
git add data/raw
git commit -m "Add data folder"
# ❌ Pasta não aparece no repositório

# Com .gitkeep - pasta é commitada
mkdir data/raw
touch data/raw/.gitkeep
git add data/raw/.gitkeep
git commit -m "Add data folder structure"
# ✅ Pasta aparece no repositório
```

---

**Resumo**: `.gitkeep` é um "truque" para fazer o Git rastrear pastas vazias, mantendo a estrutura do projeto organizada.

