# Guia para Compilação do Relatório LaTeX

## Pré-requisitos

Para compilar o relatório LaTeX, você precisa ter instalado:

1. **LaTeX Distribution**:
   - Windows: MiKTeX ou TeX Live
   - Linux: TeX Live
   - macOS: MacTeX

2. **Editor LaTeX** (opcional, mas recomendado):
   - Overleaf (online)
   - TeXstudio
   - VS Code com extensão LaTeX Workshop

## Compilação

### Método 1: Overleaf (Recomendado)

1. Acesse [Overleaf](https://www.overleaf.com)
2. Crie um novo projeto
3. Faça upload do arquivo `relatorio.tex`
4. Faça upload das imagens das matrizes de confusão de `models/`
5. Compile o documento

### Método 2: Compilação Local

```bash
# Navegue até a pasta reports
cd reports

# Compile o PDF
pdflatex relatorio.tex
bibtex relatorio  # Se usar referências
pdflatex relatorio.tex
pdflatex relatorio.tex  # Duas vezes para resolver referências
```

## Estrutura do Relatório

O relatório segue a estrutura definida no `prompt_master.md`:

1. **Introdução** - Objetivo, hipótese e contexto
2. **Descrição da Base de Dados** - Características e estatísticas
3. **Métodos e Pipeline** - Embeddings, modelos, validação, LLMs
4. **Experimentos e Resultados** - Tabelas A e B, matrizes, gráficos
5. **Uso de LLMs** - Perfilamento e análise de erros
6. **Sistema de Produção** - Streamlit, logs, dashboard
7. **Discussão** - Comparações e resposta à hipótese
8. **Conclusões** - Achados, contribuições, recomendações
9. **Referências** - Bibliografia

## Inserção de Figuras

As matrizes de confusão devem ser inseridas no diretório `reports/` ou ajustar os caminhos no LaTeX:

```latex
\includegraphics[width=\textwidth]{../models/cm_tfidf_svm_test.png}
```

## Dados para Preencher

O template contém placeholders que devem ser preenchidos com:

- **Tabela A**: Dados de `models/table_a_efficiency.csv`
- **Tabela B**: Dados de `models/table_b_classes_with_names.csv`
- **Matrizes de Confusão**: Imagens de `models/cm_*.png`
- **Estatísticas**: Dados de `reports/metrics_summary.json`
- **Análise de Erros**: Dados de `models/differential_errors.json`
- **Perfis de Classes**: Dados de `models/class_profiles.json`

## Personalização

Você pode personalizar:

- **Título e autor**: Linhas 48-50
- **Abstract**: Seção após \maketitle
- **Conteúdo**: Cada seção pode ser expandida
- **Estilo**: Pacotes LaTeX podem ser adicionados/modificados

## Dicas

1. **Compile múltiplas vezes**: LaTeX precisa de múltiplas passagens para resolver referências
2. **Verifique caminhos**: Certifique-se de que os caminhos das imagens estão corretos
3. **Use Overleaf**: Facilita colaboração e não requer instalação local
4. **Backup**: Mantenha backup do código e das imagens

