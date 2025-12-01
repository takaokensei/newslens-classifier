# Metrics Summary Report

## Experiment Information

- Dataset: Base_dados_textos_6_classes.csv
- Total Samples: 315
- Classes: 6
- Train/Val/Test Split: 60/20/20

## Class Mapping

- Class 0: Economia
- Class 1: Esportes
- Class 2: Polícia e Direitos
- Class 3: Política
- Class 4: Turismo
- Class 5: Variedades e Sociedade

## Table A: Efficiency & Performance

           Setup  F1-Macro  Accuracy  Latency (ms/doc)  Cold Start (s)  Tamanho (MB)
    TF-IDF + SVM  0.967656  0.968254          0.140129        0.079099      0.182376
TF-IDF + XGBoost  0.704040  0.714286          0.420018        0.107049      0.488663
      BERT + SVM  1.000000  1.000000          0.120019        2.228482      0.875323
  BERT + XGBoost  0.966625  0.968254          0.376861        2.295988      0.428430

## Table B: F1-Score by Class

 Class               Category  TFIDF+SVM  TFIDF+XGB  BERT+SVM  BERT+XGB
     0               Economia   0.952381   0.533333       1.0  0.952381
     1               Esportes   0.952381   0.800000       1.0  0.900000
     2     Polícia e Direitos   1.000000   0.800000       1.0  1.000000
     3               Política   1.000000   0.909091       1.0  1.000000
     4                Turismo   0.960000   0.545455       1.0  1.000000
     5 Variedades e Sociedade   0.941176   0.636364       1.0  0.947368

## Key Findings

### Best Performance
- **BERT + SVM**: F1-Macro=1.0000, Accuracy=1.0000

### Best Efficiency
- **BERT + SVM**: Latency=0.1200 ms/doc, F1-Macro=1.0000

### Best Balance
- **BERT + SVM**: F1-Macro=1.0000, Latency=0.1200 ms/doc

