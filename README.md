# Previsão de Diabetes com Base em Dados Clínicos e Comportamentais para Observar o Coportamento do Algoritmos Árvore de Decisão, KNN e SVM.

## 1. Introdução

Este relatório apresenta uma análise de modelos de aprendizado supervisionado para a previsão de diabetes, com base em dados clínicos e comportamentais. Foram avaliados três algoritmos: Árvore de Decisão, K-Nearest Neighbors (KNN) e Support Vector Machine (SVM). Dois conjuntos de dados foram utilizados:  
- **Dataset 1:** Grande e altamente desbalanceado, com aproximadamente 87% dos pacientes sem diabetes e 13% com diabetes.  
- **Dataset 2:** Conjunto menor, porém com classes mais equilibradas (cerca de 50% para cada classe).

Os estão presentes na pasta `datasets`. Os dados utilizados foram:

```python
X = df[['HighBP', 'HighChol', 'CholCheck', 'Smoker', 'Stroke',
        'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
        'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost']]
y = df['Diabetes_binary']
```

O objetivo é comparar o desempenho dos modelos em cada cenário e discutir as implicações teóricas e práticas associadas às métricas de avaliação utilizadas, tais como acurácia, precisão (precision) e sensibilidade (recall).

## 2. Fundamentação Teórica

### 2.1 Árvore de Decisão

  As árvores de decisão são modelos preditivos que utilizam uma estrutura em forma de árvore para representar decisões e seus possíveis resultados. Em problemas de classificação, as folhas da árvore representam os rótulos das classes, enquanto os ramos representam conjunções de condições dos atributos.
  As árvores de decisão estão entre os algoritmos de aprendizado de máquina mais populares devido à sua inteligibilidade e simplicidade. 

### 2.2 K-Nearest Neighbors (KNN)

  O KNN é um algoritmo de aprendizado preguiçoso que armazena os dados de treinamento e, no momento da predição, classifica uma nova amostra com base nos *k* vizinhos mais próximos, utilizando métricas de distância (ex.: Euclidiana, Manhattan, Minkowski).

### 2.3 Support Vector Machine (SVM)

  O SVM procura encontrar o hiperplano de margem máxima que separa as classes. Ele utiliza os vetores de suporte, que são os pontos mais próximos do hiperplano, para definir a fronteira entre as classes. Funções kernel (linear, polinomial, RBF, sigmoide) podem ser empregadas para mapear os dados para espaços de maior dimensão e melhorar a separação.

## 3. Métricas de Avaliação

- **Acurácia:**  
  É a proporção de previsões corretas sobre o total de previsões. Embora intuitiva e simples, pode ser enganosa em casos de desbalanceamento de classes, pois um modelo que sempre prevê a classe majoritária pode obter alta acurácia sem identificar a classe de interesse (no caso, diabetes).

- **Precisão (Precision):**  
  Mede a confiabilidade das previsões positivas. Indica, dentre as amostras classificadas como positivas, quantas são de fato positivas. É especialmente relevante quando o custo de falsos positivos é alto.

- **Sensibilidade (Recall):**  
  Indica a capacidade do modelo em identificar todos os casos positivos reais. É importante quando os falsos negativos (não identificar um caso positivo) têm consequências graves, como no diagnóstico de diabetes.

## 4. Análise dos Resultados

# Tabelas de Métricas dos Modelos

## Primeiro Dataset (Desbalanceado)

| Modelo          | Acurácia | Class 0 Precision | Class 0 Recall | Class 1 Precision | Class 1 Recall |
|-----------------|----------|-------------------|----------------|-------------------|----------------|
| Decision Tree   | 86%      | 0.86              | 1.00           | 0.00              | 0.00           |
| KNN             | 85%      | 0.87              | 0.96           | 0.34              | 0.12           |
| SVM             | 86%      | 0.86              | 1.00           | 0.00              | 0.00           |

> **Observações:**  
> - Para a Decision Tree e SVM, o modelo ignora completamente a classe minoritária (diabetes), prevendo sempre "sem diabetes".  
> - O KNN apresenta uma leve capacidade de identificar diabetes, porém com baixa sensibilidade e precisão para a classe 1.

## Segundo Dataset (Mais Equilibrado)

| Modelo          | Acurácia | Class 0 Precision | Class 0 Recall | Class 1 Precision | Class 1 Recall |
|-----------------|----------|-------------------|----------------|-------------------|----------------|
| Decision Tree   | 70%      | 0.74              | 0.62           | 0.67              | 0.78           |
| KNN             | 65%      | 0.64              | 0.69           | 0.66              | 0.61           |
| SVM             | 69%      | 0.70              | 0.66           | 0.68              | 0.71           |

> **Observações:**  
> - Todos os modelos têm acurácia abaixo de 70%, indicando dificuldade em aprender padrões no dataset menor.
> - A Árvore de Decisão e o KNN têm alta taxa de falsos positivos (pacientes saudáveis classificados como diabéticos), o que pode levar a intervenções médicas desnecessárias, já o SVM tem um equilíbrio melhor, mas ainda com margem para melhorias.

## 5. Considerações e Recomendações

- **Impacto do Desbalanceamento:**  
  Nos dados desbalanceados, a acurácia elevada é ilusória, pois os modelos priorizam a classe majoritária e praticamente ignora a detecção da classe minoritária (diabetes).

- **Pré-processamento:**  
  - Para modelos baseados em distância (KNN, SVM), é importante utilizar dados normalizados e corretamente tratados, e se necessário aplicar redução de dimensionalidade (PCA) para facilitar a visualização e possivelmente melhorar o desempenho.
  - A discretização pode ser útil para a interpretabilidade da Árvore de Decisão, mas pode acarretar perda de informação para métodos que dependem de distâncias.

- **Escolha do Modelo:**  
  - Em conjuntos de dados desbalanceados, nenhum dos modelos apresentou desempenho satisfatório para a detecção da classe de diabetes.  
  - No dataset equilibrado, embora o desempenho global seja moderado, o SVM se mostrou ligeiramente superior em termos de equilíbrio entre precisão e recall.

## 6. Conclusão

Os experimentos realizados evidenciam que:
- **Datasets desbalanceados** podem levar a modelos que, embora apresentem alta acurácia, ignoram a classe de maior interesse (diabetes).  
- **Dados mais equilibrados** possibilitam um desempenho mais realista dos modelos, mas mesmo assim, os desafios de falsos positivos e falsos negativos permanecem.
