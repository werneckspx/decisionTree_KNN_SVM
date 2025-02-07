# Aplicação da Árvore de Decisão com 10 Perguntas.

A parte um deste relatório apresenta o desenvolvimento de um árvore de decisão construída via código usando `if` e `elses` com o objetivo de ajudar pessoas a decidirem entre diferentes hobbis ou carreiras, com base em 10 perguntas.

 1. Você gosta de trabalhar em equipe? (Sim/Não)
 2. Prefere atividades ao ar livre? (Sim/Não)
 3. Gosta de lidar com números? (Sim/Não)
 4. Prefere usar habilidades artísticas? (Sim/Não)
 5. Se sente confortável trabalhando com tecnologia? (Sim/Não)
 6. Gosta de resolver problemas complexos? (Sim/Não)
 7. Prefere atividades que envolvam comunicação? (Sim/Não)
 8. Se interessa por cuidar de outras pessoas? (Sim/Não)
 9. Gosta de desafios físicos? (Sim/Não)
 10. Prefere trabalhar em ambientes organizados? (Sim/Não)

Implementar este código ajudou a compreender o funcionamento da árvore de decisão e como ela pode auxiliar em diferentes cenários a partir de entradas distintas.

# Previsão de Diabetes com Base em Dados Clínicos e Comportamentais para Observar o Coportamento do Algoritmos Árvore de Decisão, KNN e SVM.

A parte dois deste relatório apresenta uma análise de modelos de aprendizado supervisionado para a previsão de diabetes, com base em dados clínicos e comportamentais. Foram avaliados três algoritmos: Árvore de Decisão, K-Nearest Neighbors (KNN) e Support Vector Machine (SVM). Dois conjuntos de dados foram utilizados:  
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

## Fundamentação Teórica

### Árvore de Decisão

  As árvores de decisão são modelos preditivos que utilizam uma estrutura em forma de árvore para representar decisões e seus possíveis resultados. Em problemas de classificação, as folhas da árvore representam os rótulos das classes, enquanto os ramos representam conjunções de condições dos atributos.
  As árvores de decisão estão entre os algoritmos de aprendizado de máquina mais populares devido à sua inteligibilidade e simplicidade. 

### K-Nearest Neighbors (KNN)

  O KNN é um algoritmo de aprendizado preguiçoso que armazena os dados de treinamento e, no momento da predição, classifica uma nova amostra com base nos *k* vizinhos mais próximos, utilizando métricas de distância (ex.: Euclidiana, Manhattan, Minkowski).

### Support Vector Machine (SVM)

  O SVM procura encontrar o hiperplano de margem máxima que separa as classes. Ele utiliza os vetores de suporte, que são os pontos mais próximos do hiperplano, para definir a fronteira entre as classes. Funções kernel (linear, polinomial, RBF, sigmoide) podem ser empregadas para mapear os dados para espaços de maior dimensão e melhorar a separação.

## Métricas de Avaliação

- **Acurácia:**  
  É a proporção de previsões corretas sobre o total de previsões. Embora intuitiva e simples, pode ser enganosa em casos de desbalanceamento de classes, pois um modelo que sempre prevê a classe majoritária pode obter alta acurácia sem identificar a classe de interesse (no caso, diabetes).

- **Precisão (Precision):**  
  Mede a confiabilidade das previsões positivas. Indica, dentre as amostras classificadas como positivas, quantas são de fato positivas. É especialmente relevante quando o custo de falsos positivos é alto.

- **Sensibilidade (Recall):**  
  Indica a capacidade do modelo em identificar todos os casos positivos reais. É importante quando os falsos negativos (não identificar um caso positivo) têm consequências graves, como no diagnóstico de diabetes.

## Análise dos Resultados

# Tabelas de Métricas dos Modelos

## Primeiro Dataset (Desbalanceado)

| Modelo          | Acurácia | Class 0 Precision | Class 0 Recall | Class 1 Precision | Class 1 Recall |
|-----------------|----------|-------------------|----------------|-------------------|----------------|
| Decision Tree   | 86%      | 0.86              | 1.00           | 0.00              | 0.00           |
| KNN             | 85%      | 0.87              | 0.96           | 0.34              | 0.12           |
| SVM             | 86%      | 0.86              | 1.00           | 0.00              | 0.00           |

### Para a classe 0 (sem diabetes) e majoritária:

#### Árvore de Decisão e SVM:

Recall 1.00: O modelo está classificando todos os pacientes sem diabetes corretamente.

Precisão 0.86: 14% das previsões "sem diabetes" são falsos positivos (pacientes que têm diabetes, mas foram classificados erroneamente como saudáveis).

#### KNN:

Recall 0.96: 4% dos pacientes sem diabetes foram classificados erroneamente como diabéticos.

Precisão 0.87: 13% das previsões "sem diabetes" são falsos positivos.


### Para classe 1 (com diabetes) minoritária:

#### Árvore de Decisão e SVM:

Recall 0.00: O modelo não identificou nenhum caso real de diabetes.

Precisão 0.00: Todas as previsões para diabetes estão erradas.

#### KNN:

Recall 0.12: Apenas 12% dos casos reais de diabetes foram detectados.

Precisão 0.34: Quando o modelo prevê diabetes, 66% das previsões estão erradas.

> **Observações:**
> - Para a Decision Tree e SVM, o modelo ignora completamente a classe minoritária (diabetes), prevendo sempre "sem diabetes".  
> - O KNN apresenta uma leve capacidade de identificar diabetes, porém com baixa sensibilidade e precisão para a classe 1.

## Segundo Dataset (Mais Equilibrado)

| Modelo          | Acurácia | Class 0 Precision | Class 0 Recall | Class 1 Precision | Class 1 Recall |
|-----------------|----------|-------------------|----------------|-------------------|----------------|
| Decision Tree   | 70%      | 0.74              | 0.62           | 0.67              | 0.78           |
| KNN             | 65%      | 0.64              | 0.69           | 0.66              | 0.61           |
| SVM             | 69%      | 0.70              | 0.66           | 0.68              | 0.71           |

### Para a classe 0 (sem diabetes) e majoritária:

#### Árvore de Decisão:

Precisão 74%: Quando prevê "sem diabetes", 26% são falsos positivos.

Recall 62%: 38% dos pacientes sem diabetes foram erroneamente classificados como diabéticos.

#### KNN:

Recall 69%: 31% dos pacientes sem diabetes foram classificados como diabéticos.

Precisão 0.87: 13% das previsões "sem diabetes" são falsos positivos.

#### SVM:

Precisão 70%: 30% das previsões "sem diabetes" são falsos positivos.

### Para classe 1 (com diabetes) minoritária:

#### Árvore de Decisão:

Recall 78%: Identifica 78% dos casos reais de diabetes (melhor desempenho).

Precisão 67%: 33% das previsões de diabetes estão erradas.

#### KNN:

Precisão 66%: 34% das previsões de diabetes estão erradas.

Recall 61%: Baixa sensibilidade para detectar casos reais.

#### SVM:

Recall 71%: Melhor que o KNN para identificar casos reais de diabetes.

> **Observações:**  
> - Todos os modelos têm acurácia abaixo de 70%, indicando dificuldade em aprender padrões no dataset menor.
> - A Árvore de Decisão e o KNN têm alta taxa de falsos positivos (pacientes saudáveis classificados como diabéticos), o que pode levar a intervenções médicas desnecessárias, já o SVM tem um equilíbrio melhor, mas ainda com margem para melhorias.

## Considerações

- **Impacto do Desbalanceamento:**  
  Nos dados desbalanceados, a acurácia elevada é ilusória, pois os modelos priorizam a classe majoritária e praticamente ignora a detecção da classe minoritária (diabetes).

- **Pré-processamento:**  
  - Para modelos baseados em distância (KNN, SVM), é importante utilizar dados normalizados e corretamente tratados, e se necessário aplicar redução de dimensionalidade (PCA) para facilitar a visualização e possivelmente melhorar o desempenho.
  - A discretização pode ser útil para a interpretabilidade da Árvore de Decisão, mas pode acarretar perda de informação para métodos que dependem de distâncias.

- **Escolha do Modelo:**  
  - Em conjuntos de dados desbalanceados, nenhum dos modelos apresentou desempenho satisfatório para a detecção da classe de diabetes.  
  - No dataset equilibrado, embora o desempenho global seja moderado, o SVM se mostrou ligeiramente superior em termos de equilíbrio entre precisão e recall.

## Conclusão

Os experimentos realizados evidenciam que:
- **Datasets desbalanceados** podem levar a modelos que, embora apresentem alta acurácia, ignoram a classe de maior interesse (diabetes).  
- **Dados mais equilibrados** possibilitam um desempenho mais realista dos modelos, mas mesmo assim, os desafios de falsos positivos e falsos negativos permanecem.
