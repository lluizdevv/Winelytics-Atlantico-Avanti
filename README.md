# 🍷 Winelytics — Análise de Vinhos Espanhóis

> **Bootcamp em Ciência de Dados — Escola Atlântico Avanti 2026.2**
> **Squad 8** — Gabriel Cavalcante, Luiz Vinícius, Isa Paula, Ítalo Marcony
<img width="1919" height="650" alt="image" src="https://github.com/user-attachments/assets/bc3b6022-76d5-4802-9d4e-7ccf071d9b3e" />

Este repositório reúne dois notebooks complementares que analisam o
[Spanish Wine Quality Dataset](https://www.kaggle.com/datasets/fedesoriano/spanish-wine-quality-dataset),
contendo ~7.500 vinhos espanhóis com informações de preço, avaliação (rating),
número de reviews, região, vinícola, tipo, corpo e acidez.

## 📁 Notebooks

| Notebook | Entrega | Descrição |
|---|---|---|
| `Winelytics_Squad_8_Analise_Exploratoria_de_Dados.ipynb` | Desafio 02 | Análise Exploratória de Dados (EDA) |
| `Winelytics_Squad_8_Machine_Learning.ipynb` | Desafio 03 | Modelagem preditiva (regressão) |

---

## 1. 📊 Análise Exploratória de Dados (EDA)

**Objetivo:** entender os fatores que influenciam a qualidade e o preço dos
vinhos espanhóis, explorando características como região, vinícola, safra e
corpo do vinho.

**Conteúdo:**
- **Dicionário de dados** — descrição de todas as colunas do dataset.
- **Tratamento de dados** — identificação e tratamento de valores ausentes,
  isolamento do valor especial `N.V.` (Non Vintage) na coluna `year`,
  padronização de tipos e categorias.
- **Análise univariada** — distribuição individual de cada variável
  (histogramas, contagens por categoria, top vinícolas/regiões/tipos).
- **Análise de outliers** — boxplots e regra do IQR para `price`, `rating` e
  `num_reviews`, com discussão sobre por que os outliers foram mantidos
  (vinhos premium legítimos, não erros de dado).
- **Análise bivariada** — relação entre pares de variáveis (preço x rating,
  matriz de correlação, preço por tipo/região).
- **Análise multivariada** — cruzamentos com 3+ variáveis (scatter colorido,
  pairplot, heatmap região x tipo, evolução temporal do rating).

**Principais achados:**
- Forte concentração de vinhos na região **Rioja** e em poucas vinícolas.
- `price` e `num_reviews` são extremamente assimétricos (poucos vinhos muito
  caros/populares puxam a distribuição) — melhor visualizados em escala log.
- Correlação positiva, porém fraca, entre preço e rating: preço não garante
  nota melhor.

---

## 2. 🤖 Machine Learning — Análise Comparativa de Modelos

**Objetivo:** prever o **preço** de um vinho (`price`, modelado como
`log1p(price)`) a partir de suas características — um problema de
**regressão** — comparando diferentes famílias de modelos.

**Estrutura do notebook** (conforme exigido na atividade):

### 1. Metodologia
Definição do problema, variáveis utilizadas, estratégia de pré-processamento
e justificativa da validação cruzada escolhida.

### 2. Configuração do Experimento
- **Pré-processamento:** imputação de dados faltantes (mediana/moda),
  one-hot encoding de variáveis categóricas (`type`, `region` agrupadas),
  normalização (`StandardScaler`) das variáveis quantitativas.
- **Validação cruzada:** Monte Carlo (`ShuffleSplit`, 10 repetições 80/20),
  complementada por uma divisão fixa de treino/teste (holdout) para
  diagnóstico de overfitting.
- **Modelos comparados (6):**
  | Modelo | Papel |
  |---|---|
  | Baseline (`DummyRegressor`) | Piso de comparação |
  | Regressão Linear | Baseline estatístico |
  | KNN | Família baseada em distância |
  | Árvore de Decisão | Família baseada em árvore |
  | Random Forest | Ensemble de árvores (bagging) |
  | Gradient Boosting | Ensemble de árvores (boosting) |
- **Métricas:** MAE, RMSE, R² e MAPE.

### 3. Resultados e Discussão
- Tabela comparativa estilizada (Treino / CV / Teste), com destaque visual
  para o melhor (verde) e o pior (vermelho) valor por coluna.
- Gráfico de diagnóstico de **overfitting** (R² Treino vs CV vs Teste).
- Gráfico **Real vs. Previsto** por modelo.
- Boxplots de distribuição das métricas entre as repetições de CV.
- **Teste estatístico pareado** (teste t e Wilcoxon) comparando Gradient
  Boosting e Random Forest, os dois modelos com melhor desempenho.
- Interpretação, limitações e próximos passos.

**Principais achados:**
- **KNN** apresentou overfitting acentuado (R² Treino ≈ 0,9998 vs Teste ≈
  0,82), devido ao uso de `weights='distance'`.
- **Random Forest** e **Gradient Boosting** foram os modelos com melhor
  desempenho (R² de Teste ≈ 0,84), com Gradient Boosting apresentando o
  menor gap de overfitting.
- O teste estatístico pareado mostrou que a diferença entre Random Forest e
  Gradient Boosting **não é estatisticamente significativa** (p > 0,68 em
  ambos os testes) — os dois modelos estão tecnicamente empatados, e a
  escolha entre eles pode considerar outros critérios (tempo de
  treinamento, interpretabilidade).
- **Regressão Linear** (R² ≈ 0,72) confirma que a relação entre as
  características do vinho e o preço não é puramente linear.

---

## 🛠️ Como executar

1. Abra os notebooks no [Google Colab](https://colab.research.google.com/).
2. Certifique-se de que o arquivo `wines_SPA.csv` esteja acessível (via
   upload direto ou pela URL configurada na célula de leitura dos dados).
3. Execute as células em ordem, de cima para baixo (`Ambiente de execução >
   Executar tudo`).
4. O notebook de EDA não tem dependências além de `pandas`, `numpy`,
   `matplotlib` e `seaborn`. O notebook de ML usa adicionalmente
   `scikit-learn` e `scipy`.

## 📚 Fonte dos dados

[Spanish Wine Quality Dataset](https://www.kaggle.com/datasets/fedesoriano/spanish-wine-quality-dataset) (Kaggle).
