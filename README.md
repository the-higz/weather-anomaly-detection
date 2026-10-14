# Weather Anomaly Detection

Projeto de Iniciação Científica voltado à análise de dados meteorológicos da estação automática de Catalão (GO), com foco na identificação de observações potencialmente anômalas.

## Objetivo

Investigar padrões meteorológicos e identificar observações que apresentam comportamento diferente daquele observado em seus vizinhos, utilizando técnicas de análise de dados e aprendizado de máquina não supervisionado.

## Tecnologias utilizadas

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- OpenPyXL

## Metodologia

O projeto utiliza o algoritmo **Local Outlier Factor (LOF)** para identificar observações potencialmente anômalas.

O processamento dos dados é dividido nas seguintes etapas:

1. Carregamento dos dados meteorológicos.
2. Organização das observações por mês.
3. Remoção de registros com valores ausentes.
4. Normalização das variáveis numéricas.
5. Cálculo dos scores LOF.
6. Identificação das 10 observações com maiores scores em cada mês.
7. Análise das observações em relação aos seus vizinhos.
8. Geração de gráficos para exploração dos resultados.
9. Exportação das anomalias identificadas para uma planilha Excel.

## Variáveis analisadas

O conjunto de dados contém as seguintes variáveis meteorológicas:

- Precipitação
- Pressão atmosférica
- Ponto de orvalho
- Temperatura máxima
- Temperatura média
- Temperatura mínima
- Umidade relativa média
- Umidade relativa mínima
- Rajada máxima do vento
- Velocidade média do vento

## Estrutura do projeto

```text
atividade_1_ic/
├── src/
│   └── main.py
├── dados.csv
├── requirements.txt
├── .gitignore
└── README.md