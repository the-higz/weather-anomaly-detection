from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt

nomes_variaveis = [
    "Precipitação",
    "Pressão",
    "Ponto de orvalho",
    "Temperatura máxima",
    "Temperatura média",
    "Temperatura mínima",
    "Umidade média",
    "Umidade mínima",
    "Rajada máxima",
    "Velocidade média do vento"
]

with open('dados.csv', encoding='utf-8') as arquivo:

    for linha in arquivo:
        if linha.startswith('Data Medicao'):
            break

    dados_por_mes = {}

    for linha in arquivo:
        if not linha.strip():
            continue

        valores = linha.strip().rstrip(';').split(';')

        if 'null' not in valores:
            tupla = tuple(valores)
            data = tupla[0]
            partes = data.split('-')
            mes = partes[1]

            if mes not in dados_por_mes:
                dados_por_mes[mes] = []

            dados_por_mes[mes].append(tupla)

dados_normalizados = {}

for mes, tuplas in dados_por_mes.items():

    dados = np.array([
        [float(valor) for valor in tupla[1:]]
        for tupla in tuplas
    ])

    media = np.mean(dados, axis=0)
    desvio = np.std(dados, axis=0)

    normalizado = (dados - media) / desvio

    dados_normalizados[mes] = normalizado

resultados_lof = {}

k = 20

for mes, dados in dados_normalizados.items():

    lof = LocalOutlierFactor(n_neighbors=k)

    labels = lof.fit_predict(dados)
    scores = -lof.negative_outlier_factor_

    resultados_lof[mes] = {
        "labels": labels,
        "scores": scores
    }

for mes in sorted(resultados_lof, key=int):
    resultado = resultados_lof[mes]

    scores = resultado["scores"]

    indices = np.argsort(scores)[-10:][::-1]

    dados_originais = dados_por_mes[mes]

    dados = np.array([
        [float(valor) for valor in tupla[1:]]
        for tupla in dados_originais
    ])

    fig, eixos = plt.subplots(
        10,
        10,
        figsize=(22, 22)
    )

    for i in range(10):
        for j in range(10):

            ax = eixos[i, j]

            if i == j:
                ax.text(
                    0.5,
                    0.5,
                    nomes_variaveis[i],
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight="bold"
                )

                ax.set_xticks([])
                ax.set_yticks([])
                continue

            if i < j:
                ax.axis("off")
                continue

            ax.scatter(
                dados[:, j],
                dados[:, i],
                s=5
            )

            ax.scatter(
                dados[indices, j],
                dados[indices, i],
                s=25
            )

            if i == 9:
                ax.set_xlabel(
                    nomes_variaveis[j],
                    fontsize=8
                )

            else:
                ax.set_xticklabels([])

            if j == 0:
                ax.set_ylabel(
                    nomes_variaveis[i],
                    fontsize=8
                )

            else:
                ax.set_yticklabels([])

            ax.tick_params(
                axis="both",
                labelsize=6
            )

    fig.suptitle(
        f"Matriz de dispersão — Mês {mes} — Top 10 LOF",
        fontsize=18,
        fontweight="bold"
    )

    plt.tight_layout(rect=(0, 0, 1, 0.97))

    plt.show()

    media = np.mean(dados, axis=0)

    vizinhos = NearestNeighbors(n_neighbors=k + 1)

    vizinhos.fit(dados)

    distancias, indices_vizinhos = vizinhos.kneighbors(dados)

    print(f"\nMÊS: {mes}")
    print("-" * 80)

    for i in indices:

        tupla = dados_originais[i]

        amostra = dados[i]

        indices_dos_vizinhos = indices_vizinhos[i][1:]

        media_vizinhos = np.mean(
            dados[indices_dos_vizinhos],
            axis=0
        )

        diferenca_vizinhos = amostra - media_vizinhos

        diferenca = amostra - media

        print(
            f"\nData: {tupla[0]} | "
            f"LOF: {scores[i]:.3f}"
        )

        print("Amostra: ", amostra)
        print("Média: ", media)
        print("Diferença: ", diferenca)
        print("Média vizinhos: ", media_vizinhos)
        print("Diferença vizinhos: ", diferenca_vizinhos)