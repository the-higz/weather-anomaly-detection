from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

def carregar_dados():
    with open('dados.csv', encoding='utf-8') as arquivo:
        for linha in arquivo:
            if linha.startswith('Data Medicao'):
                break

        return arquivo.readlines()

def agrupar_por_mes(arquivo):
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

    return dados_por_mes

def normalizar_dados(dados_por_mes):
    dados_normalizados = {}
    for mes, tuplas in dados_por_mes.items():

        dados = np.array([
            [float(valor) for valor in tupla[1:]]
            for tupla in tuplas
        ])

        media = np.mean(dados, axis=0)
        desvio = np.std(dados, axis=0)

        desvio[desvio == 0] = 1
        normalizado = (dados - media)/desvio

        dados_normalizados[mes] = normalizado

    return dados_normalizados

def calcular_lof(dados_normalizados):
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

    return resultados_lof
    
def gerar_graficos(resultados_lof, dados_por_mes):
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
        plt.close()

def analisar_vizinhos(dados, dados_normalizados, k, mes, indices, dados_originais, scores):
    media = np.mean(dados, axis=0)

    vizinhos = NearestNeighbors(n_neighbors=k + 1)
    vizinhos.fit(dados_normalizados)

    distancias, indices_vizinhos = vizinhos.kneighbors(dados_normalizados)

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


def exportar_excel(resultados_lof, dados_por_mes):
    linhas_excel = []

    for mes in sorted(resultados_lof, key=int):
        resultado = resultados_lof[mes]
        scores = resultado["scores"]
        indices = np.argsort(scores)[-10:][::-1]

        for indice in indices:
            tupla = dados_por_mes[mes][indice]

            linha = {
                "Mês": mes,
                "Data": tupla[0],
                "Score LOF": scores[indice]
            }

            for i,nome in enumerate(nomes_variaveis):
                linha[nome] = float(tupla[i + 1])

            linhas_excel.append(linha)

    tabela = pd.DataFrame(linhas_excel)
    tabela.to_excel("resultados_anomalias.xlsx", index=False)

    print("\nResultados exportados para resultados_anomalias.xlsx")


def main():
    linhas = carregar_dados()

    dados_por_mes = agrupar_por_mes(linhas)

    dados_normalizados = normalizar_dados(dados_por_mes)

    resultados_lof = calcular_lof(dados_normalizados)

    # gerar_graficos(resultados_lof, dados_por_mes)

    k = 20

    for mes in sorted(resultados_lof, key=int):
        resultado = resultados_lof[mes]

        scores = resultado["scores"]
        indices = np.argsort(scores)[-10:][::-1]

        dados_originais = dados_por_mes[mes]

        dados = np.array([
            [float(valor) for valor in tupla[1:]]
            for tupla in dados_originais
        ])

        analisar_vizinhos(
            dados,
            dados_normalizados[mes],
            k,
            mes,
            indices,
            dados_originais,
            scores
        )

    # da para usar Invoke-Item resultados_anomalias.xlsx para abrir o excel direto pelo terminal
    exportar_excel(resultados_lof, dados_por_mes)

if __name__ == "__main__":
    main()