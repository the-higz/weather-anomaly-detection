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

        indices = np.where(resultado["labels"] == -1)[0]
        indices = indices[np.argsort(scores[indices])[::-1]]

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
        indices = np.where(resultado["labels"] == -1)[0]
        indices = indices[np.argsort(scores[indices])[::-1]]

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

def exportar_dados_dashboard(dados_por_mes, resultados_lof):
    linhas = []

    for mes in sorted(dados_por_mes, key=int):
        observacoes = dados_por_mes[mes]
        scores = resultados_lof[mes]["scores"]
        labels = resultados_lof[mes]["labels"]

        for i, tupla in enumerate(observacoes):
            linha = {
                "Mês": mes,
                "Data": tupla[0],
                "Score LOF": scores[i],
                "Anomalia": labels[i] == -1
            }

            for j, nome in enumerate(nomes_variaveis):
                linha[nome] = float(tupla[j + 1])

            linhas.append(linha)

    tabela = pd.DataFrame(linhas)
    tabela.to_excel("dados_dashboard.xlsx", index=False)

    print("Dados completos exportados para dados_dashboard.xlsx")


def exportar_resumo_temporal(dados_por_mes, resultados_lof):

    linhas_resumo = []

    for mes in sorted(dados_por_mes, key=int):

        observacoes = dados_por_mes[mes]
        resultado = resultados_lof[mes]

        labels = resultado["labels"]
        scores = resultado["scores"]

        # Agrupa as observações pelo ano original.
        anos = {}

        for i, tupla in enumerate(observacoes):

            data = tupla[0]
            ano = int(data.split("-")[0])

            if ano not in anos:
                anos[ano] = {
                    "observacoes": 0,
                    "anomalias": 0,
                    "scores_anomalos": []
                }

            anos[ano]["observacoes"] += 1

            if labels[i] == -1:
                anos[ano]["anomalias"] += 1
                anos[ano]["scores_anomalos"].append(float(scores[i]))

        # Gera uma linha de resumo para cada ano e mês.
        for ano in sorted(anos):

            resumo = anos[ano]

            total = resumo["observacoes"]
            quantidade_anomalias = resumo["anomalias"]
            scores_anomalos = resumo["scores_anomalos"]

            percentual = (
                quantidade_anomalias / total * 100
                if total > 0 else 0
            )

            linha = {
                "Ano": ano,
                "Mês": int(mes),
                "Período": f"{ano}-{mes}",
                "Observações analisadas": total,
                "Anomalias": quantidade_anomalias,
                "Percentual de anomalias (%)": percentual,
                "Média dos scores anômalos": (
                    np.mean(scores_anomalos)
                    if scores_anomalos else np.nan
                ),
                "Mediana dos scores anômalos": (
                    np.median(scores_anomalos)
                    if scores_anomalos else np.nan
                ),
                "Maior score anômalo": (
                    max(scores_anomalos)
                    if scores_anomalos else np.nan
                )
            }

            linhas_resumo.append(linha)

    tabela = pd.DataFrame(linhas_resumo)

    if not tabela.empty:
        tabela = tabela.sort_values(["Ano", "Mês"])

    tabela.to_excel("resumo_temporal.xlsx", index=False)

    print("Resumo temporal exportado para resumo_temporal.xlsx\n")


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
        indices = np.where(resultado["labels"] == -1)[0]
        indices = indices[np.argsort(scores[indices])[::-1]]

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
    exportar_dados_dashboard(dados_por_mes, resultados_lof)
    exportar_resumo_temporal(dados_por_mes, resultados_lof)

if __name__ == "__main__":
    main()