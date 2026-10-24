import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.neighbors import NearestNeighbors


VARIAVEIS = [
    "Precipitação",
    "Pressão", 
    "Ponto de orvalho", 
    "Temperatura máxima",
    "Temperatura média", 
    "Temperatura mínima", 
    "Umidade média",
    "Umidade mínima", 
    "Rajada máxima", 
    "Velocidade média do vento",
]


@st.cache_data
def carregar_resultados():
    return pd.read_excel("resultados_anomalias.xlsx")


@st.cache_data
def carregar_dados_completos():
    return pd.read_excel("dados_dashboard.xlsx")


@st.cache_data
def carregar_resumo_temporal():
    return pd.read_excel("resumo_temporal.xlsx")


def comparar_com_vizinhos(dados_mes, data_anomalia, variaveis, k=20):

    """Compara uma observação com os vizinhos no espaço normalizado."""
    dados_validos = dados_mes.dropna(subset=variaveis).copy().reset_index(drop=True)

    correspondencias = dados_validos.index[
        dados_validos["Data"].astype(str) == str(data_anomalia)

    ].tolist()

    if not correspondencias:
        return None

    indice_anomalia = correspondencias[0]
    valores = dados_validos[variaveis].astype(float).to_numpy()

    if len(valores) < 2:
        return None

    medias = valores.mean(axis=0)
    desvios = valores.std(axis=0)
    
    desvios[desvios == 0] = 1.0
    normalizados = (valores - medias) / desvios
    n_vizinhos = min(k, len(valores) - 1)

    modelo = NearestNeighbors(n_neighbors=n_vizinhos + 1)
    modelo.fit(normalizados)
    _, indices = modelo.kneighbors(normalizados[indice_anomalia].reshape(1, -1))
    indices_vizinhos = [i for i in indices[0] if i != indice_anomalia][:n_vizinhos]

    if not indices_vizinhos:
        return None

    media_vizinhos = valores[indices_vizinhos].mean(axis=0)
    observado = valores[indice_anomalia]
    diferenca = observado - media_vizinhos

    return pd.DataFrame({
        "Variável": variaveis,
        "Valor observado": observado,
        "Média dos vizinhos": media_vizinhos,
        "Diferença": diferenca,
        "Diferença padronizada": diferenca / desvios,
    })


st.set_page_config(page_title="Análise de Anomalias Meteorológicas", page_icon="🌦️", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background-color:#0B1F33;
        color:#FFFFFF;
        font-family: "Segoe UI", "Inter", "Roboto", sans-serif;
    }
    html, body, [class*="css"] {
        font-family: "Segoe UI", "Inter", "Roboto", sans-serif;
    }
    h1, h2, h3 {
        color:#F4F8FC;
        letter-spacing:-0.025em;
        font-weight:650;
    }
    h1 {
        font-size:2.15rem;
        line-height:1.15;
    }
    h2 {font-size:1.55rem;}
    h3 {font-size:1.15rem;}
    p, label, [data-testid="stCaption"] {
        color:#D4DDE5;
        line-height:1.5;
    }
    [data-testid="stMetric"] {
        background-color:#153A5B;
        padding:10px 12px;
        border-radius:12px;
        border:1px solid #285477;
    }
    [data-testid="stMetricLabel"] {
        color:#C8D8E8;
        font-size:0.88rem;
        font-weight:500;
    }
    [data-testid="stMetricValue"] {
        color:#FFFFFF;
        font-size:1.8rem;
        font-weight:650;
        letter-spacing:-0.025em;
    }
    div[data-baseweb="select"] > div {
        background-color:#153A5B;
        border-radius:9px;
    }
    div[data-baseweb="select"] * {font-family:inherit;}
    hr {border-color:#285477;}
    
/* Refinamento visual dos cartões de indicadores */
div[data-testid="stMetric"] {
    background: linear-gradient(145deg, #183f60 0%, #153653 100%);
    border: 1px solid rgba(77, 163, 255, 0.30);
    border-radius: 12px;
    padding: 14px 16px;
    min-height: 88px;
}
div[data-testid="stMetric"] label {
    font-size: 0.82rem;
    opacity: 0.92;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-weight: 700;
    letter-spacing: -0.02em;
}
@media (max-width: 900px) {
    div[data-testid="stMetric"] {
        padding: 10px 12px;
    }
}

</style>
    """, unsafe_allow_html=True)

cabecalho, metodo = st.columns([3, 1])

with cabecalho:
    st.caption("INICIAÇÃO CIENTÍFICA · ANÁLISE METEOROLÓGICA")
    st.title("🌦️ Monitoramento de Anomalias (Método LOF)")
    st.markdown("Estação meteorológica de Catalão, Goiás")

with metodo:
    st.markdown(" ")
    st.markdown(" ")

st.divider()


try:
    dados = carregar_resultados()
    dados_completos = carregar_dados_completos()
    resumo_temporal = carregar_resumo_temporal()

except FileNotFoundError as erro:
    st.error("Não encontrei uma das planilhas necessárias. Execute `python src/main.py` para gerar os arquivos Excel.")
    st.caption(f"Detalhe: {erro}")
    st.stop()


if dados.empty or dados_completos.empty:
    st.warning("Não há dados disponíveis para exibir.")
    st.stop()

if not {"Mês", "Data", "Score LOF"}.issubset(dados.columns):
    st.error("A planilha resultados_anomalias.xlsx não contém todas as colunas necessárias.")
    st.stop()

if not {"Mês", "Data", "Score LOF", "Anomalia"}.issubset(dados_completos.columns):
    st.error("A planilha dados_dashboard.xlsx não contém todas as colunas necessárias.")
    st.stop()

# Verificação do resumo temporal
colunas_temporais = {
    "Ano",
    "Mês",
    "Observações analisadas",
    "Anomalias",
    "Percentual de anomalias (%)"
}

if not colunas_temporais.issubset(resumo_temporal.columns):
    st.error(
        "A planilha resumo_temporal.xlsx não contém "
        "todas as colunas necessárias."
    )
    st.stop()

variaveis_disponiveis = [v for v in VARIAVEIS if v in dados_completos.columns]


if len(variaveis_disponiveis) < 2:
    st.error("São necessárias pelo menos duas variáveis meteorológicas disponíveis para a análise.")
    st.stop()


col_visao_geral = st.container()

with col_visao_geral:
    st.subheader("Visão geral")

    # Seleção em cascata: primeiro o ano, depois o mês daquele ano.
    dados_completos["Data"] = pd.to_datetime(dados_completos["Data"], errors="coerce")
    dados_completos["Ano"] = dados_completos["Data"].dt.year
    dados_completos["Mes_num"] = pd.to_numeric(dados_completos["Mês"], errors="coerce").astype("Int64")

    # Disponibiliza apenas anos com pelo menos uma anomalia identificada.
    dados_anomalias_data = dados.copy()
    dados_anomalias_data["Data_dt"] = pd.to_datetime(
        dados_anomalias_data["Data"], errors="coerce"
    )
    anos_com_anomalias = sorted(
        dados_anomalias_data.loc[
            dados_anomalias_data["Data_dt"].notna(), "Data_dt"
        ].dt.year.unique().astype(int).tolist()
    )
    anos_disponiveis = ["Todos os anos"] + [
        ano for ano in anos_com_anomalias
        if (dados_anomalias_data["Data_dt"].dt.year == ano).any()
    ]
    nomes_meses = {
        1: "Janeiro",
        2: "Fevereiro",
        3: "Março",
        4: "Abril",
        5: "Maio",
        6: "Junho",
        7: "Julho",
        8: "Agosto",
        9: "Setembro",
        10: "Outubro",
        11: "Novembro",
        12: "Dezembro"
    }

    if not anos_disponiveis:
        st.warning("Não há anos disponíveis para análise.")
        st.stop()

    col_ano, col_mes = st.columns([1, 1])

    with col_ano:
        ano_selecionado = st.selectbox(
            "Ano",
            options=anos_disponiveis,
            index=0
        )

    # Dentro do recorte escolhido, oferece apenas meses com anomalias.
    if ano_selecionado == "Todos os anos":
        meses_com_anomalias = (
            dados_anomalias_data.loc[
                dados_anomalias_data["Data_dt"].notna(), "Data_dt"
            ].dt.month.unique().astype(int).tolist()
        )
    else:
        meses_com_anomalias = dados_anomalias_data.loc[
            (dados_anomalias_data["Data_dt"].dt.year == ano_selecionado)
            & dados_anomalias_data["Data_dt"].notna(),
            "Data_dt"
        ].dt.month.unique().astype(int).tolist()

    meses_disponiveis = ["Todos os meses"] + sorted(meses_com_anomalias)

    with col_mes:
        mes_selecionado = st.selectbox(
            "Mês",
            options=meses_disponiveis,
            index=0,
            format_func=lambda mes: (
                mes if isinstance(mes, str) else nomes_meses.get(int(mes), str(mes))
            )
        )

    if mes_selecionado is None:
        st.warning("Não há meses disponíveis para o ano selecionado.")
        st.stop()

    dados_mes = dados_completos.copy()

    if ano_selecionado != "Todos os anos":
        dados_mes = dados_mes[dados_mes["Ano"] == ano_selecionado]

    if mes_selecionado != "Todos os meses":
        dados_mes = dados_mes[dados_mes["Mes_num"] == int(mes_selecionado)]

    mes_texto = (
        "Todos os meses"
        if mes_selecionado == "Todos os meses"
        else f"{int(mes_selecionado):02d}"
    )


    total_observacoes = len(dados_mes)
    total_anomalias = int(dados_mes["Anomalia"].fillna(False).astype(bool).sum())
    percentual = total_anomalias / total_observacoes * 100 if total_observacoes else 0
    c1, c3 = st.columns(2, gap="small")


    c1.metric("Observações analisadas", f"{total_observacoes:,}".replace(",", "."))
    c3.metric("Proporção de anomalias", f"{percentual:.2f}%")

# Filtra a lista de anomalias conforme os filtros de ano e mês da visão geral.
dados_filtrados = dados.copy()
dados_filtrados["Data_dt"] = pd.to_datetime(dados_filtrados["Data"], errors="coerce")

if ano_selecionado != "Todos os anos":
    dados_filtrados = dados_filtrados[
        dados_filtrados["Data_dt"].dt.year == ano_selecionado
    ]

if mes_selecionado != "Todos os meses":
    dados_filtrados = dados_filtrados[
        dados_filtrados["Data_dt"].dt.month == int(mes_selecionado)
    ]

st.divider()
st.subheader("Anomalias identificadas")
st.caption(
    "Explore os outliers encontrados. Selecione uma linha da tabela para abrir "
    "a análise detalhada da observação."
)

colunas_tabela = ["Data", "Mês", "Score LOF"]
tabela_outliers = dados_filtrados[colunas_tabela].copy()
tabela_outliers["Data"] = tabela_outliers["Data"].astype(str)
tabela_outliers = tabela_outliers.sort_values("Score LOF", ascending=False).reset_index(drop=True)
config_tabela_outliers = {
    "Data": st.column_config.TextColumn("Data", width="small"),
    "Mês": st.column_config.NumberColumn("Mês", width="small", format="%d"),
    "Score LOF": st.column_config.NumberColumn("Score LOF", width="small", format="%.4f"),
}

# Centraliza a tabela em uma área mais estreita para evitar que ela ocupe a página inteira.
espaco_esquerda, area_tabela, espaco_direita = st.columns([1, 2.4, 1])
with area_tabela:
    selecao_tabela = st.dataframe(
        tabela_outliers,
        column_config=config_tabela_outliers,
        hide_index=True,
        width="stretch",
        on_select="rerun",
        selection_mode="single-row",
        key="tabela_outliers"
    )

linhas_selecionadas = selecao_tabela.selection.rows
if linhas_selecionadas:
    indice_selecionado = linhas_selecionadas[0]
    anomalia_selecionada = str(tabela_outliers.iloc[indice_selecionado]["Data"])
    dados_anomalia = dados_filtrados[
        dados_filtrados["Data"].astype(str) == anomalia_selecionada
    ]

    if not dados_anomalia.empty:
        anomalia = dados_anomalia.iloc[0]
        st.divider()
        st.subheader(f"Análise da anomalia · {anomalia_selecionada}")

        metrica_data, metrica_score = st.columns(2, gap="small")
        with metrica_data:
            st.metric("Data selecionada", anomalia_selecionada)
        with metrica_score:
            st.metric("Score LOF", f'{anomalia["Score LOF"]:.3f}')

        dados_comparacao = dados_mes
        if mes_selecionado == "Todos os meses":
            mes_anomalia = int(anomalia["Mês"])
            dados_comparacao = dados_mes[dados_mes["Mes_num"] == mes_anomalia]

        comparacao = comparar_com_vizinhos(
            dados_comparacao, anomalia_selecionada, variaveis_disponiveis, k=20
        )

        if comparacao is None:
            st.warning(
                "Não foi possível calcular a comparação. Verifique se a observação "
                "tem valores válidos nas variáveis meteorológicas."
            )
        else:
            tabela = comparacao.copy()
            numericas = [
                "Valor observado", "Média dos vizinhos",
                "Diferença", "Diferença padronizada"
            ]
            tabela[numericas] = tabela[numericas].round(2)
            tabela["Magnitude do desvio"] = tabela["Diferença padronizada"].abs()
            tabela = tabela.sort_values("Magnitude do desvio", ascending=False)

            st.markdown("#### Tabela detalhada dos desvios")
            st.dataframe(
                tabela.drop(columns=["Magnitude do desvio"]),
                hide_index=True,
                width="stretch",
                height=430
            )

            st.caption(
                "Diferenças positivas indicam valores acima da média dos vizinhos; "
                "diferenças negativas indicam valores abaixo. "
                "A diferença padronizada é descritiva e não confirma, isoladamente, "
                "um evento meteorológico extremo."
            )

        st.divider()
        st.markdown("#### Relação entre variáveis meteorológicas")
        amostra_grafico = anomalia_selecionada
        cx, cy = st.columns(2, gap="small")
        with cx:
            indice_x = (
                variaveis_disponiveis.index("Temperatura máxima")
                if "Temperatura máxima" in variaveis_disponiveis else 0
            )
            variavel_x = st.selectbox(
                "Variável do eixo X",
                options=variaveis_disponiveis,
                index=indice_x,
                key="detalhe_variavel_x"
            )

        opcoes_y = [v for v in variaveis_disponiveis if v != variavel_x]
        with cy:
            variavel_y = st.selectbox(
                "Variável do eixo Y",
                options=opcoes_y,
                index=min(5, len(opcoes_y) - 1),
                key="detalhe_variavel_y"
            )

        dados_grafico = dados_mes.dropna(subset=[variavel_x, variavel_y]).copy()
        if dados_grafico.empty:
            st.warning("Não há registros suficientes para comparar essas variáveis.")
        else:
            # Outliers do LOF ficam vermelhos; a anomalia selecionada ganha uma cor própria.
            dados_grafico["Data_texto"] = dados_grafico["Data"].astype(str)
            dados_grafico["Grupo_visual"] = dados_grafico["Anomalia"].map({
                False: "Observação normal",
                True: "Outlier LOF"
            })
            dados_grafico.loc[
                dados_grafico["Data_texto"] == anomalia_selecionada,
                "Grupo_visual"
            ] = "Anomalia em investigação"

            fig = px.scatter(
                dados_grafico,
                x=variavel_x,
                y=variavel_y,
                hover_data=["Data", "Score LOF"],
                color="Grupo_visual",
                color_discrete_map={
                    "Observação normal": "#78BDE8",
                    "Outlier LOF": "#FF4D4D",
                    "Anomalia em investigação": "#E879FF"
                },
                category_orders={
                    "Grupo_visual": [
                        "Observação normal",
                        "Outlier LOF",
                        "Anomalia em investigação"
                    ]
                },
                title=f"{variavel_y} × {variavel_x}",
                labels={
                    variavel_x: variavel_x,
                    variavel_y: variavel_y,
                    "Grupo_visual": "Tipo de observação"
                },
                template="plotly_dark"
            )
            fig.update_layout(
                paper_bgcolor="#0B1F33",
                plot_bgcolor="#153A5B",
                font=dict(family="Arial", color="#FFFFFF", size=13),
                title=dict(font=dict(size=20), x=0),
                legend_title_text="Classificação",
                margin=dict(l=20, r=20, t=60, b=20),
                xaxis=dict(showgrid=True, gridcolor="#285477", zeroline=False),
                yaxis=dict(showgrid=True, gridcolor="#285477", zeroline=False)
            )

            # A amostra em investigação já tem uma categoria e cor próprias no gráfico.
            st.plotly_chart(fig, width="stretch")

        with st.expander("Nota metodológica", expanded=False):
            st.caption(
                "Uma observação classificada como anômala pelo LOF apresenta um padrão "
                "estatisticamente incomum em relação aos vizinhos considerados pelo algoritmo. "
                "Isso não significa, por si só, que seja um erro de medição ou um evento "
                "meteorológico extremo."
            )
else:
    st.info("Clique em uma linha da tabela para visualizar a análise detalhada da anomalia.")
