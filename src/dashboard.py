import streamlit as st
import pandas as pd
import plotly.express as px

@st.cache_data
def carregar_resultados():
    return pd.read_excel("resultados_anomalias.xlsx")


@st.cache_data
def carregar_dados_completos():
    return pd.read_excel("dados_dashboard.xlsx")


st.set_page_config(
    page_title="Análise de Anomalias Meteorológicas",
    page_icon="🌦️",
    layout="wide"
)

st.title("🌦️ Weather Anomaly Detection")

st.markdown(
    """
    **Monitoramento e análise de anomalias meteorológicas**

    Estação meteorológica de Catalão — GO

    Explore observações identificadas pelo algoritmo
    **Local Outlier Factor (LOF)**, compare variáveis meteorológicas
    e investigue padrões incomuns nos dados.
    """
)

st.divider()

dados = carregar_resultados()
dados_completos = carregar_dados_completos()

st.subheader("Anomalias identificadas")

meses = sorted(dados["Mês"].unique(), key=int)

mes_selecionado = st.selectbox(
    "Selecione o mês",
    meses,
    format_func=lambda mes: f"Mês {int(mes):02d}"
)

dados_filtrados = dados[dados["Mês"] == mes_selecionado]

dados_filtrados = dados_filtrados.sort_values(
    "Score LOF",
    ascending=False
).reset_index(drop=True)

anomalias = dados_filtrados["Data"].astype(str).tolist()

anomalia_selecionada = st.selectbox(
    "Selecione uma anomalia",
    anomalias
)

dados_anomalia = dados_filtrados[
    dados_filtrados["Data"].astype(str) == anomalia_selecionada
]

st.subheader("Anomalia selecionada")

if not dados_anomalia.empty:
    anomalia = dados_anomalia.iloc[0]

    col1, col2, col3 = st.columns(3)

    col1.metric("Data", anomalia["Data"])
    col2.metric("Score LOF", f'{anomalia["Score LOF"]:.3f}')
    col3.metric("Mês", f'{int(anomalia["Mês"]):02d}')

    with st.expander("Ver todas as variáveis da observação"):
        st.dataframe(dados_anomalia, width="stretch")

st.subheader("Visualização das variáveis meteorológicas")

variaveis = [
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

coluna_x, coluna_y = st.columns(2)

with coluna_x:
    variavel_x = st.selectbox(
        "Variável do eixo X",
        variaveis,
        index=3
    )

with coluna_y:
    variavel_y = st.selectbox(
        "Variável do eixo Y",
        [v for v in variaveis if v != variavel_x],
        index=5 if variavel_x == variaveis[6] else 6
    )

dados_mes_completo = dados_completos[
    dados_completos["Mês"].astype(str).str.zfill(2)
    == str(mes_selecionado).zfill(2)
]

fig = px.scatter(
    dados_mes_completo,
    x=variavel_x,
    y=variavel_y,
    hover_data=["Data", "Score LOF"],
    color="Anomalia",
    color_discrete_map={
        False: "steelblue",
        True: "orange"
    },
    title=f"{variavel_y} × {variavel_x}"
)

fig.add_scatter(
    x=dados_anomalia[variavel_x],
    y=dados_anomalia[variavel_y],
    mode="markers",
    name="Anomalia selecionada",
    marker=dict(
        color="red",
        size=14,
        symbol="x",
        line=dict(width=2)
    ),
    hovertemplate=(
        "Anomalia selecionada<br>"
        f"{variavel_x}: %{{x}}<br>"
        f"{variavel_y}: %{{y}}"
        "<extra></extra>"
    )
)

st.plotly_chart(fig, width="stretch")