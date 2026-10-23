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
    .stApp {background-color:#0B1F33;color:#FFFFFF;}
    [data-testid="stMetric"] {background-color:#153A5B;padding:18px;border-radius:12px;border:1px solid #285477;}
    h1,h2,h3 {color:#FFFFFF;} p,label {color:#D4DDE5;}
    div[data-baseweb="select"] > div {background-color:#153A5B;}
    hr {border-color:#285477;}
    </style>
    """, unsafe_allow_html=True)

st.caption("INICIAÇÃO CIENTÍFICA · ANÁLISE METEOROLÓGICA")
st.title("🌦️ Monitoramento de Anomalias Meteorológicas")
st.markdown("Investigação de padrões incomuns em observações meteorológicas da estação de Catalão, Goiás.\n\n**Método de detecção:** Local Outlier Factor (LOF)")
st.divider()


try:
    dados = carregar_resultados()
    dados_completos = carregar_dados_completos()

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


variaveis_disponiveis = [v for v in VARIAVEIS if v in dados_completos.columns]


if len(variaveis_disponiveis) < 2:
    st.error("São necessárias pelo menos duas variáveis meteorológicas disponíveis para a análise.")
    st.stop()


st.subheader("Visão geral da análise")
meses_disponiveis = dados_completos["Mês"].dropna().unique().tolist()


if not meses_disponiveis:
    st.warning("Não há meses disponíveis para análise.")
    st.stop()


meses = sorted(meses_disponiveis, key=int)
mes_selecionado = st.selectbox("Selecione o mês para investigar", options=meses, format_func=lambda mes: f"Mês {int(mes):02d}")


if mes_selecionado is None:
    st.warning("Selecione um mês válido.")
    st.stop()


mes_texto = f"{int(mes_selecionado):02d}"
dados_mes = dados_completos[dados_completos["Mês"].astype(str).str.zfill(2) == mes_texto].copy()


total_observacoes = len(dados_mes)
total_anomalias = int(dados_mes["Anomalia"].fillna(False).astype(bool).sum())
percentual = total_anomalias / total_observacoes * 100 if total_observacoes else 0
c1, c2, c3, c4 = st.columns(4)


c1.metric("Observações analisadas", f"{total_observacoes:,}".replace(",", "."))
c2.metric("Anomalias identificadas", f"{total_anomalias:,}".replace(",", "."))
c3.metric("Proporção de anomalias", f"{percentual:.2f}%")
c4.metric("Mês selecionado", mes_texto)
st.caption("Os indicadores representam os registros disponíveis no mês selecionado.")
st.divider()
st.subheader("Anomalias identificadas")


dados_filtrados = dados[dados["Mês"].astype(str).str.zfill(2) == mes_texto].copy()
dados_filtrados = dados_filtrados.sort_values("Score LOF", ascending=False).reset_index(drop=True)


if dados_filtrados.empty:
    st.info("Não há anomalias listadas para este mês.")
    st.stop()


anomalias = dados_filtrados["Data"].astype(str).tolist()
anomalia_selecionada = st.selectbox("Selecione uma anomalia", options=anomalias)
dados_anomalia = dados_filtrados[dados_filtrados["Data"].astype(str) == str(anomalia_selecionada)]


st.subheader("Anomalia selecionada")


if not dados_anomalia.empty:
    anomalia = dados_anomalia.iloc[0]

    c1, c2, c3 = st.columns(3)
    c1.metric("Data", str(anomalia["Data"]))
    c2.metric("Score LOF", f'{anomalia["Score LOF"]:.3f}')
    c3.metric("Mês", f'{int(anomalia["Mês"]):02d}')

    with st.expander("Ver todas as variáveis da observação"):
        st.dataframe(dados_anomalia, width="stretch", hide_index=True)

    st.divider()
    st.subheader("🔬 Análise comparativa da anomalia")
    st.markdown("Compare os valores observados com a média das observações mais próximas no espaço das variáveis meteorológicas normalizadas.")

    comparacao = comparar_com_vizinhos(dados_mes, anomalia_selecionada, variaveis_disponiveis, k=20)

    if comparacao is None:
        st.warning("Não foi possível calcular a comparação. Verifique se a observação tem valores válidos nas variáveis meteorológicas.")

    else:
        tabela = comparacao.copy()
        numericas = ["Valor observado", "Média dos vizinhos", "Diferença", "Diferença padronizada"]
        tabela[numericas] = tabela[numericas].round(2)
        tabela["Magnitude do desvio"] = tabela["Diferença padronizada"].abs()
        tabela = tabela.sort_values("Magnitude do desvio", ascending=False)

        st.markdown("#### Variáveis com maiores desvios")
        st.dataframe(tabela.drop(columns=["Magnitude do desvio"]), hide_index=True, width="stretch")

        fig_desvios = px.bar(tabela, x="Diferença padronizada", y="Variável", orientation="h", color="Diferença padronizada", color_continuous_scale=["#78BDE8", "#FFF0A6", "#F2C94C"], title="Desvios em relação à média dos vizinhos", labels={"Diferença padronizada":"Diferença padronizada", "Variável":"Variável meteorológica"}, template="plotly_dark")
        
        fig_desvios.update_layout(paper_bgcolor="#0B1F33", plot_bgcolor="#153A5B", font=dict(color="#FFFFFF"), coloraxis_colorbar=dict(title="Desvio"), xaxis=dict(zeroline=True, zerolinecolor="#FFFFFF", gridcolor="#285477"), yaxis=dict(autorange="reversed", gridcolor="#285477"))
        
        st.plotly_chart(fig_desvios, width="stretch")
        st.caption("Diferenças positivas indicam valores acima da média dos vizinhos; diferenças negativas indicam valores abaixo. A diferença padronizada divide o desvio pela variabilidade mensal da variável. Essas medidas são descritivas e não demonstram causalidade nem confirmam, isoladamente, um evento meteorológico extremo.")


st.divider()
st.subheader("Visualização das variáveis meteorológicas")


cx, cy = st.columns(2)


with cx:
    indice_x = variaveis_disponiveis.index("Temperatura máxima") if "Temperatura máxima" in variaveis_disponiveis else 0
    variavel_x = st.selectbox("Variável do eixo X", options=variaveis_disponiveis, index=indice_x)


opcoes_y = [v for v in variaveis_disponiveis if v != variavel_x]


with cy:
    variavel_y = st.selectbox("Variável do eixo Y", options=opcoes_y, index=min(5, len(opcoes_y)-1))
dados_grafico = dados_mes.dropna(subset=[variavel_x, variavel_y]).copy()


if dados_grafico.empty:
    st.warning("Não há registros suficientes para comparar essas variáveis no mês selecionado.")
    st.stop()


fig = px.scatter(dados_grafico, x=variavel_x, y=variavel_y, hover_data=["Data", "Score LOF"], color="Anomalia", color_discrete_map={False:"#78BDE8", True:"#F2C94C"}, title=f"{variavel_y} × {variavel_x}", labels={variavel_x:variavel_x, variavel_y:variavel_y, "Anomalia":"Classificada como anomalia"}, template="plotly_dark")
fig.update_layout(paper_bgcolor="#0B1F33", plot_bgcolor="#153A5B", font=dict(family="Arial", color="#FFFFFF", size=13), title=dict(font=dict(size=20), x=0), legend_title_text="Classificação", margin=dict(l=20,r=20,t=60,b=20), xaxis=dict(showgrid=True,gridcolor="#285477",zeroline=False), yaxis=dict(showgrid=True,gridcolor="#285477",zeroline=False))


if not dados_anomalia.empty:
    ponto = dados_anomalia.iloc[0]

    if variavel_x in ponto.index and variavel_y in ponto.index:
        fig.add_scatter(x=[ponto[variavel_x]], y=[ponto[variavel_y]], mode="markers", name="Anomalia selecionada", marker=dict(color="#FFF0A6",size=14,symbol="x",line=dict(width=2)), hovertemplate=f"Anomalia selecionada<br>{variavel_x}: %{{x}}<br>{variavel_y}: %{{y}}<extra></extra>")


st.plotly_chart(fig, width="stretch")
st.caption("Nota metodológica: uma observação classificada como anômala pelo LOF apresenta um padrão estatisticamente incomum em relação aos vizinhos considerados pelo algoritmo. Isso não significa, por si só, que seja um erro de medição ou um evento meteorológico extremo.")
