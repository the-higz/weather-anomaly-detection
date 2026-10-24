import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.neighbors import NearestNeighbors

VARIAVEIS = [
    "Precipitação", "Pressão", "Ponto de orvalho", "Temperatura máxima",
    "Temperatura média", "Temperatura mínima", "Umidade média",
    "Umidade mínima", "Rajada máxima", "Velocidade média do vento",
]
NOMES_MESES = {
    1: "Janeiro", 2: "Fevereiro", 3: "Março", 4: "Abril",
    5: "Maio", 6: "Junho", 7: "Julho", 8: "Agosto",
    9: "Setembro", 10: "Outubro", 11: "Novembro", 12: "Dezembro",
}


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
    """Compara uma observação com vizinhos do mesmo grupo mensal, de forma descritiva."""
    validos = dados_mes.copy()
    validos["Data_comparacao"] = pd.to_datetime(
        validos["Data"], errors="coerce"
    ).dt.normalize()
    for variavel in variaveis:
        validos[variavel] = pd.to_numeric(validos[variavel], errors="coerce")
    validos = validos.dropna(subset=variaveis + ["Data_comparacao"]).reset_index(drop=True)

    data_alvo = pd.to_datetime(data_anomalia, errors="coerce")
    if pd.isna(data_alvo):
        return None
    data_alvo = data_alvo.normalize()
    correspondencias = validos.index[
        validos["Data_comparacao"] == data_alvo
    ].tolist()
    if not correspondencias:
        return None

    indice = correspondencias[0]
    valores = validos[variaveis].astype(float).to_numpy()
    if len(valores) < 2:
        return None

    desvios = valores.std(axis=0)
    desvios[desvios == 0] = 1.0
    normalizados = (valores - valores.mean(axis=0)) / desvios
    n_vizinhos = min(k, len(valores) - 1)
    modelo = NearestNeighbors(n_neighbors=n_vizinhos + 1)
    modelo.fit(normalizados)
    _, indices = modelo.kneighbors(normalizados[indice].reshape(1, -1))
    vizinhos = [i for i in indices[0] if i != indice][:n_vizinhos]
    if not vizinhos:
        return None

    observado = valores[indice]
    media_vizinhos = valores[vizinhos].mean(axis=0)
    diferenca = observado - media_vizinhos
    return pd.DataFrame({
        "Variável": variaveis,
        "Valor observado": observado,
        "Média dos vizinhos": media_vizinhos,
        "Diferença": diferenca,
        "Diferença padronizada": diferenca / desvios,
    })


def selecionar_todos_meses_temporais():
    st.session_state["meses_temporais_selecionados"] = list(NOMES_MESES.values())


def limpar_meses_temporais():
    st.session_state["meses_temporais_selecionados"] = []


st.set_page_config(
    page_title="Análise de Anomalias Meteorológicas",
    page_icon="🌦️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0B1F33;
        color: #FFFFFF;
        font-family: "Segoe UI", "Inter", "Roboto", sans-serif;
    }
    html, body, [class*="css"] {
        font-family: "Segoe UI", "Inter", "Roboto", sans-serif;
    }
    h1, h2, h3 {
        color: #F4F8FC;
        letter-spacing: -0.025em;
        font-weight: 650;
    }
    h1 { font-size: 2.15rem; line-height: 1.15; }
    h2 { font-size: 1.55rem; }
    h3 { font-size: 1.15rem; }
    p, label, [data-testid="stCaption"] {
        color: #D4DDE5;
        line-height: 1.5;
    }
    [data-testid="stMetric"] {
        background: linear-gradient(145deg, #183F60 0%, #153653 100%);
        border: 1px solid rgba(77, 163, 255, 0.30);
        border-radius: 12px;
        padding: 14px 16px;
        min-height: 88px;
    }
    [data-testid="stMetricLabel"] {
        color: #C8D8E8;
        font-size: 0.82rem;
        font-weight: 500;
    }
    [data-testid="stMetricValue"] {
        color: #FFFFFF;
        font-size: 1.8rem;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    div[data-baseweb="select"] > div {
        background-color: #153A5B;
        border-radius: 9px;
    }
    div[data-baseweb="select"] * { font-family: inherit; }
    [data-testid="stSidebar"] {
        background-color: #102A43;
        border-right: 1px solid #285477;
    }
    [data-testid="stTabs"] button { font-weight: 600; }
    hr { border-color: #285477; }
    @media (max-width: 900px) {
        [data-testid="stMetric"] { padding: 10px 12px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

try:
    dados_anomalias = carregar_resultados()
    dados_completos = carregar_dados_completos()
    resumo_temporal = carregar_resumo_temporal()
except FileNotFoundError as erro:
    st.error(
        "Não encontrei uma das planilhas necessárias. "
        "Execute `python src/main.py` para gerar os arquivos Excel."
    )
    st.caption(f"Detalhe: {erro}")
    st.stop()

if dados_anomalias.empty or dados_completos.empty:
    st.warning("Não há dados disponíveis para exibir.")
    st.stop()

if not {"Mês", "Data", "Score LOF"}.issubset(dados_anomalias.columns):
    st.error("A planilha resultados_anomalias.xlsx não contém todas as colunas necessárias.")
    st.stop()

if not {"Mês", "Data", "Score LOF", "Anomalia"}.issubset(dados_completos.columns):
    st.error("A planilha dados_dashboard.xlsx não contém todas as colunas necessárias.")
    st.stop()

colunas_temporais = {
    "Ano", "Mês", "Observações analisadas", "Anomalias",
    "Percentual de anomalias (%)",
}
if not colunas_temporais.issubset(resumo_temporal.columns):
    st.error("A planilha resumo_temporal.xlsx não contém todas as colunas necessárias.")
    st.stop()

variaveis_disponiveis = [v for v in VARIAVEIS if v in dados_completos.columns]
if len(variaveis_disponiveis) < 2:
    st.error("São necessárias pelo menos duas variáveis meteorológicas disponíveis para a análise.")
    st.stop()

dados_completos = dados_completos.copy()
dados_completos["Data"] = pd.to_datetime(dados_completos["Data"], errors="coerce")
dados_completos["Ano"] = dados_completos["Data"].dt.year
dados_completos["Mes_num"] = pd.to_numeric(
    dados_completos["Mês"], errors="coerce"
).astype("Int64")

dados_anomalias = dados_anomalias.copy()
dados_anomalias["Data_dt"] = pd.to_datetime(dados_anomalias["Data"], errors="coerce")
anos_disponiveis = ["Todos os anos"] + sorted(
    dados_completos.loc[dados_completos["Ano"].notna(), "Ano"]
    .unique().astype(int).tolist()
)

with st.sidebar:
    st.caption("INICIAÇÃO CIENTÍFICA")
    st.header("Filtros")
    st.write("Escolha o período para explorar as observações.")
    ano_selecionado = st.selectbox("Ano", options=anos_disponiveis)

    dados_para_meses = dados_completos
    if ano_selecionado != "Todos os anos":
        dados_para_meses = dados_para_meses[
            dados_para_meses["Ano"] == ano_selecionado
        ]
    meses = sorted(
        dados_para_meses["Mes_num"].dropna().astype(int).unique().tolist()
    )

    mes_selecionado = st.selectbox(
        "Mês",
        options=["Todos os meses"] + meses,
        format_func=lambda mes: (
            mes if isinstance(mes, str) else NOMES_MESES.get(int(mes), str(mes))
        ),
    )
    st.divider()
    st.caption(
        "Os filtros incluem todos os períodos com observações, mesmo quando "
        "nenhuma anomalia foi identificada neles."
    )
    st.caption(
        "O LOF identifica observações incomuns em relação aos vizinhos; "
        "isso não confirma, isoladamente, um evento meteorológico extremo."
    )

dados_filtrados = dados_completos.copy()
if ano_selecionado != "Todos os anos":
    dados_filtrados = dados_filtrados[dados_filtrados["Ano"] == ano_selecionado]
if mes_selecionado != "Todos os meses":
    dados_filtrados = dados_filtrados[
        dados_filtrados["Mes_num"] == int(mes_selecionado)
    ]

anomalias_filtradas = dados_anomalias.copy()
if ano_selecionado != "Todos os anos":
    anomalias_filtradas = anomalias_filtradas[
        anomalias_filtradas["Data_dt"].dt.year == ano_selecionado
    ]
if mes_selecionado != "Todos os meses":
    anomalias_filtradas = anomalias_filtradas[
        anomalias_filtradas["Data_dt"].dt.month == int(mes_selecionado)
    ]

st.caption("INICIAÇÃO CIENTÍFICA · ANÁLISE METEOROLÓGICA")
st.title("🌦️ Monitoramento de Anomalias Meteorológicas")
st.markdown("Estação meteorológica de Catalão, Goiás")
st.caption(
    "🧪 **Versão experimental:** filtros, comparações com vizinhos e métricas "
    "complementares estão em avaliação; confira os resultados antes de utilizá-los "
    "como conclusão da pesquisa."
)
st.divider()

aba_geral, aba_investigacao, aba_meteorologia, aba_evolucao, aba_sobre = st.tabs([
    "📊 Visão geral",
    "🔎 Investigar anomalias",
    "🌡️ Análise meteorológica",
    "📈 Evolução temporal",
    "ℹ️ Sobre o projeto",
])

with aba_geral:
    st.subheader("Visão geral do período selecionado")
    total_observacoes = len(dados_filtrados)
    total_anomalias = int(
        dados_filtrados["Anomalia"].fillna(False).astype(bool).sum()
    )
    percentual = total_anomalias / total_observacoes * 100 if total_observacoes else 0

    m1, m2, m3 = st.columns(3, gap="small")
    m1.metric("Observações analisadas", f"{total_observacoes:,}".replace(",", "."))
    m2.metric("Anomalias identificadas", f"{total_anomalias:,}".replace(",", "."))
    m3.metric("Proporção de anomalias", f"{percentual:.2f}%")
    st.caption(
        "A proporção é a fração de registros classificados como anomalias "
        "no período filtrado, não a probabilidade de um evento extremo."
    )

    if not dados_filtrados.empty:
        resumo_ano = (
            dados_filtrados.dropna(subset=["Data"])
            .assign(Ano_grafico=lambda df: df["Data"].dt.year)
            .groupby("Ano_grafico", as_index=False)
            .agg(
                Observações=("Anomalia", "size"),
                Anomalias=(
                    "Anomalia",
                    lambda s: s.fillna(False).astype(bool).sum(),
                ),
            )
        )
        if not resumo_ano.empty:
            resumo_long = resumo_ano.melt(
                id_vars="Ano_grafico",
                value_vars=["Observações", "Anomalias"],
                var_name="Tipo",
                value_name="Quantidade",
            )
            fig_ano = px.bar(
                resumo_long,
                x="Ano_grafico",
                y="Quantidade",
                color="Tipo",
                barmode="group",
                text_auto=True,
                title="Observações e anomalias por ano",
                labels={
                    "Ano_grafico": "Ano",
                    "Quantidade": "Quantidade de registros",
                    "Tipo": "Série",
                },
                template="plotly_dark",
                color_discrete_map={
                    "Observações": "#78BDE8",
                    "Anomalias": "#FF6B6B",
                },
            )
            fig_ano.update_traces(textposition="outside", cliponaxis=False)
            fig_ano.update_layout(
                paper_bgcolor="#0B1F33",
                plot_bgcolor="#153A5B",
                margin=dict(l=15, r=15, t=60, b=15),
                legend_title_text="",
            )
            st.plotly_chart(fig_ano, width="stretch")

            # A cobertura é calculada no conjunto completo para não confundir
            # um filtro de mês com falta de dados do ano inteiro.
            anos_exibidos = set(resumo_ano["Ano_grafico"].astype(int).tolist())
            cobertura_por_ano = (
                dados_completos.dropna(subset=["Data"])
                .groupby("Ano")["Mes_num"]
                .agg(lambda serie: sorted(set(serie.dropna().astype(int).tolist())))
            )
            anos_parciais = []
            for ano in sorted(anos_exibidos):
                meses_ano = cobertura_por_ano.get(ano, [])
                if 0 < len(meses_ano) < 12:
                    faixa = f"{NOMES_MESES[meses_ano[0]][:3].lower()}–{NOMES_MESES[meses_ano[-1]][:3].lower()}"
                    anos_parciais.append(f"{ano} ({faixa}; {len(meses_ano)}/12 meses)")
            if anos_parciais:
                st.caption(
                    "⚠️ **Cobertura parcial:** " + ", ".join(anos_parciais) + ". "
                    "Os totais desses anos não são diretamente comparáveis aos anos "
                    "com 12 meses representados."
                )

        st.markdown("#### Distribuição mensal das anomalias")
        distribuicao = (
            dados_filtrados.assign(
                Mes_nome=dados_filtrados["Mes_num"].map(NOMES_MESES)
            )
            .groupby(["Mes_num", "Mes_nome"], dropna=True, as_index=False)
            .agg(
                Observações=("Anomalia", "size"),
                Anomalias=(
                    "Anomalia",
                    lambda s: s.fillna(False).astype(bool).sum(),
                ),
            )
            .sort_values("Mes_num")
        )
        if not distribuicao.empty:
            distribuicao["Percentual de anomalias (%)"] = (
                distribuicao["Anomalias"] / distribuicao["Observações"] * 100
            )
            tipo_distribuicao = st.radio(
                "Indicador do gráfico mensal",
                options=["Quantidade de anomalias", "Percentual de anomalias"],
                horizontal=True,
                key="indicador_distribuicao_mensal",
            )
            if tipo_distribuicao == "Quantidade de anomalias":
                coluna_y = "Anomalias"
                distribuicao["Texto_barra"] = distribuicao["Anomalias"].astype(int).astype(str)
                titulo_mes = "Quantidade de anomalias por mês"
                rotulo_y = "Anomalias identificadas"
            else:
                coluna_y = "Percentual de anomalias (%)"
                distribuicao["Texto_barra"] = distribuicao[coluna_y].map(lambda valor: f"{valor:.2f}%")
                titulo_mes = "Percentual de anomalias por mês"
                rotulo_y = "Anomalias (% das observações do mês)"

            fig_mes = px.bar(
                distribuicao,
                x="Mes_nome",
                y=coluna_y,
                text="Texto_barra",
                title=titulo_mes,
                labels={
                    "Mes_nome": "Mês",
                    coluna_y: rotulo_y,
                    "Observações": "Observações analisadas",
                    "Anomalias": "Anomalias identificadas",
                },
                hover_data=["Observações", "Anomalias", "Percentual de anomalias (%)"],
                template="plotly_dark",
                color_discrete_sequence=["#78BDE8"],
            )
            fig_mes.update_traces(textposition="outside", cliponaxis=False)
            fig_mes.update_layout(
                paper_bgcolor="#0B1F33",
                plot_bgcolor="#153A5B",
                margin=dict(l=15, r=15, t=60, b=15),
            )
            st.plotly_chart(fig_mes, width="stretch")
            st.caption(
                "O percentual mensal é calculado como anomalias ÷ total de observações "
                "do mês no período filtrado. Veja também o número de observações no hover."
            )
    else:
        st.info("Não há observações para os filtros selecionados.")

with aba_investigacao:
    st.subheader("Anomalias identificadas")

    st.caption("Selecione uma linha para consultar os detalhes da observação.")
    colunas_tabela = ["Data", "Mês", "Score LOF"]
    tabela_outliers = anomalias_filtradas[colunas_tabela].copy()
    tabela_outliers["Data"] = tabela_outliers["Data"].astype(str)
    tabela_outliers = tabela_outliers.sort_values(
        "Score LOF", ascending=False
    ).reset_index(drop=True)

    config_tabela = {
        "Data": st.column_config.TextColumn("Data", width="medium"),
        "Mês": st.column_config.NumberColumn("Mês", width="small", format="%d"),
        "Score LOF": st.column_config.NumberColumn(
            "Score LOF", width="medium", format="%.4f"
        ),
    }

    if tabela_outliers.empty:
        st.info("Não foram encontradas anomalias para os filtros selecionados.")
    else:
        _, area_tabela, _ = st.columns([1, 3, 1])
        with area_tabela:
            selecao = st.dataframe(
                tabela_outliers,
                column_config=config_tabela,
                hide_index=True,
                width="stretch",
                on_select="rerun",
                selection_mode="single-row",
                # A chave muda com os filtros para não reutilizar uma seleção
                # antiga quando a tabela é atualizada.
                key=f"tabela_outliers_{ano_selecionado}_{mes_selecionado}",
            )

        if selecao.selection.rows:
            idx = selecao.selection.rows[0]
            # Protege contra índices de seleção antigos após uma mudança nos filtros.
            if 0 <= idx < len(tabela_outliers):
                data_selecionada = str(tabela_outliers.iloc[idx]["Data"])
                registro = anomalias_filtradas[
                    anomalias_filtradas["Data"].astype(str) == data_selecionada
                ]
            else:
                registro = pd.DataFrame()
                st.info("A tabela foi atualizada. Selecione novamente uma anomalia.")
            if not registro.empty:
                anomalia = registro.iloc[0]
                st.divider()
                st.markdown(f"### Análise da anomalia · {data_selecionada}")
                c_data, c_score = st.columns(2, gap="small")
                c_data.metric("Data selecionada", data_selecionada)
                c_score.metric("Score LOF", f'{anomalia["Score LOF"]:.3f}')

                mes_anomalia = int(pd.to_datetime(data_selecionada).month)
                # O LOF agrupou o histórico por mês do calendário, juntando anos.
                # A análise detalhada usa o mesmo universo de referência.
                dados_comparacao = dados_completos[
                    dados_completos["Mes_num"] == mes_anomalia
                ]
                st.caption(
                    "A comparação usa observações do mesmo mês do calendário, "
                    "reunindo todos os anos disponíveis, para manter a lógica "
                    "coerente com o agrupamento usado no LOF."
                )
                st.caption(
                    "O score LOF não é uma probabilidade nem uma medida direta "
                    "da gravidade meteorológica; indica uma diferença de densidade local."
                )
                comparacao = comparar_com_vizinhos(
                    dados_comparacao,
                    data_selecionada,
                    variaveis_disponiveis,
                    k=20,
                )

                if comparacao is None:
                    st.warning(
                        "Não foi possível calcular a comparação. Verifique se "
                        "a observação possui valores válidos nas variáveis."
                    )
                else:
                    tabela = comparacao.copy()
                    numericas = [
                        "Valor observado", "Média dos vizinhos",
                        "Diferença", "Diferença padronizada",
                    ]
                    tabela[numericas] = tabela[numericas].round(2)
                    tabela["Magnitude do desvio"] = tabela[
                        "Diferença padronizada"
                    ].abs()
                    tabela = tabela.sort_values(
                        "Magnitude do desvio", ascending=False
                    )
                    st.markdown("#### Comparação com observações vizinhas")
                    st.dataframe(
                        tabela.drop(columns=["Magnitude do desvio"]),
                        hide_index=True,
                        width="stretch",
                        height=430,
                    )

                    st.markdown("#### Desvios em relação aos vizinhos")
                    grafico_desvios = tabela.copy()
                    grafico_desvios["Direção do desvio"] = grafico_desvios[
                        "Diferença padronizada"
                    ].apply(
                        lambda valor: (
                            "Acima da média dos vizinhos"
                            if valor > 0
                            else "Abaixo da média dos vizinhos"
                            if valor < 0
                            else "Igual à média dos vizinhos"
                        )
                    )
                    fig_desvios = px.bar(
                        grafico_desvios.sort_values(
                            "Diferença padronizada", ascending=True
                        ),
                        x="Diferença padronizada",
                        y="Variável",
                        orientation="h",
                        color="Direção do desvio",
                        color_discrete_map={
                            "Acima da média dos vizinhos": "#FF6B6B",
                            "Abaixo da média dos vizinhos": "#78BDE8",
                            "Igual à média dos vizinhos": "#B8C4D0",
                        },
                        title="Diferença padronizada em relação à média dos vizinhos",
                        labels={
                            "Diferença padronizada": "Desvio padronizado",
                            "Variável": "Variável meteorológica",
                            "Direção do desvio": "Direção",
                        },
                        template="plotly_dark",
                    )
                    fig_desvios.add_vline(
                        x=0,
                        line_width=1,
                        line_dash="dash",
                        line_color="#FFFFFF",
                    )
                    fig_desvios.update_layout(
                        paper_bgcolor="#0B1F33",
                        plot_bgcolor="#153A5B",
                        font=dict(family="Arial", color="#FFFFFF", size=12),
                        margin=dict(l=15, r=15, t=65, b=15),
                        legend_title_text="",
                    )
                    st.plotly_chart(fig_desvios, width="stretch")

                    st.caption(
                        "Diferenças positivas indicam valores acima da média dos "
                        "vizinhos; negativas indicam valores abaixo. Essa comparação "
                        "é descritiva e não confirma, isoladamente, um evento extremo."
                    )

                with st.expander("Nota metodológica", expanded=False):
                    st.write(
                        "O LOF identifica observações com densidade local diferente "
                        "da de seus vizinhos. Uma observação anômala não é necessariamente "
                        "um erro de medição nem um evento meteorológico extremo."
                    )
        else:
            st.info("Clique em uma linha da tabela para visualizar a análise detalhada.")

with aba_meteorologia:
    st.subheader("Relação entre variáveis meteorológicas")
    st.info(
        "🖱️ **Como usar a investigação automática:** clique em um ponto do gráfico "
        "para abrir a análise dessa observação logo abaixo. "
        "🔴 **Ponto vermelho:** anomalia identificada pelo LOF — clique para investigar. "
        "🔵 **Ponto azul:** observação classificada como normal. "
        "Para analisar outra observação, basta clicar em outro ponto."
    )
    col_x, col_y = st.columns(2, gap="small")
    indice_x = (
        variaveis_disponiveis.index("Temperatura máxima")
        if "Temperatura máxima" in variaveis_disponiveis else 0
    )
    with col_x:
        variavel_x = st.selectbox(
            "Variável do eixo X",
            variaveis_disponiveis,
            index=indice_x,
            key="meteorologia_variavel_x",
        )
    opcoes_y = [v for v in variaveis_disponiveis if v != variavel_x]
    with col_y:
        variavel_y = st.selectbox(
            "Variável do eixo Y",
            opcoes_y,
            index=min(5, len(opcoes_y) - 1),
            key="meteorologia_variavel_y",
        )

    grafico = dados_filtrados.dropna(subset=[variavel_x, variavel_y]).copy()
    if grafico.empty:
        st.warning("Não há registros suficientes para comparar essas variáveis.")
    else:
        grafico["Grupo_visual"] = (
            grafico["Anomalia"].fillna(False).astype(bool).map({
                False: "Observação normal",
                True: "Outlier LOF",
            })
        )
        fig = px.scatter(
            grafico,
            x=variavel_x,
            y=variavel_y,
            hover_data=["Data", "Score LOF"],
            custom_data=["Data"],
            color="Grupo_visual",
            color_discrete_map={
                "Observação normal": "#78BDE8",
                "Outlier LOF": "#FF6B6B",
            },
            title=f"{variavel_y} × {variavel_x}",
            labels={
                variavel_x: variavel_x,
                variavel_y: variavel_y,
                "Grupo_visual": "Tipo de observação",
            },
            template="plotly_dark",
        )
        fig.update_layout(
            paper_bgcolor="#0B1F33",
            plot_bgcolor="#153A5B",
            font=dict(family="Arial", color="#FFFFFF", size=13),
            title=dict(font=dict(size=20), x=0),
            legend_title_text="Classificação",
            margin=dict(l=20, r=20, t=60, b=20),
            xaxis=dict(showgrid=True, gridcolor="#285477", zeroline=False),
            yaxis=dict(showgrid=True, gridcolor="#285477", zeroline=False),
        )
        selecao_grafico = st.plotly_chart(
            fig,
            width="stretch",
            key="grafico_relacao_variaveis",
            on_select="rerun",
            selection_mode="points",
        )

        # Opção A: ao selecionar um ponto, a investigação aparece automaticamente
        # abaixo do gráfico, sem exigir navegação para outra aba.
        pontos_selecionados = selecao_grafico.selection.points
        if pontos_selecionados:
            ponto = pontos_selecionados[0]
            customdata = ponto.get("customdata", [])
            if customdata:
                st.session_state["anomalia_selecionada_data"] = str(customdata[0])

        data_selecionada_grafico = st.session_state.get("anomalia_selecionada_data")
        if data_selecionada_grafico:
            data_dt_selecionada = pd.to_datetime(
                data_selecionada_grafico, errors="coerce"
            )
            if pd.notna(data_dt_selecionada):
                registro_selecionado = dados_filtrados[
                    dados_filtrados["Data"].dt.normalize()
                    == data_dt_selecionada.normalize()
                ]
            else:
                registro_selecionado = pd.DataFrame()

            if not registro_selecionado.empty:
                observacao = registro_selecionado.iloc[0]
                data_formatada = observacao["Data"].strftime("%d/%m/%Y")
                eh_anomalia = bool(observacao["Anomalia"]) if pd.notna(
                    observacao["Anomalia"]
                ) else False

                st.divider()
                st.subheader(f"🔎 Investigação automática · {data_formatada}")
                st.caption(
                    "A análise corresponde ao ponto selecionado no gráfico. "
                    "Clique em outro ponto para atualizar este painel."
                )

                col_data, col_classificacao, col_score = st.columns(3, gap="small")
                col_data.metric("Data", data_formatada)
                col_classificacao.metric(
                    "Classificação", "🚨 Anomalia LOF" if eh_anomalia else "Observação normal"
                )
                score_observacao = observacao.get("Score LOF")
                col_score.metric(
                    "Score LOF",
                    f"{float(score_observacao):.4f}"
                    if pd.notna(score_observacao) else "Não disponível",
                )
                st.caption(
                    "O score LOF não é uma probabilidade de evento extremo nem uma "
                    "medida direta de gravidade. Ele se relaciona à densidade local "
                    "da observação em comparação com os vizinhos."
                )

                st.markdown("#### Valores meteorológicos observados")
                valores_observados = pd.DataFrame({
                    "Variável": variaveis_disponiveis,
                    "Valor observado": [observacao[v] for v in variaveis_disponiveis],
                })
                valores_observados["Valor observado"] = pd.to_numeric(
                    valores_observados["Valor observado"], errors="coerce"
                ).round(2)
                st.dataframe(
                    valores_observados,
                    hide_index=True,
                    width="stretch",
                    height=min(390, 38 * (len(valores_observados) + 1)),
                )

                if eh_anomalia:
                    st.markdown("#### Comparação com observações do mesmo mês")
                    # O modelo LOF foi calculado agrupando o mesmo mês do calendário
                    # ao longo dos anos. Por isso a comparação também usa esse grupo.
                    mes_observacao = int(observacao["Mes_num"])
                    dados_comparacao = dados_completos[
                        dados_completos["Mes_num"] == mes_observacao
                    ]
                    comparacao = comparar_com_vizinhos(
                        dados_comparacao,
                        observacao["Data"].strftime("%Y-%m-%d"),
                        variaveis_disponiveis,
                        k=20,
                    )

                    if comparacao is None:
                        st.warning(
                            "Não foi possível calcular a comparação com vizinhos. "
                            "A observação pode não ter valores válidos suficientes."
                        )
                    else:
                        tabela_comparacao = comparacao.copy()
                        numericas_comparacao = [
                            "Valor observado", "Média dos vizinhos", "Diferença",
                            "Diferença padronizada",
                        ]
                        tabela_comparacao[numericas_comparacao] = tabela_comparacao[
                            numericas_comparacao
                        ].round(2)
                        tabela_comparacao["Magnitude do desvio"] = tabela_comparacao[
                            "Diferença padronizada"
                        ].abs()
                        tabela_comparacao = tabela_comparacao.sort_values(
                            "Magnitude do desvio", ascending=False
                        )
                        st.dataframe(
                            tabela_comparacao.drop(columns=["Magnitude do desvio"]),
                            hide_index=True,
                            width="stretch",
                            height=430,
                        )

                        grafico_desvios_auto = tabela_comparacao.copy()
                        grafico_desvios_auto["Direção do desvio"] = grafico_desvios_auto[
                            "Diferença padronizada"
                        ].apply(
                            lambda valor: (
                                "Acima da média dos vizinhos" if valor > 0
                                else "Abaixo da média dos vizinhos" if valor < 0
                                else "Igual à média dos vizinhos"
                            )
                        )
                        fig_desvios_auto = px.bar(
                            grafico_desvios_auto.sort_values(
                                "Diferença padronizada", ascending=True
                            ),
                            x="Diferença padronizada",
                            y="Variável",
                            orientation="h",
                            color="Direção do desvio",
                            color_discrete_map={
                                "Acima da média dos vizinhos": "#FF6B6B",
                                "Abaixo da média dos vizinhos": "#78BDE8",
                                "Igual à média dos vizinhos": "#B8C4D0",
                            },
                            title="Diferença padronizada em relação aos vizinhos",
                            labels={
                                "Diferença padronizada": "Desvio padronizado",
                                "Variável": "Variável meteorológica",
                                "Direção do desvio": "Direção",
                            },
                            template="plotly_dark",
                        )
                        fig_desvios_auto.add_vline(
                            x=0, line_width=1, line_dash="dash", line_color="#FFFFFF"
                        )
                        fig_desvios_auto.update_layout(
                            paper_bgcolor="#0B1F33",
                            plot_bgcolor="#153A5B",
                            font=dict(family="Arial", color="#FFFFFF", size=12),
                            margin=dict(l=15, r=20, t=65, b=15),
                            legend_title_text="",
                        )
                        st.plotly_chart(fig_desvios_auto, width="stretch")
                        st.caption(
                            "Diferenças positivas indicam valores acima da média dos vizinhos; "
                            "negativas indicam valores abaixo. Essa comparação é descritiva "
                            "e não confirma, isoladamente, um evento meteorológico extremo."
                        )
                else:
                    st.info(
                        "Este ponto foi classificado pelo LOF como uma observação normal. "
                        "Selecione um ponto vermelho para investigar uma anomalia."
                    )
            else:
                st.info(
                    "A observação selecionada não está nos filtros atuais. "
                    "Selecione um ponto do gráfico para atualizar a investigação."
                )

with aba_evolucao:
    st.subheader("📈 Evolução temporal das anomalias")
    st.write(
        "Explore como a proporção de observações classificadas como anômalas "
        "varia entre anos e meses."
    )
    st.caption(
        "Os gráficos usam o histórico completo, independentemente dos filtros. "
        "O percentual representa classificações do LOF, não a probabilidade "
        "de ocorrência de eventos meteorológicos extremos."
    )
    st.caption(
        "🧪 Os controles de visualização e as comparações com vizinhos são recursos "
        "experimentais de exploração; eles não recalculam nem alteram as classificações do LOF."
    )

    resumo_graficos = resumo_temporal.copy()
    resumo_graficos["Ano"] = pd.to_numeric(
        resumo_graficos["Ano"], errors="coerce"
    )
    resumo_graficos["Mês"] = pd.to_numeric(
        resumo_graficos["Mês"], errors="coerce"
    )
    resumo_graficos["Percentual de anomalias (%)"] = pd.to_numeric(
        resumo_graficos["Percentual de anomalias (%)"], errors="coerce"
    )
    resumo_graficos = resumo_graficos.dropna(
        subset=["Ano", "Mês", "Percentual de anomalias (%)"]
    )
    resumo_graficos["Ano"] = resumo_graficos["Ano"].astype(int)
    resumo_graficos["Mês"] = resumo_graficos["Mês"].astype(int)
    resumo_graficos["Mês nome"] = resumo_graficos["Mês"].map(NOMES_MESES)

    if not resumo_graficos.empty:
        fig_temporal = px.line(
            resumo_graficos.sort_values(["Ano", "Mês"]),
            x="Ano",
            y="Percentual de anomalias (%)",
            color="Mês nome",
            # Mantém os meses na ordem cronológica, de janeiro a dezembro,
            # em vez da ordenação alfabética padrão do Plotly.
            category_orders={"Mês nome": list(NOMES_MESES.values())},
            markers=True,
            title="Percentual de anomalias ao longo dos anos",
            labels={
                "Ano": "Ano",
                "Percentual de anomalias (%)": "Anomalias (%)",
                "Mês nome": "Mês",
            },
            template="plotly_dark",
            hover_data=["Mês", "Observações analisadas", "Anomalias"],
        )
        st.markdown("**Meses exibidos no gráfico**")
        col_todos, col_limpar, col_instrucao = st.columns([1, 1, 2])
        with col_todos:
            st.button(
                "✅ Selecionar todos",
                key="selecionar_todos_meses_temporais",
                on_click=selecionar_todos_meses_temporais,
                width="stretch",
            )
        with col_limpar:
            st.button(
                "🧹 Limpar seleção",
                key="limpar_meses_temporais",
                on_click=limpar_meses_temporais,
                width="stretch",
            )
        with col_instrucao:
            st.caption("Você também pode escolher meses específicos abaixo.")
        meses_visiveis = st.multiselect(
            "Selecione um ou mais meses",
            options=list(NOMES_MESES.values()),
            key="meses_temporais_selecionados",
        )

        # Sem seleção, as linhas ficam ocultas como na versão anterior.
        # O botão Selecionar todos liga todas as séries de janeiro a dezembro.
        fig_temporal.for_each_trace(
            lambda trace: trace.update(
                visible=True if trace.name in meses_visiveis else "legendonly"
            )
        )

        fig_temporal.update_layout(
            paper_bgcolor="#0B1F33",
            plot_bgcolor="#153A5B",
            margin=dict(l=15, r=15, t=65, b=15),
            legend_title_text="Clique nos meses para exibir/ocultar",
            xaxis=dict(dtick=1, showgrid=True, gridcolor="#285477"),
            yaxis=dict(showgrid=True, gridcolor="#285477"),
        )
        st.plotly_chart(fig_temporal, width="stretch")

        matriz_calor = resumo_graficos.pivot_table(
            index="Ano",
            columns="Mês",
            values="Percentual de anomalias (%)",
            aggfunc="mean",
        ).reindex(columns=range(1, 13))
        matriz_calor.columns = [
            NOMES_MESES.get(int(mes), str(mes))
            for mes in matriz_calor.columns
        ]
        fig_calor = px.imshow(
            matriz_calor,
            aspect="auto",
            color_continuous_scale="YlOrRd",
            labels={
                "x": "Mês",
                "y": "Ano",
                "color": "Anomalias (%)",
            },
            title="Mapa de calor do percentual de anomalias por ano e mês",
            template="plotly_dark",
            origin="lower",
        )
        fig_calor.update_layout(
            paper_bgcolor="#0B1F33",
            plot_bgcolor="#153A5B",
            margin=dict(l=15, r=15, t=65, b=15),
            xaxis=dict(side="bottom"),
            yaxis=dict(dtick=1, showgrid=False),
        )
        st.plotly_chart(fig_calor, width="stretch")
        st.caption(
            "Células sem cor indicam períodos sem dados no resumo. Como o número "
            "de observações pode variar entre meses, interprete percentuais junto "
            "da quantidade de observações de cada período."
        )



with aba_sobre:
    st.subheader("Sobre o projeto")
    st.markdown(
        """
        Este dashboard faz parte de um projeto de iniciação científica voltado
        à exploração de observações meteorológicas da estação de Catalão, Goiás.

        **Método de detecção:** Local Outlier Factor (LOF).

        O LOF identifica observações com densidade local diferente da de seus
        vizinhos. O resultado aponta padrões estatisticamente incomuns, mas não
        determina sozinho se houve erro de medição ou um evento meteorológico extremo.
        """
    )
    st.info(
        "A proporção de anomalias exibida corresponde à fração dos registros "
        "classificados pelo LOF no recorte selecionado. Não representa a "
        "probabilidade de ocorrência de um evento meteorológico extremo."
    )
