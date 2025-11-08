# =======================================================
# APP INSTITUCIONAL - PREVISÃO DE PREÇOS UBER
# NCIA / FPF TECH – Equipe A (Vesp.)
# =======================================================

import streamlit as st      # 👈 precisa estar aqui no topo
import pandas as pd
import numpy as np
import base64
from sklearn.ensemble import HistGradientBoostingRegressor
from projeto_uber_final import limparDados

# =======================================================
# APLICAR TEMA VISUAL FPF TECH / NCIA
# =======================================================
import base64

with open("fpf_theme.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# =======================================================
# FUNÇÃO PARA EXIBIR IMAGENS EMBUTIDAS (BASE64)
# =======================================================
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()


# =======================================================
# CABEÇALHO INSTITUCIONAL
# =======================================================
try:
    banner = get_base64_image("imagens/start.png")
    st.markdown(
        f"""
        <div style="text-align: center; margin-bottom: -2rem;">
            <img src="data:image/png;base64,{banner}" style="width:100%; border-radius:10px;">
        </div>
        """,
        unsafe_allow_html=True,
    )
except FileNotFoundError:
    st.warning("⚠️ Imagem de cabeçalho 'start.png' não encontrada na pasta 'imagens/'. Verifique o caminho.")


# =======================================================
# TÍTULO PRINCIPAL
# =======================================================
st.title("🚗 Previsão de Preços de Corridas Uber – NCIA / FPF Tech")
st.markdown(
    """
    <div style="color:#003366; font-weight:500; font-size:18px; margin-top:-10px;">
        <em>Projeto desenvolvido pela Equipe A (Vesp.) – FPF Tech / NCIA (2025)</em>
    </div>
    <hr>
    """,
    unsafe_allow_html=True,
)


# ===========================================
# CARREGAR DADOS (upload desaparece após carregar)
# ===========================================

# Usa session_state pra lembrar se já foi feito o upload
if "data_uploaded" not in st.session_state:
    st.session_state.data_uploaded = False
    st.session_state.df = None

# Se ainda não foi feito o upload → mostra o componente
if not st.session_state.data_uploaded:
    uploaded_file = st.file_uploader(
        "📂 Envie o dataset `rideshare_uber.csv` para iniciar a análise:",
        type=["csv"]
    )

    if uploaded_file is not None:
        # Lê e processa o dataset
        df = pd.read_csv(uploaded_file)
        df = limparDados(df)
        df = df[df["cab_type"].str.lower() == "uber"]

        # Armazena no session_state
        st.session_state.df = df
        st.session_state.data_uploaded = True

        # Mensagem de sucesso + força recarregamento
        st.success(f"✅ Dataset carregado com {df.shape[0]:,} registros.")
        st.rerun()  # 👈 forçar nova renderização
else:
    # Se já foi carregado → recupera o dataframe e pula upload
    df = st.session_state.df
    st.success(f"✅ Dataset carregado com {df.shape[0]:,} registros.")



# ===========================================
# ABAS PRINCIPAIS
# ===========================================
tabs = st.tabs(["📘 Introdução", "📊 Análise Exploratória", "🤖 Modelos", "💵 Simulador", "📈 Conclusões"])

# ===========================================
# 📘 INTRODUÇÃO
# ===========================================
with tabs[0]:
    st.header("Contexto e Motivação") 
    st.markdown("""
O crescimento dos serviços de mobilidade sob demanda, como a **Uber**, trouxe a necessidade
de **modelos de precificação dinâmica** baseados em dados.  
Contudo, essa variação em tempo real pode gerar **incerteza para clientes e motoristas**.""")
    st.image("imagens/uber_driver.webp", use_container_width=True)
    st.markdown("""
💡 Este projeto aplica **algoritmos de Machine Learning** para prever o preço das corridas,
buscando maior transparência e previsibilidade na precificação.

**Dataset:** *Uber Ride Analytics Dashboard* (Boston, EUA)  
**Tamanho:** ~148 mil corridas, 57 atributos  
**Variável alvo:** `price`
""")

    
    st.info("Este trabalho foi desenvolvido no âmbito da FPF Tech / NCIA, aplicando regressão supervisionada com foco em precificação urbana.")

# ===========================================
# 📊 ANÁLISE EXPLORATÓRIA
# ===========================================
with tabs[1]:
    st.header("Exploração de Dados (EDA)")
    st.markdown("""
O conjunto de dados contém informações de **preço, distância, tempo, tipo de corrida e clima**.  
A seguir, alguns padrões importantes identificados durante a análise:
""")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Distribuição de Preços")
        st.image("imagens/distribuicao_precos.png", use_container_width=True)
        st.caption("A maioria das corridas tem preço baixo, com poucos valores muito altos (distribuição assimétrica à direita).")

    with col2:
        st.subheader("Preço x Distância")
        st.image("imagens/preco_vs_distancia.png", use_container_width=True)
        st.caption("Correlação positiva: quanto maior a distância, maior o preço da corrida.")

    st.subheader("Matriz de Correlação das Variáveis Principais")
    st.image("imagens/matriz_correlacao.png", use_container_width=True)
    st.caption("O preço apresenta correlação positiva com distância e duração, e efeito moderado de `surge_multiplier` (demanda).")

    st.markdown("""
**Principais observações:**
- `distance` e `duration` correlacionam-se fortemente com `price`  
- `surge_multiplier` indica o efeito da alta demanda  
- `name_encoded` representa as categorias Uber (UberX, Black, etc.)
""")

# ===========================================
# 🤖 COMPARAÇÃO DE MODELOS
# ===========================================
with tabs[2]:
    st.header("Comparação de Modelos de Regressão")
    st.markdown("""
Foram testados diversos algoritmos de aprendizado supervisionado para prever o preço das corridas Uber.
A tabela e os gráficos abaixo apresentam as métricas de desempenho obtidas.
""")

    # --- TABELA DE RESULTADOS ---
    data = {
        "Modelo": [
            "Linear Regression", "Random Forest", "SVR",
            "AdaBoost", "HistGradientBoosting", "Bagging", "Stacking"
        ],
        "RMSE_CV": [2.4045, 2.2313, 2.0952, 4.3347, 1.8679, 1.9586, np.nan],
        "RMSE_test": [2.3944, 2.1744, 2.0292, 4.3274, 1.8483, 1.9474, 1.8486],
        "MAE_test": [1.6377, 1.4702, 1.2048, 3.3622, 1.1390, 1.2055, 1.1379],
        "R²_test": [0.9208, 0.9347, 0.9431, 0.7413, 0.9528, 0.9476, 0.9528],
    }
    df_models = pd.DataFrame(data)

    def highlight_best_model(row):
        """Destaque especial para o melhor modelo"""
        if row["Modelo"] == "HistGradientBoosting":
            return ['background-color: #FFF2CC; font-weight: bold; border: 2px solid #FFD966; color: #003366;'] * len(row)
        else:
            return ['color: #003366; background-color: #E6EEF7;'] * len(row)

    # --- EXIBIR TABELA ESTILIZADA ---
    st.dataframe(
        df_models.style
        .format(precision=4)
        .apply(highlight_best_model, axis=1)
        .set_table_styles([
            {"selector": "thead tr", "props": [
                ("background-color", "#FFD966"),
                ("color", "#003366"),
                ("font-weight", "700"),
                ("text-align", "center")
            ]},
            {"selector": "tbody td", "props": [
                ("text-align", "center"),
                ("font-weight", "500"),
                ("border", "1px solid #C5D4E4")
            ]},
            {"selector": "tbody tr:nth-child(odd)", "props": [("background-color", "#E6EEF7")]},
            {"selector": "tbody tr:nth-child(even)", "props": [("background-color", "#D4E4F4")]},
            {"selector": "tbody tr:hover", "props": [("background-color", "#FFF2CC")]}
        ])
    )

    # --- GRÁFICOS DE COMPARAÇÃO ---
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### Visualização Comparativa das Métricas")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Comparação de Performance — RMSE")
        st.image("imagens/comparacao_modelos_rmse.jpg", use_container_width=True)
    with col2:
        st.subheader("Comparação de Modelos — R²")
        st.image("imagens/comparacao_modelos_r2.jpeg", use_container_width=True)

    st.success("""
🏆 **Melhor modelo:** HistGradientBoosting Regressor  
R² ≈ 0.95 · RMSE ≈ 1.85 · MAE ≈ 1.13  
Desempenho consistente e superior entre todos os algoritmos testados.
""")



# ===========================================
# 💵 SIMULADOR DE PREÇOS
# ===========================================
with tabs[3]:
    st.header("Simulador de Preço de Corrida Uber")
    st.markdown("Insira os parâmetros para prever o valor estimado da corrida:")

    col1, col2, col3 = st.columns(3)
    dist = col1.number_input("Distância (milhas):", min_value=0.1, max_value=8.0, value=3.5)
    hora = col2.slider("Hora do dia:", 0, 23, 17)
    surge = 1 #col3.slider("Surge Multiplier (demanda):", 1.0, 3.0, 1.0, 0.1)
    servico = col3.selectbox("Tipo de Serviço Uber:", ['UberX','UberXL','Black','Select','WAV'])
    dur = (dist / 20) * 60  # duração estimada

    features = ["distance", "duration", "surge_multiplier", "hour"]
    model = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(df[features], df["price"])

    pred = model.predict(pd.DataFrame([[dist, dur, surge, hora]], columns=features))[0]
    st.success(f"💰 **Preço estimado: US$ {pred:.2f}**")

    st.info("ℹ️ O modelo utilizado é o **HistGradientBoosting Regressor**, o mais preciso entre todos os testados.")

# ===========================================
# 📈 CONCLUSÕES
# ===========================================
with tabs[4]:
    st.header("Conclusões e Impacto")
    st.markdown("""
O modelo **HistGradientBoosting Regressor** foi o mais eficiente, com **R² = 0.95** e **RMSE ≈ 1.85**, demonstrando excelente capacidade de generalização.

💡 **Principais fatores de influência:**
- `distance` → principal determinante do preço  
- `duration` → reflete o tempo de deslocamento  
- `surge_multiplier` → indica períodos de alta demanda  
- `name_encoded` → diferencia categorias de serviço  

🧠 **Aplicações práticas:**
- Apoiar estratégias de precificação dinâmica  
- Aumentar transparência e previsibilidade para usuários e motoristas  
- Servir como base para **sistemas inteligentes de recomendação de tarifas**

📘 Estes resultados confirmam achados da literatura recente ([Sindhu et al. 2022], [Bhardwaj et al. 2024], [Khedekar et al. 2025]) que apontam o **Gradient Boosting** como o estado da arte para predição de preços na Uber.

---
Desenvolvido pela Equipe A (Vesp.) – **NCIA / FPF Tech (2025)**
""")

# =======================================================
# RODAPÉ INSTITUCIONAL
# =======================================================
try:
    footer = get_base64_image("imagens/end.png")
    st.markdown(
        f"""
        <hr style="margin-top:3rem;">
        <div style="text-align: center; margin-top: -1rem;">
            <img src="data:image/png;base64,{footer}" style="width:100%; border-radius:10px;">
        </div>
        """,
        unsafe_allow_html=True,
    )
except FileNotFoundError:
    st.warning("⚠️ Imagem de rodapé 'end.png' não encontrada na pasta 'imagens/'.")


