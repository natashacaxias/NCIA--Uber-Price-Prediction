import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from projeto_uber_final import limparDados

# ===========================================
# CONFIGURAÇÃO INICIAL
# ===========================================
st.set_page_config(page_title="Previsão de Preços Uber | NCIA", layout="wide")
st.title("🚗 Previsão de Preços de Corridas Uber – NCIA/FPF Tech")

st.markdown("""
Bem-vindo(a)! Este painel apresenta os resultados do projeto de **Machine Learning**
para prever o preço de corridas Uber, desenvolvido pela **Equipe A Vesp. NCIA – FPF Tech (2025)**.

**Integrantes:**  
👩‍💻 Natasha Caxias · 👨‍💻 Gustavo Lima · 👩‍💻 Alessandra Bentes · 👨‍💻 Tedy Prist · 👨‍💻 Kevyn Goldim  

---
""")

# ===========================================
# CARREGAR DADOS
# ===========================================
@st.cache_data
def load_data():
    df = pd.read_csv("data/rideshare_kaggle.csv")
    df = limparDados(df)
    df = df[df["cab_type"] == "Uber"]
    return df

with st.spinner("Carregando dados..."):
    df = load_data()

# ===========================================
# DEFINIR ABAS
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
Contudo, essa variação em tempo real pode gerar **incerteza para clientes e motoristas**.

💡 Este projeto aplica **algoritmos de Machine Learning** para prever o preço das corridas,
buscando maior transparência e previsibilidade na precificação.

**Dataset:** *Uber Ride Analytics Dashboard* (Boston, EUA)  
**Tamanho:** ~148 mil corridas, 57 atributos.  
**Variável alvo:** `price`
""")

    st.image("https://upload.wikimedia.org/wikipedia/commons/c/cc/Uber_logo_2018.png", width=150)
    st.info("Este trabalho foi desenvolvido no âmbito da FPF Tech / NCIA, aplicando regressão supervisionada com foco em precificação urbana.")

# ===========================================
# 📊 ANÁLISE EXPLORATÓRIA (com imagens estáticas)
# ===========================================
with tabs[1]:
    st.header("Exploração de Dados (EDA)")
    st.markdown("""
O conjunto de dados contém informações de **preço, distância, tempo, tipo de corrida e clima**.  
A seguir, alguns padrões importantes identificados durante a análise:
""")

    # Exibir imagens já geradas
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
# 🤖 COMPARAÇÃO DE MODELOS (tabela + imagens)
# ===========================================
with tabs[2]:
    st.header("Comparação de Modelos de Regressão")
    st.markdown("""
Para avaliar o desempenho dos algoritmos de Machine Learning, 
foram utilizadas as métricas **RMSE** (Root Mean Squared Error), **MAE** (Mean Absolute Error)** 
e **R²** (Coeficiente de Determinação).  
A tabela e os gráficos abaixo resumem os resultados obtidos.
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

    # Função para destacar o melhor modelo (HistGradientBoosting)
    def highlight_best_model(row):
        if row["Modelo"] == "HistGradientBoosting":
            return ['background-color: #FFF3B0; font-weight: bold;'] * len(row)
        else:
            return [''] * len(row)

    # Exibir tabela formatada
    st.dataframe(
        df_models.style
        .format(precision=4)
        .apply(highlight_best_model, axis=1)
        .set_properties(**{
            "text-align": "center",
            "background-color": "#000000",
            "color": "white"
        })
    )

    st.markdown("---")

    # --- IMAGENS DE COMPARAÇÃO ---
    st.markdown("### Visualização Comparativa das Métricas")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Comparação de Performance — RMSE")
        st.image("imagens/comparacao_modelos_rmse.jpg", use_container_width=True)
        st.caption("Modelos com menor RMSE apresentam menor erro médio na predição do preço.")

    with col2:
        st.subheader("Comparação de Modelos — R²")
        st.image("imagens/comparacao_modelos_r2.jpeg", use_container_width=True)
        st.caption("Modelos com R² mais próximo de 1 explicam melhor a variação dos preços observados.")

    # --- CONCLUSÃO DA SEÇÃO ---
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

    col1, col2 = st.columns(2)
    dist = col1.number_input("Distância (milhas):", min_value=0.1, max_value=8.0, value=3.5)
    hora = col2.slider("Hora do dia:", 0, 23, 17)
    surge = 1 #.slider("Surge Multiplier (demanda):", 1.0, 3.0, 1.0, 0.1)
    servico = 'UberX' #col3.selectbox("Tipo de Serviço Uber:", ['UberX','UberXL','Black','Select','WAV'])
    dur = (dist / 20) * 60  # duração estimada

    features = ["distance", "duration", "surge_multiplier", "hour"]
    model = HistGradientBoostingRegressor(min_samples_leaf= 100, max_leaf_nodes= 63, max_iter= 800, max_depth= None, learning_rate=0.1778279410038923, l2_regularization= 1.0, random_state=42)
    model.fit(df[features], df["price"])

    pred = model.predict(pd.DataFrame([[dist, dur, surge, hora]], columns=features))[0]
    st.success(f"💰 **Preço estimado: US$ {pred:.2f}**")

    st.caption("Previsão aproximada baseada em dados históricos da Uber (Boston, 2018).")

# ===========================================
# 📈 CONCLUSÕES
# ===========================================
with tabs[4]:
    st.header("Conclusões e Impacto")

    st.markdown("""
Os resultados mostraram que o modelo **HistGradientBoosting Regressor**
foi o mais eficiente, com **R² = 0.95** e **RMSE ≈ 1.85**, superando todos os demais.

💡 **Principais fatores de influência:**
- **distance** — principal determinante do preço  
- **duration** — reflete o tempo de deslocamento  
- **surge_multiplier** — ajusta preço conforme demanda  
- **name_encoded** — diferencia categorias do serviço  

🧠 **Aplicações práticas:**
- Apoiar estratégias de precificação dinâmica  
- Melhorar transparência e previsibilidade para usuários e motoristas  
- Servir como base para **sistemas inteligentes de recomendação de tarifas**

📊 O projeto confirma achados da literatura recente ([Sindhu et al. 2022], [Bhardwaj et al. 2024], [Khedekar et al. 2025])  
ao apontar o **Gradient Boosting** como o estado da arte em predição de preços na Uber.

---
Desenvolvido pela Equipe A Vesp.**NCIA – FPF Tech**  
""")
