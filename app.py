# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import folium
from streamlit_folium import folium_static
from branca.colormap import LinearColormap 

# --- Configuracion inicial ---
st.set_page_config(page_title="Dashboard Electoral", layout="wide")

# --- Cargar estilos personalizados ---
st.markdown("""
    <style>
    div.stButton > button:hover {
        background-color: #ff5733;
        color: white;
        transition: 0.3s ease;
    }
    .hover-box {
        background-color: #f0f0f0;
        padding: 10px;
        border-radius: 10px;
        transition: 0.3s;
        margin-bottom: 1rem;
    }
    .hover-box:hover {
        background-color: #d0e6f7;
    }
    </style>
""", unsafe_allow_html=True)

# --- Cargar datos ---
df = pd.read_csv("dataset_electoral.csv")

# --- Titulo principal ---
st.title("🌏 Predicción Electoral Interactiva")

# --- Tabs ---
tabs = st.tabs(["🌐 Resumen Nacional", "🌍 Análisis Regional", "🧑‍🏫 Demografía", "🔮 Modelo de Predicción"])

# ----------- TAB 1: Resumen Nacional -----------
with tabs[0]:
    st.subheader("Resumen Nacional de Ganadores")

    col1, col2 = st.columns(2)
    with col1:
        fig_ganador = px.histogram(df, x="ganador", color="ganador",
                                   title="Distribución de Ganadores",
                                   labels={"ganador": "Ganador (1=Sí, 0=No)"},
                                   barmode="group")
        st.plotly_chart(fig_ganador, use_container_width=True)

    with col2:
        fig_prob = px.histogram(df, x="probabilidad", nbins=20, color_discrete_sequence=["royalblue"],
                                title="Distribución de Probabilidades de Victoria")
        st.plotly_chart(fig_prob, use_container_width=True)


    st.markdown("---")
    st.subheader("🌞 Distribución jerárquica por región y candidato")

    fig_sunburst = px.sunburst(df[df["ganador"] == 1],
                               path=["region", "candidato"],
                               values="poblacion_region",
                               color="probabilidad",
                               color_continuous_scale="Blues",
                               title="Distribución de población ganada por región y candidato")
    st.plotly_chart(fig_sunburst, use_container_width=True)

# ----------- TAB 3: Demografía -----------
with tabs[2]:
    st.subheader("Análisis Demográfico")

    col1, col2 = st.columns(2)
    sexo_sel = col1.radio("Sexo:", df["sexo"].unique())
    edad_sel = col2.selectbox("Grupo Etario:", df["edad_grupo"].unique())

    df_demo = df[(df["sexo"] == sexo_sel) & (df["edad_grupo"] == edad_sel)]

    fig_demo = px.histogram(df_demo, x="candidato", color="ganador",
                            title=f"Apoyo Electoral - {sexo_sel}, Edad {edad_sel}",
                            barmode="group")
    st.plotly_chart(fig_demo, use_container_width=True)