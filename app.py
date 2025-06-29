# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

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
    st.markdown("**Mapa de Regiones con Promedio de Probabilidad:**")
    df_map = df.groupby("region")["probabilidad"].mean().reset_index()
    fig_map = px.choropleth(df_map, locations="region", locationmode="geojson-id",
                            color="probabilidad", scope="south america",
                            color_continuous_scale="blues", title="Probabilidad promedio por región")
    st.plotly_chart(fig_map, use_container_width=True)

# ----------- TAB 2: Análisis Regional -----------
with tabs[1]:
    st.subheader("Análisis por Región")

    region_sel = st.selectbox("Selecciona una región:", sorted(df['region'].unique()))
    df_region = df[df['region'] == region_sel]

    col1, col2 = st.columns(2)
    with col1:
        fig_score = px.box(df_region, x="candidato", y="score", color="candidato",
                           title="Score por Candidato en " + region_sel)
        st.plotly_chart(fig_score, use_container_width=True)

    with col2:
        fig_sent = px.bar(df_region.groupby("candidato")["sentimiento"].mean().reset_index(),
                          x="candidato", y="sentimiento", color="candidato",
                          title="Sentimiento Promedio por Candidato")
        st.plotly_chart(fig_sent, use_container_width=True)

    st.markdown("---")
    st.dataframe(df_region.head(10))

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