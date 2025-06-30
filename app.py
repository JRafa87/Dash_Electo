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
    st.markdown("**📈 Distribución de Indicadores por Región (Gráfico de Pastel)**")

    df_map = df.groupby("region").agg({
        "probabilidad": "mean",
        "poblacion_region": "first",
        "indecisos": "mean",
        "score": "mean",
        "sentimiento": "mean"
    }).reset_index()

    fig_pie = px.pie(
        df_map,
        values='poblacion_region',
        names='region',
        title='Distribución de Población por Región',
        hover_data=['indecisos', 'probabilidad', 'score', 'sentimiento'],
        labels={
            'region': 'Región',
            'poblacion_region': 'Población',
            'indecisos': '% Indecisos',
            'probabilidad': 'Probabilidad',
            'score': 'Score',
            'sentimiento': 'Sentimiento'
        }
    )

    fig_pie.update_traces(
        hovertemplate="<b>%{label}</b><br>" +
                      "Población: %{value:,.0f}<br>" +
                      "Indecisos: %{customdata[0]:.2%}<br>" +
                      "Probabilidad: %{customdata[1]:.2%}<br>" +
                      "Score: %{customdata[2]:.1f}<br>" +
                      "Sentimiento: %{customdata[3]:.2f}<extra></extra>",
        textinfo='percent+label',
        textposition='inside',
        insidetextorientation='radial',
        marker=dict(line=dict(color='#000000', width=1))
    )

    fig_pie.update_layout(
        uniformtext_minsize=10,
        uniformtext_mode='hide',
        height=600,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        hoverlabel=dict(
            bgcolor="white",
            font_size=14,
            font_family="Arial",
            bordercolor="gray"
        )
    )

    st.plotly_chart(fig_pie, use_container_width=True)


# ----------- TAB 2: Análisis Regional -----------
with tabs[1]:
    st.subheader("🔍 Análisis por Región")

    region_seleccionada = st.selectbox("Selecciona una región", sorted(df["region"].unique()))
    df_region = df[df["region"] == region_seleccionada]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Boxplot de Probabilidad por Candidato**")
        fig_box = px.box(df_region, x="candidato", y="probabilidad", color="candidato",
                         points="all", hover_data=df_region.columns,
                         title=f"Probabilidad de Victoria por Candidato en {region_seleccionada}")
        st.plotly_chart(fig_box, use_container_width=True)

    with col2:
        st.markdown("**Indecisos por Candidato (Media)**")
        df_indecisos = df_region.groupby("candidato")["indecisos"].mean().reset_index()
        fig_bar = px.bar(df_indecisos, x="candidato", y="indecisos", color="candidato",
                         text_auto=True,
                         title=f"Porcentaje Promedio de Indecisos por Candidato en {region_seleccionada}")
        fig_bar.update_traces(hovertemplate='Candidato: %{x}<br>Indecisos: %{y:.2f}%')
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")
    st.markdown("**📋 Tabla de datos detallados para la región seleccionada**")
    st.dataframe(df_region.reset_index(drop=True), use_container_width=True, height=400)


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