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
st.markdown("**Mapa de Regiones con Datos Completos:**")

# Preprocesamiento de datos
df_map = df.groupby("region").agg({
    "probabilidad": "mean",
    "poblacion_region": "first",
    "indecisos": "mean",
    "score": "mean",
    "sentimiento": "mean"
}).reset_index()

# Coordenadas mejoradas con capitales regionales
region_coords = {
    "Lima": [-77.0282, -12.0433],
    "Cusco": [-71.9673, -13.5250],
    "Arequipa": [-71.5350, -16.3988],
    "Piura": [-80.6328, -5.1945],
    "La Libertad": [-78.9994, -8.1119],
    "Junín": [-75.2099, -11.7833],
    "Puno": [-70.0199, -15.8402],
    "Loreto": [-73.2532, -3.7493],
    "Ancash": [-77.6253, -9.5280],
    "Tacna": [-70.2544, -18.0140],
    "Callao": [-77.1182, -12.0500],
    "Huánuco": [-76.2410, -9.9294],
    "Ayacucho": [-74.2236, -13.1631],
    "San Martín": [-76.1572, -6.5069],
    "Ica": [-75.7358, -14.0678],
    "Moquegua": [-70.9400, -17.1956],
    "Tumbes": [-80.4526, -3.5669],
    "Ucayali": [-73.0877, -8.3792],
    "Apurímac": [-72.8902, -13.6337],
    "Pasco": [-76.2562, -10.6864],
    "Madre de Dios": [-69.1909, -12.5999],
    "Cajamarca": [-78.5000, -7.1611],
    "Huancavelica": [-74.9764, -12.7864],
    "Amazonas": [-78.5000, -5.0667]
}

# Añadir coordenadas y normalizar tamaños
df_map["lon"] = df_map["region"].apply(lambda x: region_coords.get(x, [None])[0])
df_map["lat"] = df_map["region"].apply(lambda x: region_coords.get(x, [None, None])[1])
df_map = df_map.dropna(subset=["lon", "lat"])

# Escalar el tamaño para mejor visualización
df_map["size"] = (df_map["poblacion_region"] / df_map["poblacion_region"].max()) * 50 + 10

# Crear el mapa interactivo
fig = go.Figure()

# Añadir capa base de Perú
fig.add_trace(go.Scattermapbox(
    lat=df_map["lat"],
    lon=df_map["lon"],
    mode='markers',
    marker=go.scattermapbox.Marker(
        size=df_map["size"],
        color=df_map["indecisos"],
        colorscale="Rainbow",
        cmin=df_map["indecisos"].min(),
        cmax=df_map["indecisos"].max(),
        colorbar_title="% Indecisos",
        opacity=0.8,
        sizemode='diameter'
    ),
    text=df_map.apply(lambda row: (
        f"<b>{row['region']}</b><br>"
        f"Población: {row['poblacion_region']:,.0f}<br>"
        f"Indecisos: {row['indecisos']:.2%}<br>"
        f"Score: {row['score']:.1f}<br>"
        f"Sentimiento: {row['sentimiento']:.2f}<br>"
        f"Probabilidad: {row['probabilidad']:.2%}"
    ), axis=1),
    hoverinfo='text'
))

# Personalizar el layout
fig.update_layout(
    mapbox_style="stamen-terrain",
    mapbox_zoom=4.2,
    mapbox_center={"lat": -9.5, "lon": -75},
    margin={"r": 0, "t": 0, "l": 0, "b": 0},
    height=600,
    hoverlabel=dict(
        bgcolor="white",
        font_size=14,
        font_family="Arial"
    )
)

# Añadir etiquetas de texto para las regiones
fig.add_trace(go.Scattermapbox(
    lat=df_map["lat"],
    lon=df_map["lon"],
    mode='text',
    text=df_map["region"],
    textfont=dict(size=10, color='black'),
    hoverinfo='none'
))

st.plotly_chart(fig, use_container_width=True)

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