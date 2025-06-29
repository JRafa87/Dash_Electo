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
    st.markdown("**Mapa de Regiones con Datos Completos:**")

    df_map = df.groupby("region").agg({
        "probabilidad": "mean",
        "poblacion_region": "first",
        "indecisos": "mean",
        "score": "mean",
        "sentimiento": "mean"
    }).reset_index()

    region_coords = {
        "Lima": [-77.0428, -12.0464],
        "Cusco": [-71.9675, -13.5319],
        "Arequipa": [-71.5375, -16.4090],
        "Piura": [-80.6333, -5.1945],
        "La Libertad": [-79.0333, -8.1150],
        "Junín": [-75.0000, -11.2500],
        "Puno": [-70.0152, -15.8402],
        "Loreto": [-73.2472, -3.7491],
        "Ancash": [-77.6047, -9.5261],
        "Tacna": [-70.2486, -18.0066],
        "Callao": [-77.129, -12.05],
        "Huánuco": [-76.2422, -9.9306],
        "Ayacucho": [-74.2167, -13.1588],
        "San Martín": [-76.5527, -7.0083],
        "Ica": [-75.7306, -14.0678],
        "Moquegua": [-70.9342, -17.198],
        "Tumbes": [-80.4531, -3.5669],
        "Ucayali": [-74.3797, -8.3791],
        "Apurímac": [-73.0385, -13.6512],
        "Pasco": [-75.25, -10.6864],
        "Madre de Dios": [-70.2479, -12.5933],
        "Cajamarca": [-78.5003, -7.1638],
        "Huancavelica": [-74.9479, -12.7864],
        "Amazonas": [-77.8691, -5.0722]
    }

    df_map["lon"] = df_map["region"].map(lambda x: region_coords.get(x, [None, None])[0])
    df_map["lat"] = df_map["region"].map(lambda x: region_coords.get(x, [None, None])[1])
    df_map = df_map.dropna(subset=["lon", "lat"])

    fig_map_points = px.scatter_mapbox(
        df_map, lat="lat", lon="lon", color="indecisos",
        size="poblacion_region", hover_name="region",
        hover_data={
            "poblacion_region": True,
            "indecisos": ":.2f",
            "score": ":.1f",
            "sentimiento": ":.2f",
            "probabilidad": ":.2f",
            "lat": False,
            "lon": False
        },
        color_continuous_scale="bluered", zoom=4.5, height=500
    )
    fig_map_points.update_layout(
        mapbox_style="carto-positron",
        mapbox_center={"lat": -9.189967, "lon": -75.015152},
        mapbox_zoom=4.5
    )
    fig_map_points.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

    st.plotly_chart(fig_map_points, use_container_width=True)

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