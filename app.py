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


from streamlit_folium import folium_static
from branca.colormap import LinearColormap

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

# Coordenadas actualizadas y verificadas (lat, lon)
region_coords = {
    "Lima": [-12.0433, -77.0282],
    "Cusco": [-13.5250, -71.9673],
    "Arequipa": [-16.3988, -71.5350],
    "Piura": [-5.1945, -80.6328],
    "La Libertad": [-8.1119, -78.9994],
    "Junín": [-11.7833, -75.2099],
    "Puno": [-15.8402, -70.0199],
    "Loreto": [-3.7493, -73.2532],
    "Ancash": [-9.5280, -77.6253],
    "Tacna": [-18.0140, -70.2544],
    "Callao": [-12.0500, -77.1182],
    "Huánuco": [-9.9294, -76.2410],
    "Ayacucho": [-13.1631, -74.2236],
    "San Martín": [-6.5069, -76.1572],
    "Ica": [-14.0678, -75.7358],
    "Moquegua": [-17.1956, -70.9400],
    "Tumbes": [-3.5669, -80.4526],
    "Ucayali": [-8.3792, -73.0877],
    "Apurímac": [-13.6337, -72.8902],
    "Pasco": [-10.6864, -76.2562],
    "Madre de Dios": [-12.5999, -69.1909],
    "Cajamarca": [-7.1611, -78.5000],
    "Huancavelica": [-12.7864, -74.9764],
    "Amazonas": [-5.0667, -78.5000]
}

# Añadir coordenadas como columnas separadas
df_map["lat"] = df_map["region"].apply(lambda x: region_coords.get(x, [None])[0])
df_map["lon"] = df_map["region"].apply(lambda x: region_coords.get(x, [None, None])[1] if region_coords.get(x) else None)

# Filtrar y convertir a float
df_map = df_map.dropna(subset=["lat", "lon"])
df_map["lat"] = df_map["lat"].astype(float)
df_map["lon"] = df_map["lon"].astype(float)

# Crear mapa centrado en Perú
m = folium.Map(location=[-9.5, -75], zoom_start=5, tiles='CartoDB positron')

# Crear escala de colores
colormap = LinearColormap(
    colors=['blue', 'red'],
    vmin=df_map['indecisos'].min(),
    vmax=df_map['indecisos'].max()
)

# Añadir marcadores verificados
for index, row in df_map.iterrows():
    if pd.notnull(row['lat']) and pd.notnull(row['lon']):
        popup_content = f"""
        <div style="width:250px;">
            <h4 style="margin-bottom:5px; border-bottom:1px solid #ccc; padding-bottom:5px;">{row['region']}</h4>
            <table style="width:100%; font-size:12px;">
                <tr><td><b>Población:</b></td><td style="text-align:right;">{row['poblacion_region']:,.0f}</td></tr>
                <tr><td><b>Indecisos:</b></td><td style="text-align:right;">{row['indecisos']:.2%}</td></tr>
                <tr><td><b>Score:</b></td><td style="text-align:right;">{row['score']:.1f}</td></tr>
                <tr><td><b>Sentimiento:</b></td><td style="text-align:right;">{row['sentimiento']:.2f}</td></tr>
                <tr><td><b>Probabilidad:</b></td><td style="text-align:right;">{row['probabilidad']:.2%}</td></tr>
            </table>
        </div>
        """
        
        # Tamaño proporcional con límites
        marker_size = max(5, min(30, 5 + (row['poblacion_region'] / df_map['poblacion_region'].max()) * 25))
        
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=marker_size,
            popup=folium.Popup(popup_content, max_width=300),
            color=colormap(row['indecisos']),
            fill=True,
            fill_color=colormap(row['indecisos']),
            fill_opacity=0.7,
            weight=1
        ).add_to(m)

# Añadir leyenda
colormap.caption = 'Porcentaje de Indecisos'
colormap.add_to(m)

# Mostrar el mapa
folium_static(m, width=800, height=600)

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