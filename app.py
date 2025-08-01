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
tabs = st.tabs(["🌐 Resumen Nacional", "🌍 Análisis Regional", "🧑‍🏫 Demografía", "🔮 Modelo de Predicción", "🧠 Simulación de Escenarios"])

# ----------- TAB 1: Resumen Nacional ----------- 
with tabs[0]:
    st.subheader("Resumen Nacional de Ganadores")

    # Dividir en columnas para ajustar los gráficos
    col1, col2 = st.columns(2)

    # Gráfico de ganadores (Distribución de Ganadores)
    with col1:
        st.markdown("**🔎 ¿Quién lidera a nivel nacional?**")
        st.caption("Este gráfico muestra la cantidad de veces que cada candidato aparece como ganador según las simulaciones del modelo.")
        fig_ganador = px.histogram(
            df.sort_values("ganador"), x="ganador", color="ganador",
            title="Distribución de Ganadores",
            category_orders={"ganador": df["ganador"].value_counts().index.tolist()},
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        fig_ganador.update_layout(showlegend=False)
        st.plotly_chart(fig_ganador, use_container_width=True)

    # Gráfico de probabilidades (Distribución de Probabilidades de Victoria)
    with col2:
        st.markdown("**📊 ¿Cuán segura es la predicción?**")
        st.caption("Una mayor probabilidad implica mayor confianza en que ese candidato ganará.")
        fig_prob = px.histogram(
            df, x="probabilidad", nbins=20,
            color_discrete_sequence=["#1f77b4"],
            title="Distribución de Probabilidades de Victoria"
        )
        fig_prob.update_layout(xaxis_title="Probabilidad de Victoria", yaxis_title="Frecuencia")
        st.plotly_chart(fig_prob, use_container_width=True)

    # Gráfico de comparación regional (Solo en la pestaña de Análisis Regional)
    st.subheader("Población vs Indecisos por región")
    df_map = df.groupby("region").agg({
        "probabilidad": "mean",
        "poblacion_region": "first",
        "indecisos": "mean"
    }).reset_index()

    fig_bar_stacked = px.bar(
        df_map.sort_values("poblacion_region", ascending=False),
        x='region',
        y='poblacion_region',
        color='indecisos',
        labels={'region': 'Región', 'poblacion_region': 'Población', 'indecisos': 'Indecisos (%)'},
        color_continuous_scale='viridis',
        hover_data={'probabilidad': ':.2f', 'indecisos': ':.2%'}
    )

    fig_bar_stacked.update_traces(
        hovertemplate="<b>%{x}</b><br>" +
                      "Población: %{y:,.0f}<br>" +
                      "Indecisos: %{marker.color:.2%}<br>" +
                      "Probabilidad: %{customdata[0]:.2%}<extra></extra>",
        texttemplate='%{y:,.0f}',
        textposition='outside'
    )

    fig_bar_stacked.update_layout(
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

    st.markdown("**🌎 Comparación Regional**")
    st.caption("Este gráfico permite comparar el tamaño poblacional y el nivel de indecisión electoral en cada región.")
    st.plotly_chart(fig_bar_stacked, use_container_width=True)


# ----------- TAB 2: Análisis Regional ----------- 
with tabs[1]:
    st.subheader("🔍 Análisis por Región")

    region_seleccionada = st.selectbox("Selecciona una región", sorted(df["region"].unique()))
    df_region = df[df["region"] == region_seleccionada]

    col1, col2 = st.columns(2)

    # Gráfico de Probabilidad de Victoria en la región seleccionada
    with col1:
        st.markdown(f"**📊 Distribución de Probabilidad de Victoria ({region_seleccionada})**")
        st.caption("Cada punto representa una estimación individual. Las cajas agrupan el rango típico de probabilidades por candidato.")
        fig_box = px.box(
            df_region, x="candidato", y="probabilidad", color="candidato",
            points="all", hover_data=df_region.columns,
            category_orders={"candidato": df_region.groupby("candidato")["probabilidad"].mean().sort_values(ascending=False).index.tolist()},
            color_discrete_sequence=px.colors.qualitative.Set2,
            title=f"Probabilidad de Victoria por Candidato en {region_seleccionada}"
        )
        st.plotly_chart(fig_box, use_container_width=True)

    # Gráfico de Indecisos en la región seleccionada
    with col2:
        st.markdown("**🧭 Nivel de Indecisión por Candidato**")
        st.caption("Muestra el porcentaje promedio de personas indecisas por cada candidato en la región.")
        df_indecisos = df_region.groupby("candidato")["indecisos"].mean().round(2).reset_index()
        fig_bar = px.bar(
            df_indecisos.sort_values("indecisos", ascending=False),
            x="candidato", y="indecisos", color="candidato",
            text="indecisos",
            color_discrete_sequence=px.colors.qualitative.Set1,
            title=f"Porcentaje Promedio de Indecisos por Candidato en {region_seleccionada}"
        )
        fig_bar.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        fig_bar.update_layout(yaxis_title="Indecisos (%)")
        st.plotly_chart(fig_bar, use_container_width=True)

    # Mostrar tabla detallada para la región seleccionada debajo de los gráficos
    st.markdown("---")
    st.markdown(f"**📋 Datos detallados de la región: {region_seleccionada}**")
    st.caption("Visualiza los registros individuales correspondientes a la región seleccionada.")

    st.markdown(""" 
    <div class="hover-box">
        Esta tabla muestra los valores de probabilidad, indecisión y características demográficas 
        para analizar en detalle el comportamiento electoral regional.
    </div>
    """, unsafe_allow_html=True)

    # Asegurar que la columna 'region' esté visible aunque esté filtrado
    cols = ["region"] + [col for col in df_region.columns if col != "region"]

    # Mostrar la tabla ocupando el ancho completo
    st.dataframe(df_region[cols].reset_index(drop=True), use_container_width=True, height=400)



# ----------- TAB 3: Demografía ----------- 
with tabs[2]:
    st.subheader("Análisis Demográfico")
    st.caption("Explora el apoyo electoral cruzado con sexo y grupo etario.")

    col1, col2 = st.columns(2)
    sexo_sel = col1.radio("Sexo:", df["sexo"].unique())
    edad_sel = col2.selectbox("Grupo Etario:", df["edad_grupo"].unique())

    # Filtrar el DataFrame según las selecciones de sexo y grupo etario
    df_filtrado = df[(df["sexo"] == sexo_sel) & (df["edad_grupo"] == edad_sel)]

    # Crear el gráfico de barras
    fig_demo = px.histogram(
        df_filtrado.sort_values("candidato"), 
        x="candidato", color="ganador",
        title=f"Apoyo Electoral - {sexo_sel}, Edad {edad_sel}",
        barmode="group",
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    fig_demo.update_layout(
        xaxis_title="Candidato", 
        yaxis_title="Frecuencia de Apoyo"
    )
    st.plotly_chart(fig_demo, use_container_width=True)

# ----------- TAB 4: Predicción ----------- 

with tabs[3]:  # Esta es tu pestaña "Modelo de Predicción"
    st.subheader("🔍 Predicción de Resultados Electorales")
    st.markdown("Completa los datos para estimar la **probabilidad de victoria** y si el candidato **ganaría o no**.")

    # Cargar modelos si no se han cargado ya
    if "reg_model" not in st.session_state:
        import joblib
        reg_model = joblib.load("models/reg_arbol.pkl")
        class_model = joblib.load("models/modelo_xgb.pkl")
        label_encoders = joblib.load("models/label_encoders.pkl")
        st.session_state.reg_model = reg_model
        st.session_state.class_model = class_model
        st.session_state.label_encoders = label_encoders
    else:
        reg_model = st.session_state.reg_model
        class_model = st.session_state.class_model
        label_encoders = st.session_state.label_encoders

        #st.write("Label Encoders cargados:", list(label_encoders.keys()))

    def codificar_input(input_dict, label_encoders):
        df = pd.DataFrame([input_dict])
        for col, le in label_encoders.items():
            df[col] = le.transform(df[col])
        return df

    with st.form("formulario_prediccion"):
        col1, col2 = st.columns(2)

        with col1:
            region = st.selectbox("🗺️ Región", label_encoders["region"].classes_)
            candidato = st.selectbox("👤 Candidato", label_encoders["candidato"].classes_)
            sexo = st.selectbox("🧬 Sexo", label_encoders["sexo"].classes_)
            edad_grupo = st.selectbox("🎂 Grupo Etario", label_encoders["edad_grupo"].classes_)
            

        with col2:
            ingreso_promedio = st.slider("💰 Ingreso Promedio", 500.0, 5000.0, 1500.0)
            score = st.slider("📈 Score", 0.0, 1.0, 0.5)
            indecisos = st.slider("🌀 Nivel de Indecisos (%)", 0.0, 100.0, 20.0)
            porcentaje_grupo = st.slider("👥 Porcentaje del Grupo Etario (%)", 0.0, 100.0, 30.0)
            poblacion_region = st.slider("🏙️ Población Regional", 1000, 1000000, 50000)
            sentimiento = st.slider("💬 Sentimiento", min_value=0.2, max_value=0.8, step=0.01)


        submitted = st.form_submit_button("🔍 Predecir")

    if submitted:
        input_dict = {
            "region": region,
            "candidato": candidato,
            "sexo": sexo,
            "edad_grupo": edad_grupo,
            "poblacion_region": poblacion_region,
            "porcentaje_grupo": porcentaje_grupo,
            "ingreso_promedio": ingreso_promedio,
            "score": score,
            "sentimiento": sentimiento,
            "indecisos": indecisos,
            
            
        }

        X_input = codificar_input(input_dict, label_encoders)

        # Predicción regresiva
        probabilidad_estim = reg_model.predict(X_input)[0]
        st.metric("📊 Probabilidad estimada de victoria", f"{probabilidad_estim*100:.2f}%")

        # Clasificación
        class_pred = class_model.predict(X_input)[0]
        class_prob = class_model.predict_proba(X_input)[0][class_pred]

        # Guardar en sesión
        st.session_state.prediccion_resultado = {
           "input_data": input_dict,
        "X_input": X_input,
        "probabilidad": float(probabilidad_estim),
        "ganador": int(class_pred),
        "confianza": float(class_prob)
        }

        #resultado = "GANARÍA" if class_pred == 1 else "NO GANARÍA"
        if class_pred == 1:
           st.success(f"✅ El candidato seleccionado **GANARÍA** en este escenario. 📢")
        else:
           st.warning(f"❌ El candidato seleccionado **NO GANARÍA** según la predicción actual. ⚠️")

        st.info(f"🔍 Confianza del modelo: {class_prob*100:.2f}%")

# ----------- TAB 5: Simulación ----------- 
with tabs[4]:
    st.title("🧠 Escenarios y Simulaciones")
    st.markdown("Simula diferentes escenarios con base en las predicciones o ingresa tus propios valores.")

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import graphviz as graphviz

    st.subheader("⚙️ Fuente de datos para simulación")
    modo_entrada = st.radio("¿Cómo deseas cargar los datos?", ["📥 Ingresar manualmente", "📊 Usar datos predichos de la pestaña anterior"])

    if modo_entrada == "📊 Usar datos predichos de la pestaña anterior":
        try:
            if "resultado_prediccion" in st.session_state:
                datos_pred = st.session_state.resultado_prediccion
                probabilidad = datos_pred["probabilidad"]
                gana = datos_pred["ganador"]
                st.success("✅ Datos cargados desde la predicción anterior.")
            else:
                st.warning("⚠️ No se encontró una predicción previa. Ingresa los valores manualmente.")
                modo_entrada = "📥 Ingresar manualmente"
        except:
            st.error("Error al cargar datos predichos.")
            modo_entrada = "📥 Ingresar manualmente"

    if modo_entrada == "📥 Ingresar manualmente":
        col1, col2 = st.columns(2)
        with col1:
            probabilidad = st.slider("Probabilidad de ganar (%)", 0.0, 100.0, 75.0, step=0.1)
        with col2:
            gana = st.checkbox("¿Ganador en escenario?", value=True)

    col3, col4 = st.columns(2)
    with col3:
        presupuesto = st.number_input("Presupuesto estimado (en miles)", min_value=0, step=10)
        exposicion = st.selectbox("Nivel de exposición", ["baja", "media", "alta"])
    with col4:
        influencia = st.selectbox("Influencia en redes", ["baja", "media", "alta"])

    st.markdown("---")
    if st.button("▶️ Ejecutar Simulación"):
        prob_decimal = probabilidad / 100

        # Árbol de decisión visual e interactivo
        def mostrar_arbol_decision(prob, ganador, exp, infl):
            st.subheader("📍 Árbol de decisión")
            dot = graphviz.Digraph()
            dot.node("A", "¿Probabilidad > 60%?")
            if prob > 0.6:
                dot.node("B", "¿Influencia Alta?")
                dot.edge("A", "B", "Sí")
                if infl == "alta":
                    dot.node("C", "✅ Apoyar (Alta influencia)")
                    dot.edge("B", "C", "Sí")
                elif exp == "alta":
                    dot.node("D", "✅ Apoyar (Alta exposición)")
                    dot.edge("B", "D", "No, pero exposición alta")
                else:
                    dot.node("E", "🤔 Evaluar más indicadores")
                    dot.edge("B", "E", "No")
            else:
                dot.node("F", "❌ No apoyar")
                dot.edge("A", "F", "No")

            st.graphviz_chart(dot)

        # Matriz de pago mejorada
        def mostrar_matriz_pago():
            st.subheader("📊 Matriz de Pago (Decisión de campaña)")
            matriz = pd.DataFrame({
                "Decisión ↓ / Resultado →": ["Apoyar", "No Apoyar"],
                "Gana": [100, -50],
                "Pierde": [-100, 0]
            }).set_index("Decisión ↓ / Resultado →")
            st.dataframe(matriz, use_container_width=True)

        # Simulación de Montecarlo
        def simular_montecarlo(prob, n=1000):
            st.subheader("🎲 Simulación de Montecarlo")
            simulaciones = np.random.rand(n) < prob
            tasa_ganadora = np.mean(simulaciones)

            col_sim1, col_sim2 = st.columns(2)
            with col_sim1:
                st.metric("Tasa estimada de victoria", f"{tasa_ganadora*100:.2f} %")
            with col_sim2:
                st.metric("Número de simulaciones", f"{n:,}")

            fig, ax = plt.subplots()
            ax.hist(simulaciones.astype(int), bins=[-0.5, 0.5, 1.5], edgecolor='black', rwidth=0.6, color="#3b82f6")
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['Pierde', 'Gana'])
            ax.set_ylabel("Frecuencia")
            st.pyplot(fig)

            # Opcional: descarga CSV
            if st.checkbox("📥 Descargar resultados de la simulación"):
                df_sim = pd.DataFrame({"Resultado": np.where(simulaciones, "Gana", "Pierde")})
                st.download_button("Descargar CSV", df_sim.to_csv(index=False), "simulacion_montecarlo.csv")

        # Insights accionables
        def mostrar_insights(prob, ganador, exp, infl):
            st.subheader("🔍 Insights accionables")
            insights = []

            if prob > 0.8 and ganador:
                insights.append("✅ Alta probabilidad de victoria. Invertir más en zonas con exposición baja.")
            if prob < 0.5:
                insights.append("⚠️ Riesgo elevado de derrota. Enfocar recursos en redes y discurso positivo.")
            if infl == "alta" and exp != "alta":
                insights.append("🎯 Fortalezca la visibilidad en medios para potenciar su influencia actual.")
            if exp == "media" and not ganador:
                insights.append("📢 Reforzar presencia territorial con actos públicos.")
            if presupuesto < 50:
                insights.append("💰 Presupuesto limitado. Optimizar microsegmentación de mensajes.")

            if insights:
                for i, tip in enumerate(insights[:3]):
                    st.info(f"Insight {i+1}: {tip}")
            else:
                st.info("📌 Mantenga seguimiento regular de exposición, presupuesto e intención de voto.")

        # Ejecutar todo
        mostrar_arbol_decision(prob_decimal, gana, exposicion, influencia)
        mostrar_matriz_pago()
        simular_montecarlo(prob_decimal)
        mostrar_insights(prob_decimal, gana, exposicion, influencia)

