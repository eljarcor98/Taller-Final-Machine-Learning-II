"""
app.py - Interfaz de Usuario Interactiva en Streamlit
Sistema de Vigilancia y Segmentación Epidemiológica (SIVIGILA)
Dataset Oficial: Portal Nacional de Datos Abiertos (datos.gov.co)
"""

from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

# Configuración inicial de la página
st.set_page_config(
    page_title="SIVIGILA | Segmentación Epidemiológica",
    page_icon="🏥",
    layout="centered"
)

MODEL_PATH = Path("model.joblib")


@st.cache_resource
def load_artifact():
    """Carga el modelo y sus metadatos desde el archivo serializado."""
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def main():
    st.title("🏥 SIVIGILA: Segmentación Epidemiológica")
    st.caption("Caracterización de eventos de salud pública con K-Means y Docker | Datos Abiertos Colombia")

    artifact = load_artifact()

    if artifact is None:
        st.error(
            f"⚠️ No se encontró el archivo del modelo `{MODEL_PATH.name}`.\n\n"
            "Por favor ejecuta primero en la consola:\n"
            "```bash\npython train.py\n```"
        )
        st.stop()

    model = artifact["model"]
    features = artifact["features"]
    cluster_descriptions = artifact.get("cluster_descriptions", {})
    profiles = artifact.get("profiles", {})
    top_events = artifact.get("top_events", {})
    numeric_stats = artifact.get("numeric_stats", {})
    n_clusters = artifact.get("n_clusters", 4)
    silhouette = artifact.get("silhouette", "N/A")
    total_records = artifact.get("total_records", 1226)

    st.markdown("### Ingrese los datos del evento epidemiológico a evaluar:")
    st.info("El modelo normaliza las variables y asigna el caso al perfil epidemiológico y nivel de prioridad correspondiente.")

    user_inputs = {}

    with st.form("epidemiology_form"):
        col1, col2 = st.columns(2)

        # Edad del paciente
        stats_edad = numeric_stats.get("edad", {"min": 0.0, "max": 90.0, "median": 20.0})
        user_inputs["edad"] = col1.number_input(
            "👶/👵 Edad del Paciente (años)",
            min_value=0.0,
            max_value=110.0,
            value=float(stats_edad.get("median", 20.0)),
            step=1.0,
            help="Edad cronológica en años cumplidos"
        )

        # Semana epidemiológica del año
        user_inputs["semana_epidemiologica"] = col2.slider(
            "📅 Semana Epidemiológica del Reporte",
            min_value=1,
            max_value=52,
            value=25,
            help="Semana del calendario epidemiológico (1 a 52) en la que ocurrió la notificación"
        )

        col3, col4 = st.columns(2)

        # Estrato socioeconómico
        user_inputs["estrato"] = float(col3.selectbox(
            "🏠 Estrato Socioeconómico",
            options=[1, 2, 3, 4],
            index=0,
            help="Estrato de residencia registrado en la ficha epidemiológica"
        ))

        # Hospitalización
        hosp_choice = col4.selectbox(
            "🚑 ¿Requirió Hospitalización?",
            options=["NO (Manejo ambulatorio)", "SI (Internación hospitalaria)"],
            index=0,
            help="Indica si el paciente requirió internación en centro hospitalario"
        )
        user_inputs["hospitalizado"] = 1.0 if hosp_choice.startswith("SI") else 0.0

        submitted = st.form_submit_button("🔍 Caracterizar Evento Epidemiológico", use_container_width=True)

    if submitted:
        # Vector ordenado según las características del modelo
        X_new = pd.DataFrame([user_inputs], columns=features)

        try:
            # Inferencia a través del Pipeline
            cluster_id = int(model.predict(X_new)[0])
            cluster_info = cluster_descriptions.get(cluster_id, "Sin descripción disponible.")

            st.markdown("---")
            st.subheader(f"🎯 Perfil Asignado: `Clúster {cluster_id}`")
            
            # Badge de alerta según severidad
            if cluster_id == 2:
                st.error(f"🚨 **Alerta de Alta Severidad / Hospitalario:** {cluster_info}")
            elif cluster_id == 3:
                st.warning(f"⚠️ **Alerta Urbana / Zoonosis:** {cluster_info}")
            else:
                st.success(f"✅ **Manejo Comunitario / Ambulatorio:** {cluster_info}")

            # Patologías frecuentes en este clúster
            events = top_events.get(cluster_id, {})
            if events:
                st.markdown("#### 🦠 Patologías y Eventos Frecuentes en este Grupo:")
                event_items = [f"**{k}** ({v} casos)" for k, v in events.items()]
                st.markdown(" • " + "  \n • ".join(event_items))

            # Comparativa métrica frente al promedio del clúster
            if cluster_id in profiles:
                c_prof = profiles[cluster_id]
                st.markdown("#### 📊 Comparativa vs Promedios del Clúster:")
                m1, m2, m3 = st.columns(3)
                m1.metric(
                    label="Edad del Paciente",
                    value=f"{user_inputs['edad']:.0f} años",
                    delta=f"{user_inputs['edad'] - c_prof['edad']:+.1f} vs promedio ({c_prof['edad']} a)"
                )
                m2.metric(
                    label="Semana del Brote",
                    value=f"Semana {user_inputs['semana_epidemiologica']:.0f}",
                    delta=f"{user_inputs['semana_epidemiologica'] - c_prof['semana_epidemiologica']:+.1f} vs media (Sem {c_prof['semana_epidemiologica']})"
                )
                m3.metric(
                    label="Estrato",
                    value=f"Estrato {user_inputs['estrato']:.0f}",
                    delta=f"{user_inputs['estrato'] - c_prof['estrato']:+.2f} vs media ({c_prof['estrato']})"
                )

            with st.expander("Ver vector de características procesado"):
                st.dataframe(X_new)

        except Exception as e:
            st.error(f"Error al calcular la asignación del clúster: {e}")

    # Barra lateral técnica y contexto
    with st.sidebar:
        st.header("📌 Parámetros del Modelo")
        st.write(f"- **Algoritmo:** K-Means Clustering")
        st.write(f"- **Número de Clústers (k):** {n_clusters}")
        st.write(f"- **Silhouette Score:** `{silhouette}`")
        st.write(f"- **Casos Históricos Analizados:** `{total_records:,}`")
        st.markdown("---")
        st.subheader("🌐 Fuente de Datos Oficial")
        st.markdown("[Eventos Epidemiológicos | datos.gov.co](https://www.datos.gov.co/Salud-y-Protecci-n-Social/Eventos-Epidemiol-gicos/afi7-frbs/about_data)")
        st.caption("Sistema de Vigilancia en Salud Pública (SIVIGILA), Ministerio de Salud y Protección Social de Colombia.")
        st.markdown("---")
        st.subheader("📖 Clústeres Definidos")
        for cid, desc in cluster_descriptions.items():
            st.markdown(f"**Clúster {cid}:** {desc[:80]}...")


if __name__ == "__main__":
    main()
