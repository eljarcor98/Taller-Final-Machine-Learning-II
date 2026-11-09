"""
app.py - Interfaz interactiva en Streamlit para Segmentación (Ruta B: Clustering K-Means)
Consume el artefacto model.joblib generado por train.py sin volver a entrenar.
"""

from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

# Configuración inicial de la página
st.set_page_config(
    page_title="Segmentación de Clientes | K-Means",
    page_icon="👥",
    layout="centered"
)

MODEL_PATH = Path("model.joblib")


@st.cache_resource
def load_artifact():
    """Carga el artefacto serializado del modelo y sus metadatos."""
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def main():
    st.title("👥 Segmentación Inteligente de Clientes")
    st.caption("Despliegue de modelo no supervisado K-Means con Pipeline de Scikit-Learn y Docker")

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
    numeric_features = artifact.get("numeric_features", [])
    cluster_descriptions = artifact.get("cluster_descriptions", {})
    profiles = artifact.get("profiles", {})
    numeric_stats = artifact.get("numeric_stats", {})
    n_clusters = artifact.get("n_clusters", 4)
    silhouette = artifact.get("silhouette", "N/A")

    st.markdown("### Ingrese las características del cliente a segmentar:")
    st.info("El modelo normalizará los datos con StandardScaler y asignará automáticamente el perfil correspondiente.")

    user_inputs = {}

    with st.form("clustering_form"):
        # Formulario amigable en columnas
        col1, col2, col3 = st.columns(3)

        # Edad
        stats_edad = numeric_stats.get("edad", {"min": 18.0, "max": 75.0, "median": 35.0})
        user_inputs["edad"] = col1.number_input(
            "Edad (años)",
            min_value=float(stats_edad.get("min", 18.0)),
            max_value=float(stats_edad.get("max", 100.0)),
            value=float(stats_edad.get("median", 35.0)),
            step=1.0,
            help="Edad cronológica del cliente"
        )

        # Ingreso anual
        stats_ingreso = numeric_stats.get("ingreso_anual", {"min": 10.0, "max": 150.0, "median": 60.0})
        user_inputs["ingreso_anual"] = col2.number_input(
            "Ingreso Anual (k$ USD)",
            min_value=float(stats_ingreso.get("min", 10.0)),
            max_value=float(stats_ingreso.get("max", 200.0)),
            value=float(stats_ingreso.get("median", 60.0)),
            step=1.0,
            help="Ingresos anuales estimados en miles de dólares"
        )

        # Puntaje de gasto
        stats_gasto = numeric_stats.get("puntaje_gasto", {"min": 1.0, "max": 100.0, "median": 50.0})
        user_inputs["puntaje_gasto"] = col3.slider(
            "Puntaje de Gasto (1-100)",
            min_value=1.0,
            max_value=100.0,
            value=float(stats_gasto.get("median", 50.0)),
            step=1.0,
            help="Puntaje de consumo basado en historial de compras y comportamiento"
        )

        submitted = st.form_submit_button("🔍 Asignar Clúster", use_container_width=True)

    if submitted:
        # Construir DataFrame con el orden exacto de características
        X_new = pd.DataFrame([user_inputs], columns=features)

        try:
            # Predicción mediante el Pipeline (preprocesador + kmeans.predict)
            cluster_id = int(model.predict(X_new)[0])
            cluster_info = cluster_descriptions.get(cluster_id, "Sin descripción disponible.")

            st.markdown("---")
            st.subheader(f"🎯 Clúster Asignado: `Clúster {cluster_id}`")
            st.success(f"**Perfil Interpretado:** {cluster_info}")

            # Comparativa con el centroide / promedio del clúster
            if cluster_id in profiles:
                cluster_profile = profiles[cluster_id]
                st.markdown("#### 📊 Comparativa: Cliente vs Promedio del Clúster")
                comp_cols = st.columns(3)
                comp_cols[0].metric(
                    label="Edad",
                    value=f"{user_inputs['edad']:.0f} años",
                    delta=f"{user_inputs['edad'] - cluster_profile['edad']:+.1f} vs promedio ({cluster_profile['edad']} a)"
                )
                comp_cols[1].metric(
                    label="Ingreso Anual",
                    value=f"${user_inputs['ingreso_anual']:.1f}k",
                    delta=f"{user_inputs['ingreso_anual'] - cluster_profile['ingreso_anual']:+.1f}k vs promedio (${cluster_profile['ingreso_anual']}k)"
                )
                comp_cols[2].metric(
                    label="Puntaje de Gasto",
                    value=f"{user_inputs['puntaje_gasto']:.0f} pts",
                    delta=f"{user_inputs['puntaje_gasto'] - cluster_profile['puntaje_gasto']:+.1f} vs promedio ({cluster_profile['puntaje_gasto']} pts)"
                )

            with st.expander("Ver vector de datos crudos procesado"):
                st.dataframe(X_new)

        except Exception as e:
            st.error(f"Error al calcular la asignación del clúster: {e}")

    # Barra lateral informativa
    with st.sidebar:
        st.header("📌 Parámetros del Modelo")
        st.write(f"- **Algoritmo:** K-Means Clustering")
        st.write(f"- **Número de Clústers (k):** {n_clusters}")
        st.write(f"- **Silhouette Score:** `{silhouette}`")
        st.write(f"- **Variables de Entrada:** `{', '.join(features)}`")
        st.markdown("---")
        st.subheader("📖 Resumen de Perfiles")
        for cid, desc in cluster_descriptions.items():
            st.markdown(f"**Clúster {cid}:** {desc}")
        st.markdown("---")
        st.caption("Reto ML | Scikit-Learn + Streamlit + Docker + GCP")


if __name__ == "__main__":
    main()
