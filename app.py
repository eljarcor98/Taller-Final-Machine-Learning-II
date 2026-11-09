"""
app.py - Interfaz de Usuario Interactiva en Streamlit (Ruta A: Clasificación Supervisada)
Sistema de Triaje Epidemiológico y Predicción de Hospitalización (SIVIGILA)
Dataset Oficial: Portal Nacional de Datos Abiertos de Colombia (datos.gov.co)
"""

from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

# Configuración inicial de la página
st.set_page_config(
    page_title="SIVIGILA | Predicción de Hospitalización",
    page_icon="🏥",
    layout="centered"
)

MODEL_PATH = Path("model.joblib")


@st.cache_resource
def load_artifact():
    """Carga el modelo y sus metadatos desde el artefacto serializado."""
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def main():
    st.title("🏥 SIVIGILA: Predicción de Hospitalización")
    st.caption("Sistema de triaje y priorización clínica con Machine Learning (Scikit-Learn + Docker)")

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
    categorical_features = artifact.get("categorical_features", [])
    category_options = artifact.get("category_options", {})
    numeric_stats = artifact.get("numeric_stats", {})
    metrics = artifact.get("metrics", {})
    total_cases = artifact.get("total_cases", 1226)

    st.markdown("### Ingrese los datos del paciente y del evento notificado:")
    st.info("El modelo evalúa los factores de riesgo clínico y estima la probabilidad de requerir internación hospitalaria.")

    user_inputs = {}

    with st.form("triage_form"):
        st.subheader("📋 Información del Evento y Demografía")
        c1, c2 = st.columns(2)

        # Evento reportado
        event_opts = category_options.get("evento_reportado", ["MALARIA", "DENGUE"])
        user_inputs["evento_reportado"] = c1.selectbox(
            "🦠 Evento Epidemiológico Reportado",
            options=event_opts,
            index=0 if "MALARIA" not in event_opts else event_opts.index("MALARIA")
        )

        # Edad del paciente
        stats_edad = numeric_stats.get("edad", {"median": 20.0})
        user_inputs["edad"] = float(c2.number_input(
            "👶/👵 Edad del Paciente (años)",
            min_value=0.0,
            max_value=110.0,
            value=float(stats_edad.get("median", 20.0)),
            step=1.0
        ))

        c3, c4, c5 = st.columns(3)
        # Semana epidemiológica
        user_inputs["semana_epidemiologica"] = float(c3.slider(
            "📅 Semana Epidemiológica (1-52)",
            min_value=1,
            max_value=52,
            value=25
        ))

        # Estrato
        user_inputs["estrato"] = float(c4.selectbox(
            "🏠 Estrato Socioeconómico",
            options=[1.0, 2.0, 3.0, 4.0],
            index=0
        ))

        # Sexo
        user_inputs["sexo"] = c5.selectbox(
            "⚧ Sexo",
            options=category_options.get("sexo", ["F", "M"])
        )

        st.subheader("🛡️ Factores de Vulnerabilidad y Seguridad Social")
        c6, c7 = st.columns(2)

        # Régimen de salud con etiquetas amigables
        regimen_map = {
            "S": "S - Subsidiado",
            "C": "C - Contributivo",
            "I": "I - Indeterminado / Sin afiliación",
            "E": "E - Especial / Magisterio",
            "P": "P - Excepción / Policía / FFMM",
            "N": "N - No asegurado"
        }
        reg_opts = category_options.get("regimen_salud", ["S", "C", "I"])
        selected_reg = c6.selectbox(
            "🏥 Régimen de Seguridad Social",
            options=reg_opts,
            format_func=lambda x: regimen_map.get(x, x)
        )
        user_inputs["regimen_salud"] = selected_reg

        # Gestante
        user_inputs["gestante"] = c7.selectbox(
            "🤰 ¿Paciente Gestante?",
            options=category_options.get("gestante", ["NO", "SI"])
        )

        c8, c9 = st.columns(2)
        # Grupo de riesgo
        user_inputs["grupo_riesgo"] = c8.selectbox(
            "⚠️ ¿Pertenece a Grupo de Riesgo Comunitario?",
            options=category_options.get("grupo_riesgo", ["SI", "NO"]),
            index=0
        )

        # Pertenencia indígena
        user_inputs["indigena"] = c9.selectbox(
            "🌿 ¿Pertenencia Étnica Indígena?",
            options=category_options.get("indigena", ["NO", "SI"]),
            index=0
        )

        submitted = st.form_submit_button("🔮 Evaluar Riesgo y Predecir Hospitalización", use_container_width=True)

    if submitted:
        # Construir DataFrame en el orden exacto de características
        X_new = pd.DataFrame([user_inputs], columns=features)

        try:
            # Predicción con el Pipeline
            pred = str(model.predict(X_new)[0])
            probs = model.predict_proba(X_new)[0]
            classes = list(model.classes_)
            prob_dict = {cls: prob for cls, prob in zip(classes, probs)}

            prob_si = prob_dict.get("SI", 0.0)
            prob_no = prob_dict.get("NO", 0.0)

            st.markdown("---")
            st.subheader("📋 Dictamen del Modelo de Triaje")

            if pred == "SI":
                st.error(
                    "🚨 **ALERTA CLÍNICA: Paciente con Alto Riesgo de Requerir Hospitalización**\n\n"
                    "*Recomendación:* Se sugiere priorizar asignación de cama hospitalaria, aislamiento según protocolo "
                    "y valoración médica presencial de urgencia."
                )
            else:
                st.success(
                    "✅ **MANEJO AMBULATORIO SUGERIDO**\n\n"
                    "*Recomendación:* El paciente presenta un perfil compatible con manejo domiciliario controlado, "
                    "prescripción de tratamiento ambulatorio y seguimiento epidemiológico comunitario."
                )

            # Visualización de probabilidades
            st.markdown("#### 🎯 Distribución de Certeza del Modelo:")
            p_col1, p_col2 = st.columns(2)
            p_col1.metric("Probabilidad de Hospitalización (SI)", f"{prob_si * 100:.1f}%")
            p_col1.progress(float(prob_si))

            p_col2.metric("Probabilidad de Manejo Ambulatorio (NO)", f"{prob_no * 100:.1f}%")
            p_col2.progress(float(prob_no))

            with st.expander("Ver vector de datos enviado al Pipeline de Scikit-Learn"):
                st.dataframe(X_new)

        except Exception as e:
            st.error(f"Error al procesar la predicción: {e}")

    # Barra lateral
    with st.sidebar:
        st.header("📌 Parámetros del Modelo")
        st.write("- **Tipo:** Clasificación Supervisada")
        st.write("- **Algoritmo:** Regresión Logística (Class-Weighted)")
        st.write(f"- **F1-Score Ponderado:** `{metrics.get('f1_score', 0.7364)}`")
        st.write(f"- **Exactitud (Accuracy):** `{metrics.get('accuracy', 0.7276) * 100:.1f}%`")
        st.write(f"- **Casos Históricos:** `{total_cases:,}`")
        st.markdown("---")
        st.subheader("🌐 Fuente Oficial")
        st.markdown("[Eventos Epidemiológicos | datos.gov.co](https://www.datos.gov.co/Salud-y-Protecci-n-Social/Eventos-Epidemiol-gicos/afi7-frbs/about_data)")
        st.caption("Sistema de Vigilancia en Salud Pública (SIVIGILA) - Ministerio de Salud.")


if __name__ == "__main__":
    main()
