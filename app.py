"""
app.py - Interfaz de Usuario para Triaje Epidemiológico (Ruta A: Clasificación Supervisada)
Sistema de Vigilancia en Salud Pública (SIVIGILA)
Dataset Oficial: Portal Nacional de Datos Abiertos de Colombia (datos.gov.co)
"""

from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

# Configuración de página institucional
st.set_page_config(
    page_title="SIVIGILA | Predicción de Hospitalización",
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
    st.title("SIVIGILA: Sistema de Predicción de Hospitalización")
    st.caption("Modelo predictivo para el apoyo a la toma de decisiones clínicas y priorización hospitalaria")

    artifact = load_artifact()

    if artifact is None:
        st.error(
            f"No se encontró el archivo del modelo ({MODEL_PATH.name}). "
            "Ejecute el script de entrenamiento previo: python train.py"
        )
        st.stop()

    model = artifact["model"]
    features = artifact["features"]
    category_options = artifact.get("category_options", {})
    numeric_stats = artifact.get("numeric_stats", {})
    metrics = artifact.get("metrics", {})
    total_cases = artifact.get("total_cases", 1226)

    st.markdown("#### Formulario de Notificación y Evaluación del Paciente")
    st.write(
        "Ingrese las variables demográficas, clínicas y epidemiológicas observadas en la ficha de notificación. "
        "El sistema aplicará el pipeline de preprocesamiento y estimará la necesidad de internación hospitalaria."
    )

    user_inputs = {}

    with st.form("clinical_triage_form"):
        st.markdown("**Variables Demográficas y Evento Epidemiológico**")
        c1, c2 = st.columns(2)

        event_opts = category_options.get("evento_reportado", ["MALARIA", "DENGUE"])
        user_inputs["evento_reportado"] = c1.selectbox(
            "Evento o patología notificada",
            options=event_opts,
            index=0 if "MALARIA" not in event_opts else event_opts.index("MALARIA")
        )

        stats_edad = numeric_stats.get("edad", {"median": 20.0})
        user_inputs["edad"] = float(c2.number_input(
            "Edad del paciente (años cumplidos)",
            min_value=0.0,
            max_value=110.0,
            value=float(stats_edad.get("median", 20.0)),
            step=1.0
        ))

        c3, c4, c5 = st.columns(3)
        user_inputs["semana_epidemiologica"] = float(c3.slider(
            "Semana epidemiológica del reporte",
            min_value=1,
            max_value=52,
            value=25
        ))

        user_inputs["estrato"] = float(c4.selectbox(
            "Estrato socioeconómico de residencia",
            options=[1.0, 2.0, 3.0, 4.0],
            index=0
        ))

        user_inputs["sexo"] = c5.selectbox(
            "Sexo biológico",
            options=category_options.get("sexo", ["F", "M"])
        )

        st.markdown("**Seguridad Social y Factores de Riesgo Institucional**")
        c6, c7 = st.columns(2)

        regimen_map = {
            "S": "S - Subsidiado",
            "C": "C - Contributivo",
            "I": "I - Indeterminado / Sin afiliación",
            "E": "E - Especial / Magisterio",
            "P": "P - Excepción / Fuerza Pública",
            "N": "N - No asegurado"
        }
        reg_opts = category_options.get("regimen_salud", ["S", "C", "I"])
        selected_reg = c6.selectbox(
            "Régimen de afiliación en salud",
            options=reg_opts,
            format_func=lambda x: regimen_map.get(x, x)
        )
        user_inputs["regimen_salud"] = selected_reg

        user_inputs["gestante"] = c7.selectbox(
            "Condición de gestante",
            options=category_options.get("gestante", ["NO", "SI"])
        )

        c8, c9 = st.columns(2)
        user_inputs["grupo_riesgo"] = c8.selectbox(
            "Pertenencia a grupo poblacional de riesgo",
            options=category_options.get("grupo_riesgo", ["SI", "NO"]),
            index=0
        )

        user_inputs["indigena"] = c9.selectbox(
            "Pertenencia étnica indígena",
            options=category_options.get("indigena", ["NO", "SI"]),
            index=0
        )

        submitted = st.form_submit_button("Evaluar Caso Clínico", use_container_width=True)

    if submitted:
        X_new = pd.DataFrame([user_inputs], columns=features)

        try:
            pred = str(model.predict(X_new)[0])
            probs = model.predict_proba(X_new)[0]
            classes = list(model.classes_)
            prob_dict = {cls: prob for cls, prob in zip(classes, probs)}

            prob_si = prob_dict.get("SI", 0.0)
            prob_no = prob_dict.get("NO", 0.0)

            st.markdown("---")
            st.subheader("Dictamen del Modelo de Triaje")

            if pred == "SI":
                st.error(
                    "**Dictamen Clínico: Hospitalización Requerida (Clase SI)**\n\n"
                    "El análisis de factores de riesgo indica una probabilidad elevada de requerir internación. "
                    "Se recomienda priorizar la disponibilidad de cama hospitalaria y proceder a la valoración médica presencial inmediata."
                )
            else:
                st.success(
                    "**Dictamen Clínico: Manejo Ambulatorio Sugerido (Clase NO)**\n\n"
                    "El perfil de variables es compatible con tratamiento domiciliario controlado. "
                    "Se sugiere prescripción ambulatoria, indicaciones de alarma al paciente y seguimiento epidemiológico comunitario."
                )

            st.markdown("#### Estimación de Probabilidades")
            p_col1, p_col2 = st.columns(2)
            p_col1.metric("Probabilidad de Hospitalización", f"{prob_si * 100:.2f}%")
            p_col1.progress(float(prob_si))

            p_col2.metric("Probabilidad de Manejo Ambulatorio", f"{prob_no * 100:.2f}%")
            p_col2.progress(float(prob_no))

            with st.expander("Detalle del vector de características procesado"):
                st.dataframe(X_new)

        except Exception as e:
            st.error(f"Error durante el procesamiento de la inferencia: {e}")

    with st.sidebar:
        st.header("Especificaciones Técnicas")
        st.write("**Tarea:** Clasificación Supervisada")
        st.write("**Modelo:** Regresión Logística con Ponderación de Clases")
        st.write(f"**F1-Score Ponderado:** `{metrics.get('f1_score', 0.7364)}`")
        st.write(f"**Exactitud (Accuracy):** `{metrics.get('accuracy', 0.7276) * 100:.2f}%`")
        st.write(f"**Registros de Validación:** `{total_cases:,}`")
        st.markdown("---")
        st.subheader("Fuente de Información")
        st.markdown("[Eventos Epidemiológicos | datos.gov.co](https://www.datos.gov.co/Salud-y-Protecci-n-Social/Eventos-Epidemiol-gicos/afi7-frbs/about_data)")
        st.caption("Sistema de Vigilancia en Salud Pública (SIVIGILA). Ministerio de Salud y Protección Social de Colombia.")


if __name__ == "__main__":
    main()
