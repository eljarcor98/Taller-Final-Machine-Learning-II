"""
app.py - Plataforma Integral de Vigilancia Epidemiológica (SIVIGILA)
Implementa en un solo despliegue interactivo:
  - Pestaña 1: Modelo Supervisado (Predicción de Hospitalización)
  - Pestaña 2: Modelo No Supervisado (Segmentación con K-Means)
Dataset Oficial: Portal Nacional de Datos Abiertos de Colombia (datos.gov.co)
"""

from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

# Configuración institucional de la aplicación
st.set_page_config(
    page_title="SIVIGILA | Plataforma Analítica",
    layout="centered"
)

MODEL_PATH = Path("model.joblib")


@st.cache_resource
def load_consolidated_artifact():
    """Carga los modelos supervisado y no supervisado desde el artefacto serializado."""
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def main():
    st.title("SIVIGILA: Plataforma Analítica de Inteligencia Epidemiológica")
    st.caption("Sistema de apoyo a la toma de decisiones clínicas y gestión poblacional en salud pública")

    artifact = load_consolidated_artifact()

    if artifact is None:
        st.error(
            f"No se encontró el archivo del modelo ({MODEL_PATH.name}). "
            "Ejecute previamente en la consola: python train.py"
        )
        st.stop()

    sup_data = artifact.get("supervised", {})
    unsup_data = artifact.get("unsupervised", {})
    total_cases = artifact.get("total_cases", 1226)

    # Navegación integrada en pestañas dentro del mismo despliegue
    tab_sup, tab_unsup = st.tabs([
        "Modelo Supervisado: Triaje de Hospitalización",
        "Modelo No Supervisado: Segmentación K-Means"
    ])

    # =========================================================================
    # PESTAÑA 1: APRENDIZAJE SUPERVISADO (PREDICCIÓN DE HOSPITALIZACIÓN)
    # =========================================================================
    with tab_sup:
        st.subheader("Predicción de Hospitalización en Pacientes Epidemiológicos")
        st.write(
            "Este modelo de clasificación estima la probabilidad de que un paciente requiera "
            "internación hospitalaria con base en sus antecedentes clínicos, demográficos y de vulnerabilidad."
        )

        sup_model = sup_data.get("model")
        sup_features = sup_data.get("features", [])
        category_options = sup_data.get("category_options", {})
        numeric_stats = sup_data.get("numeric_stats", {})

        with st.form("supervised_triage_form"):
            st.markdown("**Variables Clínicas y Demográficas**")
            c1, c2 = st.columns(2)

            event_opts = category_options.get("evento_reportado", ["MALARIA", "DENGUE"])
            idx_default = event_opts.index("MALARIA") if "MALARIA" in event_opts else 0
            evento_sel = c1.selectbox("Evento o patología notificada", options=event_opts, index=idx_default)

            stats_edad = numeric_stats.get("edad", {"median": 20.0})
            edad_val = float(c2.number_input(
                "Edad del paciente (años cumplidos)",
                min_value=0.0,
                max_value=110.0,
                value=float(stats_edad.get("median", 20.0)),
                step=1.0
            ))

            c3, c4, c5 = st.columns(3)
            semana_val = float(c3.slider("Semana epidemiológica del reporte", min_value=1, max_value=52, value=25))
            estrato_val = float(c4.selectbox("Estrato socioeconómico", options=[1.0, 2.0, 3.0, 4.0], index=0))
            sexo_val = c5.selectbox("Sexo biológico", options=category_options.get("sexo", ["F", "M"]))

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
            regimen_val = c6.selectbox(
                "Régimen de afiliación en salud",
                options=reg_opts,
                format_func=lambda x: regimen_map.get(x, x)
            )

            gestante_val = c7.selectbox("Condición de gestante", options=category_options.get("gestante", ["NO", "SI"]))

            c8, c9 = st.columns(2)
            grupo_riesgo_val = c8.selectbox(
                "Pertenencia a grupo poblacional de riesgo",
                options=category_options.get("grupo_riesgo", ["SI", "NO"]),
                index=0
            )

            indigena_val = c9.selectbox(
                "Pertenencia étnica indígena",
                options=category_options.get("indigena", ["NO", "SI"]),
                index=0
            )

            btn_evaluar_sup = st.form_submit_button("Evaluar Caso Clínico", use_container_width=True)

        if btn_evaluar_sup and sup_model is not None:
            input_dict_sup = {
                "edad": edad_val,
                "semana_epidemiologica": semana_val,
                "estrato": estrato_val,
                "evento_reportado": evento_sel,
                "sexo": sexo_val,
                "regimen_salud": regimen_val,
                "gestante": gestante_val,
                "grupo_riesgo": grupo_riesgo_val,
                "indigena": indigena_val
            }
            X_new_sup = pd.DataFrame([input_dict_sup], columns=sup_features)

            try:
                pred_sup = str(sup_model.predict(X_new_sup)[0])
                probs_sup = sup_model.predict_proba(X_new_sup)[0]
                classes = list(sup_model.classes_)
                prob_dict = {cls: prob for cls, prob in zip(classes, probs_sup)}

                prob_si = prob_dict.get("SI", 0.0)
                prob_no = prob_dict.get("NO", 0.0)

                st.markdown("---")
                st.subheader("Dictamen del Modelo de Triaje")

                if pred_sup == "SI":
                    st.error(
                        "**Dictamen Clínico: Hospitalización Requerida (Clase SI)**\n\n"
                        "El análisis multivariado identifica un perfil de alta severidad clínica. "
                        "Se recomienda priorizar la asignación inmediata de cama hospitalaria y valoración médica presencial."
                    )
                else:
                    st.success(
                        "**Dictamen Clínico: Manejo Ambulatorio Sugerido (Clase NO)**\n\n"
                        "El perfil del paciente es compatible con tratamiento ambulatorio controlado, "
                        "seguimiento domiciliario e indicaciones preventivas de alarma comunitaria."
                    )

                st.markdown("#### Estimación de Probabilidades")
                p1, p2 = st.columns(2)
                p1.metric("Probabilidad de Hospitalización", f"{prob_si * 100:.2f}%")
                p1.progress(float(prob_si))

                p2.metric("Probabilidad de Manejo Ambulatorio", f"{prob_no * 100:.2f}%")
                p2.progress(float(prob_no))

                with st.expander("Ver vector de características procesado"):
                    st.dataframe(X_new_sup)

            except Exception as e:
                st.error(f"Error en la inferencia del modelo supervisado: {e}")

    # =========================================================================
    # PESTAÑA 2: APRENDIZAJE NO SUPERVISADO (SEGMENTACIÓN K-MEANS)
    # =========================================================================
    with tab_unsup:
        st.subheader("Caracterización y Segmentación Poblacional (K-Means)")
        st.write(
            "Este modelo de clustering agrupa los eventos de salud pública en perfiles homogéneos "
            "con base en edad, estacionalidad del brote, estrato y condición de atención."
        )

        unsup_model = unsup_data.get("model")
        unsup_features = unsup_data.get("features", [])
        cluster_descriptions = unsup_data.get("cluster_descriptions", {})
        profiles = unsup_data.get("profiles", {})

        with st.form("unsupervised_clustering_form"):
            st.markdown("**Variables para Agrupamiento Cuantitativo**")
            u_c1, u_c2 = st.columns(2)

            edad_unsup = float(u_c1.number_input(
                "Edad del paciente (años)",
                min_value=0.0,
                max_value=110.0,
                value=25.0,
                step=1.0,
                key="edad_unsup"
            ))

            semana_unsup = float(u_c2.slider(
                "Semana epidemiológica del reporte",
                min_value=1,
                max_value=52,
                value=25,
                key="semana_unsup"
            ))

            u_c3, u_c4 = st.columns(2)
            estrato_unsup = float(u_c3.selectbox(
                "Estrato socioeconómico",
                options=[1.0, 2.0, 3.0, 4.0],
                index=0,
                key="estrato_unsup"
            ))

            hosp_option = u_c4.selectbox(
                "Condición de atención recibida",
                options=["Manejo ambulatorio (0)", "Internación hospitalaria (1)"],
                index=0
            )
            hosp_num = 1.0 if "Internación" in hosp_option else 0.0

            btn_evaluar_unsup = st.form_submit_button("Caracterizar Perfil Poblacional", use_container_width=True)

        if btn_evaluar_unsup and unsup_model is not None:
            input_dict_unsup = {
                "edad": edad_unsup,
                "semana_epidemiologica": semana_unsup,
                "estrato": estrato_unsup,
                "hospitalizado_num": hosp_num
            }
            X_new_unsup = pd.DataFrame([input_dict_unsup], columns=unsup_features)

            try:
                cluster_id = int(unsup_model.predict(X_new_unsup)[0])
                cluster_info = cluster_descriptions.get(cluster_id, "Sin descripción disponible.")

                st.markdown("---")
                st.subheader(f"Segmento Asignado: Clúster {cluster_id}")
                st.info(f"**Perfil Epidemiológico:** {cluster_info}")

                if cluster_id in profiles:
                    c_prof = profiles[cluster_id]
                    st.markdown("#### Comparativa del Caso vs Promedios del Clúster:")
                    m1, m2, m3 = st.columns(3)
                    m1.metric(
                        label="Edad del Paciente",
                        value=f"{edad_unsup:.0f} años",
                        delta=f"{edad_unsup - c_prof['edad']:+.1f} vs promedio ({c_prof['edad']} a)"
                    )
                    m2.metric(
                        label="Semana del Brote",
                        value=f"Semana {semana_unsup:.0f}",
                        delta=f"{semana_unsup - c_prof['semana_epidemiologica']:+.1f} vs media (Sem {c_prof['semana_epidemiologica']})"
                    )
                    m3.metric(
                        label="Estrato",
                        value=f"Estrato {estrato_unsup:.0f}",
                        delta=f"{estrato_unsup - c_prof['estrato']:+.2f} vs media ({c_prof['estrato']})"
                    )

                with st.expander("Ver vector de características cuantitativas procesado"):
                    st.dataframe(X_new_unsup)

            except Exception as e:
                st.error(f"Error en la inferencia del modelo no supervisado: {e}")

    # =========================================================================
    # BARRA LATERAL TÉCNICA E INSTITUCIONAL
    # =========================================================================
    with st.sidebar:
        st.header("Especificaciones Técnicas")
        st.markdown("**Modelo 1: Supervisado**")
        sup_metrics = sup_data.get("metrics", {})
        st.write("- **Algoritmo:** Regresión Logística Ponderada")
        st.write(f"- **F1-Score Ponderado:** `{sup_metrics.get('f1_score', 0.7364)}`")
        st.write(f"- **Exactitud:** `{sup_metrics.get('accuracy', 0.7276) * 100:.2f}%`")

        st.markdown("---")
        st.markdown("**Modelo 2: No Supervisado**")
        st.write("- **Algoritmo:** K-Means Clustering (k=4)")
        st.write(f"- **Silhouette Score:** `{unsup_data.get('silhouette', 0.3969)}`")
        st.write(f"- **Inercia:** `{unsup_data.get('inertia', 2011.28)}`")

        st.markdown("---")
        st.write(f"**Registros Analizados:** `{total_cases:,}`")
        st.subheader("Fuente Institucional")
        st.markdown("[Eventos Epidemiológicos | datos.gov.co](https://www.datos.gov.co/Salud-y-Protecci-n-Social/Eventos-Epidemiol-gicos/afi7-frbs/about_data)")
        st.caption("Sistema de Vigilancia en Salud Pública (SIVIGILA). Ministerio de Salud y Protección Social de Colombia.")


if __name__ == "__main__":
    main()
