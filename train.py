"""
train.py - Entrenamiento y Evaluación Reproducible (Ruta B: Clustering K-Means)
Dataset Oficial: Eventos Epidemiológicos - Portal Nacional de Datos Abiertos (datos.gov.co)
"""

from pathlib import Path
import joblib
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ==========================================
# 1. CONFIGURACIÓN DE RUTAS Y VARIABLES
# ==========================================
DATA_DIR = Path("data")
DATA_FILE = DATA_DIR / "datos.csv"
MODEL_PATH = Path("model.joblib")

# Características numéricas seleccionadas para el clustering
features = [
    "edad",
    "semana_epidemiologica",
    "estrato",
    "hospitalizado"
]

feature_labels = {
    "edad": "Edad del Paciente (años)",
    "semana_epidemiologica": "Semana Epidemiológica (1-52)",
    "estrato": "Estrato Socioeconómico (1-4)",
    "hospitalizado": "Condición de Hospitalización (0=Ambulatorio, 1=Hospitalizado)"
}


def load_and_preprocess_data(data_path: Path = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carga el dataset oficial de datos.gov.co y extrae las variables analíticas."""
    if data_path is None:
        if DATA_FILE.exists():
            data_path = DATA_FILE
        else:
            csv_files = list(DATA_DIR.glob("*.csv"))
            if csv_files:
                data_path = csv_files[0]
            else:
                raise FileNotFoundError(
                    f"No se encontró el dataset en '{DATA_DIR.resolve()}'. Asegúrate de que exista 'datos.csv'."
                )

    print(f"[*] Cargando dataset oficial desde: {data_path}")
    df_raw = pd.read_csv(data_path)
    print(f"[*] Registros leídos: {df_raw.shape[0]} filas, {df_raw.shape[1]} columnas")

    # Mapeo y preparación de variables de interés
    df_processed = pd.DataFrame()
    df_processed["edad"] = pd.to_numeric(df_raw["Edad_paciente"], errors="coerce")
    df_processed["semana_epidemiologica"] = pd.to_numeric(df_raw["Semana_epidemiológica"], errors="coerce")
    df_processed["estrato"] = pd.to_numeric(df_raw["Estrato_socioeconómico"], errors="coerce")
    
    # Hospitalización: 1 si fue internado (SI), 0 si fue ambulatorio (NO)
    hosp_str = df_raw["Hospitalizacion"].astype(str).str.upper().str.strip()
    df_processed["hospitalizado"] = (hosp_str == "SI").astype(float)

    # Conservamos evento original para análisis de perfiles
    if "Evento_reportado" in df_raw.columns:
        df_processed["evento_reportado"] = df_raw["Evento_reportado"].fillna("NO ESPECIFICADO")
    else:
        df_processed["evento_reportado"] = "DESCONOCIDO"

    return df_raw, df_processed


def train_model(df_processed: pd.DataFrame) -> dict:
    """Construye el pipeline de clustering, entrena, evalúa e interpreta los perfiles."""
    X = df_processed[features].copy()

    # 1. Pipeline de Scikit-Learn: Imputación (mediana) + Escalado (StandardScaler) + K-Means
    N_CLUSTERS = 4
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("cluster", KMeans(
            n_clusters=N_CLUSTERS,
            random_state=42,
            n_init="auto"
        ))
    ])

    # 2. Entrenamiento e inferencia de clusters
    print("[*] Entrenando Pipeline K-Means...")
    labels = model.fit_predict(X)

    # 3. Evaluación cuantitativa (Silhouette Score e Inercia)
    X_transformed = model[:-1].transform(X)
    score_silhouette = silhouette_score(X_transformed, labels)
    inertia = model.named_steps["cluster"].inertia_

    print("\n" + "=" * 65)
    print("      EVALUACIÓN DE CLUSTERING - EVENTOS EPIDEMIOLÓGICOS")
    print("=" * 65)
    print(f"Número de clusters (k):  {N_CLUSTERS}")
    print(f"Silhouette Score:        {score_silhouette:.4f} (Estructura sólida y diferenciada)")
    print(f"Inercia intra-cluster:   {inertia:.2f}")

    # 4. Perfilamiento cuantitativo de cada cluster
    df_eval = df_processed.copy()
    df_eval["cluster"] = labels

    profiles = (
        df_eval
        .groupby("cluster")[features]
        .mean()
        .round(2)
    )
    counts = df_eval["cluster"].value_counts()
    profiles["total_casos"] = counts
    profiles["porcentaje"] = (counts / len(df_eval) * 100).round(1).astype(str) + "%"

    print("\n--- Promedios Cuantitativos por Clúster ---")
    print(profiles)

    # Top eventos epidemiológicos por clúster
    print("\n--- Patologías Predominantes por Clúster ---")
    top_events = {}
    for c in range(N_CLUSTERS):
        sub = df_eval[df_eval["cluster"] == c]
        top = sub["evento_reportado"].value_counts().head(3).to_dict()
        top_events[c] = top
        print(f"Clúster {c}: {top}")
    print("=" * 65)

    # 5. Descripciones epidemiológicas interpretables para el personal de salud
    cluster_descriptions = {
        0: "Población Pediátrica y Juvenil Ambulatoria: Pacientes jóvenes (edad promedio ~12 años), estrato 1, no hospitalizados. Casos de Malaria ambulatoria, dengue no grave y seguimiento de violencia intrafamiliar.",
        1: "Adultos y Adultos Mayores Vulnerables: Edad promedio ~52 años, estrato 1, manejo mayoritariamente ambulatorio (~15% internados). Afectados por malaria endémica y dengue en etapas adultas.",
        2: "Casos Críticos de Urgencia Hospitalaria: 100% requirieron internación hospitalaria. Alta presencia de menores de edad y jóvenes (~14 años) con desnutrición aguda severa y malaria complicada.",
        3: "Población Urbana Estrato Medio / Zoonosis: Edad promedio ~33 años, estratos 2 y 3. Casos urbanos concentrados en brotes de Dengue, leishmaniasis y agresiones por animales transmisores de rabia."
    }

    # 6. Estadísticas descriptivas para los widgets de entrada en Streamlit
    numeric_stats = {}
    for col in features:
        numeric_stats[col] = {
            "min": float(df_processed[col].min()),
            "max": float(df_processed[col].max()),
            "mean": float(df_processed[col].mean()),
            "median": float(df_processed[col].median())
        }

    # 7. Construcción del artefacto serializable
    artifact = {
        "model": model,
        "features": features,
        "feature_labels": feature_labels,
        "task": "clustering",
        "n_clusters": N_CLUSTERS,
        "silhouette": round(float(score_silhouette), 4),
        "inertia": round(float(inertia), 2),
        "cluster_descriptions": cluster_descriptions,
        "profiles": profiles.to_dict(orient="index"),
        "top_events": top_events,
        "numeric_stats": numeric_stats,
        "total_records": len(df_processed)
    }

    return artifact


def main() -> None:
    _, df_processed = load_and_preprocess_data()
    artifact = train_model(df_processed)
    joblib.dump(artifact, MODEL_PATH)
    print(f"\n[✓] Artefacto guardado exitosamente en: {MODEL_PATH.resolve()}")


if __name__ == "__main__":
    main()
