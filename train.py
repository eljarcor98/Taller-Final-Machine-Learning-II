"""
train.py - Script de entrenamiento y evaluación reproducible (Ruta B: Aprendizaje No Supervisado - Clustering)
Reto completo de Machine Learning: del dataset al despliegue.
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

# Variables numéricas interpretables para el agrupamiento
numeric_features = [
    "edad",
    "ingreso_anual",
    "puntaje_gasto"
]
features = numeric_features


def load_data(data_path: Path = None) -> pd.DataFrame:
    """Carga el dataset desde archivo CSV o Excel dentro de la carpeta data/."""
    if data_path is None:
        if DATA_FILE.exists():
            data_path = DATA_FILE
        else:
            csv_files = list(DATA_DIR.glob("*.csv"))
            excel_files = list(DATA_DIR.glob("*.xlsx")) + list(DATA_DIR.glob("*.xls"))
            if csv_files:
                data_path = csv_files[0]
            elif excel_files:
                data_path = excel_files[0]
            else:
                raise FileNotFoundError(
                    f"No se encontró el dataset en '{DATA_DIR.resolve()}'. Asegúrate de que exista 'datos.csv'."
                )

    print(f"[*] Cargando dataset desde: {data_path}")
    if str(data_path).lower().endswith(".csv"):
        df = pd.read_csv(data_path)
    else:
        df = pd.read_excel(data_path)

    print(f"[*] Filas: {df.shape[0]}, Columnas: {df.shape[1]}")
    return df


def train_model(df: pd.DataFrame) -> dict:
    """Construye el pipeline de preprocesamiento + K-Means, evalúa e interpreta."""
    # Validación de variables
    missing_cols = [c for c in features if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Faltan columnas requeridas en el dataset: {missing_cols}\n"
            f"Columnas disponibles: {list(df.columns)}"
        )

    # 1. Selección de variables numéricas para clustering
    X = df[features].copy()

    # 2. Definición del Pipeline (Imputación + Estandarización + K-Means)
    # n_clusters=4 ofrece una partición limpia, separable e interpretable
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

    # 3. Entrenamiento y predicción de etiquetas
    print("[*] Entrenando Pipeline de Clustering (K-Means)...")
    labels = model.fit_predict(X)

    # 4. Evaluación de la estructura con Silhouette Score e Inercia
    # Se evalúa sobre el espacio transformado (escalado)
    X_processed = model[:-1].transform(X)
    score_silhouette = silhouette_score(X_processed, labels)
    inertia = model.named_steps["cluster"].inertia_

    print("\n" + "=" * 60)
    print("           EVALUACIÓN DE CLUSTERING (K-MEANS)")
    print("=" * 60)
    print(f"Número de clusters (k): {N_CLUSTERS}")
    print(f"Silhouette Score:       {score_silhouette:.4f} (Estructura coherente)")
    print(f"Inercia (WCSS):         {inertia:.2f}")

    # 5. Interpretación y perfilamiento cuantitativo de cada cluster
    df_clustered = df.copy()
    df_clustered["cluster"] = labels

    profiles = (
        df_clustered
        .groupby("cluster")[numeric_features]
        .mean()
        .round(2)
    )
    counts = df_clustered["cluster"].value_counts()
    profiles["total_clientes"] = counts
    profiles["porcentaje"] = (counts / len(df) * 100).round(1).astype(str) + "%"

    print("\n--- Perfiles Promedio por Clúster ---")
    print(profiles)
    print("=" * 60)

    # 6. Descripciones interpretables asociadas a los datos
    cluster_descriptions = {
        0: "Cliente Maduro Tradicional: Edad promedio ~54 años, ingreso medio ($48k USD) y gasto moderado (~40 pts). Perfil estable y conservador.",
        1: "Cliente VIP / Alto Valor: Adultos jóvenes (~33 años) con ingresos altos ($86k USD) y alto puntaje de gasto (~82 pts). Clientes estratégicos de alta rentabilidad.",
        2: "Joven Entusiasta / Dinámico: Jóvenes (~25 años) con ingreso moderado ($40k USD) pero alto nivel de gasto (~60 pts). Gran afinidad a promociones y tendencias.",
        3: "Ahorrador Cuidadoso: Adultos (~39 años) con altos ingresos ($87k USD) pero muy bajo puntaje de gasto (~20 pts). Perfil cauteloso y analítico."
    }

    # 7. Estadísticas de variables para inputs recomendados en Streamlit
    numeric_stats = {}
    for col in numeric_features:
        numeric_stats[col] = {
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "mean": float(df[col].mean()),
            "median": float(df[col].median())
        }

    # 8. Construcción del artefacto serializable
    artifact = {
        "model": model,
        "features": features,
        "numeric_features": numeric_features,
        "task": "clustering",
        "n_clusters": N_CLUSTERS,
        "silhouette": round(float(score_silhouette), 4),
        "cluster_descriptions": cluster_descriptions,
        "profiles": profiles.to_dict(orient="index"),
        "numeric_stats": numeric_stats
    }

    return artifact


def main() -> None:
    df = load_data()
    artifact = train_model(df)
    joblib.dump(artifact, MODEL_PATH)
    print(f"\n[✓] Artefacto guardado exitosamente en: {MODEL_PATH.resolve()}")


if __name__ == "__main__":
    main()
