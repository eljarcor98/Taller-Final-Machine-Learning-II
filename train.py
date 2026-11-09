"""
train.py - Entrenamiento y Evaluación Reproducible de Modelos de Machine Learning
Plataforma Integral de Vigilancia Epidemiológica (SIVIGILA)
Dataset Oficial: Portal Nacional de Datos Abiertos de Colombia (datos.gov.co)

Incluye:
  1. Ruta A: Aprendizaje Supervisado (Clasificación de Hospitalización)
  2. Ruta B: Aprendizaje No Supervisado (Segmentación con K-Means)
"""

from pathlib import Path
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, silhouette_score

# ==========================================
# 1. CONFIGURACIÓN DE RUTAS Y CONSTANTES
# ==========================================
DATA_DIR = Path("data")
DATA_FILE = DATA_DIR / "datos.csv"
MODEL_PATH = Path("model.joblib")


def load_and_prepare_data(data_path: Path = None) -> pd.DataFrame:
    """Carga y estructura el dataset oficial de datos.gov.co."""
    if data_path is None:
        if DATA_FILE.exists():
            data_path = DATA_FILE
        else:
            csv_files = list(DATA_DIR.glob("*.csv"))
            if csv_files:
                data_path = csv_files[0]
            else:
                raise FileNotFoundError(
                    f"No se encontró el dataset en '{DATA_DIR.resolve()}'. Asegúrese de que exista 'datos.csv'."
                )

    print(f"[INFO] Cargando dataset oficial desde: {data_path}")
    df_raw = pd.read_csv(data_path)
    print(f"[INFO] Registros leídos: {df_raw.shape[0]} filas, {df_raw.shape[1]} columnas")

    df = pd.DataFrame()
    df["edad"] = pd.to_numeric(df_raw["Edad_paciente"], errors="coerce")
    df["semana_epidemiologica"] = pd.to_numeric(df_raw["Semana_epidemiológica"], errors="coerce")
    df["estrato"] = pd.to_numeric(df_raw["Estrato_socioeconómico"], errors="coerce")
    df["sexo"] = df_raw["Sexo_paciente"].astype(str).str.upper().str.strip().fillna("F")
    df["regimen_salud"] = df_raw["Tipo Seguridad Social"].astype(str).str.upper().str.strip().fillna("S")
    df["gestante"] = df_raw["Gestante"].astype(str).str.upper().str.strip().fillna("NO")
    df["grupo_riesgo"] = df_raw["Otros_grupos_riesgo"].astype(str).str.upper().str.strip().fillna("SI")
    df["indigena"] = df_raw["Pertenencia_indígena"].astype(str).str.upper().str.strip().fillna("NO")
    df["evento_reportado"] = df_raw["Evento_reportado"].astype(str).str.upper().str.strip().fillna("OTRO")

    # Indicador binario numérico y etiqueta para supervisado
    df["hospitalizado_num"] = (df_raw["Hospitalizacion"].astype(str).str.upper().str.strip() == "SI").astype(float)
    df["hospitalizacion"] = df_raw["Hospitalizacion"].astype(str).str.upper().str.strip()
    df = df[df["hospitalizacion"].isin(["SI", "NO"])].copy()

    return df


def train_supervised_model(df: pd.DataFrame) -> dict:
    """Construye y evalúa el Pipeline de Clasificación Supervisada."""
    num_features = ["edad", "semana_epidemiologica", "estrato"]
    cat_features = ["evento_reportado", "sexo", "regimen_salud", "gestante", "grupo_riesgo", "indigena"]
    features = num_features + cat_features

    X = df[features].copy()
    y = df["hospitalizacion"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, num_features),
        ("cat", categorical_pipe, cat_features)
    ])

    classifier = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    print("\n" + "=" * 65)
    print("      ENTRENAMIENTO: MODELO SUPERVISADO (HOSPITALIZACIÓN)")
    print("=" * 65)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average="weighted")
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=["NO", "SI"])

    print(f"F1-Score Ponderado: {f1:.4f}")
    print(f"Exactitud (Accuracy): {acc * 100:.2f}%\n")
    print("Matriz de Confusión [NO, SI]:")
    print(cm)
    print("\nReporte de Clasificación:")
    print(classification_report(y_test, y_pred))

    category_options = {
        col: sorted(df[col].dropna().unique().tolist())
        for col in cat_features
    }

    numeric_stats = {}
    for col in num_features:
        numeric_stats[col] = {
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "median": float(df[col].median()),
            "mean": float(df[col].mean())
        }

    return {
        "model": model,
        "features": features,
        "numeric_features": num_features,
        "categorical_features": cat_features,
        "category_options": category_options,
        "numeric_stats": numeric_stats,
        "target_name": "Hospitalización",
        "classes": ["NO", "SI"],
        "metrics": {
            "f1_score": round(float(f1), 4),
            "accuracy": round(float(acc), 4)
        }
    }


def train_unsupervised_model(df: pd.DataFrame) -> dict:
    """Construye y evalúa el Pipeline de Agrupamiento No Supervisado (K-Means)."""
    features = ["edad", "semana_epidemiologica", "estrato", "hospitalizado_num"]
    X = df[features].copy()

    N_CLUSTERS = 4
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("cluster", KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init="auto"))
    ])

    print("\n" + "=" * 65)
    print("      ENTRENAMIENTO: MODELO NO SUPERVISADO (K-MEANS)")
    print("=" * 65)
    labels = model.fit_predict(X)

    X_transformed = model[:-1].transform(X)
    score_silhouette = silhouette_score(X_transformed, labels)
    inertia = model.named_steps["cluster"].inertia_

    print(f"Número de Clusters (k): {N_CLUSTERS}")
    print(f"Silhouette Score:       {score_silhouette:.4f}")
    print(f"Inercia (WCSS):         {inertia:.2f}")

    df_eval = df.copy()
    df_eval["cluster"] = labels

    profiles = df_eval.groupby("cluster")[features].mean().round(2)
    counts = df_eval["cluster"].value_counts()
    profiles["total_casos"] = counts
    profiles["porcentaje"] = (counts / len(df_eval) * 100).round(1).astype(str) + "%"

    print("\nPerfiles Promedio por Clúster:")
    print(profiles)

    cluster_descriptions = {
        0: "Población Pediátrica y Juvenil Ambulatoria: Pacientes jóvenes (edad promedio ~12 años), estrato 1, no hospitalizados. Casos de Malaria ambulatoria, dengue no grave y seguimiento de violencia intrafamiliar.",
        1: "Adultos y Adultos Mayores Vulnerables: Edad promedio ~52 años, estrato 1, manejo mayoritariamente ambulatorio (~15% internados). Afectados por malaria endémica y dengue en etapas adultas.",
        2: "Casos Críticos de Urgencia Hospitalaria: 100% requirieron internación hospitalaria. Alta presencia de menores de edad y jóvenes (~14 años) con desnutrición aguda severa y malaria complicada.",
        3: "Población Urbana Estrato Medio / Zoonosis: Edad promedio ~33 años, estratos 2 y 3. Casos urbanos concentrados en brotes de Dengue, leishmaniasis y agresiones por animales transmisores de rabia."
    }

    numeric_stats = {}
    for col in features:
        numeric_stats[col] = {
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "median": float(df[col].median()),
            "mean": float(df[col].mean())
        }

    return {
        "model": model,
        "features": features,
        "n_clusters": N_CLUSTERS,
        "silhouette": round(float(score_silhouette), 4),
        "inertia": round(float(inertia), 2),
        "cluster_descriptions": cluster_descriptions,
        "profiles": profiles.to_dict(orient="index"),
        "numeric_stats": numeric_stats
    }


def main() -> None:
    df = load_and_prepare_data()
    sup_artifact = train_supervised_model(df)
    unsup_artifact = train_unsupervised_model(df)

    consolidated_artifact = {
        "supervised": sup_artifact,
        "unsupervised": unsup_artifact,
        "total_cases": len(df)
    }

    joblib.dump(consolidated_artifact, MODEL_PATH)
    print("\n" + "=" * 65)
    print(f"[INFO] Artefacto consolidado guardado exitosamente en:\n       {MODEL_PATH.resolve()}")
    print("=" * 65)


if __name__ == "__main__":
    main()
