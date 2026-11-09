"""
train.py - Entrenamiento y Evaluación Reproducible (Ruta A: Aprendizaje Supervisado - Clasificación)
Problema: Predicción de Hospitalización en Pacientes Epidemiológicos (SIVIGILA)
Dataset Oficial: Portal Nacional de Datos Abiertos de Colombia (datos.gov.co)
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
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

# ==========================================
# 1. CONFIGURACIÓN DE RUTAS Y VARIABLES
# ==========================================
DATA_DIR = Path("data")
DATA_FILE = DATA_DIR / "datos.csv"
MODEL_PATH = Path("model.joblib")

# Variable objetivo a predecir
TARGET = "hospitalizacion"

# Variables explicativas
numeric_features = [
    "edad",
    "semana_epidemiologica",
    "estrato"
]

categorical_features = [
    "evento_reportado",
    "sexo",
    "regimen_salud",
    "gestante",
    "grupo_riesgo",
    "indigena"
]

features = numeric_features + categorical_features


def load_and_clean_data(data_path: Path = None) -> pd.DataFrame:
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
                    f"No se encontró el dataset en '{DATA_DIR.resolve()}'. Asegúrate de que exista 'datos.csv'."
                )

    print(f"[*] Cargando dataset oficial desde: {data_path}")
    df_raw = pd.read_csv(data_path)
    print(f"[*] Registros leídos: {df_raw.shape[0]} filas, {df_raw.shape[1]} columnas")

    # Mapeo de variables analíticas
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

    # Variable objetivo (Hospitalización: SI / NO)
    df[TARGET] = df_raw["Hospitalizacion"].astype(str).str.upper().str.strip()
    df = df[df[TARGET].isin(["SI", "NO"])].copy()

    print(f"[*] Casos analizados con etiqueta válida: {len(df)}")
    print(f"[*] Distribución de {TARGET}:")
    print(df[TARGET].value_counts(normalize=True).round(3) * 100)

    return df


def train_model(df: pd.DataFrame) -> dict:
    """Construye el pipeline de preprocesamiento + clasificación, entrena y evalúa."""
    X = df[features].copy()
    y = df[TARGET].copy()

    # 1. Separación de datos de entrenamiento (80%) y prueba (20%) estratificado
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 2. Pipeline de preprocesamiento unificado (ColumnTransformer)
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, numeric_features),
        ("cat", categorical_pipe, categorical_features)
    ])

    # 3. Clasificador con balanceo de clases para alta sensibilidad clínica
    classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    )

    # 4. Pipeline consolidado
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    # 5. Entrenamiento
    print("\n[*] Entrenando Pipeline de Clasificación Supervisada...")
    model.fit(X_train, y_train)

    # 6. Evaluación rigurosa en conjunto de prueba
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average="weighted")
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=["NO", "SI"])
    report = classification_report(y_test, y_pred)

    print("\n" + "=" * 65)
    print("      EVALUACIÓN DEL MODELO SUPERVISADO (HOSPITALIZACIÓN)")
    print("=" * 65)
    print(f"F1-Score ponderado: {f1:.4f}")
    print(f"Exactitud (Accuracy): {acc * 100:.2f}%\n")
    print("--- Matriz de Confusión [NO, SI] ---")
    print(cm)
    print("\n--- Reporte de Clasificación Detallado ---")
    print(report)
    print("=" * 65)

    # 7. Opciones de categorías observadas durante el entrenamiento
    category_options = {
        col: sorted(df[col].dropna().unique().tolist())
        for col in categorical_features
    }

    # 8. Estadísticas numéricas para los widgets de Streamlit
    numeric_stats = {}
    for col in numeric_features:
        numeric_stats[col] = {
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "median": float(df[col].median()),
            "mean": float(df[col].mean())
        }

    # 9. Artefacto serializable
    artifact = {
        "model": model,
        "features": features,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "category_options": category_options,
        "numeric_stats": numeric_stats,
        "task": "classification",
        "target_name": "Hospitalización",
        "classes": ["NO", "SI"],
        "metrics": {
            "f1_score": round(float(f1), 4),
            "accuracy": round(float(acc), 4)
        },
        "total_cases": len(df)
    }

    return artifact


def main() -> None:
    df = load_and_clean_data()
    artifact = train_model(df)
    joblib.dump(artifact, MODEL_PATH)
    print(f"\n[✓] Artefacto guardado exitosamente en: {MODEL_PATH.resolve()}")


if __name__ == "__main__":
    main()
