# Plataforma Analítica de Vigilancia Epidemiológica (SIVIGILA) con Streamlit, Docker y GCP

## 1. Información General
* **Título del Proyecto:** Plataforma Analítica Integral de Inteligencia Epidemiológica
* **Enfoque Desarrollado:** Solución Dual Unificada (Ruta A: Aprendizaje Supervisado + Ruta B: Aprendizaje No Supervisado)
* **Autores:** Arnold Santiago Torres (@eljarcor98) & [Compañero/a de pareja]
* **Asignatura:** Machine Learning 2
* **Repositorio Oficial:** [github.com/eljarcor98/Taller-Final-Machine-Learning-II](https://github.com/eljarcor98/Taller-Final-Machine-Learning-II)

---

## 2. Objetivo y Descripción del Dataset
* **Objetivo Institucional / Salud Pública:** Proveer una herramienta de apoyo a la toma de decisiones clínicas y gestión poblacional mediante dos capacidades analíticas complementarias:
  1. **Triaje Predictivo (Supervisado):** Estimar la probabilidad de requerir hospitalización para optimizar la asignación preventiva de camas e internación urgente.
  2. **Caracterización Poblacional (No Supervisado):** Agrupar y segmentar los casos notificados en perfiles epidemiológicos homogéneos según demografía, estacionalidad y vulnerabilidad.
* **Unidad de Observación:** Notificación individual de evento de salud pública reportada al Sistema de Vigilancia en Salud Pública (SIVIGILA).
* **Origen de los datos:** Portal Nacional de Datos Abiertos de Colombia ([datos.gov.co - Eventos Epidemiológicos](https://www.datos.gov.co/Salud-y-Protecci-n-Social/Eventos-Epidemiol-gicos/afi7-frbs/about_data)).
* **Ubicación en el repositorio:** `data/datos.csv` (1,226 casos notificados).

---

## 3. Variables Utilizadas

### Modelo Supervisado (Triaje de Hospitalización)
* **Variable Objetivo (`TARGET`):** `hospitalizacion` (`SI`: 30.7%, `NO`: 69.3%).
* **Variables Numéricas:** `edad`, `semana_epidemiologica`, `estrato`.
* **Variables Categóricas:** `evento_reportado`, `sexo`, `regimen_salud`, `gestante`, `grupo_riesgo`, `indigena`.

### Modelo No Supervisado (Segmentación K-Means)
* **Variables de Agrupamiento:** `edad`, `semana_epidemiologica`, `estrato`, `hospitalizado_num` (indicador 0/1).

---

## 4. Preparación de Datos y Pipelines de Scikit-Learn
Para garantizar total reproducibilidad y prevenir la fuga de datos (*data leakage*), ambos modelos se encapsulan en Pipelines independientes:
1. **Pipeline Supervisado:**
   * Rama numérica: `SimpleImputer(strategy="median")` + `StandardScaler()`.
   * Rama categórica: `SimpleImputer(strategy="most_frequent")` + `OneHotEncoder(handle_unknown="ignore")`.
   * Estimador: `LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)`.
2. **Pipeline No Supervisado:**
   * Imputación mediana + `StandardScaler()`.
   * Estimador: `KMeans(n_clusters=4, random_state=42, n_init="auto")`.

---

## 5. Justificación y Métricas de Rendimiento

### Modelo Supervisado (Clasificación de Hospitalización)
* **F1-Score Ponderado:** `0.7364`
* **Exactitud Global (Accuracy):** `72.76%`
* **Sensibilidad / Recall para Hospitalización (`SI`):** `72.0%` (alta capacidad para identificar casos que requieren internación).
* **Precisión para Manejo Ambulatorio (`NO`):** `86.0%`
* **Justificación:** El balanceo de clases prioriza la reducción de falsos negativos críticos en salud pública, generando probabilidades calibradas mediante `predict_proba()`.

### Modelo No Supervisado (Clustering K-Means)
* **Número de Clusters ($k$):** `4`
* **Silhouette Score:** `0.3969` (estructura diferenciada en el espacio normalizado).
* **Inercia intra-cluster (WCSS):** `2011.28`
* **Perfiles Identificados:**
  * **Clúster 0 (45.1%):** Población Pediátrica y Juvenil Ambulatoria (Edad media: 12.2 años, Estrato 1, 0% Hospitalizados).
  * **Clúster 1 (15.4%):** Adultos Mayores Vulnerables (Edad media: 51.9 años, Estrato 1, 15.3% Hospitalizados).
  * **Clúster 2 (26.3%):** Casos Críticos de Urgencia Hospitalaria (Edad media: 14.4 años, 100% Hospitalizados, Desnutrición y malaria severa).
  * **Clúster 3 (13.1%):** Población Urbana Estrato Medio / Zoonosis (Edad media: 32.5 años, Estrato 2.1, Brotes de Dengue y zoonosis).

---

## 6. Arquitectura de la Aplicación en Streamlit
La aplicación [`app.py`](app.py) integra ambos modelos en un solo despliegue interactivo mediante pestañas (`st.tabs`):
* **Pestaña 1 (Triaje Clínico Supervisado):** Formulario clínico completo, dictamen de hospitalización o manejo ambulatorio, y visualización de probabilidades con barras de progreso.
* **Pestaña 2 (Segmentación No Supervisada):** Formulario simplificado, asignación de clúster K-Means, descripción cualitativa del perfil y comparación métrica frente a los promedios del clúster.

---

## 7. Instrucciones de Reproducción Local

### Paso 1: Instalar dependencias
```bash
pip install -r requirements.txt
```

### Paso 2: Entrenar los modelos y generar el artefacto consolidado
```bash
python train.py
```
> Entrena simultáneamente el modelo supervisado y el no supervisado, guardando ambos pipelines en `model.joblib`.

### Paso 3: Ejecutar la aplicación web
```bash
python -m streamlit run app.py
```
Acceder en el navegador a: `http://localhost:8501`.

---

## 8. Despliegue con Docker

### Paso 1: Construir la imagen Docker
```bash
docker build -t ml-streamlit-app .
```

### Paso 2: Ejecutar el contenedor
```bash
docker run -d --name ml-streamlit --restart unless-stopped -p 8501:8501 ml-streamlit-app
```

### Paso 3: Verificar estado y logs
```bash
docker ps
docker logs ml-streamlit
```

---

## 9. Despliegue en Google Cloud Platform (Compute Engine)

1. En Google Cloud Console, ir a **Compute Engine** $\rightarrow$ **Instancias de VM** $\rightarrow$ **Crear instancia**.
2. Seleccionar Linux **Ubuntu 24.04 LTS**, serie económica `e2-micro` o `e2-small`.
3. Asignar la etiqueta de red `streamlit-server` y confirmar dirección IP externa.
4. Crear regla de Firewall en Cloud Shell:
   ```bash
   gcloud compute firewall-rules create allow-streamlit-8501 \
       --allow=tcp:8501 \
       --direction=INGRESS \
       --target-tags=streamlit-server
   ```
5. Acceder por SSH e instalar Docker y Git:
   ```bash
   sudo apt update
   sudo apt install -y docker.io git
   sudo systemctl enable --now docker
   sudo usermod -aG docker $USER
   ```
6. Clonar el repositorio y desplegar:
   ```bash
   git clone https://github.com/eljarcor98/Taller-Final-Machine-Learning-II.git
   cd Taller-Final-Machine-Learning-II
   docker build -t ml-streamlit-app .
   docker run -d --name ml-streamlit --restart unless-stopped -p 8501:8501 ml-streamlit-app
   ```
7. Abrir en el navegador: `http://<IP_EXTERNA_VM>:8501`.

### 9.1 Despliegue Alternativo en Servidor Físico / Raspberry Pi (Edge Computing)
1. En la terminal de la Raspberry Pi (ARM64):
   ```bash
   sudo apt update && sudo apt install -y docker.io git
   git clone https://github.com/eljarcor98/Taller-Final-Machine-Learning-II.git
   cd Taller-Final-Machine-Learning-II
   docker build -t ml-streamlit-app .
   docker run -d --name ml-streamlit --restart unless-stopped -p 8501:8501 ml-streamlit-app
   ```
2. Para acceso público remoto seguro sin abrir puertos en el router:
   ```bash
   npx localtunnel --port 8501
   # O mediante Cloudflare Tunnel: cloudflared tunnel --url http://localhost:8501
   ```

### 9.2 Despliegue Alternativo PaaS (Render.com / Koyeb)
1. Conectar la cuenta de GitHub con [Render.com](https://render.com) o [Koyeb.com](https://www.koyeb.com).
2. Crear un nuevo servicio apuntando a `Taller-Final-Machine-Learning-II`.
3. Seleccionar el entorno **Docker** (detectará automáticamente el `Dockerfile` y la variable de puerto `$PORT`).
4. La plataforma construirá el contenedor y asignará una URL pública segura con HTTPS.

---

## 10. Casos de Prueba para Evaluación Cruzada

Para que otra pareja compruebe la reproducibilidad completa de ambos modelos en la interfaz:

### Pruebas en Pestaña 1 (Modelo Supervisado):
* **Caso A (Alta Complejidad - Hospitalización Requerida):**
  * Entradas: `Edad = 28`, `Semana = 20`, `Estrato = 1`, `Evento = MORBILIDAD MATERNA EXTREMA`, `Sexo = F`, `Régimen = S`, `Gestante = SI`, `Grupo de Riesgo = SI`, `Indígena = NO`
  * Salida Esperada: **`Clase SI`** (Probabilidad de Hospitalización: **~88.8%**).
* **Caso B (Manejo Ambulatorio):**
  * Entradas: `Edad = 35`, `Semana = 35`, `Estrato = 2`, `Evento = AGRESIONES POR ANIMALES POTENCIALMENTE TRANSMISORES DE RABIA`, `Sexo = M`, `Régimen = C`, `Gestante = NO`, `Grupo de Riesgo = NO`, `Indígena = NO`
  * Salida Esperada: **`Clase NO`** (Probabilidad de Manejo Ambulatorio: **~93.6%**).

### Pruebas en Pestaña 2 (Modelo No Supervisado - K-Means):
* **Caso C (Población Pediátrica Ambulatoria):**
  * Entradas: `Edad = 8`, `Semana = 28`, `Estrato = 1`, `Condición = Manejo ambulatorio (0)`
  * Salida Esperada: **`Clúster 0`** (Población Pediátrica y Juvenil Ambulatoria).
* **Caso D (Urgencia Hospitalaria):**
  * Entradas: `Edad = 15`, `Semana = 20`, `Estrato = 1`, `Condición = Internación hospitalaria (1)`
  * Salida Esperada: **`Clúster 2`** (Casos Críticos de Urgencia Hospitalaria).
