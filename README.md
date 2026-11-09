# Reto ML: Segmentación de Eventos Epidemiológicos con Streamlit, Docker y GCP

## 1. Información General
* **Título del Proyecto:** Sistema Inteligente de Segmentación y Caracterización de Eventos Epidemiológicos (SIVIGILA)
* **Ruta Elegida:** Ruta B - Aprendizaje No Supervisado (Clustering con K-Means)
* **Autores:** Arnold Santiago Torres (@eljarcor98) & [Compañero/a de pareja]
* **Asignatura:** Machine Learning 2

---

## 2. Objetivo y Descripción del Dataset
* **Objetivo de Negocio / Salud Pública:** Descubrir patrones poblacionales, demográficos y clínicos en los eventos epidemiológicos reportados en el sistema de salud pública, permitiendo priorizar recursos hospitalarios, triaje preventivo y focalización de campañas de salud comunitaria.
* **Unidad de Observación:** Cada registro representa una **notificación individual de un evento de salud pública** reportado por una Unidad Generadora de Datos (IPS / Hospitales).
* **Origen de los datos:** Portal Nacional de Datos Abiertos de Colombia ([datos.gov.co - Eventos Epidemiológicos](https://www.datos.gov.co/Salud-y-Protecci-n-Social/Eventos-Epidemiol-gicos/afi7-frbs/about_data)).
* **Ubicación en el repositorio:** `data/datos.csv` (1,226 casos notificados).

---

## 3. Variables Utilizadas

| Variable | Tipo | Descripción | Rango / Valores |
| :--- | :--- | :--- | :--- |
| `edad` | Numérica (Entera) | Edad cronológica del paciente al momento del evento | 0 a 90 años |
| `semana_epidemiologica` | Numérica (Entera) | Semana del calendario epidemiológico del reporte | 1 a 52 |
| `estrato` | Numérica / Ordinal | Estrato socioeconómico de la residencia | 1 a 4 |
| `hospitalizado` | Numérica / Binaria | Condición de atención hospitalaria del caso | 0 = Ambulatorio, 1 = Hospitalizado |

> **Nota Metodológica:** Se excluyeron identificadores administrativos, códigos geográficos redundantes y fechas largas de texto para evitar ruido y fugas de información (*data leakage*).

---

## 4. Preparación de Datos y Pipeline
Para garantizar que las transformaciones se repliquen de forma idéntica en entrenamiento e inferencia, se construyó un `Pipeline` unificado de Scikit-Learn:
1. **Imputación:** `SimpleImputer(strategy="median")` para manejar posibles campos vacíos en estrato o edad.
2. **Estandarización:** `StandardScaler()`. Al utilizar K-Means basado en distancias euclidianas, es indispensable estandarizar las variables (media 0, desviación estándar 1) para que magnitudes dispares (ej. semanas epidemiológicas 1-52 vs estrato 1-4) no distorsionen los centroides.
3. **Clustering:** `KMeans(n_clusters=4, random_state=42, n_init="auto")`.

---

## 5. Modelo Seleccionado y Justificación
* **Algoritmo:** **K-Means**.
* **Justificación para despliegue:** Permite inferencia en tiempo real en la aplicación web mediante el método `.predict()`, proyectando nuevos casos sobre los centroides calculados durante el entrenamiento en tiempo $O(k \cdot d)$.
* **Criterios y Métricas de Evaluación:**
  * **Silhouette Score:** `0.3969` (indica clusters bien formados y estadísticamente diferenciados en el espacio transformado).
  * **Inercia intra-cluster (WCSS):** `2011.28`.
  * **Interpretabilidad clínica:** Cada clúster aísla perfiles con implicaciones operativas claras en la gestión hospitalaria.

---

## 6. Interpretación de los Clústers (Perfiles Epidemiológicos)

* **Clúster 0 - Población Pediátrica y Juvenil Ambulatoria (45.1% de los casos):**
  * *Edad promedio:* 12.2 años | *Semana promedio:* 28.7 | *Estrato:* 1.0 | *Hospitalizados:* 0.0%
  * *Eventos frecuentes:* Malaria ambulatoria, violencia intrafamiliar y dengue no grave.
  * *Estrategia:* Monitoreo ambulatorio y refuerzo de visitas comunitarias.
* **Clúster 1 - Adultos Mayores Vulnerables (15.4% de los casos):**
  * *Edad promedio:* 51.9 años | *Semana promedio:* 25.9 | *Estrato:* 1.0 | *Hospitalizados:* 15.3%
  * *Eventos frecuentes:* Malaria en edad adulta, dengue y agresiones por animales (rabia).
  * *Estrategia:* Control de comorbilidades y seguimiento a infecciones endémicas.
* **Clúster 2 - Casos Críticos de Urgencia Hospitalaria (26.3% de los casos):**
  * *Edad promedio:* 14.4 años | *Semana promedio:* 20.3 | *Estrato:* 1.0 | *Hospitalizados:* **100.0%**
  * *Eventos frecuentes:* Malaria complicada, desnutrición aguda severa en menores de 5 años e intentos de autolesión.
  * *Estrategia:* Priorización inmediata de camas hospitalarias y alerta al servicio de urgencias.
* **Clúster 3 - Población Urbana Estrato Medio / Zoonosis (13.1% de los casos):**
  * *Edad promedio:* 32.5 años | *Semana promedio:* 32.0 | *Estrato:* 2.1 | *Hospitalizados:* 16.1%
  * *Eventos frecuentes:* Dengue urbano, agresiones por animales transmisores de rabia y leishmaniasis.
  * *Estrategia:* Control vectorial urbano y vigilancia epidemiológica focalizada.

---

## 7. Instrucciones de Reproducción Local

### Paso 1: Instalar dependencias
```bash
pip install -r requirements.txt
```

### Paso 2: Entrenar el modelo y generar el artefacto
```bash
python train.py
```
> Ejecuta el pipeline completo y genera el archivo `model.joblib`.

### Paso 3: Iniciar la aplicación web en Streamlit
```bash
python -m streamlit run app.py
```
Abrir en el navegador: `http://localhost:8501`.

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

---

## 10. Casos de Prueba para Evaluación Cruzada

Para que otra pareja reproduzca y compruebe el proyecto:

### Caso 1: Paciente Pediátrico Ambulatorio
* **Entradas:** `Edad = 8`, `Semana Epidemiológica = 28`, `Estrato = 1`, `Hospitalizado = NO`
* **Salida Esperada:** **`Clúster 0`** (Población Pediátrica y Juvenil Ambulatoria).

### Caso 2: Adulto Mayor Vulnerable
* **Entradas:** `Edad = 60`, `Semana Epidemiológica = 26`, `Estrato = 1`, `Hospitalizado = NO`
* **Salida Esperada:** **`Clúster 1`** (Adultos y Adultos Mayores Vulnerables).

### Caso 3: Paciente Hospitalizado de Urgencia
* **Entradas:** `Edad = 15`, `Semana Epidemiológica = 20`, `Estrato = 1`, `Hospitalizado = SI`
* **Salida Esperada:** **`Clúster 2`** (Casos Críticos de Urgencia Hospitalaria).

### Caso 4: Caso Urbano Estrato Medio
* **Entradas:** `Edad = 32`, `Semana Epidemiológica = 32`, `Estrato = 3`, `Hospitalizado = NO`
* **Salida Esperada:** **`Clúster 3`** (Población Urbana Estrato Medio / Zoonosis).
