# Reto ML: Predicción de Hospitalización Epidemiológica (SIVIGILA) con Streamlit, Docker y GCP

## 1. Información General
* **Título del Proyecto:** Sistema Inteligente de Triaje Epidemiológico y Predicción de Hospitalización
* **Ruta Elegida:** Ruta A - Aprendizaje Supervisado (Clasificación Binaria)
* **Autores:** Arnold Santiago Torres (@eljarcor98) & [Compañero/a de pareja]
* **Asignatura:** Machine Learning 2
* **Repositorio Oficial:** [github.com/eljarcor98/Taller-Final-Machine-Learning-II](https://github.com/eljarcor98/Taller-Final-Machine-Learning-II)

---

## 2. Objetivo y Descripción del Dataset
* **Objetivo de Negocio / Salud Pública:** Predecir oportunamente si un paciente notificado en el sistema de vigilancia epidemiológica requerirá **Hospitalización (`SI` / `NO`)**, optimizando la asignación preventiva de camas hospitalarias, el triaje clínico y la activación de protocolos de aislamiento.
* **Unidad de Observación:** Cada registro representa una **notificación individual de un evento de salud pública** reportado por una Institución Prestadora de Salud (IPS).
* **Origen de los datos:** Portal Nacional de Datos Abiertos de Colombia ([datos.gov.co - Eventos Epidemiológicos SIVIGILA](https://www.datos.gov.co/Salud-y-Protecci-n-Social/Eventos-Epidemiol-gicos/afi7-frbs/about_data)).
* **Ubicación en el repositorio:** `data/datos.csv` (1,226 casos notificados).

---

## 3. Variables Utilizadas

### Variable Objetivo (`TARGET`)
* **`hospitalizacion`**: Condición de internación hospitalaria del paciente (`SI`: 30.7%, `NO`: 69.3%).

### Variables Numéricas (`numeric_features`)
* **`edad`**: Edad cronológica cumplida del paciente (0 a 110 años).
* **`semana_epidemiologica`**: Semana del calendario epidemiológico del reporte (1 a 52).
* **`estrato`**: Estrato socioeconómico de la vivienda (1 a 4).

### Variables Categóricas (`categorical_features`)
* **`evento_reportado`**: Evento o patología notificada (Dengue, Malaria, Morbilidad Materna, Desnutrición, Agresiones por animales, etc.).
* **`sexo`**: Sexo biológico del paciente (`F`, `M`).
* **`regimen_salud`**: Régimen de afiliación a seguridad social (`S`: Subsidiado, `C`: Contributivo, `I`: Indeterminado, `E`: Especial, etc.).
* **`gestante`**: Condición de embarazo (`SI`, `NO`).
* **`grupo_riesgo`**: Pertenencia a grupos comunitarios en riesgo epidemiológico (`SI`, `NO`).
* **`indigena`**: Pertenencia a población étnica indígena (`SI`, `NO`).

> **Nota Metodológica:** Se excluyeron identificadores administrativos, llaves primarias, códigos de municipio redundantes y fechas para prevenir sesgos y fugas de datos (*data leakage*).

---

## 4. Preparación de Datos y Pipeline
Para asegurar total reproducibilidad y evitar diferencias entre entrenamiento e inferencia, el preprocesamiento y el modelo se encapsulan en un `Pipeline` único de Scikit-Learn:
1. **Rama Numérica:**
   * Imputación de valores faltantes con la **mediana** (`SimpleImputer(strategy="median")`).
   * Estandarización a media 0 y varianza unitaria (`StandardScaler()`).
2. **Rama Categórica:**
   * Imputación con la moda / categoría más frecuente (`SimpleImputer(strategy="most_frequent")`).
   * Codificación One-Hot (`OneHotEncoder(handle_unknown="ignore")`) para tolerar categorías nuevas en producción sin generar excepciones.
3. **Consolidación:** Ensamblado con `ColumnTransformer` que alimenta directamente al estimador.

---

## 5. Modelo Seleccionado y Justificación
* **Algoritmo:** **Regresión Logística con Balanceo de Clases (`class_weight="balanced"`)**.
* **Justificación Clínica y Técnica:** En triaje médico, el costo de un Falso Negativo (enviar a casa a un paciente que requería internación urgente) es crítico. El balanceo de pesos prioriza la sensibilidad clínica de la clase minoritaria (`SI`), permitiendo además calcular probabilidades exactas y calibradas con `predict_proba()` para dar certidumbre al personal de salud.
* **Métricas Obtenidas en Conjunto de Prueba Independiente (20% Test):**
  * **F1-Score Ponderado:** `0.7364`
  * **Exactitud Global (Accuracy):** `72.76%`
  * **Sensibilidad / Recall para Hospitalización (`SI`):** `72.0%` (detecta 54 de cada 75 casos que requieren hospitalización).
  * **Precisión para Manejo Ambulatorio (`NO`):** `86.0%`

---

## 6. Interpretación de la Salida en la Aplicación
La interfaz en Streamlit recibe las variables del nuevo caso y entrega:
1. **Dictamen Clínico:**
   * **Hospitalización Requerida (Clase SI):** Recomienda priorizar la disponibilidad de cama hospitalaria y valoración médica presencial de urgencia.
   * **Manejo Ambulatorio Sugerido (Clase NO):** Recomienda tratamiento domiciliario controlado y seguimiento epidemiológico comunitario.
2. **Distribución de Certeza:** Visualización gráfica mediante métricas y barras de progreso del porcentaje de probabilidad de internación (`Probabilidad SI`) vs manejo ambulatorio (`Probabilidad NO`).

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
> Ejecuta el pipeline, genera las métricas de evaluación y exporta `model.joblib`.

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

### 9.1 Despliegue Alternativo en Servidor Físico / Raspberry Pi (Edge Computing)
Si se desea desplegar en hardware propio (Raspberry Pi 3/4/5 con arquitectura ARM64 o Linux local):
1. Instalar Docker y Git:
   ```bash
   sudo apt update && sudo apt install -y docker.io git
   sudo systemctl enable --now docker
   sudo usermod -aG docker $USER
   ```
2. Clonar y ejecutar el contenedor:
   ```bash
   git clone https://github.com/eljarcor98/Taller-Final-Machine-Learning-II.git
   cd Taller-Final-Machine-Learning-II
   docker build -t ml-streamlit-app .
   docker run -d --name ml-streamlit --restart unless-stopped -p 8501:8501 ml-streamlit-app
   ```
3. Exposición a Internet mediante túnel seguro (sin necesidad de abrir puertos en el router residencial):
   ```bash
   npx localtunnel --port 8501
   # O mediante Cloudflare Tunnel:
   # cloudflared tunnel --url http://localhost:8501
   ```

### 9.2 Despliegue Alternativo PaaS (Render.com / Koyeb)
1. Conectar la cuenta de GitHub con [Render.com](https://render.com) o [Koyeb.com](https://www.koyeb.com).
2. Crear un nuevo servicio apuntando al repositorio `Taller-Final-Machine-Learning-II`.
3. Seleccionar el entorno **Docker** (detectará automáticamente el `Dockerfile` y la variable de puerto `$PORT`).
4. La plataforma construirá el contenedor y asignará una URL pública segura con HTTPS.

---

## 10. Casos de Prueba para Evaluación Cruzada

Para que otra pareja compruebe la reproducibilidad del sistema, ingresar estos casos en la aplicación web:

### Caso 1: Morbilidad Materna Extrema (Alta Complejidad)
* **Entradas:** `Edad = 28`, `Semana = 20`, `Estrato = 1`, `Evento = MORBILIDAD MATERNA EXTREMA`, `Sexo = F`, `Régimen = S`, `Gestante = SI`, `Grupo de Riesgo = SI`, `Indígena = NO`
* **Salida Esperada:** **`Clase SI`** (Probabilidad de Hospitalización: **~88.8%**).

### Caso 2: Agresión por Animal / Rabia (Manejo Ambulatorio)
* **Entradas:** `Edad = 35`, `Semana = 35`, `Estrato = 2`, `Evento = AGRESIONES POR ANIMALES POTENCIALMENTE TRANSMISORES DE RABIA`, `Sexo = M`, `Régimen = C`, `Gestante = NO`, `Grupo de Riesgo = NO`, `Indígena = NO`
* **Salida Esperada:** **`Clase NO`** (Probabilidad de Manejo Ambulatorio: **~93.6%**).

### Caso 3: Desnutrición Aguda Pediátrica
* **Entradas:** `Edad = 3`, `Semana = 15`, `Estrato = 1`, `Evento = DESNUTRICIÓN AGUDA EN MENORES DE 5 AÑOS`, `Sexo = M`, `Régimen = S`, `Gestante = NO`, `Grupo de Riesgo = SI`, `Indígena = NO`
* **Salida Esperada:** **`Clase SI`** (Probabilidad de Hospitalización: **~71.2%**).

### Caso 4: Malaria Ambulatoria en Adulto
* **Entradas:** `Edad = 28`, `Semana = 35`, `Estrato = 2`, `Evento = MALARIA`, `Sexo = M`, `Régimen = C`, `Gestante = NO`, `Grupo de Riesgo = NO`, `Indígena = NO`
* **Salida Esperada:** **`Clase NO`** (Probabilidad de Manejo Ambulatorio: **~65.9%**).
