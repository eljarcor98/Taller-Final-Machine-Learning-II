# Taller Final: Del Dataset al Despliegue en Producción (MLOps con Streamlit y Docker)

## 1. Información General
* **Proyecto:** Plataforma de Analítica y Triaje Epidemiológico con Machine Learning
* **Integrantes:** Arnold Santiago Torres Anzola
* **Materia:** Machine Learning 2
* **Repositorio en GitHub:** https://github.com/eljarcor98/Taller-Final-Machine-Learning-II
* **Aplicación en Vivo (Activa en Internet):** [https://aircraft-bruce-mark-zone.trycloudflare.com/](https://aircraft-bruce-mark-zone.trycloudflare.com/)

---

## 2. ¿De qué trata el proyecto y qué datos usamos?

Para este taller final decidimos no quedarnos solo entrenando un modelo en un notebook de Jupyter o Colab. Lo que hicimos fue montar todo el ciclo completo de Machine Learning (MLOps): preparamos los datos, entrenamos los pipelines, creamos una aplicación web en Streamlit y la empaquetamos en Docker para dejarla corriendo en un servidor real con acceso público en Internet.

Para trabajar con datos reales de nuestro contexto, usamos una base de datos oficial de Colombia descargada directamente desde el portal **datos.gov.co**: el dataset de **Eventos Epidemiológicos (SIVIGILA)** reportado por el Ministerio de Salud y Protección Social.

* **¿Qué representa cada fila?** Cada fila es la notificación individual de un paciente atendido en un centro de salud o IPS por algún evento epidemiológico.
* **Tamaño del archivo:** Son 1.226 registros reales con datos demográficos, clínicos y sociales.
* **El objetivo del proyecto:** Desarrollamos una herramienta web con dos funciones principales:
  1. **Predecir si un paciente va a requerir hospitalización o si se puede manejar de forma ambulatoria**, lo que ayuda a priorizar camas y recursos en urgencias.
  2. **Segmentar y agrupar los casos con K-Means**, para identificar qué perfiles de población y qué enfermedades se concentran en diferentes semanas del año.

---

## 3. Variables que seleccionamos

Antes de entrenar cualquier modelo, revisamos las columnas y eliminamos identificadores, códigos de municipios y fechas largas en texto para no generar fugas de información (*data leakage*). Nos quedamos únicamente con las variables que un médico o epidemiólogo tiene a la mano al momento de atender a la persona:

### Variables para el modelo supervisado (Predecir si se hospitaliza o no):
* **Variable objetivo a predecir (`TARGET`):** `hospitalizacion` (valores: `SI` o `NO`). En los datos reales, el 30.7% de los pacientes fue hospitalizado y el 69.3% tuvo manejo ambulatorio.
* **Variables numéricas:**
  * `edad`: Edad del paciente en años cumplidos (de 0 a 110 años).
  * `semana_epidemiologica`: Semana del año en la que se reporta el caso (semanas 1 a 52).
  * `estrato`: Estrato socioeconómico de la vivienda (1 a 4).
* **Variables categóricas:**
  * `evento_reportado`: La enfermedad o evento diagnosticado (Malaria, Dengue, Desnutrición aguda en menores de 5 años, Morbilidad materna extrema, Agresiones por animales transmisores de rabia, etc.).
  * `sexo`: Sexo biológico del paciente (`F` o `M`).
  * `regimen_salud`: Tipo de afiliación en salud (`S`: Subsidiado, `C`: Contributivo, `I`: Indeterminado, etc.).
  * `gestante`: Si la paciente está embarazada (`SI` o `NO`).
  * `grupo_riesgo`: Si pertenece a algún grupo poblacional de riesgo (`SI` o `NO`).
  * `indigena`: Si tiene pertenencia étnica indígena (`SI` o `NO`).

### Variables para el modelo no supervisado (K-Means):
* Usamos las variables cuantitativas del caso: `edad`, `semana_epidemiologica`, `estrato` y si el paciente fue hospitalizado o no (`hospitalizado_num`: 0 o 1).

---

## 4. ¿Cómo preparamos los datos? (Pipeline de Scikit-Learn)

Para evitar que la aplicación falle cuando reciba datos nuevos o categorías que no estaban en el entrenamiento, metimos todo el preprocesamiento dentro de un **`Pipeline`** unificado de Scikit-Learn usando `ColumnTransformer`:

1. **Para los números:** Si llega a faltar algún valor (como en estrato), lo rellenamos automáticamente con la mediana usando `SimpleImputer`. Después escalamos todas las variables numéricas con `StandardScaler` (media 0 y varianza 1) para que los números grandes (como semanas 1-52) no aplasten a los números pequeños (como estrato 1-4).
2. **Para las categorías:** Si falta algún dato, colocamos el valor más común (`most_frequent`). Luego aplicamos `OneHotEncoder(handle_unknown="ignore")`. Esto es clave porque si en producción el usuario ingresa una categoría no vista, el modelo simplemente la ignora y no rompe la aplicación.

---

## 5. Modelos que entrenamos y resultados

Quisimos implementar **ambos enfoques (supervisado y no supervisado)** dentro del mismo proyecto para demostrar que la arquitectura de software sirve para cualquier tipo de tarea de Machine Learning:

### Modelo 1: Clasificación Supervisada (Regresión Logística Balanceada)
* **¿Por qué este modelo?** En un triaje médico, cometer un falso negativo (mandar para la casa a un paciente grave que necesitaba hospitalización) es muy peligroso. Por eso usamos `class_weight="balanced"`, obligando al algoritmo a prestarle atención a la clase minoritaria (`SI`). Además, la Regresión Logística nos da probabilidades reales y calibradas usando `predict_proba()`.
* **Métricas en los datos de prueba (el 20% del dataset que el modelo nunca vio):**
  * **F1-Score ponderado:** `0.7364`
  * **Exactitud (Accuracy):** `72.76%`
  * **Sensibilidad / Recall para Hospitalización (`SI`):** **`72.0%`** (el modelo detecta correctamente a 54 de cada 75 pacientes que requerían hospitalización).
  * **Precisión para Manejo Ambulatorio (`NO`):** **`86.0%`**

### Modelo 2: Segmentación No Supervisada (K-Means con k=4)
* **¿Por qué K-Means?** Porque calcula centroides fijos en el entrenamiento y permite proyectar pacientes nuevos en tiempo real con el método `.predict()`.
* **Métricas:**
  * **Silhouette Score:** `0.3969` (confirma que los grupos están bien formados y separados).
  * **Inercia:** `2011.28`
* **Los 4 perfiles que encontró el algoritmo:**
  * **Clúster 0 (45.1% de los casos):** Población pediátrica y juvenil ambulatoria (edad promedio 12 años, estrato 1, 0% hospitalizados). Concentra casos leves de malaria y seguimiento social.
  * **Clúster 1 (15.4% de los casos):** Adultos mayores vulnerables (edad promedio 52 años, estrato 1, 15% hospitalizados). Casos de malaria y dengue en edad adulta.
  * **Clúster 2 (26.3% de los casos):** Casos críticos de urgencia (edad promedio 14 años, estrato 1, **100% hospitalizados**). Casos graves de desnutrición aguda en menores de 5 años y malaria complicada.
  * **Clúster 3 (13.1% de los casos):** Población urbana de estrato medio (edad promedio 33 años, estratos 2 y 3). Brotes de dengue urbano y mordeduras de animales (riesgo de rabia).

---

## 6. La Aplicación Web en Streamlit (`app.py`)

Hicimos una aplicación web limpia, rápida y sin emojis, organizada en dos pestañas:

* **Pestaña 1 (Triaje Clínico):** Permite ingresar la patología, edad, estrato y antecedentes del paciente. Al darle al botón "Evaluar Caso Clínico", la app entrega el dictamen formal (`Hospitalización Requerida` o `Manejo Ambulatorio Sugerido`) junto con barras de porcentaje mostrando la probabilidad de cada opción.
* **Pestaña 2 (Segmentación K-Means):** Permite ingresar las variables cuantitativas y asigna automáticamente al paciente a uno de los 4 grupos, mostrando una comparativa numérica de sus datos frente al promedio de ese clúster.

> **Importante para la evaluación:** La aplicación **nunca vuelve a entrenar los modelos**. Carga directamente el archivo serializado `model.joblib` al iniciar, respondiendo en milisegundos.

---

## 7. Cómo correr el proyecto en local

Si alguien quiere probar el proyecto en su computador:

1. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Entrenar los modelos (opcional, el archivo `model.joblib` ya está generado y listo):**
   ```bash
   python train.py
   ```

3. **Iniciar la aplicación en Streamlit:**
   ```bash
   python -m streamlit run app.py
   ```
   Abre automáticamente en el navegador en: `http://localhost:8501`.

---

## 8. Cómo empaquetarlo y correrlo con Docker

Escribimos un `Dockerfile` usando `python:3.12-slim`, aprovechando la caché de capas para que compile rápido:

1. **Construir la imagen Docker:**
   ```bash
   docker build -t ml-streamlit-app .
   ```

2. **Correr el contenedor en segundo plano:**
   ```bash
   docker run -d --name ml-streamlit --restart unless-stopped -p 8501:8501 ml-streamlit-app
   ```

3. **Verificar que esté arriba y revisar logs:**
   ```bash
   docker ps
   docker logs ml-streamlit
   ```

---

## 9. Despliegue en Servidores Reales (Producción y Nube)

### 9.1 Nuestro Despliegue Activo: Servidor en Raspberry Pi (Linux ARM64) con Docker y Cloudflare Tunnel

Para llevar la práctica a un entorno de hardware real (*Edge Computing*), montamos el contenedor en una **Raspberry Pi física** y lo sacamos a Internet de forma segura usando **Cloudflare Tunnel**:

* **Enlace público activo para pruebas:** [https://aircraft-bruce-mark-zone.trycloudflare.com/](https://aircraft-bruce-mark-zone.trycloudflare.com/)

**Los pasos exactos con los que lo montamos en la Raspberry:**
1. Instalamos Docker en la Raspberry Pi:
   ```bash
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh
   sudo usermod -aG docker $USER
   newgrp docker
   ```
2. Clonamos nuestro repo y construimos la imagen de Docker en ARM:
   ```bash
   git clone https://github.com/eljarcor98/Taller-Final-Machine-Learning-II.git
   cd Taller-Final-Machine-Learning-II
   docker build -t ml-streamlit-app .
   ```
3. Levantamos el contenedor con reinicio automático:
   ```bash
   docker run -d --name ml-streamlit --restart unless-stopped -p 8501:8501 ml-streamlit-app
   ```
4. Creamos el túnel seguro de Cloudflare para que cualquier persona en el mundo pueda entrar sin contraseñas ni abrir puertos en el router:
   ```bash
   curl -L --output cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64.deb
   sudo dpkg -i cloudflared.deb
   nohup cloudflared tunnel --url http://127.0.0.1:8501 > cloudflare.log 2>&1 & disown
   ```

---

### 9.2 Despliegue en Google Cloud Platform (Compute Engine)

Si se quiere replicar en una Máquina Virtual de GCP como lo indica la guía:

1. En Google Cloud Console, creamos una VM con **Ubuntu 24.04 LTS** (tipo `e2-micro` o `e2-small`).
2. Le asignamos la etiqueta de red `streamlit-server` y confirmamos que tenga IP externa.
3. Abrimos el puerto 8501 en el firewall desde Cloud Shell:
   ```bash
   gcloud compute firewall-rules create allow-streamlit-8501 \
       --allow=tcp:8501 \
       --direction=INGRESS \
       --target-tags=streamlit-server
   ```
4. Entramos por la terminal SSH de la VM y corremos:
   ```bash
   sudo apt update && sudo apt install -y docker.io git
   sudo systemctl enable --now docker
   sudo usermod -aG docker $USER

   git clone https://github.com/eljarcor98/Taller-Final-Machine-Learning-II.git
   cd Taller-Final-Machine-Learning-II
   docker build -t ml-streamlit-app .
   docker run -d --name ml-streamlit --restart unless-stopped -p 8501:8501 ml-streamlit-app
   ```
5. Abrimos en el navegador: `http://<IP_EXTERNA_DE_LA_VM>:8501`.

---

### 9.3 Despliegue en Render.com o Koyeb (PaaS Gratuito)
1. Conectamos la cuenta de GitHub en [Render.com](https://render.com).
2. Creamos un Web Service seleccionando este repositorio.
3. Seleccionamos el entorno **Docker** (detecta el `Dockerfile` y la variable `$PORT` automáticamente).
4. Render compila la imagen y nos da una URL pública con HTTPS gratis.

---

## 10. Casos de Prueba para la Evaluación Cruzada

Para que la pareja que haga la prueba cruzada verifique que el modelo arroja los mismos resultados, pueden ingresar a [https://aircraft-bruce-mark-zone.trycloudflare.com/](https://aircraft-bruce-mark-zone.trycloudflare.com/) y probar estos 4 casos:

### Casos de prueba en Pestaña 1 (Modelo Supervisado):
* **Caso 1: Paciente de alta gravedad (Complicación materna / gestante):**
  * Entradas: `Edad = 28`, `Semana = 20`, `Estrato = 1`, `Evento = MORBILIDAD MATERNA EXTREMA`, `Sexo = F`, `Régimen = S`, `Gestante = SI`, `Grupo de Riesgo = SI`, `Indígena = NO`.
  * **Resultado esperado:** **`Clase SI`** (Dictamen de Hospitalización Requerida con probabilidad cercana al **`88.8%`**).

* **Caso 2: Paciente para manejo ambulatorio (Mordedura de animal con sospecha de rabia):**
  * Entradas: `Edad = 35`, `Semana = 35`, `Estrato = 2`, `Evento = AGRESIONES POR ANIMALES POTENCIALMENTE TRANSMISORES DE RABIA`, `Sexo = M`, `Régimen = C`, `Gestante = NO`, `Grupo de Riesgo = NO`, `Indígena = NO`.
  * **Resultado esperado:** **`Clase NO`** (Dictamen de Manejo Ambulatorio Sugerido con probabilidad ambulatoria cercana al **`93.6%`**).

### Casos de prueba en Pestaña 2 (Modelo No Supervisado - K-Means):
* **Caso 3: Paciente pediátrico ambulatorio:**
  * Entradas: `Edad = 8`, `Semana = 28`, `Estrato = 1`, `Condición = Manejo ambulatorio (0)`.
  * **Resultado esperado:** **`Clúster 0`** (Población Pediátrica y Juvenil Ambulatoria).

* **Caso 4: Paciente hospitalizado de urgencia:**
  * Entradas: `Edad = 15`, `Semana = 20`, `Estrato = 1`, `Condición = Internación hospitalaria (1)`.
  * **Resultado esperado:** **`Clúster 2`** (Casos Críticos de Urgencia Hospitalaria).
