# Taller Final: Del Dataset al Despliegue en Producción (MLOps con Streamlit y Docker)

## 1. Información General
* **Proyecto:** Plataforma de Analítica y Triaje Epidemiológico con Machine Learning
* **Integrantes:** Arnold Santiago Torres (@eljarcor98) y compañero de pareja
* **Materia:** Machine Learning 2
* **Repositorio Oficial en GitHub:** https://github.com/eljarcor98/Taller-Final-Machine-Learning-II
* **Aplicación en Vivo (Despliegue Público Activo):** [https://aircraft-bruce-mark-zone.trycloudflare.com/](https://aircraft-bruce-mark-zone.trycloudflare.com/)

---

## 2. ¿De qué trata este proyecto y qué datos usamos?

En este proyecto construimos una solución completa de Machine Learning de punta a punta. En lugar de quedarnos solo en un notebook, llevamos el modelo hasta una aplicación web interactiva empaquetada con Docker y desplegada públicamente en un servidor en vivo.

Para que el proyecto tuviera impacto real, trabajamos con datos oficiales de Colombia tomados del portal **datos.gov.co**: el conjunto de datos de **Eventos Epidemiológicos (SIVIGILA)** del Ministerio de Salud y Protección Social. 

* **¿Qué representa cada fila del dataset?** Representa un caso o notificación individual de un evento de salud pública atendido por una IPS (hospital o centro de salud).
* **Tamaño del dataset:** 1.226 registros reales con variables demográficas, clínicas y sociales.
* **El reto que resolvemos:** Diseñamos una herramienta que ayuda al personal médico y epidemiológico en dos tareas clave:
  1. **Predecir si un paciente necesita hospitalización urgente o manejo ambulatorio**, ayudando a priorizar camas y recursos clínicos.
  2. **Segmentar los casos en grupos o perfiles**, para entender qué tipos de pacientes y enfermedades se concentran en cada etapa del año.

---

## 3. Variables que utilizamos

Para evitar sesgos y problemas de fuga de datos (*data leakage*), eliminamos columnas que no aportan a la predicción antes de tiempo (como códigos internos, identificadores y fechas largas en texto). Nos quedamos con las variables que un médico o epidemiólogo tiene a la mano al momento de atender al paciente:

### Para el modelo supervisado (Predecir Hospitalización):
* **Variable objetivo a predecir (`TARGET`):** `hospitalizacion` (valores: `SI` o `NO`). En los datos, el 30.7% de los casos requirió hospitalización y el 69.3% manejo ambulatorio.
* **Variables numéricas:**
  * `edad`: Edad del paciente en años cumplidos (0 a 110 años).
  * `semana_epidemiologica`: Semana del año en que se reporta el caso (semanas 1 a 52).
  * `estrato`: Estrato socioeconómico de la vivienda del paciente (1 a 4).
* **Variables categóricas:**
  * `evento_reportado`: La enfermedad o evento diagnosticado (Malaria, Dengue, Desnutrición aguda, Morbilidad materna, Agresiones por animales con riesgo de rabia, etc.).
  * `sexo`: Sexo biológico (`F` o `M`).
  * `regimen_salud`: Régimen de seguridad social (`S`: Subsidiado, `C`: Contributivo, `I`: Indeterminado, etc.).
  * `gestante`: Si la paciente está en embarazo (`SI` o `NO`).
  * `grupo_riesgo`: Si pertenece a algún grupo poblacional de riesgo comunitario (`SI` o `NO`).
  * `indigena`: Si tiene pertenencia étnica indígena (`SI` o `NO`).

### Para el modelo no supervisado (Clustering con K-Means):
* Usamos las variables numéricas y cuantitativas: `edad`, `semana_epidemiologica`, `estrato` y la condición de `hospitalizado` (0 para ambulatorio, 1 para hospitalizado).

---

## 4. ¿Cómo preparamos los datos? (Pipeline de Scikit-Learn)

Para asegurarnos de que el modelo sea 100% reproducible y que Streamlit no falle con datos nuevos, metimos todo el preprocesamiento dentro de un **`Pipeline`** unificado usando `ColumnTransformer`:

1. **Variables numéricas:** Si llega algún valor faltante, lo completamos con la mediana (`SimpleImputer`). Luego escalamos los valores con `StandardScaler` (media 0 y varianza 1) para que variables con números grandes (como las semanas 1-52) no dominen a variables pequeñas (como el estrato 1-4).
2. **Variables categóricas:** Si falta algún dato, colocamos la categoría más frecuente. Luego aplicamos `OneHotEncoder(handle_unknown="ignore")`, lo que permite que si en el futuro llega una categoría no vista, la app no se caiga sino que la ignore de forma segura.

---

## 5. Modelos elegidos y resultados obtenidos

Decidimos integrar **ambos enfoques (supervisado y no supervisado)** para demostrar que la misma arquitectura de MLOps sirve para cualquier tipo de problema:

### Modelo 1: Clasificación Supervisada (Regresión Logística con Balanceo de Clases)
* **¿Por qué lo elegimos?** Porque en salud pública es mucho más grave cometer un falso negativo (enviar a casa a una persona que en realidad necesitaba hospitalización urgente). Usamos el parámetro `class_weight="balanced"` para que el algoritmo le dé la misma importancia a la clase minoritaria (`SI`), y además porque nos permite calcular probabilidades reales y calibradas con `predict_proba()`.
* **Resultados en el conjunto de prueba (20% de los datos que el modelo nunca vio):**
  * **F1-Score ponderado:** `0.7364`
  * **Exactitud global (Accuracy):** `72.76%`
  * **Sensibilidad / Recall para Hospitalización (`SI`):** **`72.0%`** (detecta 54 de cada 75 pacientes que requerían hospitalización).
  * **Precisión para Ambulatorio (`NO`):** **`86.0%`**

### Modelo 2: Agrupamiento No Supervisado (K-Means con k=4)
* **¿Por qué lo elegimos?** K-Means es ideal para producción porque calcula centroides fijos durante el entrenamiento y permite usar `.predict()` sobre pacientes nuevos.
* **Resultados:**
  * **Silhouette Score:** `0.3969` (muestra grupos bien separados y consistentes).
  * **Inercia:** `2011.28`
* **Los 4 perfiles que descubrió el algoritmo:**
  * **Clúster 0 (45.1% de los casos):** Niños y jóvenes ambulatorios (edad promedio 12 años, estrato 1, 0% hospitalizados). Concentra casos leves de malaria y seguimiento social.
  * **Clúster 1 (15.4% de los casos):** Adultos mayores vulnerables (edad promedio 52 años, estrato 1, 15% hospitalizados). Afectados por malaria y dengue en edad adulta.
  * **Clúster 2 (26.3% de los casos):** Urgencias críticas (edad promedio 14 años, estrato 1, **100% hospitalizados**). Casos graves de desnutrición aguda y malaria complicada.
  * **Clúster 3 (13.1% de los casos):** Casos urbanos de estrato medio (edad promedio 33 años, estrato 2 a 3). Concentra brotes de dengue urbano y mordeduras de animales (riesgo de rabia).

---

## 6. La Aplicación Web en Streamlit (`app.py`)

Creamos una aplicación limpia, profesional y sin emojis innecesarios, organizada en dos pestañas principales:

* **Pestaña 1 (Triaje Clínico Supervisado):** El usuario ingresa la patología, edad, estrato y antecedentes del paciente. Al hacer clic en "Evaluar Caso Clínico", la app entrega un dictamen formal (`Hospitalización Requerida` o `Manejo Ambulatorio Sugerido`) junto con barras de porcentaje mostrando la probabilidad exacta de cada resultado.
* **Pestaña 2 (Caracterización K-Means):** Permite ingresar los datos demográficos y asigna automáticamente al paciente a uno de los 4 grupos, mostrando una comparativa numérica de sus datos frente al promedio del clúster.

> **Regla de oro cumplida:** La aplicación **nunca vuelve a entrenar el modelo**. Carga el archivo `model.joblib` directamente al inicio, respondiendo en milisegundos.

---

## 7. Cómo ejecutar el proyecto en tu computador local

Si quieres probarlo en tu propia máquina:

1. **Instalar las librerías necesarias:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Entrenar los modelos (opcional, ya dejamos el `model.joblib` listo):**
   ```bash
   python train.py
   ```

3. **Iniciar la aplicación en Streamlit:**
   ```bash
   python -m streamlit run app.py
   ```
   Se abrirá en tu navegador en: `http://localhost:8501`.

---

## 8. Cómo empaquetar y correr con Docker

Configuramos un `Dockerfile` basado en `python:3.12-slim`, optimizando las capas para que la imagen se construya rápido:

1. **Construir la imagen de Docker:**
   ```bash
   docker build -t ml-streamlit-app .
   ```

2. **Iniciar el contenedor en segundo plano:**
   ```bash
   docker run -d --name ml-streamlit --restart unless-stopped -p 8501:8501 ml-streamlit-app
   ```

3. **Revisar que esté corriendo y ver los logs:**
   ```bash
   docker ps
   docker logs ml-streamlit
   ```

---

## 9. Despliegue en Producción y Servidores Cloud / Edge

### 9.1 Despliegue Activo en Producción: Servidor Edge en Raspberry Pi (Linux ARM64) con Docker y Cloudflare Tunnel

Nuestra aplicación se encuentra **actualmente desplegada y funcionando en vivo** en una Raspberry Pi física con arquitectura ARM64, demostrando la portabilidad real de los contenedores Docker en hardware dedicado:

* **Enlace público activo:** [https://aircraft-bruce-mark-zone.trycloudflare.com/](https://aircraft-bruce-mark-zone.trycloudflare.com/)

**Pasos exactos con los que se montó en la Raspberry Pi:**
1. **Instalación de Docker en la Raspberry Pi:**
   ```bash
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh
   sudo usermod -aG docker $USER
   newgrp docker
   ```
2. **Descarga del proyecto y construcción de la imagen nativa:**
   ```bash
   git clone https://github.com/eljarcor98/Taller-Final-Machine-Learning-II.git
   cd Taller-Final-Machine-Learning-II
   docker build -t ml-streamlit-app .
   ```
3. **Ejecución del contenedor en segundo plano con reinicio automático:**
   ```bash
   docker run -d --name ml-streamlit --restart unless-stopped -p 8501:8501 ml-streamlit-app
   ```
4. **Exposición pública segura a Internet mediante Cloudflare Tunnel:**
   ```bash
   # Descarga e instalación de cloudflared
   curl -L --output cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64.deb
   sudo dpkg -i cloudflared.deb

   # Ejecución en segundo plano persistente (sin morir al cerrar SSH)
   nohup cloudflared tunnel --url http://127.0.0.1:8501 > cloudflare.log 2>&1 & disown
   ```
   *Ventajas técnicas:* No requiere abrir puertos inseguros en el router residencial, cuenta con certificado SSL/HTTPS nativo y es accesible desde cualquier red en el mundo.

---

### 9.2 Despliegue en Google Cloud Platform (Compute Engine)

Para replicar el despliegue en una Máquina Virtual de GCP:

1. En Google Cloud Console, crear una máquina virtual con sistema operativo **Ubuntu 24.04 LTS** (serie `e2-micro` o `e2-small`).
2. Asignar la etiqueta de red `streamlit-server` y confirmar dirección IP externa.
3. Abrir el puerto 8501 en el firewall desde Cloud Shell:
   ```bash
   gcloud compute firewall-rules create allow-streamlit-8501 \
       --allow=tcp:8501 \
       --direction=INGRESS \
       --target-tags=streamlit-server
   ```
4. Conectarse por SSH a la VM y ejecutar:
   ```bash
   sudo apt update && sudo apt install -y docker.io git
   sudo systemctl enable --now docker
   sudo usermod -aG docker $USER

   git clone https://github.com/eljarcor98/Taller-Final-Machine-Learning-II.git
   cd Taller-Final-Machine-Learning-II
   docker build -t ml-streamlit-app .
   docker run -d --name ml-streamlit --restart unless-stopped -p 8501:8501 ml-streamlit-app
   ```
5. Abrir en el navegador: `http://<IP_EXTERNA_DE_LA_VM>:8501`.

---

### 9.3 Despliegue Alternativo en PaaS (Render.com / Koyeb)
1. Conectar la cuenta de GitHub con [Render.com](https://render.com).
2. Crear un nuevo Web Service seleccionando el repositorio `Taller-Final-Machine-Learning-II`.
3. Seleccionar el entorno **Docker** (detectará automáticamente el `Dockerfile` y la variable de puerto `$PORT`).
4. Render compilará la imagen y asignará una URL pública segura con HTTPS.

---

## 10. Casos de Prueba para la Evaluación Cruzada

Para la prueba entre parejas, la otra pareja puede acceder a la URL pública [https://aircraft-bruce-mark-zone.trycloudflare.com/](https://aircraft-bruce-mark-zone.trycloudflare.com/) (o clonar el repositorio) y probar estos 4 casos para comprobar que arroja exactamente los mismos resultados:

### Pruebas en Pestaña 1 (Modelo Supervisado):
* **Caso 1: Paciente de alta gravedad (Gestante con complicación materna):**
  * Entradas: `Edad = 28`, `Semana = 20`, `Estrato = 1`, `Evento = MORBILIDAD MATERNA EXTREMA`, `Sexo = F`, `Régimen = S`, `Gestante = SI`, `Grupo de Riesgo = SI`, `Indígena = NO`.
  * **Resultado esperado:** **`Clase SI`** (Dictamen de Hospitalización Requerida, probabilidad cercana al **`88.8%`**).

* **Caso 2: Paciente para manejo ambulatorio (Mordedura de animal / sospecha de rabia):**
  * Entradas: `Edad = 35`, `Semana = 35`, `Estrato = 2`, `Evento = AGRESIONES POR ANIMALES POTENCIALMENTE TRANSMISORES DE RABIA`, `Sexo = M`, `Régimen = C`, `Gestante = NO`, `Grupo de Riesgo = NO`, `Indígena = NO`.
  * **Resultado esperado:** **`Clase NO`** (Dictamen de Manejo Ambulatorio Sugerido, probabilidad de ambulatorio cercana al **`93.6%`**).

### Pruebas en Pestaña 2 (Modelo No Supervisado - K-Means):
* **Caso 3: Paciente pediátrico ambulatorio:**
  * Entradas: `Edad = 8`, `Semana = 28`, `Estrato = 1`, `Condición = Manejo ambulatorio (0)`.
  * **Resultado esperado:** **`Clúster 0`** (Población Pediátrica y Juvenil Ambulatoria).

* **Caso 4: Paciente hospitalizado de urgencia:**
  * Entradas: `Edad = 15`, `Semana = 20`, `Estrato = 1`, `Condición = Internación hospitalaria (1)`.
  * **Resultado esperado:** **`Clúster 2`** (Casos Críticos de Urgencia Hospitalaria).
