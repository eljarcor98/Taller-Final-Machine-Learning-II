# Reto ML: Del Dataset al Despliegue con Streamlit, Docker y GCP

## 1. Información General
* **Título del Proyecto:** Segmentación de Clientes con K-Means y Despliegue en Contenedor
* **Ruta Elegida:** Ruta B - Aprendizaje No Supervisado (Clustering con K-Means)
* **Autores:** [Nombre Integrante 1] & [Nombre Integrante 2]
* **Asignatura:** Machine Learning 2

---

## 2. Objetivo y Descripción del Dataset
* **Objetivo de Negocio / Técnico:** Descubrir patrones naturales y agrupar a los clientes de un centro comercial en segmentos homogéneos según su perfil demográfico y hábitos de compra, permitiendo personalizar estrategias comerciales y de marketing.
* **Unidad de Observación:** Cada registro representa un **cliente individual** del centro comercial.
* **Origen de los datos:** Archivo `data/datos.csv` (Dataset Mall Customers).

---

## 3. Variables Utilizadas

| Variable | Tipo | Descripción | Rango / Unidad |
| :--- | :--- | :--- | :--- |
| `edad` | Numérica (Continua/Entera) | Edad cronológica del cliente | 18 a 70 años |
| `ingreso_anual` | Numérica (Continua) | Ingresos anuales estimados del cliente | 15k a 137k USD |
| `puntaje_gasto` | Numérica (Discreta) | Calificación de propensión y volumen de gasto asignado por el comercio | 1 a 99 puntos |

> **Nota Metodológica:** Se excluyó la columna identificadora (`id_cliente`) para no introducir sesgos ni distancias espurias en el cálculo de proximidad del algoritmo euclidiano.

---

## 4. Preparación de Datos y Pipeline
Para evitar fuga de datos y asegurar consistencia total en inferencia, el preprocesamiento y el modelo se encapsulan en un solo `Pipeline` de Scikit-Learn:
1. **Imputación:** `SimpleImputer(strategy="median")` para tratar posibles valores atípicos o nulos.
2. **Estandarización:** `StandardScaler()`. Dado que K-Means se basa en distancia euclidiana, el escalado es indispensable para que una variable con magnitud grande (como `ingreso_anual`) no opaque a variables de menor magnitud (como `edad`).
3. **Clustering:** `KMeans(n_clusters=4, random_state=42, n_init="auto")`.

---

## 5. Modelo Seleccionado y Justificación
* **Algoritmo:** **K-Means**.
* **Justificación para despliegue:** Es el algoritmo de clustering óptimo para este reto porque calcula centroides fijos en el entrenamiento y permite proyectar e inferir nuevos casos mediante el método `.predict()`, asignando cada nueva observación al centroide más cercano en tiempo constante $O(k \cdot d)$.
* **Criterios de Evaluación:**
  * **Silhouette Score:** `0.4040` (indica una estructura de agrupamiento consistente, compacta y separable).
  * **Inercia intra-cluster (WCSS):** `205.23`.
  * **Interpretabilidad cuantitativa:** Los 4 clusters presentan claras diferencias en poder adquisitivo y hábitos de consumo.

---

## 6. Interpretación de los Clústers (Perfiles de Negocio)

A partir del análisis cuantitativo de medias:
* **Clúster 0 - Cliente Maduro Tradicional:**
  * *Edad media:* 54 años | *Ingreso anual medio:* $48k USD | *Gasto medio:* 40 pts.
  * *Perfil:* Clientes conservadores de edad madura con gasto moderado y controlado.
* **Clúster 1 - Cliente VIP / Alto Valor:**
  * *Edad media:* 33 años | *Ingreso anual medio:* $86k USD | *Gasto medio:* 82 pts.
  * *Perfil:* Adultos jóvenes con alta capacidad adquisitiva y alta disposición al gasto. Segmento estratégico para ventas de alto margen y fidelización premium.
* **Clúster 2 - Joven Entusiasta / Dinámico:**
  * *Edad media:* 25 años | *Ingreso anual medio:* $40k USD | *Gasto medio:* 60 pts.
  * *Perfil:* Clientes jóvenes con ingresos moderados pero gran disposición al consumo de tendencias, ocio y moda.
* **Clúster 3 - Ahorrador Cuidadoso:**
  * *Edad media:* 39 años | *Ingreso anual medio:* $87k USD | *Gasto medio:* 20 pts.
  * *Perfil:* Clientes con ingresos elevados pero bajo consumo. Perfil analítico y cauto, sensible a ofertas de valor e inversiones.

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
> Esto ejecuta el pipeline, calcula las métricas y genera el archivo `model.joblib`.

### Paso 3: Ejecutar la aplicación Streamlit
```bash
streamlit run app.py
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

### Paso 3: Comprobar contenedor y logs
```bash
docker ps
docker logs ml-streamlit
```

---

## 9. Despliegue en Google Cloud Platform (Compute Engine)

1. En la consola de GCP, ir a **Compute Engine** -> **Instancias de VM** -> **Crear instancia**.
2. Seleccionar SO **Ubuntu 24.04 LTS**, tipo de máquina económica (ej. `e2-micro` o `e2-small`).
3. En la sección de redes, asignar la etiqueta de red `streamlit-server` y confirmar que tenga IP externa.
4. Crear la regla de Firewall para habilitar el puerto 8501:
   ```bash
   gcloud compute firewall-rules create allow-streamlit-8501 \
       --allow=tcp:8501 \
       --direction=INGRESS \
       --target-tags=streamlit-server
   ```
5. Abrir la terminal SSH de la VM en el navegador e instalar Docker y Git:
   ```bash
   sudo apt update
   sudo apt install -y docker.io git
   sudo systemctl enable --now docker
   sudo usermod -aG docker $USER
   ```
6. Clonar el repositorio y levantar la solución:
   ```bash
   git clone <URL_DEL_REPOSITORIO>
   cd ml_streamlit_project
   docker build -t ml-streamlit-app .
   docker run -d --name ml-streamlit --restart unless-stopped -p 8501:8501 ml-streamlit-app
   ```
7. Probar en el navegador:
   `http://<IP_EXTERNA_DE_LA_VM>:8501`

---

## 10. Ejemplo de Prueba Cruzada (Para verificación por otra pareja)

Cualquier pareja que reproduzca el proyecto puede ingresar los siguientes casos de prueba en la interfaz web:

### Caso de Prueba 1: Perfil VIP
* **Entradas:** `edad = 32`, `ingreso_anual = 85.0`, `puntaje_gasto = 80`
* **Salida esperada:** **`Clúster 1`** (Cliente VIP / Alto Valor).

### Caso de Prueba 2: Perfil Joven Dinámico
* **Entradas:** `edad = 24`, `ingreso_anual = 38.0`, `puntaje_gasto = 65`
* **Salida esperada:** **`Clúster 2`** (Joven Entusiasta / Dinámico).

### Caso de Prueba 3: Perfil Maduro Conservador
* **Entradas:** `edad = 56`, `ingreso_anual = 45.0`, `puntaje_gasto = 35`
* **Salida esperada:** **`Clúster 0`** (Cliente Maduro Tradicional).
