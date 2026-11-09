FROM python:3.12-slim

WORKDIR /app

# Optimización de caché de capas de Docker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código y artefactos
COPY . .

# Exponer el puerto predeterminado de Streamlit
EXPOSE 8501

# Comando de inicio del servidor Streamlit escuchando en todas las interfaces de red
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
