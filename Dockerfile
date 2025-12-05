# ==============================================
# Stock Predictor - Dockerfile Principal (Backend)
# Tech Challenge Fase 4 - FIAP Pos-Tech ML Engineering
# ==============================================
FROM python:3.10-slim

WORKDIR /app

# Instalar dependencias do sistema
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements e instalar
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar codigo fonte
COPY src/ ./src/
COPY config/ ./config/
COPY railway_app/backend/ ./railway_app/backend/

# Criar diretorio de modelos
RUN mkdir -p models

# Definir PYTHONPATH
ENV PYTHONPATH=/app

# Diretorio de trabalho
WORKDIR /app/railway_app/backend

# Porta
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Comando
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
