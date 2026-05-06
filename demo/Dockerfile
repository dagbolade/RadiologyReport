# ─── Radiology Report Generator Demo ─────────────────────────────────────────
# Single container: FastAPI backend serves the frontend as static files.
# Model files are expected to be mounted or included at /app/Radiography/
# ──────────────────────────────────────────────────────────────────────────────

FROM python:3.11.7-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY demo/backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY demo/backend/ /app/
COPY demo/frontend/ /app/frontend/

# Copy model weights and tokenizer
COPY Radiography/Attention_with_cheXNet_full_model1.h5 /app/Radiography/
COPY ["Radiography/Copy of Copy of brucechou1983_CheXNet_Keras_0.3.0_weights.h5", "/app/Radiography/"]
COPY Radiography/tokenizer/ /app/Radiography/tokenizer/

# Environment
ENV MODEL_BASE_DIR=/app
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
