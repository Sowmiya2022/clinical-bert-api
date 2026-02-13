# ============================================================
# Clinical BERT Assertion API
# Base image: python:3.12-slim (as specified in requirements)
# ============================================================

FROM python:3.12-slim AS base

# ---- System dependencies ----
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ---- Build stage: install Python packages ----
FROM base AS builder

WORKDIR /app

# Copy and install requirements before source code for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---- Runtime stage ----
FROM base AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY app/ ./app/

# Pre-download and cache the HuggingFace model during image build so the
# container starts without internet access at runtime.
# Set HUGGINGFACE_HUB_DISABLE_IMPLICIT_TOKEN to suppress auth warnings.
ENV HUGGINGFACE_HUB_DISABLE_IMPLICIT_TOKEN=1 \
    TRANSFORMERS_OFFLINE=0 \
    HF_HOME=/app/.hf_cache \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Download model weights at build time
RUN python -c "\
from transformers import AutoModelForSequenceClassification, AutoTokenizer; \
model_name = 'bvanaken/clinical-assertion-negation-bert'; \
AutoTokenizer.from_pretrained(model_name); \
AutoModelForSequenceClassification.from_pretrained(model_name); \
print('Model cached successfully.')"

# Switch to offline mode so runtime never needs outbound network
ENV TRANSFORMERS_OFFLINE=1

# Non-root user for security
RUN adduser --disabled-password --gecos "" appuser \
 && chown -R appuser:appuser /app
USER appuser

# Cloud Run expects the server on the PORT env-var (default 8080)
ENV PORT=8080
EXPOSE 8080

# Health check — matches our /health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Start Uvicorn
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1 --log-level info"]