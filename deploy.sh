#!/usr/bin/env bash
# ============================================================
# deploy.sh — Manual deployment script for GCP Cloud Run
#
# Usage:
#   ./deploy.sh                          # uses defaults below
#   PROJECT=my-proj REGION=us-east1 ./deploy.sh
#
# Prerequisites:
#   - gcloud CLI installed and authenticated (gcloud auth login)
#   - Docker installed and running
#   - Artifact Registry API enabled in your GCP project
#   - Cloud Run API enabled in your GCP project
# ============================================================

set -euo pipefail

# ---- Configuration (override via environment or edit here) ----
PROJECT="${PROJECT:-your-gcp-project-id}"
REGION="${REGION:-us-central1}"
SERVICE="${SERVICE:-clinical-bert-api}"
AR_REPO="${AR_REPO:-clinical-bert}"
IMAGE_NAME="clinical-bert-api"
IMAGE_TAG="${IMAGE_TAG:-latest}"

# Full image URI
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT}/${AR_REPO}/${IMAGE_NAME}:${IMAGE_TAG}"

echo "============================================================"
echo "  Clinical BERT API — GCP Deployment"
echo "============================================================"
echo "  Project  : ${PROJECT}"
echo "  Region   : ${REGION}"
echo "  Service  : ${SERVICE}"
echo "  Image    : ${IMAGE_URI}"
echo "============================================================"
echo ""

# ---- Step 1: Enable required APIs ----
echo "[1/6] Enabling GCP APIs …"
gcloud services enable \
  artifactregistry.googleapis.com \
  run.googleapis.com \
  --project="${PROJECT}"

# ---- Step 2: Create Artifact Registry repository (idempotent) ----
echo "[2/6] Ensuring Artifact Registry repository exists …"
gcloud artifacts repositories create "${AR_REPO}" \
  --repository-format=docker \
  --location="${REGION}" \
  --description="Clinical BERT API images" \
  --project="${PROJECT}" 2>/dev/null || true

# ---- Step 3: Configure Docker ----
echo "[3/6] Configuring Docker for Artifact Registry …"
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

# ---- Step 4: Build Docker image ----
echo "[4/6] Building Docker image …"
docker build \
  --platform linux/amd64 \
  -t "${IMAGE_URI}" \
  .

# ---- Step 5: Push to Artifact Registry ----
echo "[5/6] Pushing image to Artifact Registry …"
docker push "${IMAGE_URI}"

# ---- Step 6: Deploy to Cloud Run ----
echo "[6/6] Deploying to Cloud Run …"
gcloud run deploy "${SERVICE}" \
  --image="${IMAGE_URI}" \
  --region="${REGION}" \
  --platform=managed \
  --allow-unauthenticated \
  --memory=2Gi \
  --cpu=2 \
  --min-instances=0 \
  --max-instances=5 \
  --concurrency=4 \
  --timeout=60 \
  --port=8080 \
  --project="${PROJECT}"

# ---- Print service URL ----
SERVICE_URL=$(gcloud run services describe "${SERVICE}" \
  --region="${REGION}" \
  --project="${PROJECT}" \
  --format="value(status.url)")

echo ""
echo "============================================================"
echo "  Deployment complete!"
echo "  Service URL: ${SERVICE_URL}"
echo "============================================================"
echo ""
echo "  Quick test:"
echo "  curl -X POST ${SERVICE_URL}/predict \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"sentence\": \"The patient denies chest pain.\"}'"
echo ""
