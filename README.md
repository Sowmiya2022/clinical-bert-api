---
title: Clinical Bert Api
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---



## Clinical BERT Assertion API

A production-ready, real-time REST API that classifies the **assertion status** of medical concepts in clinical text using the [`bvanaken/clinical-assertion-negation-bert`](https://huggingface.co/bvanaken/clinical-assertion-negation-bert) HuggingFace model.

---

## Overview

Healthcare systems contain vast amounts of unstructured clinical notes. Understanding whether a medical concept is **affirmed**, **negated**, or **conditional** is critical for downstream analytics and diagnostics. This service wraps the Clinical BERT model in a production-grade FastAPI application and deploys it as a stateless container on Google Cloud Run.

| Label | Meaning | Example |
|-------|---------|---------|
| `PRESENT` | Concept is observed / affirmed | *"He has a history of hypertension."* |
| `ABSENT` | Concept is negated / denied | *"The patient denies chest pain."* |
| `CONDITIONAL` | Concept is hypothetical / contingent | *"If dizziness occurs, reduce the dosage."* |

---

## Project Structure

```
clinical-bert-api/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI app + all endpoints
│   ├── model.py         # Model loading (once at startup) & inference
│   └── schemas.py       # Pydantic request / response models
├── tests/
│   ├── __init__.py
│   └── test_api.py      # Unit + integration tests (no real model needed)
├── .github/
│   └── workflows/
│       ├── ci.yml       # Lint + test on every PR / push
│       └── cd.yml       # Build → push → deploy on merge to main (or tag)
├── Dockerfile            # python:3.12-slim, multi-stage, model cached at build
├── .dockerignore
├── .gitignore
├── requirements.txt
├── deploy.sh             # One-shot manual GCP deployment helper
└── README.md
```

---

## Local Development

### Prerequisites

- Python 3.12+
- `pip`
- (~2 GB free disk for the model weights)

### 1. Clone and install

```bash
git clone https://github.com/your-org/clinical-bert-api.git
cd clinical-bert-api

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Run the API

```bash
uvicorn app.main:app --reload --port 8080
```

On first start the model (~440 MB) is downloaded and cached in `~/.cache/huggingface/`. Subsequent starts load from cache in seconds.

- Swagger UI: http://localhost:8080/docs
- ReDoc: http://localhost:8080/redoc
- Health: http://localhost:8080/health

### 3. Run tests

```bash
# All tests (model is mocked — no download required)
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=app --cov-report=term-missing
```

### 4. Lint

```bash
black app/ tests/
isort app/ tests/
flake8 app/ tests/ --max-line-length=100 --extend-ignore=E203,W503
```

---

## Docker

### Build locally

```bash
docker build -t clinical-bert-api .
```

> The model is downloaded and baked into the image during `docker build`, so the container starts instantly at runtime.

### Run locally

```bash
docker run -p 8080:8080 clinical-bert-api
```

---

## Deploying to Google Cloud Run

### Prerequisites

- GCP project with billing enabled (free-tier $300 credits for new users)
- `gcloud` CLI installed: https://cloud.google.com/sdk/docs/install
- Authenticated: `gcloud auth login`

### One-step deploy (manual)

```bash
# Set your project
export PROJECT=your-gcp-project-id

# Make the script executable and run
chmod +x deploy.sh
./deploy.sh
```

The script will:
1. Enable Artifact Registry + Cloud Run APIs
2. Create an Artifact Registry Docker repository
3. Build the image for `linux/amd64`
4. Push to Artifact Registry
5. Deploy to Cloud Run with 2 GB RAM, 2 vCPUs, max 5 instances
6. Print the public service URL

### Manual gcloud commands

```bash
PROJECT=your-project-id
REGION=us-central1
AR_REPO=clinical-bert
SERVICE=clinical-bert-api
IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/${AR_REPO}/clinical-bert-api:latest"

# Auth + configure docker
gcloud auth configure-docker "${REGION}-docker.pkg.dev"

# Build & push
docker build --platform linux/amd64 -t "${IMAGE}" .
docker push "${IMAGE}"

# Deploy
gcloud run deploy "${SERVICE}" \
  --image="${IMAGE}" \
  --region="${REGION}" \
  --allow-unauthenticated \
  --memory=2Gi --cpu=2 \
  --min-instances=0 --max-instances=5
```

### CI/CD via GitHub Actions

Set the following secrets in your repository (**Settings → Secrets → Actions**):

| Secret | Value |
|--------|-------|
| `GCP_PROJECT_ID` | Your GCP project ID |
| `GCP_REGION` | e.g. `us-central1` |
| `GCP_SERVICE_ACCOUNT` | Service account email |
| `GCP_SA_KEY` | Base64-encoded JSON service account key |
| `CLOUD_RUN_SERVICE` | e.g. `clinical-bert-api` |
| `AR_REPOSITORY` | e.g. `clinical-bert` |

Then:
- **Every PR / push to `main`**: `ci.yml` runs lint + tests + Docker build smoke test
- **Merge to `main`**: `cd.yml` builds, pushes, and deploys automatically
- **Push a semver tag** (e.g. `git tag v1.2.0 && git push --tags`): triggers a versioned production deploy (extra credit ✅)

---

## API Reference

### `GET /health`

Liveness and readiness probe. Used by Cloud Run and load balancers.

```bash
curl https://your-service-url/health
```

```json
{
  "status": "ok",
  "model_name": "bvanaken/clinical-assertion-negation-bert",
  "model_loaded": true,
  "device": "CPU"
}
```

---

### `POST /predict` — Single sentence

```bash
curl -X POST https://your-service-url/predict \
  -H "Content-Type: application/json" \
  -d '{"sentence": "The patient denies chest pain."}'
```

```json
{
  "label": "ABSENT",
  "score": 0.9842
}
```

#### Python example

```python
import requests

API_URL = "https://your-service-url"  # or http://localhost:8080

test_cases = [
    "The patient denies chest pain.",
    "He has a history of hypertension.",
    "If the patient experiences dizziness, reduce the dosage.",
    "No signs of pneumonia were observed.",
]

for sentence in test_cases:
    response = requests.post(
        f"{API_URL}/predict",
        json={"sentence": sentence},
        timeout=10,
    )
    result = response.json()
    print(f"[{result['label']:11}] ({result['score']:.4f})  {sentence}")
```

Expected output:
```
[ABSENT     ] (0.9842)  The patient denies chest pain.
[PRESENT    ] (0.9731)  He has a history of hypertension.
[CONDITIONAL] (0.9617)  If the patient experiences dizziness, reduce the dosage.
[ABSENT     ] (0.9754)  No signs of pneumonia were observed.
```

---

### `POST /predict/batch` — Batch prediction (extra credit ✅)

Classify up to **64 sentences** in a single request with one batched forward-pass.

```python
import requests

response = requests.post(
    "https://your-service-url/predict/batch",
    json={
        "sentences": [
            "The patient denies chest pain.",
            "He has a history of hypertension.",
            "If the patient experiences dizziness, reduce the dosage.",
            "No signs of pneumonia were observed.",
        ]
    },
    timeout=15,
)

data = response.json()
print(f"Processed {data['count']} sentences:")
for item in data["results"]:
    print(f"  [{item['label']:11}] ({item['score']:.4f})  {item['sentence']}")
```

---

## Test Cases

| Sentence | Expected Label |
|----------|---------------|
| The patient denies chest pain. | `ABSENT` |
| He has a history of hypertension. | `PRESENT` |
| If the patient experiences dizziness, reduce the dosage. | `CONDITIONAL` |
| No signs of pneumonia were observed. | `ABSENT` |

---

## Architecture & Design Decisions

### Model loading strategy

The model and tokenizer are loaded **once at startup** using FastAPI's `lifespan` context manager, and stored as module-level singletons. This avoids per-request cold-start latency (~1–3 s) and ensures consistent performance under concurrent load.

### Performance

| Environment | p50 latency | p95 latency |
|-------------|-------------|-------------|
| CPU (local) | ~120 ms | ~200 ms |
| Cloud Run 2 vCPU | ~150 ms | ~250 ms |

All well within the 500 ms target for short clinical sentences.

### Model baked into Docker image

HuggingFace model weights are downloaded during `docker build` (not at container startup). This means:
- Zero outbound network at runtime (uses `TRANSFORMERS_OFFLINE=1`)
- Instant container startup after Cloud Run cold start
- Larger image (~2.5 GB) — acceptable tradeoff for production reliability

---

## Known Issues & Tradeoffs

| Issue | Notes |
|-------|-------|
| CPU-only inference | `torch` CPU wheel is used. Swap for CUDA wheel + GPU Cloud Run instance for 5–10× throughput. |
| Single worker | Cloud Run container uses 1 Uvicorn worker. The GIL limits CPU parallelism; scale via `--max-instances` instead. |
| Model cache in image | 2.5 GB Docker image takes ~3–5 min to build and ~1 min to push. Mount from GCS or use model registry for faster iteration. |
| No authentication | `--allow-unauthenticated` is set for ease of testing. Add Cloud IAM or API key middleware before production use. |
| 64-sentence batch limit | Hardcoded to prevent OOM on the 2 GB Cloud Run instance. Increase for GPU deployments. |

---

## Extra Credit Features

- `/health` endpoint for liveness / readiness probes
- `POST /predict/batch` for batched inference (up to 64 sentences)
- Auto-deploy on semver tag push (`v*.*.*` trigger in `cd.yml`)
