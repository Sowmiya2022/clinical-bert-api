"""
Clinical BERT Real-Time Inference API
=====================================
Classifies the assertion status of medical concepts in clinical text.

Labels:
  PRESENT      - concept is affirmed / observed
  ABSENT       - concept is negated / denied
  CONDITIONAL  - concept depends on a condition or is hypothetical

Endpoints:
  GET  /health           - liveness/readiness probe
  GET  /                 - API info
  POST /predict          - single sentence classification
  POST /predict/batch    - batch classification (up to 64 sentences)
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.model import get_model_info, load_model, predict_batch, predict_single
from app.schemas import (
    BatchPredictItem,
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan — load model once at startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the ML model before serving any requests."""
    logger.info("=== Application startup: loading Clinical BERT model ===")
    load_model()
    logger.info("=== Model ready — API is live ===")
    yield
    logger.info("=== Application shutdown ===")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Clinical BERT Assertion API",
    description=(
        "Real-time inference API for clinical assertion/negation classification "
        "using the bvanaken/clinical-assertion-negation-bert HuggingFace model."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — tighten in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request timing middleware
# ---------------------------------------------------------------------------
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
    return response


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled error on {request.method} {request.url.path}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please try again later."},
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", summary="API info", tags=["Meta"])
async def root():
    """Return basic API metadata."""
    return {
        "name": "Clinical BERT Assertion API",
        "version": "1.0.0",
        "model": "bvanaken/clinical-assertion-negation-bert",
        "endpoints": {
            "health": "GET /health",
            "predict": "POST /predict",
            "batch_predict": "POST /predict/batch",
            "docs": "GET /docs",
        },
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health / readiness check",
    tags=["Meta"],
)
async def health():
    """
    Liveness and readiness probe.
    Returns HTTP 200 when the model is loaded and ready to serve requests.
    Returns HTTP 503 if the model failed to load.
    """
    info = get_model_info()
    if not info["loaded"]:
        raise HTTPException(
            status_code=503,
            detail="Model is not yet loaded. Service is not ready.",
        )
    return HealthResponse(
        status="ok",
        model_name=info["model_name"],
        model_loaded=info["loaded"],
        device=info["device"],
    )


@app.post(
    "/predict",
    response_model=PredictResponse,
    summary="Classify a single clinical sentence",
    tags=["Inference"],
)
async def predict(request: PredictRequest):
    """
    Classify the assertion status of a clinical sentence.

    **Labels:**
    - `PRESENT` - the medical concept is affirmed (e.g. *"The patient has fever."*)
    - `ABSENT` - the concept is negated (e.g. *"No signs of pneumonia."*)
    - `CONDITIONAL` - the concept is hypothetical (e.g. *"If dizziness occurs, reduce dose."*)

    **Performance target:** < 500 ms for short clinical sentences on CPU.
    """
    try:
        result = predict_single(request.sentence)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.exception(f"Prediction error for sentence: {request.sentence!r}")
        raise HTTPException(status_code=500, detail="Prediction failed.") from exc

    return PredictResponse(**result)


@app.post(
    "/predict/batch",
    response_model=BatchPredictResponse,
    summary="Classify multiple clinical sentences in one request",
    tags=["Inference"],
)
async def predict_batch_endpoint(request: BatchPredictRequest):
    """
    Classify the assertion status of up to **64** clinical sentences in a single
    request. Sentences are processed in one batched forward-pass for efficiency.

    Returns a list of results preserving input order.
    """
    try:
        results = predict_batch(request.sentences)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.exception("Batch prediction error")
        raise HTTPException(status_code=500, detail="Batch prediction failed.") from exc

    items = [BatchPredictItem(**r) for r in results]
    return BatchPredictResponse(results=items, count=len(items))
