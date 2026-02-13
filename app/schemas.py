"""
Pydantic schemas for request / response validation.
"""

from typing import Annotated

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Single prediction
# ---------------------------------------------------------------------------


class PredictRequest(BaseModel):
    """Input schema for POST /predict"""

    sentence: Annotated[
        str,
        Field(
            min_length=1,
            max_length=2048,
            description="Clinical sentence to classify.",
            examples=["The patient denies chest pain."],
        ),
    ]

    @field_validator("sentence")
    @classmethod
    def sentence_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("sentence must not be blank or whitespace only.")
        return v.strip()


class PredictResponse(BaseModel):
    """Output schema for POST /predict"""

    label: Annotated[
        str,
        Field(
            description="Assertion label: PRESENT | ABSENT | CONDITIONAL",
            examples=["ABSENT"],
        ),
    ]
    score: Annotated[
        float,
        Field(
            ge=0.0,
            le=1.0,
            description="Confidence score for the predicted label (0-1).",
            examples=[0.9842],
        ),
    ]


# ---------------------------------------------------------------------------
# Batch prediction
# ---------------------------------------------------------------------------


class BatchPredictRequest(BaseModel):
    """Input schema for POST /predict/batch"""

    sentences: Annotated[
        list[str],
        Field(
            min_length=1,
            max_length=64,
            description="List of clinical sentences (1-64 items).",
        ),
    ]

    @field_validator("sentences")
    @classmethod
    def validate_sentences(cls, v: list[str]) -> list[str]:
        cleaned = []
        for i, s in enumerate(v):
            if not isinstance(s, str) or not s.strip():
                raise ValueError(f"sentences[{i}] is empty or not a string.")
            cleaned.append(s.strip())
        return cleaned


class BatchPredictItem(BaseModel):
    """Single item in the batch response."""

    sentence: str
    label: str
    score: float = Field(ge=0.0, le=1.0)


class BatchPredictResponse(BaseModel):
    """Output schema for POST /predict/batch"""

    results: list[BatchPredictItem]
    count: int


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """Output schema for GET /health"""

    status: str = Field(description="'ok' when the service is healthy.")
    model_name: str
    model_loaded: bool
    device: str
