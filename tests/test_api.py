"""
Test suite for the Clinical BERT Assertion API.

Run with:
    pytest tests/ -v
    pytest tests/ -v --cov=app --cov-report=term-missing
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mock_predict_single():
    """
    Patch app.model.predict_single and app.model.load_model so tests run
    without downloading the full HuggingFace model (~440 MB).
    """
    label_map = {
        "The patient denies chest pain.": ("ABSENT", 0.9842),
        "He has a history of hypertension.": ("PRESENT", 0.9731),
        "If the patient experiences dizziness, reduce the dosage.": ("CONDITIONAL", 0.9617),
        "No signs of pneumonia were observed.": ("ABSENT", 0.9754),
    }

    def _fake_predict(sentence: str) -> dict:
        label, score = label_map.get(sentence, ("PRESENT", 0.85))
        return {"label": label, "score": score}

    def _fake_batch(sentences: list) -> list:
        return [
            {"sentence": s, **_fake_predict(s)} for s in sentences
        ]

    with (
        patch("app.model.load_model"),
        patch("app.model._pipeline", new=MagicMock()),
        patch("app.model.predict_single", side_effect=_fake_predict),
        patch("app.model.predict_batch", side_effect=_fake_batch),
        patch(
            "app.model.get_model_info",
            return_value={
                "model_name": "bvanaken/clinical-assertion-negation-bert",
                "loaded": True,
                "load_time_ms": 1234.5,
                "device": "CPU",
                "labels": ["PRESENT", "ABSENT", "CONDITIONAL"],
            },
        ),
    ):
        # Import app AFTER patches are applied
        from app.main import app
        with TestClient(app) as client:
            yield client


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_returns_200(self, mock_predict_single):
        resp = mock_predict_single.get("/health")
        assert resp.status_code == 200

    def test_health_response_structure(self, mock_predict_single):
        data = mock_predict_single.get("/health").json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True
        assert "model_name" in data
        assert "device" in data


# ---------------------------------------------------------------------------
# Root endpoint
# ---------------------------------------------------------------------------

class TestRoot:
    def test_root_returns_200(self, mock_predict_single):
        resp = mock_predict_single.get("/")
        assert resp.status_code == 200

    def test_root_contains_endpoints(self, mock_predict_single):
        data = mock_predict_single.get("/").json()
        assert "predict" in data["endpoints"]
        assert "health" in data["endpoints"]


# ---------------------------------------------------------------------------
# Single prediction — required test cases from assignment
# ---------------------------------------------------------------------------

REQUIRED_CASES = [
    ("The patient denies chest pain.", "ABSENT"),
    ("He has a history of hypertension.", "PRESENT"),
    ("If the patient experiences dizziness, reduce the dosage.", "CONDITIONAL"),
    ("No signs of pneumonia were observed.", "ABSENT"),
]


class TestPredict:
    @pytest.mark.parametrize("sentence,expected_label", REQUIRED_CASES)
    def test_required_cases(self, mock_predict_single, sentence, expected_label):
        resp = mock_predict_single.post("/predict", json={"sentence": sentence})
        assert resp.status_code == 200
        data = resp.json()
        assert data["label"] == expected_label
        assert 0.0 <= data["score"] <= 1.0

    def test_response_has_label_and_score(self, mock_predict_single):
        resp = mock_predict_single.post(
            "/predict", json={"sentence": "Patient has no fever."}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "label" in data
        assert "score" in data

    def test_score_is_between_0_and_1(self, mock_predict_single):
        resp = mock_predict_single.post(
            "/predict", json={"sentence": "He has a history of hypertension."}
        )
        assert 0.0 <= resp.json()["score"] <= 1.0

    def test_label_is_valid(self, mock_predict_single):
        resp = mock_predict_single.post(
            "/predict", json={"sentence": "He has a history of hypertension."}
        )
        assert resp.json()["label"] in {"PRESENT", "ABSENT", "CONDITIONAL"}

    # --- Input validation ---

    def test_empty_sentence_returns_422(self, mock_predict_single):
        resp = mock_predict_single.post("/predict", json={"sentence": ""})
        assert resp.status_code == 422

    def test_whitespace_only_sentence_returns_422(self, mock_predict_single):
        resp = mock_predict_single.post("/predict", json={"sentence": "   "})
        assert resp.status_code == 422

    def test_missing_sentence_field_returns_422(self, mock_predict_single):
        resp = mock_predict_single.post("/predict", json={})
        assert resp.status_code == 422

    def test_sentence_too_long_returns_422(self, mock_predict_single):
        resp = mock_predict_single.post(
            "/predict", json={"sentence": "word " * 600}
        )
        assert resp.status_code == 422

    def test_non_string_sentence_returns_422(self, mock_predict_single):
        resp = mock_predict_single.post("/predict", json={"sentence": 12345})
        assert resp.status_code == 422

    def test_get_method_not_allowed(self, mock_predict_single):
        resp = mock_predict_single.get("/predict")
        assert resp.status_code == 405


# ---------------------------------------------------------------------------
# Batch prediction
# ---------------------------------------------------------------------------

class TestBatchPredict:
    def test_batch_returns_correct_count(self, mock_predict_single):
        sentences = [s for s, _ in REQUIRED_CASES]
        resp = mock_predict_single.post(
            "/predict/batch", json={"sentences": sentences}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == len(sentences)
        assert len(data["results"]) == len(sentences)

    def test_batch_results_have_correct_fields(self, mock_predict_single):
        resp = mock_predict_single.post(
            "/predict/batch",
            json={"sentences": ["The patient denies chest pain."]},
        )
        result = resp.json()["results"][0]
        assert "sentence" in result
        assert "label" in result
        assert "score" in result

    def test_batch_preserves_order(self, mock_predict_single):
        sentences = [s for s, _ in REQUIRED_CASES]
        resp = mock_predict_single.post(
            "/predict/batch", json={"sentences": sentences}
        )
        returned_sentences = [r["sentence"] for r in resp.json()["results"]]
        assert returned_sentences == sentences

    def test_batch_required_labels(self, mock_predict_single):
        sentences = [s for s, _ in REQUIRED_CASES]
        expected_labels = [l for _, l in REQUIRED_CASES]
        resp = mock_predict_single.post(
            "/predict/batch", json={"sentences": sentences}
        )
        returned_labels = [r["label"] for r in resp.json()["results"]]
        assert returned_labels == expected_labels

    def test_batch_empty_list_returns_422(self, mock_predict_single):
        resp = mock_predict_single.post(
            "/predict/batch", json={"sentences": []}
        )
        assert resp.status_code == 422

    def test_batch_with_blank_sentence_returns_422(self, mock_predict_single):
        resp = mock_predict_single.post(
            "/predict/batch", json={"sentences": ["valid sentence", ""]}
        )
        assert resp.status_code == 422

    def test_batch_single_sentence(self, mock_predict_single):
        resp = mock_predict_single.post(
            "/predict/batch",
            json={"sentences": ["He has a history of hypertension."]},
        )
        assert resp.status_code == 200
        assert resp.json()["count"] == 1