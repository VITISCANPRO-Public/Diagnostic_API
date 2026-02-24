"""
test_api_integration.py — Integration tests for the Vitiscan Diagnostic API.

These tests verify that all API endpoints respond correctly.
They run inside GitHub Actions using a mock model (TESTING=true),
without needing to load the real ResNet18 model from MLflow.

Response formats (matching schemas.py):
    GET  /           → { "message": str, "status": str }
    GET  /diseases   → DiseasesResponse  { "diseases": dict, "dataset_name": str }
    POST /diagno     → PredictionResponse { "predictions": list[DiseasePrediction],
                                            "model_version": str }
                        DiseasePrediction  { "disease": str, "confidence": float }
"""

import io
import os

import pytest
from fastapi.testclient import TestClient
from PIL import Image

# ── Setup ──────────────────────────────────────────────────────────────────────

# Set TESTING=true before importing the app.
# This triggers the mock model branch in startup(), skipping MLflow and S3.
os.environ["TESTING"] = "true"

from app import app

client = TestClient(app)

# The 7 expected INRAE classes — must match CLASS_NAMES in app.py exactly
EXPECTED_CLASSES = sorted([
    "colomerus_vitis",
    "elsinoe_ampelina",
    "erysiphe_necator",
    "guignardia_bidwellii",
    "healthy",
    "phaeomoniella_chlamydospora",
    "plasmopara_viticola",
])


def create_test_image(width: int = 224, height: int = 224) -> bytes:
    """
    Creates a fake RGB image in memory for testing purposes.
    Returns the image as JPEG bytes.

    We use 224x224 because that is the input size expected by ResNet18.
    """
    image  = Image.new("RGB", (width, height), color=(34, 139, 34))  # leaf green
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)
    return buffer.read()


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestHealthCheck:
    """Verifies that the API starts and responds correctly."""

    def test_root_returns_200(self):
        """The root endpoint / must respond with HTTP status 200."""
        response = client.get("/")
        assert response.status_code == 200, (
            f"API not responding. Status received: {response.status_code}"
        )

    def test_root_returns_correct_fields(self):
        """The root response must contain 'message' and 'status' fields."""
        response = client.get("/")
        data     = response.json()
        assert "message" in data, "Missing field 'message' in root response"
        assert "status"  in data, "Missing field 'status' in root response"

    def test_root_status_is_ok(self):
        """The 'status' field must equal 'ok'."""
        response = client.get("/")
        assert response.json()["status"] == "ok", (
            f"Expected status 'ok', received: {response.json()['status']}"
        )


class TestDiseasesEndpoint:
    """Verifies that /diseases returns the correct INRAE classes."""

    def test_diseases_returns_200(self):
        response = client.get("/diseases")
        assert response.status_code == 200

    def test_diseases_response_has_correct_fields(self):
        """
        The /diseases response must contain:
        - diseases    : dict mapping class names to human-readable labels
        - dataset_name: name of the dataset used ('inrae' by default)
        """
        response = client.get("/diseases")
        data     = response.json()

        assert "diseases"     in data, "Missing field 'diseases' in /diseases response"
        assert "dataset_name" in data, "Missing field 'dataset_name' in /diseases response"

    def test_diseases_returns_all_7_classes(self):
        """
        The API must return exactly the 7 INRAE disease classes.
        If any class is missing or misspelled, this test will fail.
        """
        response         = client.get("/diseases")
        returned_classes = sorted(response.json()["diseases"].keys())

        assert returned_classes == EXPECTED_CLASSES, (
            f"Incorrect disease classes.\n"
            f"Expected : {EXPECTED_CLASSES}\n"
            f"Received : {returned_classes}"
        )

    def test_diseases_have_non_empty_labels(self):
        """Each disease key must have a non-empty label as its value."""
        diseases = client.get("/diseases").json()["diseases"]
        for class_name, label in diseases.items():
            assert label, f"Empty label for class '{class_name}' in /diseases"


class TestDiagnoEndpoint:
    """Verifies that /diagno accepts images and returns valid predictions."""

    def test_diagno_returns_200_with_valid_image(self):
        """Sending a valid JPEG image must return HTTP 200."""
        response = client.post(
            "/diagno",
            files={"file": ("test_leaf.jpg", create_test_image(), "image/jpeg")},
        )
        assert response.status_code == 200, (
            f"/diagno returned {response.status_code} instead of 200.\n"
            f"Response: {response.text}"
        )

    def test_diagno_response_has_correct_fields(self):
        """
        The /diagno response must contain:
        - predictions  : list of DiseasePrediction objects
        - model_version: string identifying the model used
        """
        response = client.post(
            "/diagno",
            files={"file": ("test_leaf.jpg", create_test_image(), "image/jpeg")},
        )
        data = response.json()

        assert "predictions"   in data, "Missing field 'predictions' in /diagno response"
        assert "model_version" in data, "Missing field 'model_version' in /diagno response"

    def test_diagno_returns_7_predictions(self):
        """
        The predictions list must contain exactly 7 entries — one per INRAE class.
        """
        response    = client.post(
            "/diagno",
            files={"file": ("test_leaf.jpg", create_test_image(), "image/jpeg")},
        )
        predictions = response.json()["predictions"]

        assert len(predictions) == 7, (
            f"Expected 7 predictions (one per class), received: {len(predictions)}"
        )

    def test_diagno_each_prediction_has_disease_and_confidence(self):
        """
        Each prediction in the list must have:
        - disease   : the class name (string)
        - confidence: the probability score (float)
        """
        response    = client.post(
            "/diagno",
            files={"file": ("test_leaf.jpg", create_test_image(), "image/jpeg")},
        )
        predictions = response.json()["predictions"]

        for pred in predictions:
            assert "disease"    in pred, f"Missing 'disease' field in prediction: {pred}"
            assert "confidence" in pred, f"Missing 'confidence' field in prediction: {pred}"

    def test_diagno_all_predicted_classes_are_valid(self):
        """All predicted class names must belong to the 7 known INRAE classes."""
        response    = client.post(
            "/diagno",
            files={"file": ("test_leaf.jpg", create_test_image(), "image/jpeg")},
        )
        predictions = response.json()["predictions"]

        for pred in predictions:
            assert pred["disease"] in EXPECTED_CLASSES, (
                f"Unknown class in predictions: '{pred['disease']}'\n"
                f"Accepted classes: {EXPECTED_CLASSES}"
            )

    def test_diagno_confidence_scores_are_valid_floats(self):
        """Every confidence score must be a float between 0.0 and 1.0."""
        response    = client.post(
            "/diagno",
            files={"file": ("test_leaf.jpg", create_test_image(), "image/jpeg")},
        )
        predictions = response.json()["predictions"]

        for pred in predictions:
            c = pred["confidence"]
            assert isinstance(c, float), (
                f"Confidence must be a float, received {type(c)} for class '{pred['disease']}'"
            )
            assert 0.0 <= c <= 1.0, (
                f"Confidence must be between 0 and 1, received {c} for class '{pred['disease']}'"
            )

    def test_diagno_confidence_scores_sum_to_one(self):
        """
        The sum of all confidence scores must be approximately 1.0.
        This is a mathematical property of a valid softmax output.
        """
        response    = client.post(
            "/diagno",
            files={"file": ("test_leaf.jpg", create_test_image(), "image/jpeg")},
        )
        predictions = response.json()["predictions"]
        total       = sum(pred["confidence"] for pred in predictions)

        assert abs(total - 1.0) < 0.01, (
            f"Confidence scores should sum to ~1.0, received: {total}"
        )

    def test_diagno_predictions_are_sorted_by_confidence_descending(self):
        """
        Predictions must be returned sorted from highest to lowest confidence.
        The best prediction comes first.
        """
        response    = client.post(
            "/diagno",
            files={"file": ("test_leaf.jpg", create_test_image(), "image/jpeg")},
        )
        predictions = response.json()["predictions"]
        confidences = [pred["confidence"] for pred in predictions]

        assert confidences == sorted(confidences, reverse=True), (
            "Predictions are not sorted by confidence (descending)"
        )

    def test_diagno_model_version_is_string(self):
        """The model_version field must be a non-empty string."""
        response = client.post(
            "/diagno",
            files={"file": ("test_leaf.jpg", create_test_image(), "image/jpeg")},
        )
        model_version = response.json()["model_version"]

        assert isinstance(model_version, str), "model_version must be a string"
        assert model_version, "model_version must not be empty"


class TestDiagnoErrorHandling:
    """Verifies that the API handles invalid inputs gracefully."""

    def test_diagno_without_file_returns_422(self):
        """
        Calling /diagno without a file must return HTTP 422 (Unprocessable Entity).
        This is FastAPI's standard behavior for missing required fields.
        """
        response = client.post("/diagno")
        assert response.status_code == 422, (
            f"Without a file, /diagno should return 422, received: {response.status_code}"
        )

    def test_diagno_with_non_image_file_returns_error(self):
        """
        Sending a text file instead of an image must return an error (400 or 500).
        The API must not crash silently.
        """
        fake_text = b"this is not an image"
        response  = client.post(
            "/diagno",
            files={"file": ("not_an_image.txt", fake_text, "text/plain")},
        )
        assert response.status_code in [400, 422, 500], (
            f"A non-image file should return 400, 422 or 500, "
            f"received: {response.status_code}"
        )