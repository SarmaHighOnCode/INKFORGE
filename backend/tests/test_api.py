"""
INKFORGE — API Tests

Tests for FastAPI endpoints: /generate, /export, /styles, /health.
"""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_returns_200(self) -> None:
        """Health endpoint should return 200 with status info."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_contains_required_fields(self) -> None:
        """Health response should contain status, model_loaded."""
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        if "gpu" in data:
            assert "available" in data["gpu"]


class TestStylesEndpoint:
    """Tests for GET /styles."""

    def test_list_styles_returns_presets(self) -> None:
        """Should return all 5 style presets."""
        response = client.get("/api/styles")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 5

    def test_style_preset_has_required_fields(self) -> None:
        """Each preset should have id, name, description."""
        response = client.get("/api/styles")
        data = response.json()
        preset = data[0]
        assert "id" in preset
        assert "name" in preset
        assert "description" in preset


class TestGenerateEndpoint:
    """Tests for POST /generate."""

    def test_generate_accepts_valid_request(self) -> None:
        """Should accept valid text with default params."""
        response = client.post("/api/generate", json={"text": "Hello World"})
        assert response.status_code == 202
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "queued"

    def test_generate_rejects_empty_text(self) -> None:
        """Should return 422 for empty text input."""
        response = client.post("/api/generate", json={"text": ""})
        assert response.status_code == 422

    def test_generate_rejects_text_over_2000_chars(self) -> None:
        """Should reject text exceeding MVP 2,000 char limit."""
        response = client.post("/api/generate", json={"text": "a" * 2001})
        assert response.status_code == 422


class TestExportEndpoint:
    """Tests for POST /export."""

    def test_export_requires_job_id(self) -> None:
        """Should return 422 if job_id is missing."""
        response = client.post("/api/export", json={"format": "png"})
        assert response.status_code == 422
