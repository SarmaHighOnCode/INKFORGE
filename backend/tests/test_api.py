"""
INKFORGE — API Tests

Tests for FastAPI endpoints: /generate, /export, /styles, /health.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_returns_200(self) -> None:
        """Health endpoint should return 200 with status info."""
        # TODO: Implement when health route is registered
        pass

    def test_health_contains_required_fields(self) -> None:
        """Health response should contain status, model_loaded, gpu_available."""
        # TODO: Implement
        pass


class TestStylesEndpoint:
    """Tests for GET /styles."""

    def test_list_styles_returns_presets(self) -> None:
        """Should return all 5 style presets."""
        # TODO: Implement when styles route is registered
        pass

    def test_style_preset_has_required_fields(self) -> None:
        """Each preset should have id, name, description."""
        # TODO: Implement
        pass


class TestGenerateEndpoint:
    """Tests for POST /generate."""

    def test_generate_accepts_valid_request(self) -> None:
        """Should accept valid text with default params."""
        # TODO: Implement
        pass

    def test_generate_rejects_empty_text(self) -> None:
        """Should return 422 for empty text input."""
        # TODO: Implement
        pass

    def test_generate_rejects_text_over_2000_chars(self) -> None:
        """Should reject text exceeding MVP 2,000 char limit."""
        # TODO: Implement
        pass


class TestExportEndpoint:
    """Tests for POST /export."""

    def test_export_requires_job_id(self) -> None:
        """Should return 422 if job_id is missing."""
        # TODO: Implement
        pass
