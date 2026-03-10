"""
Tests for the FastAPI endpoints in src/main.py.

The conftest.py mocks torch/transformers and sets SKIP_MODEL_LOAD=true so
the server starts without downloading any model.
"""
import pytest
from fastapi.testclient import TestClient

from src.main import app, enhance_with_emojis, ChatRequest

client = TestClient(app)


# ---------------------------------------------------------------------------
# Root endpoint
# ---------------------------------------------------------------------------
class TestRootEndpoint:
    def test_returns_200(self):
        response = client.get("/")
        assert response.status_code == 200

    def test_service_name(self):
        data = client.get("/").json()
        assert data["service"] == "BlackRoad AI - Qwen2.5"

    def test_status_online(self):
        data = client.get("/").json()
        assert data["status"] == "online"

    def test_features_present(self):
        data = client.get("/").json()
        assert "features" in data
        features = data["features"]
        assert "memory_integration" in features
        assert "emoji_support" in features
        assert "action_execution" in features

    def test_model_name_present(self):
        data = client.get("/").json()
        assert "model" in data


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------
class TestHealthEndpoint:
    def test_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_status_healthy(self):
        data = client.get("/health").json()
        assert data["status"] == "healthy"

    def test_model_not_loaded_in_test_mode(self):
        # SKIP_MODEL_LOAD=true means model is None
        data = client.get("/health").json()
        assert data["model_loaded"] is False

    def test_memory_not_connected_when_disabled(self):
        # BLACKROAD_MEMORY_ENABLED=false means memory_bridge is None
        data = client.get("/health").json()
        assert data["memory_connected"] is False


# ---------------------------------------------------------------------------
# Chat endpoint
# ---------------------------------------------------------------------------
class TestChatEndpoint:
    def test_503_when_model_not_loaded(self):
        # Model is not loaded (SKIP_MODEL_LOAD=true), so chat must return 503
        response = client.post("/chat", json={"message": "Hello"})
        assert response.status_code == 503

    def test_503_error_detail(self):
        response = client.post("/chat", json={"message": "Hello"})
        assert "not loaded" in response.json()["detail"].lower()

    def test_invalid_request_missing_message(self):
        response = client.post("/chat", json={})
        assert response.status_code == 422  # Pydantic validation error


# ---------------------------------------------------------------------------
# Emoji enhancement helper
# ---------------------------------------------------------------------------
class TestEnhanceWithEmojis:
    def test_success_keyword_gets_emoji(self):
        result = enhance_with_emojis("success")
        assert "✅" in result

    def test_error_keyword_gets_emoji(self):
        result = enhance_with_emojis("error occurred")
        assert "❌" in result

    def test_rocket_keyword_gets_emoji(self):
        result = enhance_with_emojis("rocket launch")
        assert "🚀" in result

    def test_brain_keyword_gets_emoji(self):
        result = enhance_with_emojis("brain power")
        assert "🧠" in result

    def test_no_duplicate_emoji(self):
        # If emoji already in text, should not be added again
        result = enhance_with_emojis("✅ success complete")
        # emoji should appear exactly once
        assert result.count("✅") == 1

    def test_plain_text_unchanged(self):
        text = "nothing special here"
        result = enhance_with_emojis(text)
        assert result == text

    def test_blackroad_keyword(self):
        result = enhance_with_emojis("blackroad is awesome")
        assert "🖤🛣️" in result


# ---------------------------------------------------------------------------
# ChatRequest model validation
# ---------------------------------------------------------------------------
class TestChatRequestDefaults:
    def test_default_max_tokens(self):
        req = ChatRequest(message="hello")
        assert req.max_tokens == 512

    def test_default_temperature(self):
        req = ChatRequest(message="hello")
        assert req.temperature == 0.7

    def test_default_use_memory(self):
        req = ChatRequest(message="hello")
        assert req.use_memory is True

    def test_default_enable_actions(self):
        req = ChatRequest(message="hello")
        assert req.enable_actions is True

    def test_session_id_optional(self):
        req = ChatRequest(message="hello")
        assert req.session_id is None
