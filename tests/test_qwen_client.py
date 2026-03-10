"""
Tests for src/qwen_client.py — exercises chat, code, embed and vision
methods against a mocked httpx backend so no live gateway is required.
"""
import sys
import os
import pytest
from unittest.mock import MagicMock, patch

# Ensure src is importable from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.qwen_client import QwenClient, ask, code, analyze_image


def _mock_response(payload: dict, status_code: int = 200):
    """Return a minimal httpx.Response-like mock."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = payload
    resp.raise_for_status = MagicMock()
    return resp


# ---------------------------------------------------------------------------
# QwenClient.chat
# ---------------------------------------------------------------------------
class TestQwenClientChat:
    def test_returns_content_string(self):
        with patch("httpx.post", return_value=_mock_response({"content": "4"})) as mock_post:
            client = QwenClient()
            result = client.chat("What is 2+2?")
        assert result == "4"

    def test_sends_user_message(self):
        with patch("httpx.post", return_value=_mock_response({"content": "hi"})) as mock_post:
            client = QwenClient()
            client.chat("hello")
            call_json = mock_post.call_args.kwargs["json"]
            messages = call_json["messages"]
            assert any(m["role"] == "user" and m["content"] == "hello" for m in messages)

    def test_system_message_included_when_provided(self):
        with patch("httpx.post", return_value=_mock_response({"content": "ok"})) as mock_post:
            client = QwenClient()
            client.chat("hi", system="Be concise.")
            messages = mock_post.call_args.kwargs["json"]["messages"]
            assert messages[0]["role"] == "system"
            assert messages[0]["content"] == "Be concise."

    def test_no_system_message_when_empty(self):
        with patch("httpx.post", return_value=_mock_response({"content": "ok"})) as mock_post:
            client = QwenClient()
            client.chat("hi")
            messages = mock_post.call_args.kwargs["json"]["messages"]
            roles = [m["role"] for m in messages]
            assert "system" not in roles

    def test_temperature_forwarded(self):
        with patch("httpx.post", return_value=_mock_response({"content": "ok"})) as mock_post:
            client = QwenClient()
            client.chat("hi", temperature=0.1)
            body = mock_post.call_args.kwargs["json"]
            assert body["temperature"] == 0.1

    def test_empty_content_key_returns_empty_string(self):
        with patch("httpx.post", return_value=_mock_response({})):
            client = QwenClient()
            result = client.chat("hi")
        assert result == ""

    def test_model_sent_in_request(self):
        with patch("httpx.post", return_value=_mock_response({"content": "ok"})) as mock_post:
            client = QwenClient(model="qwen2.5:14b")
            client.chat("hi")
            body = mock_post.call_args.kwargs["json"]
            assert body["model"] == "qwen2.5:14b"


# ---------------------------------------------------------------------------
# QwenClient.code
# ---------------------------------------------------------------------------
class TestQwenClientCode:
    def test_returns_code_string(self):
        code_snippet = "def fib(n):\n    pass"
        with patch("httpx.post", return_value=_mock_response({"content": code_snippet})):
            client = QwenClient()
            result = client.code("fibonacci sequence")
        assert result == code_snippet

    def test_low_temperature_used(self):
        with patch("httpx.post", return_value=_mock_response({"content": ""})) as mock_post:
            client = QwenClient()
            client.code("hello world")
            body = mock_post.call_args.kwargs["json"]
            assert body["temperature"] == 0.1

    def test_language_in_prompt(self):
        with patch("httpx.post", return_value=_mock_response({"content": ""})) as mock_post:
            client = QwenClient()
            client.code("sort a list", language="javascript")
            messages = mock_post.call_args.kwargs["json"]["messages"]
            user_content = next(m["content"] for m in messages if m["role"] == "user")
            assert "javascript" in user_content.lower()


# ---------------------------------------------------------------------------
# QwenClient.embed
# ---------------------------------------------------------------------------
class TestQwenClientEmbed:
    def test_returns_list_of_embeddings(self):
        vectors = [[0.1, 0.2], [0.3, 0.4]]
        with patch("httpx.post", return_value=_mock_response({"embeddings": vectors})):
            client = QwenClient()
            result = client.embed(["hello", "world"])
        assert result == vectors

    def test_texts_sent_in_request(self):
        with patch("httpx.post", return_value=_mock_response({"embeddings": []})) as mock_post:
            client = QwenClient()
            client.embed(["foo", "bar"])
            body = mock_post.call_args.kwargs["json"]
            assert body["texts"] == ["foo", "bar"]


# ---------------------------------------------------------------------------
# QwenClient.vision
# ---------------------------------------------------------------------------
class TestQwenClientVision:
    def test_returns_content(self, tmp_path):
        img = tmp_path / "test.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n")  # minimal PNG header bytes
        with patch("httpx.post", return_value=_mock_response({"content": "a cat"})):
            client = QwenClient()
            result = client.vision(str(img), "What is this?")
        assert result == "a cat"

    def test_image_sent_as_base64(self, tmp_path):
        img = tmp_path / "photo.png"
        img.write_bytes(b"PNGDATA")
        with patch("httpx.post", return_value=_mock_response({"content": ""})) as mock_post:
            client = QwenClient()
            client.vision(str(img), "describe")
            messages = mock_post.call_args.kwargs["json"]["messages"]
            image_part = next(
                p for m in messages for p in m["content"]
                if isinstance(p, dict) and p.get("type") == "image_url"
            )
            assert image_part["image_url"]["url"].startswith("data:image/png;base64,")


# ---------------------------------------------------------------------------
# Module-level convenience shortcuts
# ---------------------------------------------------------------------------
class TestConvenienceShortcuts:
    def test_ask_returns_string(self):
        with patch("httpx.post", return_value=_mock_response({"content": "42"})):
            result = ask("What is 6 times 7?")
        assert result == "42"

    def test_code_shortcut(self):
        snippet = "print('hello')"
        with patch("httpx.post", return_value=_mock_response({"content": snippet})):
            result = code("print hello world")
        assert result == snippet

    def test_analyze_image_shortcut(self, tmp_path):
        img = tmp_path / "pic.jpg"
        img.write_bytes(b"JPGDATA")
        with patch("httpx.post", return_value=_mock_response({"content": "a dog"})):
            result = analyze_image(str(img))
        assert result == "a dog"
