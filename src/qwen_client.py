#!/usr/bin/env python3
"""
BlackRoad Qwen — Multi-modal model integration
Supports Qwen2.5 (text), Qwen-VL (vision), Qwen-Audio.
"""
import os, base64, httpx
from pathlib import Path

GATEWAY_URL = os.environ.get("BLACKROAD_GATEWAY_URL", "http://127.0.0.1:8787")

class QwenClient:
    def __init__(self, model: str = "qwen2.5:7b"):
        self.model = model
        self.base = GATEWAY_URL

    def chat(self, message: str, system: str = "", temperature: float = 0.7) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": message})
        resp = httpx.post(f"{self.base}/chat", json={
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data.get("content", "")

    def vision(self, image_path: str, prompt: str) -> str:
        """Analyze image with Qwen-VL (requires qwen2-vl model)."""
        img_b64 = base64.b64encode(Path(image_path).read_bytes()).decode()
        ext = Path(image_path).suffix.lstrip(".")
        messages = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/{ext};base64,{img_b64}"}},
            {"type": "text", "text": prompt},
        ]}]
        resp = httpx.post(f"{self.base}/chat", json={
            "model": "qwen2-vl:7b",
            "messages": messages,
        }, timeout=120)
        resp.raise_for_status()
        return resp.json().get("content", "")

    def code(self, task: str, language: str = "python") -> str:
        """Generate code using Qwen2.5-Coder."""
        return self.chat(
            f"Write {language} code for: {task}",
            system="You are an expert programmer. Output only clean, well-commented code.",
            temperature=0.1,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Get text embeddings for semantic search."""
        resp = httpx.post(f"{self.base}/embed", json={
            "model": "nomic-embed-text",
            "texts": texts,
        }, timeout=60)
        resp.raise_for_status()
        return resp.json()["embeddings"]


# Convenience shortcuts
_client = QwenClient()

def ask(question: str) -> str:
    return _client.chat(question)

def code(task: str, language: str = "python") -> str:
    return _client.code(task, language)

def analyze_image(path: str, question: str = "Describe this image") -> str:
    return _client.vision(path, question)


if __name__ == "__main__":
    print(ask("What is 2+2? Just say the number."))
    print(code("fibonacci sequence"))
