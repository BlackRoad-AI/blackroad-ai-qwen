"""
Pytest configuration: mock heavy ML dependencies so tests run without GPU or large
model downloads.  These mocks are installed into sys.modules before any test
module is imported.
"""
import os
import sys
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Mock torch so module-level code in src/main.py doesn't require a real install
# ---------------------------------------------------------------------------
mock_torch = MagicMock()
mock_torch.cuda.is_available.return_value = False
mock_torch.float32 = "float32"
mock_torch.float16 = "float16"

sys.modules.setdefault("torch", mock_torch)
sys.modules.setdefault("transformers", MagicMock())
sys.modules.setdefault("accelerate", MagicMock())
sys.modules.setdefault("bitsandbytes", MagicMock())
sys.modules.setdefault("sentencepiece", MagicMock())

# ---------------------------------------------------------------------------
# Test environment: disable memory system & skip model loading
# ---------------------------------------------------------------------------
os.environ.setdefault("BLACKROAD_MEMORY_ENABLED", "false")
os.environ.setdefault("SKIP_MODEL_LOAD", "true")
