import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_settings(monkeypatch):
    """Default test config: no LiveKit, no STT, dev helpers on."""
    for key in ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET", "TRANSCRIPTION_PROVIDER"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("DEV_MODE", "true")
    settings.reload()
    yield
    settings.reload()


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    import main

    with TestClient(main.app) as c:
        yield c
