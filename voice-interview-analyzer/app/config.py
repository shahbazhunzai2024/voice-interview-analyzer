"""Application settings loaded from environment variables (.env in development).

Secrets (LiveKit API secret, transcription API key) stay server-side and are
never returned by any endpoint.
"""

from __future__ import annotations

import os

try:  # python-dotenv is optional
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


class Settings:
    """Runtime configuration. Re-instantiate (or call ``reload``) in tests."""

    def __init__(self) -> None:
        self.reload()

    def reload(self) -> None:
        # LiveKit (media plane)
        self.livekit_url: str = os.getenv("LIVEKIT_URL", "")
        self.livekit_api_key: str = os.getenv("LIVEKIT_API_KEY", "")
        self.livekit_api_secret: str = os.getenv("LIVEKIT_API_SECRET", "")
        self.join_token_ttl_s: int = _int("JOIN_TOKEN_TTL_SECONDS", 600)

        # Analysis worker
        self.analysis_worker_enabled: bool = _bool("ANALYSIS_WORKER_ENABLED", True)
        self.analysis_sample_rate_hz: int = _int("ANALYSIS_SAMPLE_RATE_HZ", 16000)

        # Transcription
        self.transcription_provider: str = os.getenv("TRANSCRIPTION_PROVIDER", "none")
        self.transcription_api_key: str = os.getenv("TRANSCRIPTION_API_KEY", "")

        # Privacy / ops
        self.audio_retention_hours: int = _int("AUDIO_RETENTION_HOURS", 0)
        self.max_session_minutes: int = _int("MAX_SESSION_MINUTES", 90)
        self.log_level: str = os.getenv("LOG_LEVEL", "INFO")

        # Enables /api/sessions/{id}/dev/* helpers (simulated transcripts).
        self.dev_mode: bool = _bool("DEV_MODE", True)

    @property
    def livekit_configured(self) -> bool:
        return bool(self.livekit_url and self.livekit_api_key and self.livekit_api_secret)


settings = Settings()
