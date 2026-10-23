"""Speech-to-text integration (Section 5.4).

One ``TranscriptionStream`` is opened per role so transcripts stay separate.
Providers are pluggable; credentials stay server-side.

Built-in providers
------------------
* ``none``  — accepts audio, produces no text. The pipeline still runs VAD
  and voice metrics. Use the dev transcript endpoint to simulate text.
* ``openai`` — placeholder adapter for a realtime transcription API. Fill in
  ``_run`` following the provider's realtime transcription guide (see
  docs/project_documentation.pdf, Section 14). Evaluate accuracy, latency,
  cost and privacy before selecting a provider (Section 6).
"""

from __future__ import annotations

import abc
import asyncio
import logging
from dataclasses import dataclass
from typing import AsyncIterator, Callable

from app.config import Settings
from app.schemas.events import Role

log = logging.getLogger(__name__)


@dataclass
class TranscriptResult:
    role: Role
    text: str
    start_ms: int
    end_ms: int
    is_final: bool
    confidence: float | None = None
    item_id: str | None = None  # provider's id; partials of one utterance share it


class TranscriptionStream(abc.ABC):
    """Streaming transcription for a single role/track."""

    def __init__(self, role: Role, sample_rate_hz: int) -> None:
        self.role = role
        self.sample_rate_hz = sample_rate_hz
        self._queue: asyncio.Queue[TranscriptResult | None] = asyncio.Queue()

    @abc.abstractmethod
    async def start(self) -> None: ...

    @abc.abstractmethod
    async def send_audio(self, pcm16_mono: bytes, start_ms: int) -> None:
        """Send mono PCM16 at ``sample_rate_hz`` positioned at ``start_ms``."""

    async def close(self) -> None:
        await self._queue.put(None)

    async def results(self) -> AsyncIterator[TranscriptResult]:
        while True:
            item = await self._queue.get()
            if item is None:
                return
            yield item


class NullTranscriptionStream(TranscriptionStream):
    async def start(self) -> None:
        log.info("Transcription disabled for role=%s (provider=none)", self.role)

    async def send_audio(self, pcm16_mono: bytes, start_ms: int) -> None:
        return None


class OpenAIRealtimeTranscriptionStream(TranscriptionStream):  # pragma: no cover - needs network
    """Adapter skeleton for a realtime transcription WebSocket API.

    Implementation outline:
      1. Open a WSS connection with the server-side API key.
      2. Configure an English transcription session with server VAD.
      3. Forward base64 PCM16 from ``send_audio`` (resample to the provider's
         required rate in app/audio/normalization.py).
      4. Map delta/completed events to ``TranscriptResult`` and ``put`` them on
         ``self._queue``; translate provider times onto the session timeline
         using the ``start_ms`` of the audio sent.
      5. Surface rate limits / disconnects as recoverable errors.
    """

    def __init__(self, role: Role, sample_rate_hz: int, api_key: str) -> None:
        super().__init__(role, sample_rate_hz)
        self._api_key = api_key

    async def start(self) -> None:
        raise NotImplementedError(
            "OpenAI realtime transcription adapter is not implemented yet. "
            "Set TRANSCRIPTION_PROVIDER=none or implement this class."
        )

    async def send_audio(self, pcm16_mono: bytes, start_ms: int) -> None:
        raise NotImplementedError


ProviderFactory = Callable[[Role, int], TranscriptionStream]


def create_stream(settings: Settings, role: Role, sample_rate_hz: int) -> TranscriptionStream:
    provider = settings.transcription_provider.lower()
    if provider in {"", "none", "disabled"}:
        return NullTranscriptionStream(role, sample_rate_hz)
    if provider == "openai":
        if not settings.transcription_api_key or settings.transcription_api_key == "replace_me":
            raise RuntimeError("TRANSCRIPTION_API_KEY is not configured")
        return OpenAIRealtimeTranscriptionStream(role, sample_rate_hz, settings.transcription_api_key)
    raise ValueError(f"Unknown TRANSCRIPTION_PROVIDER '{settings.transcription_provider}'")
