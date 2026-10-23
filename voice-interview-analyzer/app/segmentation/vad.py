"""Voice activity detection and speech segmentation (Section 5.3).

``EnergyVAD`` is a dependency-free, streaming, adaptive-threshold detector.
It is a reasonable baseline; evaluate a model-based VAD (e.g. Silero, WebRTC
VAD) for noisy environments before production (see Section 6).

A pause does not automatically end a turn: segments close only after
``min_silence_ms`` of continuous non-speech (hangover).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

import numpy as np


@dataclass
class SpeechSegment:
    segment_id: str
    start_ms: int
    end_ms: int

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


@dataclass
class EnergyVAD:
    sample_rate_hz: int = 16000
    frame_ms: int = 30
    min_speech_ms: int = 250
    min_silence_ms: int = 600
    min_pause_ms: int = 150  # shorter gaps are not counted as pauses
    # speech if frame RMS > noise_floor * ratio and > absolute_floor
    threshold_ratio: float = 3.0
    absolute_floor: float = 0.005
    noise_adapt: float = 0.05

    noise_floor: float = 0.003
    _buffer: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    _buffer_start_ms: float | None = None
    _in_speech: bool = False
    _speech_start_ms: float = 0.0
    _last_speech_ms: float = 0.0
    # (start_ms, end_ms) of every silent gap >= min_pause_ms between speech frames
    pauses: list[tuple[int, int]] = field(default_factory=list)

    @property
    def frame_len(self) -> int:
        return int(self.sample_rate_hz * self.frame_ms / 1000)

    def is_speech_frame(self, frame: np.ndarray) -> bool:
        rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2))) if frame.size else 0.0
        speech = rms > max(self.noise_floor * self.threshold_ratio, self.absolute_floor)
        if not speech:  # adapt the noise floor only on non-speech frames
            self.noise_floor = (1 - self.noise_adapt) * self.noise_floor + self.noise_adapt * rms
        return speech

    def process(self, samples: np.ndarray, start_ms: float) -> list[SpeechSegment]:
        """Feed mono float32 samples starting at ``start_ms``; return closed segments."""
        if self._buffer_start_ms is None or self._buffer.size == 0:
            self._buffer_start_ms = start_ms
        self._buffer = np.concatenate([self._buffer, samples.astype(np.float32)])

        closed: list[SpeechSegment] = []
        fl = self.frame_len
        n_frames = self._buffer.size // fl
        for i in range(n_frames):
            frame = self._buffer[i * fl:(i + 1) * fl]
            t0 = self._buffer_start_ms + i * self.frame_ms
            t1 = t0 + self.frame_ms
            if self.is_speech_frame(frame):
                gap = t0 - self._last_speech_ms
                if self._last_speech_ms and gap >= self.min_pause_ms:
                    self.pauses.append((int(self._last_speech_ms), int(t0)))
                if not self._in_speech:
                    self._in_speech = True
                    self._speech_start_ms = t0
                self._last_speech_ms = t1
            elif self._in_speech and t1 - self._last_speech_ms >= self.min_silence_ms:
                seg = self._close()
                if seg:
                    closed.append(seg)

        consumed = n_frames * fl
        self._buffer = self._buffer[consumed:]
        self._buffer_start_ms += n_frames * self.frame_ms
        return closed

    def flush(self) -> list[SpeechSegment]:
        """Close any open segment (end of track / session)."""
        if self._in_speech:
            seg = self._close()
            return [seg] if seg else []
        return []

    def _close(self) -> SpeechSegment | None:
        self._in_speech = False
        start, end = self._speech_start_ms, self._last_speech_ms
        if end - start < self.min_speech_ms:
            return None
        return SpeechSegment(f"seg-{uuid.uuid4().hex[:10]}", int(start), int(end))


def segment_audio(samples: np.ndarray, sample_rate_hz: int, **kwargs) -> list[SpeechSegment]:
    """Offline convenience wrapper for a whole buffer."""
    vad = EnergyVAD(sample_rate_hz=sample_rate_hz, **kwargs)
    return vad.process(samples, 0) + vad.flush()
