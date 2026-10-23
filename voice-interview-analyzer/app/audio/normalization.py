"""Audio format adapters (Section 4.3).

Internal analysis format: mono float32 in [-1, 1] at ``ANALYSIS_SAMPLE_RATE_HZ``
(16 kHz by default). Every conversion is explicit; arbitrary byte chunks are
never treated as audio without framing metadata.
"""

from __future__ import annotations

import numpy as np

PCM16_BYTES = 2
_PCM16_SCALE = 32768.0


class AudioFormatError(ValueError):
    """Raised when a frame does not match its declared format."""


def validate_frame(data: bytes | memoryview, channels: int, sample_rate_hz: int,
                   sample_width: int = PCM16_BYTES) -> int:
    """Validate framing metadata and return samples-per-channel."""
    if channels < 1:
        raise AudioFormatError("channels must be >= 1")
    if sample_rate_hz <= 0:
        raise AudioFormatError("sample_rate_hz must be > 0")
    n_bytes = len(data)
    frame_bytes = channels * sample_width
    if n_bytes % frame_bytes != 0:
        raise AudioFormatError(
            f"{n_bytes} bytes is not a whole number of {channels}-channel "
            f"{sample_width * 8}-bit frames"
        )
    return n_bytes // frame_bytes


def pcm16_to_float32(data: bytes | memoryview) -> np.ndarray:
    """Little-endian signed 16-bit PCM -> float32 in [-1, 1]."""
    arr = np.frombuffer(data, dtype="<i2")
    return arr.astype(np.float32) / _PCM16_SCALE


def float32_to_pcm16(samples: np.ndarray) -> bytes:
    clipped = np.clip(samples, -1.0, 1.0 - 1.0 / _PCM16_SCALE)
    return (clipped * _PCM16_SCALE).astype("<i2").tobytes()


def to_mono(samples: np.ndarray, channels: int) -> np.ndarray:
    """Down-mix interleaved samples to mono by averaging channels."""
    if channels == 1:
        return samples
    if samples.size % channels:
        raise AudioFormatError("interleaved sample count not divisible by channels")
    return samples.reshape(-1, channels).mean(axis=1).astype(np.float32)


def resample(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Resample by linear interpolation.

    Adequate for VAD / energy / F0 features. For transcription-quality
    resampling consider a polyphase filter (e.g. scipy.signal.resample_poly)
    or request the target rate directly from the media SDK.
    """
    if src_rate == dst_rate or samples.size == 0:
        return samples.astype(np.float32, copy=False)
    n_out = int(round(samples.size * dst_rate / src_rate))
    if n_out <= 0:
        return np.zeros(0, dtype=np.float32)
    x_old = np.arange(samples.size, dtype=np.float64) / src_rate
    x_new = np.arange(n_out, dtype=np.float64) / dst_rate
    return np.interp(x_new, x_old, samples).astype(np.float32)


def normalize_frame(data: bytes | memoryview, sample_rate_hz: int, channels: int,
                    target_rate_hz: int) -> np.ndarray:
    """Full adapter: PCM16 interleaved frame -> mono float32 at target rate."""
    validate_frame(data, channels, sample_rate_hz)
    samples = pcm16_to_float32(data)
    mono = to_mono(samples, channels)
    return resample(mono, sample_rate_hz, target_rate_hz)
