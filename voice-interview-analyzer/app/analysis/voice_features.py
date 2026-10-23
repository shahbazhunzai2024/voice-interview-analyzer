"""Candidate voice feature extraction (Section 5.6).

* RMS energy per window — a *relative* signal; never compared across
  microphones/participants unless calibrated.
* F0 (pitch) on voiced windows only — autocorrelation by default, librosa
  pYIN when installed.
* Speech rate from transcript word counts.
* Changes are expressed relative to the candidate's own running baseline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

F0_MIN_HZ = 70.0
F0_MAX_HZ = 400.0

try:  # optional, higher-quality pitch tracker
    import librosa  # type: ignore

    HAVE_LIBROSA = True
except ImportError:  # pragma: no cover
    HAVE_LIBROSA = False


def rms(samples: np.ndarray) -> float:
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))


def estimate_f0_autocorr(samples: np.ndarray, sample_rate_hz: int,
                         fmin: float = F0_MIN_HZ, fmax: float = F0_MAX_HZ,
                         voicing_threshold: float = 0.45) -> float | None:
    """Estimate F0 of a short window via normalized autocorrelation.

    Returns ``None`` for unvoiced / silent / noisy windows.
    """
    x = samples.astype(np.float64)
    if x.size < int(sample_rate_hz / fmin) * 2 or rms(x) < 1e-3:
        return None
    x = x - x.mean()
    n = x.size
    # FFT-based autocorrelation
    spec = np.fft.rfft(x, n=2 * n)
    ac = np.fft.irfft(spec * np.conj(spec))[:n]
    if ac[0] <= 0:
        return None
    ac /= ac[0]
    lag_min = max(1, int(sample_rate_hz / fmax))
    lag_max = min(n - 1, int(sample_rate_hz / fmin))
    if lag_max <= lag_min:
        return None
    seg = ac[lag_min:lag_max]
    peak = int(np.argmax(seg)) + lag_min
    if ac[peak] < voicing_threshold:
        return None
    # Parabolic interpolation for sub-sample lag precision
    if 1 <= peak < n - 1:
        a, b, c = ac[peak - 1], ac[peak], ac[peak + 1]
        denom = a - 2 * b + c
        shift = 0.5 * (a - c) / denom if denom else 0.0
    else:
        shift = 0.0
    return float(sample_rate_hz / (peak + shift))


def estimate_f0(samples: np.ndarray, sample_rate_hz: int) -> tuple[float | None, float]:
    """Return (median F0 over voiced frames or None, voiced ratio)."""
    if HAVE_LIBROSA and samples.size >= 2048:  # pragma: no cover - optional path
        f0, voiced, _ = librosa.pyin(samples.astype(np.float32), fmin=F0_MIN_HZ,
                                     fmax=F0_MAX_HZ, sr=sample_rate_hz)
        voiced_f0 = f0[voiced & ~np.isnan(f0)]
        ratio = float(np.mean(voiced)) if voiced.size else 0.0
        return (float(np.median(voiced_f0)) if voiced_f0.size else None), ratio

    frame = int(0.04 * sample_rate_hz)  # 40 ms frames, 20 ms hop
    hop = frame // 2
    values = []
    total = 0
    for start in range(0, max(0, samples.size - frame + 1), hop):
        total += 1
        f0 = estimate_f0_autocorr(samples[start:start + frame], sample_rate_hz)
        if f0 is not None:
            values.append(f0)
    if total == 0:
        return None, 0.0
    return (float(np.median(values)) if values else None), len(values) / total


def speech_rate_wpm(text: str, duration_ms: int) -> float | None:
    words = len([w for w in text.split() if any(ch.isalnum() for ch in w)])
    if duration_ms <= 0 or words == 0:
        return None
    return words / (duration_ms / 60000.0)


@dataclass
class RunningStat:
    """Welford running mean/variance."""

    n: int = 0
    mean: float = 0.0
    _m2: float = 0.0

    def add(self, x: float) -> None:
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        self._m2 += d * (x - self.mean)

    @property
    def std(self) -> float:
        return math.sqrt(self._m2 / (self.n - 1)) if self.n > 1 else 0.0

    @property
    def cv(self) -> float | None:
        return self.std / self.mean if self.n > 1 and self.mean else None


@dataclass
class CandidateBaseline:
    """Within-candidate baseline built from the first ``warmup_windows`` voiced windows.

    After warm-up the baseline is frozen so later changes are measured against
    the candidate's early-interview delivery, not against other people.
    """

    warmup_windows: int = 10
    rms_stat: RunningStat = field(default_factory=RunningStat)
    f0_stat: RunningStat = field(default_factory=RunningStat)
    rate_stat: RunningStat = field(default_factory=RunningStat)

    @property
    def ready(self) -> bool:
        return self.rms_stat.n >= self.warmup_windows

    def update(self, rms_value: float | None, f0_hz: float | None) -> None:
        if self.ready:
            return
        if rms_value is not None and rms_value > 0:
            self.rms_stat.add(rms_value)
        if f0_hz is not None:
            self.f0_stat.add(f0_hz)

    def add_rate(self, wpm: float | None) -> None:
        if wpm is not None and self.rate_stat.n < self.warmup_windows:
            self.rate_stat.add(wpm)

    def ratio(self, kind: str, value: float | None) -> float | None:
        stat = {"rms": self.rms_stat, "f0": self.f0_stat, "rate": self.rate_stat}[kind]
        if value is None or stat.n < 3 or not stat.mean:
            return None
        return value / stat.mean


@dataclass
class WindowFeatures:
    start_ms: int
    end_ms: int
    rms: float
    f0_hz: float | None
    voiced_ratio: float


def compute_window_features(samples: np.ndarray, sample_rate_hz: int,
                            start_ms: int) -> WindowFeatures:
    f0, voiced_ratio = estimate_f0(samples, sample_rate_hz)
    end_ms = start_ms + int(samples.size * 1000 / sample_rate_hz)
    return WindowFeatures(start_ms, end_ms, rms(samples), f0, voiced_ratio)
