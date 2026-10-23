"""Timing and synchronization (Section 4.2).

Participants are on different computers, so their clocks are never assumed to
match. Each track is placed on a server-side *session timeline*:

* ``SessionClock`` defines t=0 as the moment the session was created on the
  server (monotonic clock).
* ``TrackTimeline`` advances by the number of samples received, anchored to
  the server receive time of the first frame. When the receive time drifts
  ahead of the sample-count position by more than ``gap_threshold_ms`` (mute,
  reconnect, dropped media) the timeline is re-anchored and the gap recorded.

This gives frame-accurate timing *within* a track and receive-time alignment
*across* tracks. It does not claim sample-accurate cross-track sync.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


class SessionClock:
    def __init__(self) -> None:
        self._t0_ns = time.monotonic_ns()
        self.started_at_ms = int(time.time() * 1000)

    def now_ms(self) -> int:
        return (time.monotonic_ns() - self._t0_ns) // 1_000_000


@dataclass
class TimingGap:
    start_ms: int
    end_ms: int


@dataclass
class TrackTimeline:
    sample_rate_hz: int
    gap_threshold_ms: int = 250
    _anchor_ms: int | None = None
    _samples_since_anchor: int = 0
    gaps: list[TimingGap] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be > 0")
        if self.gap_threshold_ms < 0:
            raise ValueError("gap_threshold_ms must be >= 0")

    @property
    def position_ms(self) -> int | None:
        if self._anchor_ms is None:
            return None
        return self._anchor_ms + int(self._samples_since_anchor * 1000 / self.sample_rate_hz)

    def place(self, n_samples: int, server_received_ms: int) -> tuple[int, int]:
        """Return (start_ms, end_ms) on the session timeline for a frame.

        ``server_received_ms`` is the session-clock time the frame arrived.
        """
        if n_samples <= 0:
            raise ValueError("n_samples must be > 0")
        if server_received_ms < 0:
            raise ValueError("server_received_ms must be >= 0")
        duration_ms = n_samples * 1000 / self.sample_rate_hz
        expected_start = self.position_ms
        arrival_start = int(server_received_ms - duration_ms)

        if expected_start is None:
            self._anchor_ms = max(0, arrival_start)
            self._samples_since_anchor = 0
        elif arrival_start - expected_start > self.gap_threshold_ms:
            self.gaps.append(TimingGap(expected_start, arrival_start))
            self._anchor_ms = arrival_start
            self._samples_since_anchor = 0

        start = self.position_ms
        assert start is not None
        self._samples_since_anchor += n_samples
        end = self.position_ms
        assert end is not None
        return start, end

    def reset(self) -> None:
        """Call after an unpublish/republish so the next frame re-anchors."""
        self._anchor_ms = None
        self._samples_since_anchor = 0
