import numpy as np
import pytest
from pydantic import ValidationError

from app.audio.normalization import (
    AudioFormatError,
    float32_to_pcm16,
    normalize_frame,
    pcm16_to_float32,
    resample,
    to_mono,
    validate_frame,
)
from app.audio.timestamps import TrackTimeline
from app.schemas.events import TrackMetadata


def test_pcm16_roundtrip():
    x = np.linspace(-0.9, 0.9, 480).astype(np.float32)
    y = pcm16_to_float32(float32_to_pcm16(x))
    assert np.max(np.abs(x - y)) < 1e-4


def test_validate_frame_rejects_partial_frames():
    assert validate_frame(b"\x00" * 960, channels=1, sample_rate_hz=48000) == 480
    assert validate_frame(b"\x00" * 960, channels=2, sample_rate_hz=48000) == 240
    with pytest.raises(AudioFormatError):
        validate_frame(b"\x00" * 961, channels=1, sample_rate_hz=48000)
    with pytest.raises(AudioFormatError):
        validate_frame(b"\x00" * 6, channels=4, sample_rate_hz=48000)
    with pytest.raises(AudioFormatError):
        validate_frame(b"\x00" * 4, channels=1, sample_rate_hz=0)


def test_to_mono_averages_interleaved_channels():
    stereo = np.array([1.0, 0.0, 0.5, 0.5], dtype=np.float32)
    assert np.allclose(to_mono(stereo, 2), [0.5, 0.5])


def test_resample_length_and_frequency_preserved():
    sr_in, sr_out = 48000, 16000
    t = np.arange(sr_in) / sr_in
    tone = np.sin(2 * np.pi * 220 * t).astype(np.float32)
    out = resample(tone, sr_in, sr_out)
    assert out.size == sr_out
    spectrum = np.abs(np.fft.rfft(out))
    assert abs(np.argmax(spectrum) - 220) <= 1  # 1 Hz bins for a 1 s signal


def test_normalize_frame_full_adapter():
    frame = float32_to_pcm16(np.zeros(960, dtype=np.float32))  # 20 ms stereo @48k
    out = normalize_frame(frame, 48000, 2, 16000)
    assert out.dtype == np.float32 and out.size == 160


def test_track_metadata_schema():
    meta = TrackMetadata(session_id="s1", role="candidate", track_id="TR_1",
                         server_received_ms=168000, sample_rate_hz=48000, channels=1)
    assert meta.encoding == "pcm_s16le"
    with pytest.raises(ValidationError):
        TrackMetadata(session_id="s1", role="observer", track_id="x",
                      server_received_ms=0, sample_rate_hz=48000, channels=1)
    with pytest.raises(ValidationError):
        TrackMetadata(session_id="s1", role="candidate", track_id="x",
                      server_received_ms=0, sample_rate_hz=0, channels=1)


def test_timeline_is_contiguous_then_rebases_after_gap():
    tl = TrackTimeline(sample_rate_hz=16000, gap_threshold_ms=250)
    a = tl.place(320, server_received_ms=1020)   # 20 ms frame
    b = tl.place(320, server_received_ms=1045)   # small jitter -> contiguous
    assert a == (1000, 1020)
    assert b == (1020, 1040)
    c = tl.place(320, server_received_ms=3020)   # 2 s gap (mute / reconnect)
    assert c == (3000, 3020)
    assert len(tl.gaps) == 1 and tl.gaps[0].start_ms == 1040


def test_timeline_reset_reanchors():
    tl = TrackTimeline(sample_rate_hz=16000)
    tl.place(320, 500)
    tl.reset()
    assert tl.position_ms is None
    assert tl.place(320, 900) == (880, 900)


def test_timeline_rejects_invalid_configuration_and_frames():
    with pytest.raises(ValueError, match="sample_rate_hz"):
        TrackTimeline(sample_rate_hz=0)
    with pytest.raises(ValueError, match="gap_threshold_ms"):
        TrackTimeline(sample_rate_hz=16000, gap_threshold_ms=-1)

    timeline = TrackTimeline(sample_rate_hz=16000)
    with pytest.raises(ValueError, match="n_samples"):
        timeline.place(0, server_received_ms=100)
    with pytest.raises(ValueError, match="n_samples"):
        timeline.place(-1, server_received_ms=100)
    with pytest.raises(ValueError, match="server_received_ms"):
        timeline.place(320, server_received_ms=-1)
