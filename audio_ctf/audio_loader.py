"""Unified audio loading supporting WAV, MP3, FLAC, and OGG formats."""

import io
import numpy as np
import soundfile as sf


def load_audio(path: str, mono: bool = False) -> tuple[np.ndarray, int]:
    """Load an audio file and return (samples, sample_rate).

    samples is a float32 numpy array, shape (n_samples,) for mono or
    (n_samples, n_channels) for stereo/multi-channel.

    Args:
        path: Path to audio file (WAV, FLAC, OGG, or MP3).
        mono: If True, downmix to mono before returning.

    Returns:
        Tuple of (samples, sample_rate). Samples are float32 in [-1, 1].
    """
    if path.lower().endswith(".mp3"):
        data, sr = _load_mp3(path)
    else:
        data, sr = sf.read(path, dtype="float32", always_2d=False)

    if mono and data.ndim == 2:
        data = data.mean(axis=1)

    if data.dtype != np.float32:
        data = data.astype(np.float32)

    return data, sr


def _load_mp3(path: str) -> tuple[np.ndarray, int]:
    """Load MP3 via pydub → in-memory WAV → soundfile."""
    try:
        from pydub import AudioSegment
    except ImportError as e:
        raise ImportError(
            "pydub is required for MP3 support. Install it with: pip install pydub"
        ) from e

    audio = AudioSegment.from_mp3(path)
    buf = io.BytesIO()
    audio.export(buf, format="wav")
    buf.seek(0)
    data, sr = sf.read(buf, dtype="float32", always_2d=False)
    return data, sr


def to_mono(data: np.ndarray) -> np.ndarray:
    """Downmix multi-channel audio to mono."""
    if data.ndim == 1:
        return data
    return data.mean(axis=1).astype(np.float32)


def get_channel(data: np.ndarray, channel: int = 0) -> np.ndarray:
    """Extract a single channel from multi-channel audio."""
    if data.ndim == 1:
        return data
    return data[:, channel].astype(np.float32)
