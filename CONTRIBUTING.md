# Contributing to audio-ctf

Thank you for your interest in contributing! This document covers how to set up a development environment, the conventions used in the codebase, and what to keep in mind when submitting changes.

## Getting started

```bash
git clone <repo-url>
cd audio-ctf-tool
pip install -e ".[sstv]"
```

Requires Python 3.9+ and `ffmpeg` on PATH for MP3 support.

## Project structure

```
audio_ctf/
  cli.py              # Click CLI entry point
  audio_loader.py     # Audio loading and normalization
  analyzers/
    morse.py          # CW Morse code detector
    dtmf.py           # DTMF tone detector
    modem.py          # FSK modem decoder (Bell 103/202, AFSK)
    lsb.py            # LSB steganography extractor
    sstv.py           # SSTV image decoder
    rtty.py           # Baudot/RTTY decoder
    metadata.py       # Audio metadata reporter
```

## Analyzer contract

Every analyzer module must expose exactly two public functions:

```python
def analyze(data: np.ndarray, sr: int, **kwargs) -> dict:
    # Returns:
    # {
    #   "detected": bool,
    #   "confidence": float,   # 0.0 when detected=False
    #   "data": dict,          # protocol-specific details
    # }

def format_result(result: dict) -> str:
    # Returns a human-readable string summary.
```

Key rules:
- `confidence` must be `0.0` when `detected=False`.
- `detected=True` requires a strong signal: file magic bytes, high printable-text ratio, or correct protocol framing — not just "non-random" data.
- `data` is always a plain dict (JSON-serializable).

## Adding a new analyzer

1. Create `audio_ctf/analyzers/<name>.py` implementing `analyze()` and `format_result()`.
2. Register it in `audio_ctf/analyzers/__init__.py`.
3. Add a CLI command in `audio_ctf/cli.py` using the `_<name>` import alias convention (see below).
4. Document the new command in `README.md`.

### CLI import alias convention

All analyzer imports in `cli.py` use a `_` prefix to prevent Click command functions from shadowing module names:

```python
from audio_ctf.analyzers import morse as _morse
```

Always follow this pattern when adding new commands.

### FSK demodulation

Use the per-bit DFT projection pattern (not bandpass + envelope or instantaneous frequency):

```python
chunk = data[s:e].astype(np.float64)
tc = np.arange(len(chunk)) / sr
mark_pwr  = abs(np.dot(chunk, np.exp(-1j * 2 * np.pi * mark_hz  * tc))) ** 2
space_pwr = abs(np.dot(chunk, np.exp(-1j * 2 * np.pi * space_hz * tc))) ** 2
bit = 1 if mark_pwr > space_pwr else 0
```

Generate test WAVs at **48000 Hz** — at 44100 Hz, 1200 baud gives 36.75 samples/bit (non-integer), causing cumulative timing drift.

## Testing

There is no automated test suite yet. Verify analyzers by generating synthetic test WAVs and scanning them:

```bash
# Example: generate a test WAV and scan it
python -c "
import numpy as np, scipy.io.wavfile as wv
sr = 48000
# ... generate signal ...
wv.write('/tmp/test.wav', sr, data.astype(np.float32))
"
audioctf scan /tmp/test.wav
```

Contributions that add a `tests/` directory with pytest-based tests are very welcome.

## Submitting changes

1. Fork the repository and create a branch from `main`.
2. Make your changes following the conventions above.
3. Verify your analyzer works against a synthetic test WAV.
4. Open a pull request with a clear description of what the change does and why.

For bug reports or feature requests, open a GitHub issue.

## Code style

- Follow PEP 8. No formatter is enforced yet, but keep lines under 100 characters.
- Avoid unnecessary abstractions — three similar lines of code is better than a premature helper.
- Don't add docstrings or type annotations to code you didn't write.
