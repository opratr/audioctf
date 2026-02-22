# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup and Running

```bash
# Install in editable mode (includes SSTV support)
pip install -e ".[sstv]"

# Run a command
audioctf scan unknown.wav
audioctf sstv signal.wav -o decoded.png
audioctf lsb stego.wav --bits 2 --output-dir /tmp

# Run a single analyzer directly (useful during development)
.venv/bin/python -c "
from audio_ctf.audio_loader import load_audio, to_mono
from audio_ctf.analyzers import sstv as _sstv
data, sr = load_audio('file.wav')
result = _sstv.analyze(to_mono(data), sr, output_path='/tmp/out.png')
print(_sstv.format_result(result))
"
```

No test suite exists yet. Verify analyzers by generating synthetic test WAVs and scanning them.

## Architecture

### Data flow

```
audio file → audio_loader.load_audio() → numpy float32 array
                                               ↓
                          analyzer.analyze(data, sr, **opts)
                                               ↓
                          {"detected": bool, "confidence": float, "data": dict}
                                               ↓
                          analyzer.format_result(result) → str
```

Every analyzer module exposes exactly two public functions: `analyze()` and `format_result()`. The `scan` CLI command calls all analyzers and aggregates results using this uniform interface.

### Analyzer contract

- `analyze()` always returns `{"detected": bool, "confidence": float, "data": dict}`
- `confidence` must be `0.0` when `detected=False` — never return high confidence with a False detection
- `detected=True` requires a strong positive signal: file magic bytes, high printable-text ratio, or correct framing — not just "non-random" data

### CLI import aliasing (important)

All analyzer imports in `cli.py` use `_` prefix aliases (`from audio_ctf.analyzers import morse as _morse`) to prevent Click command functions from shadowing module names. Always maintain this pattern when adding new commands or analyzers.

### FSK demodulation pattern

All FSK-based analyzers (modem, baudot/RTTY) share the same demodulation approach — **per-bit DFT projection** — not bandpass + envelope or instantaneous frequency:

```python
chunk = data[s:e].astype(np.float64)
tc = np.arange(len(chunk)) / sr
mark_pwr  = abs(np.dot(chunk, np.exp(-1j * 2 * np.pi * mark_hz  * tc))) ** 2
space_pwr = abs(np.dot(chunk, np.exp(-1j * 2 * np.pi * space_hz * tc))) ** 2
bit = 1 if mark_pwr > space_pwr else 0
```

**Critical**: generate test WAVs at 48000 Hz. At 44100 Hz, `44100/1200 = 36.75` samples/bit (non-integer) causes cumulative timing drift. At 48000 Hz, `48000/1200 = 40.0` exactly.

### SSTV VIS detection

Uses Hilbert instantaneous frequency (not zero-crossing rate — ZCR quantizes to 100 Hz steps and misses the 10 ms break). Downsampled to 1 ms resolution. Detects the pattern: leader (≥150 ms @ 1900 Hz) → gap (≤60 ms) → leader (≥150 ms) → skip start bit (30 ms) → read 7 VIS data bits @ middle of each 30 ms slot.

`pysstv` is encode-only (WAV generator from images). It has no decode API. Robot 36 and Martin M1 are decoded manually using Hilbert instantaneous frequency per scan segment.

### False positive behavior

SSTV signals (1200–2300 Hz) overlap with Morse, DTMF, and RTTY frequency ranges, causing false positives in those detectors during `scan`. This is expected. Confidence scores disambiguate: the correct detector scores significantly higher. Known cross-detector triggers:
- RTTY mark frequency (2125 Hz) → Morse carrier false positive at 100% if ≥3 elements decoded
- Morse requires `total_elements >= 3` to set `detected=True`
- LSB entropy threshold is `< 3.0 bits/byte` (not 5.0) — sine waves have ~4 bits/byte naturally
