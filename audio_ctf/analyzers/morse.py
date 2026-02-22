"""Morse code detector and decoder using envelope detection."""

from typing import Any

import numpy as np
from scipy import signal


# ITU international Morse code table
MORSE_TABLE = {
    ".-": "A", "-...": "B", "-.-.": "C", "-..": "D", ".": "E",
    "..-.": "F", "--.": "G", "....": "H", "..": "I", ".---": "J",
    "-.-": "K", ".-..": "L", "--": "M", "-.": "N", "---": "O",
    ".--.": "P", "--.-": "Q", ".-.": "R", "...": "S", "-": "T",
    "..-": "U", "...-": "V", ".--": "W", "-..-": "X", "-.--": "Y",
    "--..": "Z",
    "-----": "0", ".----": "1", "..---": "2", "...--": "3", "....-": "4",
    ".....": "5", "-....": "6", "--...": "7", "---..": "8", "----.": "9",
    ".-.-.-": ".", "--..--": ",", "..--..": "?", ".----.": "'",
    "-.-.--": "!", "-..-.": "/", "-.--.": "(", "-.--.-": ")",
    ".-...": "&", "---...": ":", "-.-.-.": ";", "-...-": "=",
    ".-.-.": "+", "-....-": "-", "..--.-": "_", ".-..-.": '"',
    "...-..-": "$", ".--.-.": "@", "...---...": "SOS",
}


def analyze(data: np.ndarray, sr: int, carrier_hz: float | None = None) -> dict[str, Any]:
    """Detect and decode Morse code from audio.

    Args:
        data: Mono float32 audio array.
        sr: Sample rate in Hz.
        carrier_hz: If known, the CW carrier frequency. Auto-detected if None.

    Returns:
        Result dict with detected, confidence, data (decoded text, timing info).
    """
    # Auto-detect carrier frequency via FFT if not provided
    if carrier_hz is None:
        carrier_hz = _detect_carrier(data, sr)
        if carrier_hz is None:
            return {"detected": False, "confidence": 0.0, "data": {"error": "No carrier found"}}

    # Bandpass filter around carrier
    filtered = _bandpass(data, sr, carrier_hz, bandwidth=200)

    # Compute envelope via Hilbert transform
    analytic = signal.hilbert(filtered)
    envelope = np.abs(analytic)

    # Smooth envelope
    kernel_size = max(1, int(sr * 0.005))  # 5ms smoothing
    kernel = np.ones(kernel_size) / kernel_size
    envelope = np.convolve(envelope, kernel, mode="same")

    # Threshold to binary
    threshold = (envelope.max() - envelope.min()) * 0.3 + envelope.min()
    binary = (envelope > threshold).astype(np.int8)

    # Find runs of 0s and 1s
    runs = _find_runs(binary)
    if not runs:
        return {"detected": False, "confidence": 0.0, "data": {"error": "No signal runs found"}}

    # Calculate timing
    sample_dur = 1.0 / sr
    run_durations = [(val, count * sample_dur) for val, count in runs]

    # Decode using timing analysis
    decoded, confidence, timing_info = _decode_runs(run_durations)

    n_elements = timing_info.get("total_elements", 0)
    return {
        "detected": confidence > 0.3 and n_elements >= 3,
        "confidence": confidence if n_elements >= 3 else 0.0,
        "data": {
            "carrier_hz": round(carrier_hz, 1),
            "decoded_text": decoded,
            "timing": timing_info,
        },
    }


def _detect_carrier(data: np.ndarray, sr: int, min_hz: float = 200, max_hz: float = 4000) -> float | None:
    """Find the dominant carrier frequency via FFT."""
    n = len(data)
    fft = np.abs(np.fft.rfft(data * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, 1.0 / sr)

    # Restrict to audible CW range
    mask = (freqs >= min_hz) & (freqs <= max_hz)
    if not np.any(mask):
        return None

    peak_idx = np.argmax(fft * mask)
    if fft[peak_idx] < 1e-6:
        return None
    return float(freqs[peak_idx])


def _bandpass(data: np.ndarray, sr: int, center: float, bandwidth: float) -> np.ndarray:
    """Apply a bandpass filter centered at `center` Hz with given bandwidth."""
    low = max(1, center - bandwidth / 2)
    high = min(sr / 2 - 1, center + bandwidth / 2)
    nyq = sr / 2.0
    sos = signal.butter(6, [low / nyq, high / nyq], btype="band", output="sos")
    return signal.sosfilt(sos, data)


def _find_runs(binary: np.ndarray) -> list[tuple[int, int]]:
    """Return list of (value, run_length) pairs from a binary array."""
    if len(binary) == 0:
        return []
    runs = []
    current = binary[0]
    count = 1
    for val in binary[1:]:
        if val == current:
            count += 1
        else:
            runs.append((int(current), count))
            current = val
            count = 1
    runs.append((int(current), count))
    return runs


def _decode_runs(runs: list[tuple[int, float]]) -> tuple[str, float, dict]:
    """Decode Morse timing into text.

    Args:
        runs: List of (signal_state, duration_seconds). signal_state=1 is tone-on.

    Returns:
        (decoded_text, confidence, timing_info)
    """
    # Separate on-durations (dits/dahs) and off-durations (gaps)
    on_durs = [d for v, d in runs if v == 1 and d > 0.005]
    off_durs = [d for v, d in runs if v == 0 and d > 0.005]

    if not on_durs:
        return "", 0.0, {}

    # Estimate dit duration as the most common (median of short) on-times
    on_sorted = sorted(on_durs)
    # Use lower third as dit candidates
    n_dits = max(1, len(on_sorted) // 2)
    dit_dur = float(np.median(on_sorted[:n_dits]))
    if dit_dur < 0.01:
        dit_dur = 0.05  # minimum 50ms

    # Classify: dit < 1.5*dit, dah > 1.5*dit
    # Gaps: inter-element < 1.5*dit, inter-char 2-5*dit, word > 5*dit
    def classify_on(d: float) -> str:
        if d < 1.5 * dit_dur:
            return "."
        return "-"

    def classify_off(d: float) -> str:
        if d < 1.5 * dit_dur:
            return "element"  # within character
        if d < 5 * dit_dur:
            return "char"     # between characters
        return "word"         # between words

    # Build element sequence
    morse_words: list[list[str]] = [[]]
    current_chars: list[str] = []
    current_elements: list[str] = []

    for val, dur in runs:
        if dur < 0.005:
            continue
        if val == 1:
            current_elements.append(classify_on(dur))
        else:
            gap_type = classify_off(dur)
            if gap_type == "element":
                pass  # continue current character
            elif gap_type == "char":
                if current_elements:
                    current_chars.append("".join(current_elements))
                    current_elements = []
            else:  # word gap
                if current_elements:
                    current_chars.append("".join(current_elements))
                    current_elements = []
                if current_chars:
                    morse_words[-1].extend(current_chars)
                    current_chars = []
                morse_words.append([])

    # Flush remaining
    if current_elements:
        current_chars.append("".join(current_elements))
    if current_chars:
        morse_words[-1].extend(current_chars)

    # Decode each character
    decoded_words = []
    total_chars = 0
    decoded_chars = 0
    for word_chars in morse_words:
        word_str = ""
        for code in word_chars:
            total_chars += 1
            if code in MORSE_TABLE:
                word_str += MORSE_TABLE[code]
                decoded_chars += 1
            else:
                word_str += f"[{code}]"
        if word_str:
            decoded_words.append(word_str)

    decoded = " ".join(decoded_words)
    confidence = (decoded_chars / total_chars) if total_chars > 0 else 0.0

    timing_info = {
        "dit_duration_ms": round(dit_dur * 1000, 1),
        "wpm_estimate": round(1.2 / dit_dur, 1) if dit_dur > 0 else 0,
        "total_elements": total_chars,
        "decoded_chars": decoded_chars,
    }

    return decoded, float(confidence), timing_info


def format_result(result: dict[str, Any]) -> str:
    lines = []
    d = result["data"]
    if not result["detected"]:
        return f"No Morse code detected. {d.get('error', '')}"

    lines.append(f"Carrier frequency: {d['carrier_hz']} Hz")
    t = d["timing"]
    lines.append(f"Dit duration:      {t['dit_duration_ms']} ms (~{t['wpm_estimate']} WPM)")
    lines.append(f"Elements decoded:  {t['decoded_chars']}/{t['total_elements']}")
    lines.append(f"Confidence:        {result['confidence']:.1%}")
    lines.append("")
    lines.append("Decoded text:")
    lines.append(f"  {d['decoded_text']}")
    return "\n".join(lines)
