"""Baudot/RTTY decoder: 5-bit codes with LTRS/FIGS shift."""

from typing import Any

import numpy as np
from scipy import signal as sp_signal

# RTTY baud rates and frequency pairs: (name, baud, mark_hz, space_hz)
RTTY_CONFIGS = [
    ("RTTY 45bd standard",  45.45, 2125, 2295),
    ("RTTY 50bd standard",  50.0,  2125, 2295),
    ("RTTY 45bd UHF",       45.45, 1275, 1445),
    ("RTTY 50bd UHF",       50.0,  1275, 1445),
    ("RTTY 75bd",           75.0,  2125, 2295),
    ("RTTY 110bd",         110.0,  2125, 2295),
]

# Baudot ITA2 character table (LTRS and FIGS)
BAUDOT_LTRS = {
    0b00000: "\x00",  # NUL
    0b00001: "E",
    0b00010: "\n",
    0b00011: "A",
    0b00100: " ",
    0b00101: "S",
    0b00110: "I",
    0b00111: "U",
    0b01000: "\r",
    0b01001: "D",
    0b01010: "R",
    0b01011: "J",
    0b01100: "N",
    0b01101: "F",
    0b01110: "C",
    0b01111: "K",
    0b10000: "T",
    0b10001: "Z",
    0b10010: "L",
    0b10011: "W",
    0b10100: "H",
    0b10101: "Y",
    0b10110: "P",
    0b10111: "Q",
    0b11000: "O",
    0b11001: "B",
    0b11010: "G",
    0b11011: "\x1b",  # FIGS shift
    0b11100: "M",
    0b11101: "X",
    0b11110: "V",
    0b11111: "\x1a",  # LTRS shift
}

BAUDOT_FIGS = {
    0b00000: "\x00",
    0b00001: "3",
    0b00010: "\n",
    0b00011: "-",
    0b00100: " ",
    0b00101: "'",
    0b00110: "8",
    0b00111: "7",
    0b01000: "\r",
    0b01001: "\x05",  # WRU (Who Are You)
    0b01010: "4",
    0b01011: "\x07",  # Bell
    0b01100: ",",
    0b01101: "!",
    0b01110: ":",
    0b01111: "(",
    0b10000: "5",
    0b10001: '"',
    0b10010: ")",
    0b10011: "2",
    0b10100: "#",
    0b10101: "6",
    0b10110: "0",
    0b10111: "1",
    0b11000: "9",
    0b11001: "?",
    0b11010: "&",
    0b11011: "\x1b",  # FIGS
    0b11100: ".",
    0b11101: "/",
    0b11110: ";",
    0b11111: "\x1a",  # LTRS
}

FIGS_SHIFT = 0b11011
LTRS_SHIFT = 0b11111


def analyze(data: np.ndarray, sr: int) -> dict[str, Any]:
    """Detect and decode Baudot/RTTY from audio.

    Args:
        data: Mono float32 audio array.
        sr: Sample rate in Hz.

    Returns:
        Result dict with detected, confidence, data (decoded text).
    """
    # Auto-detect RTTY config via FFT
    config = _detect_rtty_config(data, sr)
    if config is None:
        return {
            "detected": False,
            "confidence": 0.0,
            "data": {"error": "No RTTY signal detected"},
        }

    name, baud, mark_hz, space_hz = config

    # Demodulate FSK
    bits = _fsk_demodulate(data, sr, mark_hz, space_hz, baud)
    if not bits:
        return {
            "detected": False,
            "confidence": 0.0,
            "data": {"error": f"Config {name} detected but no bits decoded"},
        }

    # Decode Baudot 5-bit codes with start/stop framing
    decoded, n_chars, n_total = _decode_baudot(bits)

    detected = bool(decoded)
    if detected:
        confidence = (n_chars / n_total) if n_total > 0 else 0.0
        # Boost if we got printable text
        if sum(1 for c in decoded if c.isprintable()) > len(decoded) * 0.7:
            confidence = max(confidence, 0.7)
    else:
        confidence = 0.0

    return {
        "detected": detected,
        "confidence": confidence,
        "data": {
            "config": name,
            "baud_rate": baud,
            "mark_hz": mark_hz,
            "space_hz": space_hz,
            "bits_decoded": len(bits),
            "chars_decoded": n_chars,
            "decoded_text": decoded,
        },
    }


def _detect_rtty_config(
    data: np.ndarray, sr: int
) -> tuple[str, float, int, int] | None:
    """Auto-detect RTTY config.

    Step 1 – FFT: find which frequency *pair* (mark/space) has the most
    energy. This identifies the frequency set but not the baud rate when
    multiple configs share the same frequencies.

    Step 2 – Baud disambiguation: for configs that share a frequency pair,
    try each candidate baud rate and score by the fraction of valid Baudot
    start/stop framing found in the decoded bit stream.
    """
    n = len(data)
    fft = np.abs(np.fft.rfft(data * np.hanning(n)))
    freqs_arr = np.fft.rfftfreq(n, 1.0 / sr)

    def freq_power(hz: float, bw: float = 60) -> float:
        mask = np.abs(freqs_arr - hz) < bw
        return float(fft[mask].sum()) if np.any(mask) else 0.0

    # Score each unique frequency pair
    pair_scores: dict[tuple[int, int], float] = {}
    for name, baud, mark, space in RTTY_CONFIGS:
        pair = (mark, space)
        score = min(freq_power(mark), freq_power(space))
        if score > pair_scores.get(pair, 0.0):
            pair_scores[pair] = score

    if not pair_scores or max(pair_scores.values()) < 1e-5:
        return None

    best_pair = max(pair_scores, key=pair_scores.__getitem__)
    mark, space = best_pair

    # Among all configs with this frequency pair, pick the baud rate that
    # yields the most valid Baudot frames (start=0, 5 data bits, stop=1).
    candidates = [(name, baud) for name, baud, m, s in RTTY_CONFIGS
                  if (m, s) == best_pair]

    best_name, best_baud, best_frame_score = candidates[0][0], candidates[0][1], -1

    for name, baud in candidates:
        bits = _fsk_demodulate(data, sr, mark, space, baud)
        score = _frame_score(bits)
        if score > best_frame_score:
            best_frame_score = score
            best_name, best_baud = name, baud

    return best_name, best_baud, mark, space


def _frame_score(bits: list[int]) -> float:
    """Count valid-looking Baudot frames (start=0, stop=1) as a fraction."""
    valid = 0
    total = 0
    i = 0
    while i < len(bits) - 6:
        if bits[i] == 0:          # candidate start bit
            stop = bits[i + 6] if i + 6 < len(bits) else -1
            total += 1
            if stop == 1:
                valid += 1
            i += 7
        else:
            i += 1
    return valid / total if total else 0.0


def _fsk_demodulate(
    data: np.ndarray, sr: int, mark_hz: int, space_hz: int, baud: float
) -> list[int]:
    """Demodulate RTTY FSK using a per-bit DFT projection.

    Projects each bit window onto complex exponentials at mark and space
    frequencies and compares powers.  Delay-free, phase-agnostic, and
    immune to the group-delay drift of envelope-based demodulators.
    """
    spb_float = sr / baud
    # Use exact integer steps when the sample rate divides evenly
    spb_exact = round(spb_float)
    spb_use = float(spb_exact) if abs(spb_exact - spb_float) < 0.01 else spb_float

    bits = []
    pos = 0.0
    n_sig = len(data)

    while pos + spb_use <= n_sig:
        s = int(pos)
        e = int(pos + spb_use)
        if e <= s:
            e = s + 1
        chunk = data[s:e].astype(np.float64)
        tc = np.arange(len(chunk)) / sr
        mark_pwr  = float(abs(np.dot(chunk, np.exp(-1j * 2 * np.pi * mark_hz  * tc))) ** 2)
        space_pwr = float(abs(np.dot(chunk, np.exp(-1j * 2 * np.pi * space_hz * tc))) ** 2)
        bits.append(1 if mark_pwr > space_pwr else 0)
        pos += spb_use

    return bits


def _decode_baudot(bits: list[int]) -> tuple[str, int, int]:
    """Decode Baudot ITA2 with 1 start bit, 5 data bits, 1.5 stop bits.

    Returns (decoded_text, chars_decoded, total_attempts).
    """
    result = []
    in_figs = False
    n = len(bits)
    i = 0
    n_chars = 0
    n_total = 0

    while i < n - 6:
        # Look for start bit (0 = space = mark is 1)
        # RTTY: mark=1, space=0. Start bit = space (0 in our bit encoding)
        if bits[i] != 0:
            i += 1
            continue

        # Read 5 data bits (LSB first in RTTY)
        if i + 6 > n:
            break

        data_bits = bits[i + 1:i + 6]
        stop_bit = bits[i + 6]  # should be 1 (mark)

        code = 0
        for j, b in enumerate(data_bits):
            code |= b << j

        n_total += 1

        if stop_bit == 1:
            # Valid framing
            n_chars += 1
            if code == FIGS_SHIFT:
                in_figs = True
                i += 7
                continue
            elif code == LTRS_SHIFT:
                in_figs = False
                i += 7
                continue
            else:
                table = BAUDOT_FIGS if in_figs else BAUDOT_LTRS
                ch = table.get(code, "?")
                if ch and ch not in ("\x00", "\x1a", "\x1b"):
                    result.append(ch)
            i += 7
        else:
            i += 1

    return "".join(result), n_chars, n_total


def format_result(result: dict[str, Any]) -> str:
    lines = []
    d = result["data"]
    if not result["detected"]:
        return f"No Baudot/RTTY signal detected. {d.get('error', '')}"

    lines.append(f"Configuration: {d['config']}")
    lines.append(f"Baud rate:     {d['baud_rate']}")
    lines.append(f"Mark freq:     {d['mark_hz']} Hz")
    lines.append(f"Space freq:    {d['space_hz']} Hz")
    lines.append(f"Bits decoded:  {d['bits_decoded']}")
    lines.append(f"Chars decoded: {d['chars_decoded']}")
    lines.append(f"Confidence:    {result['confidence']:.1%}")
    lines.append("")
    lines.append("Decoded text:")
    for line in d["decoded_text"].splitlines()[:50]:
        lines.append(f"  {line}")
    if len(d["decoded_text"].splitlines()) > 50:
        lines.append(f"  ... ({len(d['decoded_text'].splitlines())} total lines)")
    return "\n".join(lines)
