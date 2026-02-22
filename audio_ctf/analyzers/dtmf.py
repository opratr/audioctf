"""DTMF tone detection using the Goertzel algorithm."""

from typing import Any

import numpy as np

# DTMF frequency pairs
ROW_FREQS = [697, 770, 852, 941]
COL_FREQS = [1209, 1336, 1477, 1633]

DTMF_MAP = {
    (697, 1209): "1", (697, 1336): "2", (697, 1477): "3", (697, 1633): "A",
    (770, 1209): "4", (770, 1336): "5", (770, 1477): "6", (770, 1633): "B",
    (852, 1209): "7", (852, 1336): "8", (852, 1477): "9", (852, 1633): "C",
    (941, 1209): "*", (941, 1336): "0", (941, 1477): "#", (941, 1633): "D",
}

# Minimum tone duration for valid DTMF (40ms)
MIN_TONE_DURATION = 0.040


def analyze(data: np.ndarray, sr: int) -> dict[str, Any]:
    """Detect and decode DTMF tones from audio.

    Args:
        data: Mono float32 audio array.
        sr: Sample rate in Hz.

    Returns:
        Result dict with detected, confidence, data (digits with timestamps).
    """
    window_size = int(sr * 0.04)  # 40ms windows
    hop_size = window_size // 2    # 50% overlap

    detections = []
    n = len(data)

    for start in range(0, n - window_size, hop_size):
        chunk = data[start:start + window_size]
        row_f, col_f = _detect_dtmf_window(chunk, sr)
        if row_f and col_f:
            digit = DTMF_MAP.get((row_f, col_f))
            if digit:
                t = start / sr
                detections.append((t, digit, row_f, col_f))

    # Merge consecutive detections of the same digit
    merged = _merge_detections(detections, hop_size / sr)

    digits = [d["digit"] for d in merged]
    sequence = "".join(digits)

    confidence = min(1.0, len(merged) / 3) if merged else 0.0

    return {
        "detected": bool(merged),
        "confidence": confidence,
        "data": {
            "sequence": sequence,
            "detections": merged,
        },
    }


def _goertzel(samples: np.ndarray, target_freq: float, sr: int) -> float:
    """Compute the power at a specific frequency using Goertzel algorithm."""
    n = len(samples)
    k = int(0.5 + n * target_freq / sr)
    omega = 2.0 * np.pi * k / n
    coeff = 2.0 * np.cos(omega)
    s_prev = 0.0
    s_prev2 = 0.0
    for sample in samples:
        s = sample + coeff * s_prev - s_prev2
        s_prev2 = s_prev
        s_prev = s
    power = s_prev2**2 + s_prev**2 - coeff * s_prev * s_prev2
    return power


def _detect_dtmf_window(chunk: np.ndarray, sr: int) -> tuple[int | None, int | None]:
    """Detect a single DTMF tone pair in a short audio window."""
    row_powers = {f: _goertzel(chunk, f, sr) for f in ROW_FREQS}
    col_powers = {f: _goertzel(chunk, f, sr) for f in COL_FREQS}

    best_row = max(row_powers, key=row_powers.get)
    best_col = max(col_powers, key=col_powers.get)

    row_sorted = sorted(row_powers.values(), reverse=True)
    col_sorted = sorted(col_powers.values(), reverse=True)

    # SNR check: best must be significantly stronger than second-best
    if len(row_sorted) > 1 and row_sorted[1] > 0:
        row_ratio = row_sorted[0] / row_sorted[1]
    else:
        row_ratio = float("inf")

    if len(col_sorted) > 1 and col_sorted[1] > 0:
        col_ratio = col_sorted[0] / col_sorted[1]
    else:
        col_ratio = float("inf")

    min_ratio = 2.0
    min_power = 1e-6

    if (
        row_ratio >= min_ratio
        and col_ratio >= min_ratio
        and row_powers[best_row] > min_power
        and col_powers[best_col] > min_power
    ):
        return best_row, best_col

    return None, None


def _merge_detections(
    detections: list[tuple[float, str, int, int]], hop_sec: float
) -> list[dict]:
    """Merge consecutive detections of the same digit into tone events."""
    if not detections:
        return []

    merged = []
    current_start = detections[0][0]
    current_digit = detections[0][1]
    current_row = detections[0][2]
    current_col = detections[0][3]
    prev_t = detections[0][0]

    for t, digit, row, col in detections[1:]:
        gap = t - prev_t
        if digit == current_digit and gap <= hop_sec * 2.5:
            prev_t = t
            continue
        else:
            duration = prev_t - current_start + hop_sec
            if duration >= MIN_TONE_DURATION:
                merged.append({
                    "digit": current_digit,
                    "start_sec": round(current_start, 3),
                    "duration_sec": round(duration, 3),
                    "row_hz": current_row,
                    "col_hz": current_col,
                })
            current_start = t
            current_digit = digit
            current_row = row
            current_col = col
            prev_t = t

    # Flush last
    duration = prev_t - current_start + hop_sec
    if duration >= MIN_TONE_DURATION:
        merged.append({
            "digit": current_digit,
            "start_sec": round(current_start, 3),
            "duration_sec": round(duration, 3),
            "row_hz": current_row,
            "col_hz": current_col,
        })

    return merged


def format_result(result: dict[str, Any]) -> str:
    lines = []
    d = result["data"]
    if not result["detected"]:
        return "No DTMF tones detected."

    lines.append(f"Sequence: {d['sequence']}")
    lines.append(f"Digits:   {len(d['detections'])}")
    lines.append("")
    lines.append(f"  {'Digit':>5}  {'Start (s)':>10}  {'Duration (ms)':>14}  {'Row Hz':>7}  {'Col Hz':>7}")
    lines.append(f"  {'-'*5}  {'-'*10}  {'-'*14}  {'-'*7}  {'-'*7}")
    for det in d["detections"]:
        lines.append(
            f"  {det['digit']:>5}  {det['start_sec']:>10.3f}  "
            f"{det['duration_sec']*1000:>14.1f}  {det['row_hz']:>7}  {det['col_hz']:>7}"
        )
    return "\n".join(lines)
