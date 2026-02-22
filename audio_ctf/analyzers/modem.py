"""FSK modem demodulator: Bell 103, Bell 202, AFSK/AX.25."""

from typing import Any

import numpy as np
from scipy import signal as sp_signal


# Protocol definitions: (name, baud_rate, mark_hz, space_hz, description)
PROTOCOLS = [
    ("Bell 103 Originate", 300,  1270, 1070, "Bell 103 originate (answer side)"),
    ("Bell 103 Answer",    300,  2225, 2025, "Bell 103 answer (originate side)"),
    ("Bell 202 / AFSK",   1200, 1200, 2200, "Bell 202 / AFSK 1200 baud"),
    ("V.21 Ch1",           300,   980,  1180, "V.21 channel 1"),
    ("V.21 Ch2",           300,  1650,  1850, "V.21 channel 2"),
]

# HDLC frame delimiter
HDLC_FLAG = 0x7E  # 01111110


def analyze(data: np.ndarray, sr: int) -> dict[str, Any]:
    """Auto-detect FSK modem protocol and decode.

    Args:
        data: Mono float32 audio array.
        sr: Sample rate in Hz.

    Returns:
        Result dict with detected, confidence, data (protocol, bits, decoded text).
    """
    # Find dominant frequencies to auto-select protocol
    detected_proto, mark_hz, space_hz, baud = _detect_protocol(data, sr)

    if detected_proto is None:
        return {
            "detected": False,
            "confidence": 0.0,
            "data": {"error": "No FSK modem signal detected"},
        }

    # Demodulate FSK → bitstream
    bits = _fsk_demodulate(data, sr, mark_hz, space_hz, baud)

    if not bits:
        return {
            "detected": False,
            "confidence": 0.0,
            "data": {"error": f"Protocol {detected_proto} detected but no bits decoded"},
        }

    # Try to decode bitstream as ASCII text (8N1 framing)
    ascii_text, text_confidence = _decode_ascii_8n1(bits)

    # Try HDLC decode for AFSK/AX.25
    hdlc_frames = []
    if "AFSK" in detected_proto or "202" in detected_proto:
        hdlc_frames = _decode_hdlc(bits)

    overall_confidence = text_confidence if ascii_text else (0.3 if bits else 0.0)

    return {
        "detected": True,
        "confidence": overall_confidence,
        "data": {
            "protocol": detected_proto,
            "mark_hz": mark_hz,
            "space_hz": space_hz,
            "baud_rate": baud,
            "bits_decoded": len(bits),
            "ascii_text": ascii_text,
            "hdlc_frames": hdlc_frames,
            "raw_bits": bits[:256] if bits else [],  # first 256 bits for inspection
        },
    }


def _detect_protocol(data: np.ndarray, sr: int) -> tuple[str | None, int, int, int]:
    """Detect which modem protocol is present by finding spectral peaks.

    Finds the dominant peaks in the spectrum, then matches them to the
    closest known protocol frequency pair. This avoids being fooled by
    FSK transition sidebands that land between protocol frequencies.
    """
    n = len(data)
    fft = np.abs(np.fft.rfft(data * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, 1.0 / sr)

    # Find peaks in the 500–3000 Hz modem band
    band = (freqs >= 500) & (freqs <= 3000)
    band_idx = np.where(band)[0]
    band_fft = fft[band_idx]

    from scipy.signal import find_peaks
    # minimum distance ~50 Hz between peaks
    min_dist = max(1, int(50 / (freqs[1] - freqs[0])))
    peaks, _ = find_peaks(band_fft, distance=min_dist, prominence=band_fft.max() * 0.05)

    if len(peaks) < 2:
        return None, 0, 0, 0

    # Take the top-N peaks by amplitude
    top_n = min(8, len(peaks))
    top_peaks = peaks[np.argsort(band_fft[peaks])[::-1][:top_n]]
    peak_freqs = freqs[band_idx[top_peaks]]
    peak_amps = band_fft[top_peaks]

    # Match every pair of top peaks to the nearest protocol
    best_score = 0.0
    best = (None, 0, 0, 0)

    for name, baud, mark, space, _ in PROTOCOLS:
        for i, (pf1, pa1) in enumerate(zip(peak_freqs, peak_amps)):
            for j, (pf2, pa2) in enumerate(zip(peak_freqs, peak_amps)):
                if i == j:
                    continue
                # Score: how close are these two peaks to (mark, space)?
                # Penalise by frequency error relative to the expected shift
                shift = abs(mark - space)
                err_mark  = abs(pf1 - mark)
                err_space = abs(pf2 - space)
                tolerance = max(baud * 0.6, 80)   # tighter for higher baud
                if err_mark > tolerance or err_space > tolerance:
                    continue
                # Score = combined amplitude weighted by frequency accuracy
                freq_penalty = (err_mark + err_space) / (shift + 1)
                score = (pa1 + pa2) / (1 + freq_penalty)
                if score > best_score:
                    best_score = score
                    best = (name, mark, space, baud)

    if best_score < 1e-4:
        return None, 0, 0, 0
    return best


def _fsk_demodulate(
    data: np.ndarray, sr: int, mark_hz: int, space_hz: int, baud: int
) -> list[int]:
    """Demodulate FSK signal using a non-coherent per-bit DFT.

    For each bit window, projects the samples onto complex exponentials at
    the mark and space frequencies (equivalent to two-point DFT / Goertzel)
    and compares the resulting powers. This is delay-free, phase-agnostic,
    and does not suffer from filter group-delay timing errors.
    """
    spb_float = sr / baud
    # If the sample rate gives an exact integer samples-per-bit, use it
    # to avoid cumulative timing drift; otherwise keep the float.
    spb = int(round(spb_float))
    if abs(spb - spb_float) > 0.01:
        spb_float_use = spb_float  # fractional — accept some drift
    else:
        spb_float_use = float(spb)  # exact — no drift

    bits = []
    pos = 0.0
    n_sig = len(data)

    while pos + spb_float_use <= n_sig:
        s = int(pos)
        e = int(pos + spb_float_use)
        if e <= s:
            e = s + 1
        chunk = data[s:e].astype(np.float64)
        tc = np.arange(len(chunk)) / sr
        mark_pwr  = float(abs(np.dot(chunk, np.exp(-1j * 2 * np.pi * mark_hz  * tc))) ** 2)
        space_pwr = float(abs(np.dot(chunk, np.exp(-1j * 2 * np.pi * space_hz * tc))) ** 2)
        bits.append(1 if mark_pwr > space_pwr else 0)
        pos += spb_float_use

    return bits


def _bandpass(data: np.ndarray, sr: int, center: float, bw: float) -> np.ndarray:
    nyq = sr / 2.0
    low = max(1, center - bw / 2)
    high = min(nyq - 1, center + bw / 2)
    sos = sp_signal.butter(5, [low / nyq, high / nyq], btype="band", output="sos")
    return sp_signal.sosfilt(sos, data)


def _decode_ascii_8n1(bits: list[int]) -> tuple[str, float]:
    """Try to decode bitstream as 8N1 async serial (start=0, 8 data bits, stop=1).

    Returns (text, confidence). confidence based on fraction of printable chars.
    """
    text_chars = []
    i = 0
    n = len(bits)
    found = 0

    while i < n - 10:
        # Look for start bit (0)
        if bits[i] != 0:
            i += 1
            continue

        # Read 8 data bits (LSB first)
        if i + 9 >= n:
            break
        byte_bits = bits[i + 1:i + 9]
        stop_bit = bits[i + 9]

        byte_val = 0
        for j, b in enumerate(byte_bits):
            byte_val |= b << j

        if stop_bit == 1:
            # Valid framing
            found += 1
            text_chars.append(byte_val)
            i += 10
        else:
            i += 1

    if not text_chars:
        return "", 0.0

    # Decode as text
    raw_bytes = bytes(text_chars)
    printable = sum(
        1 for c in raw_bytes if 0x20 <= c <= 0x7E or c in (0x09, 0x0A, 0x0D)
    )
    confidence = printable / len(raw_bytes) if raw_bytes else 0.0

    try:
        text = raw_bytes.decode("ascii", errors="replace")
    except Exception:
        text = raw_bytes.decode("latin-1", errors="replace")

    return text, confidence


def _decode_hdlc(bits: list[int]) -> list[dict]:
    """Attempt HDLC frame extraction (for AX.25 packet radio).

    Handles bit-stuffing removal and extracts payload bytes.
    """
    frames = []

    # Remove bit stuffing: after 5 consecutive 1s, a 0 is inserted
    destuffed = []
    ones_count = 0
    i = 0
    while i < len(bits):
        b = bits[i]
        if b == 1:
            ones_count += 1
            destuffed.append(b)
            if ones_count == 5:
                # Skip the next stuffed 0
                i += 1
                if i < len(bits) and bits[i] == 0:
                    ones_count = 0
                    i += 1
                    continue
        else:
            ones_count = 0
            destuffed.append(b)
        i += 1

    # Search for HDLC flags (01111110)
    flag_bits = [0, 1, 1, 1, 1, 1, 1, 0]
    flag_positions = []
    for j in range(len(destuffed) - 7):
        if destuffed[j:j + 8] == flag_bits:
            flag_positions.append(j)

    # Extract frames between consecutive flags
    for a, b_pos in zip(flag_positions, flag_positions[1:]):
        frame_bits = destuffed[a + 8:b_pos]
        if len(frame_bits) < 16:
            continue
        # Convert to bytes
        frame_bytes = []
        for k in range(0, len(frame_bits) - 7, 8):
            byte_val = 0
            for bit_idx, bit in enumerate(frame_bits[k:k + 8]):
                byte_val |= bit << bit_idx
            frame_bytes.append(byte_val)
        if frame_bytes:
            try:
                payload = bytes(frame_bytes).decode("ascii", errors="replace")
            except Exception:
                payload = bytes(frame_bytes).hex()
            frames.append({
                "length_bytes": len(frame_bytes),
                "payload": payload,
                "hex": bytes(frame_bytes).hex(),
            })

    return frames


def format_result(result: dict[str, Any]) -> str:
    lines = []
    d = result["data"]
    if not result["detected"]:
        return f"No FSK modem signal detected. {d.get('error', '')}"

    lines.append(f"Protocol:     {d['protocol']}")
    lines.append(f"Mark freq:    {d['mark_hz']} Hz")
    lines.append(f"Space freq:   {d['space_hz']} Hz")
    lines.append(f"Baud rate:    {d['baud_rate']}")
    lines.append(f"Bits decoded: {d['bits_decoded']}")
    lines.append(f"Confidence:   {result['confidence']:.1%}")

    if d.get("ascii_text"):
        lines.append("")
        lines.append("Decoded ASCII text:")
        text = d["ascii_text"]
        for line in text.splitlines()[:50]:
            lines.append(f"  {line}")
        if len(text.splitlines()) > 50:
            lines.append(f"  ... ({len(text.splitlines())} total lines)")

    if d.get("hdlc_frames"):
        lines.append("")
        lines.append(f"HDLC/AX.25 frames ({len(d['hdlc_frames'])}):")
        for i, frame in enumerate(d["hdlc_frames"][:5]):
            lines.append(f"  Frame {i+1}: {frame['length_bytes']} bytes")
            lines.append(f"    {frame['payload'][:80]}")

    if d.get("raw_bits"):
        bits_str = "".join(str(b) for b in d["raw_bits"][:64])
        lines.append("")
        lines.append(f"Raw bits (first 64): {bits_str}")

    return "\n".join(lines)
