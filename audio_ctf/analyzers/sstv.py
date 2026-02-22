"""SSTV (Slow Scan Television) decoder with VIS code detection."""

import os
from typing import Any

import numpy as np
from scipy import signal as sp_signal

# VIS mode codes (7-bit, LSB first)
VIS_MODES = {
    8:   ("Robot 36", "color", 36.0),
    12:  ("Robot 72", "color", 72.0),
    40:  ("FAX480",   "mono",  480.0),
    44:  ("Martin M2", "color", 58.0),
    56:  ("Martin M1", "color", 114.0),
    60:  ("Scottie S1", "color", 110.0),
    52:  ("Scottie S2", "color", 71.2),
    55:  ("Scottie DX", "color", 269.0),
    96:  ("Robot B&W 8", "mono", 8.0),
    100: ("Robot B&W 24", "mono", 24.0),
}

# SSTV frequency ranges
SYNC_HZ = 1200.0
LEADER_HZ = 1900.0
BLACK_HZ = 1500.0
WHITE_HZ = 2300.0


def analyze(
    data: np.ndarray,
    sr: int,
    output_path: str | None = None,
) -> dict[str, Any]:
    """Detect and decode an SSTV signal from audio.

    Attempts to use pysstv if available, otherwise implements VIS detection
    and Martin M1 decoding manually.

    Args:
        data: Mono float32 audio array.
        sr: Sample rate in Hz.
        output_path: Path to save decoded image PNG.

    Returns:
        Result dict with detected, confidence, data.
    """
    # Detect VIS header
    vis_result = _detect_vis(data, sr)
    if vis_result is None:
        return {
            "detected": False,
            "confidence": 0.0,
            "data": {"error": "No SSTV VIS header detected"},
        }

    vis_code, vis_start_sample, mode_name, mode_type, duration_sec = vis_result

    # Decode image data manually based on detected mode
    decoded_path = None
    decoder_name = "manual"

    if mode_name and "Robot 36" in mode_name:
        decoded_path = _decode_robot36(data, sr, vis_start_sample, output_path)
    elif mode_name and "Martin M1" in mode_name:
        decoded_path = _decode_martin_m1(data, sr, vis_start_sample, output_path)
    elif output_path:
        decoded_path = _save_frequency_image(data, sr, vis_start_sample, output_path)

    return {
        "detected": True,
        "confidence": 0.9,
        "data": {
            "vis_code": vis_code,
            "mode": mode_name or f"Unknown (VIS {vis_code})",
            "mode_type": mode_type,
            "duration_sec": duration_sec,
            "decoder": decoder_name,
            "image_path": decoded_path,
        },
    }


def _detect_vis(data: np.ndarray, sr: int) -> tuple | None:
    """Detect VIS (Vertical Interval Signaling) header in audio.

    VIS header structure:
      - 1900 Hz leader for 300 ms
      - 1200 Hz break  for  10 ms
      - 1900 Hz leader for 300 ms
      - 1200 Hz VIS start bit (30 ms)
      - 7 data bits × 30 ms: 1100 Hz = 1, 1300 Hz = 0  (LSB first)
      - 1 parity bit × 30 ms
      - 1200 Hz VIS stop bit (30 ms)

    Uses Hilbert-transform instantaneous frequency (accurate, continuous)
    with 1 ms smoothing.  The 10 ms break is detected as a gap between
    two leader runs rather than requiring a sustained 1200 Hz period.
    """
    from scipy.signal import hilbert
    from scipy.ndimage import uniform_filter1d

    # Instantaneous frequency via Hilbert transform, 1 ms smoothing
    analytic = hilbert(data.astype(np.float64))
    phase = np.unwrap(np.angle(analytic))
    inst_hz = np.diff(phase) / (2 * np.pi) * sr
    inst_hz = np.append(inst_hz, inst_hz[-1])
    smooth_n = max(1, int(sr * 0.001))          # 1 ms smoothing
    inst_hz = uniform_filter1d(inst_hz, size=smooth_n)

    # Downsample to 1 ms resolution for fast pattern matching
    step = max(1, int(sr * 0.001))
    f_ms = inst_hz[::step]
    n = len(f_ms)

    leader_min_ms = 150    # minimum leader run to accept
    gap_max_ms    = 60     # max gap between leader1 and leader2
    vis_bit_ms    = 30     # each VIS bit is 30 ms wide

    def is_leader(f): return abs(f - LEADER_HZ) < 200   # 1900 ± 200 Hz
    def is_vis1(f):   return abs(f - 1100)       < 150  # bit=1
    def is_vis0(f):   return abs(f - 1300)       < 150  # bit=0

    i = 0
    while i < n - leader_min_ms:
        # ── Find leader 1 ────────────────────────────────────────────────
        if not is_leader(f_ms[i]):
            i += 1
            continue

        l1_start = i
        while i < n and is_leader(f_ms[i]):
            i += 1
        l1_len = i - l1_start
        if l1_len < leader_min_ms:
            continue                          # too short

        # ── Find leader 2 (skip at most gap_max_ms non-leader samples) ───
        gap_start = i
        while i < n and not is_leader(f_ms[i]) and (i - gap_start) < gap_max_ms:
            i += 1
        gap_len = i - gap_start
        if gap_len == 0 or gap_len >= gap_max_ms:
            continue                          # no gap, or gap too long

        l2_start = i
        while i < n and is_leader(f_ms[i]):
            i += 1
        l2_len = i - l2_start
        if l2_len < leader_min_ms:
            continue

        # ── Skip VIS start bit (1200 Hz, ~30 ms) ────────────────────────
        # The VIS start bit is NOT a leader; just advance 30 ms past leader 2
        vis_data_start = i + vis_bit_ms      # skip start bit entirely

        if vis_data_start + 8 * vis_bit_ms >= n:
            continue

        # ── Read 8 VIS bits by sampling the middle of each 30 ms slot ───
        bits = []
        ok = True
        for bit_idx in range(8):
            mid = vis_data_start + int((bit_idx + 0.5) * vis_bit_ms)
            if mid >= n:
                ok = False
                break
            f = f_ms[mid]
            if is_vis1(f):
                bits.append(1)
            elif is_vis0(f):
                bits.append(0)
            else:
                ok = False
                break

        if not ok or len(bits) < 7:
            continue

        # ── Decode VIS code (bits 0-6, LSB first) ────────────────────────
        vis_code = sum(b << k for k, b in enumerate(bits[:7]))
        mode_info = VIS_MODES.get(vis_code)

        # Image data starts after: leader2 + start_bit + 8 data bits + stop_bit
        img_start_ms  = i + vis_bit_ms + 8 * vis_bit_ms + vis_bit_ms
        img_start_smp = min(img_start_ms * step, len(data) - 1)

        if mode_info:
            name, mtype, dur = mode_info
            return vis_code, img_start_smp, name, mtype, dur
        else:
            return vis_code, img_start_smp, None, None, 0.0

    return None


def _instantaneous_freq(
    data: np.ndarray, sr: int, window: int, hop: int
) -> np.ndarray:
    """Estimate instantaneous frequency via zero-crossing rate per window.
    Kept for backward compatibility; VIS detection now uses Hilbert transform.
    """
    freqs = []
    for start in range(0, len(data) - window, hop):
        chunk = data[start:start + window]
        crossings = np.where(np.diff(np.sign(chunk)))[0]
        zcr = len(crossings) / 2
        freq = zcr * sr / window
        freqs.append(freq)
    return np.array(freqs)


def _decode_robot36(
    data: np.ndarray, sr: int, start_sample: int, output_path: str | None
) -> str | None:
    """Manually decode Robot 36 SSTV mode.

    Robot 36: 320 columns, 240 lines, color (YCbCr)
    Per-line structure:
      sync (9 ms @ 1200 Hz) + sync_porch (3 ms @ 1500 Hz)
      + Y scan (88 ms, 320 px) + inter_ch_gap (4.5 ms)
      + C porch (1.5 ms @ 1900 Hz) + C scan (44 ms, 160 px)
    Even lines transmit Cr; odd lines transmit Cb.
    Each chrominance row is applied to both lines of its pair.
    """
    from PIL import Image
    from scipy.signal import hilbert
    from scipy.ndimage import uniform_filter1d

    IMG_W = 320
    IMG_H = 240

    SYNC_DUR   = 0.009    # 9.0 ms sync pulse
    PORCH_DUR  = 0.003    # 3.0 ms sync porch
    Y_DUR      = 0.088    # 88.0 ms luminance scan (320 px)
    IGAP_DUR   = 0.0045   # 4.5 ms inter-channel gap
    CPORCH_DUR = 0.0015   # 1.5 ms colour porch
    C_DUR      = 0.044    # 44.0 ms chrominance scan (160 px)

    def scan_pixels(sample_start: float, dur: float, n_px: int) -> np.ndarray:
        """Decode a frequency-scan segment to pixel values via Hilbert IF."""
        s = int(sample_start)
        e = min(int(sample_start + dur * sr), len(data))
        if e - s < 4:
            return np.full(n_px, 128, dtype=np.uint8)
        chunk = data[s:e].astype(np.float64)
        analytic = hilbert(chunk)
        phase = np.unwrap(np.angle(analytic))
        inst_hz = np.diff(phase) / (2 * np.pi) * sr
        inst_hz = np.append(inst_hz, inst_hz[-1])
        smooth_n = max(1, int(sr * 0.0005))   # 0.5 ms smoothing
        inst_hz = uniform_filter1d(inst_hz, size=smooth_n)
        idx = np.linspace(0, len(inst_hz) - 1, n_px).astype(int)
        freqs = inst_hz[idx]
        return np.clip(
            (freqs - BLACK_HZ) / (WHITE_HZ - BLACK_HZ) * 255, 0, 255
        ).astype(np.uint8)

    Y_arr = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
    C_arr = np.zeros((IMG_H, IMG_W), dtype=np.uint8)  # raw C scan per line

    pos = float(start_sample)
    for line in range(IMG_H):
        pos += SYNC_DUR   * sr   # skip sync pulse
        pos += PORCH_DUR  * sr   # skip sync porch

        Y_arr[line] = scan_pixels(pos, Y_DUR, IMG_W)
        pos += Y_DUR * sr

        pos += IGAP_DUR   * sr   # skip inter-channel gap
        pos += CPORCH_DUR * sr   # skip colour porch

        # C scan: 160 unique pixels → repeat each pixel twice → 320 wide
        c_raw = scan_pixels(pos, C_DUR, IMG_W // 2)
        C_arr[line] = np.repeat(c_raw, 2)[:IMG_W]
        pos += C_DUR * sr

    # Assign Cr / Cb from the raw C scan:
    # even lines → Cr for line pair (n, n+1)
    # odd  lines → Cb for line pair (n-1, n)
    Cb_arr = np.full((IMG_H, IMG_W), 128, dtype=np.uint8)
    Cr_arr = np.full((IMG_H, IMG_W), 128, dtype=np.uint8)

    for line in range(0, IMG_H, 2):       # even → Cr
        Cr_arr[line] = C_arr[line]
        if line + 1 < IMG_H:
            Cr_arr[line + 1] = C_arr[line]

    for line in range(1, IMG_H, 2):       # odd → Cb
        Cb_arr[line] = C_arr[line]
        if line - 1 >= 0:
            Cb_arr[line - 1] = C_arr[line]

    # YCbCr → RGB (BT.601 coefficients)
    Y  = Y_arr.astype(np.float32)
    Cb = Cb_arr.astype(np.float32) - 128.0
    Cr = Cr_arr.astype(np.float32) - 128.0

    R = np.clip(Y + 1.402 * Cr,                        0, 255).astype(np.uint8)
    G = np.clip(Y - 0.344136 * Cb - 0.714136 * Cr,    0, 255).astype(np.uint8)
    B = np.clip(Y + 1.772 * Cb,                        0, 255).astype(np.uint8)

    img = Image.fromarray(np.stack([R, G, B], axis=2), mode="RGB")

    if output_path:
        img.save(output_path)
        return output_path
    return None


def _decode_martin_m1(
    data: np.ndarray, sr: int, start_sample: int, output_path: str | None
) -> str | None:
    """Manually decode Martin M1 SSTV mode.

    Martin M1: 256 columns, 256 lines, color (GBR order)
    Line duration: 114.285ms
    Sync: 4.862ms at 1200 Hz
    Porch: 0.572ms at 1500 Hz
    Green: 146.432ms scan
    Porch: 0.572ms
    Blue: 146.432ms scan
    Porch: 0.572ms
    Red: 146.432ms scan
    """
    from PIL import Image

    img_width = 320
    img_height = 256
    img = Image.new("RGB", (img_width, img_height), (0, 0, 0))

    line_dur = 0.114285
    scan_dur = 0.146432
    sync_dur = 0.004862

    scan_samples = int(scan_dur * sr)

    pixel_data = np.zeros((img_height, img_width, 3), dtype=np.uint8)

    pos = start_sample
    for line in range(img_height):
        # Skip sync pulse and porches
        pos += int(sync_dur * sr)
        pos += int(0.000572 * sr)

        for channel_idx in range(3):  # G, B, R
            if pos + scan_samples > len(data):
                break
            scan_chunk = data[pos:pos + scan_samples]
            # Map frequency to pixel value: 1500 Hz = black, 2300 Hz = white
            # Use instantaneous frequency estimation
            chunk_freqs = _instantaneous_freq(scan_chunk, sr, max(1, sr // 1000), max(1, sr // 2000))
            if len(chunk_freqs) == 0:
                continue
            # Resample to img_width pixels
            x_idx = np.linspace(0, len(chunk_freqs) - 1, img_width).astype(int)
            line_freqs = chunk_freqs[x_idx]
            # Clamp and normalize: 1500 Hz → 0, 2300 Hz → 255
            pixel_vals = np.clip(
                (line_freqs - BLACK_HZ) / (WHITE_HZ - BLACK_HZ) * 255, 0, 255
            ).astype(np.uint8)
            # Martin M1 channel order: Green, Blue, Red
            ch_map = {0: 1, 1: 2, 2: 0}  # G→1, B→2, R→0
            pixel_data[line, :, ch_map[channel_idx]] = pixel_vals
            pos += scan_samples
            pos += int(0.000572 * sr)  # porch

    for y in range(img_height):
        for x in range(img_width):
            img.putpixel((x, y), tuple(pixel_data[y, x]))

    if output_path:
        img.save(output_path)
        return output_path
    return None


def _decode_with_pysstv(
    data: np.ndarray, sr: int, output_path: str | None
) -> str | None:
    """Decode SSTV audio using the pysstv library.

    pysstv works as an encoder only (it generates audio from images).
    For decoding we need to feed the raw PCM samples to its decoder API.
    Tries the documented decode() path; falls back to gen_image() if available.
    """
    try:
        from pysstv import decode as pysstv_decode
        from io import BytesIO
        import wave

        # pysstv.decode expects a wave file object
        pcm_int16 = (data * 32767).clip(-32768, 32767).astype(np.int16)
        wav_buf = BytesIO()
        with wave.open(wav_buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm_int16.tobytes())
        wav_buf.seek(0)

        sstv_obj, exc = pysstv_decode.decode(wav_buf)
        if sstv_obj is not None:
            img = sstv_obj.to_pil()
            if output_path and img:
                img.save(output_path)
                return output_path
    except Exception:
        pass
    return None


def _save_frequency_image(
    data: np.ndarray, sr: int, start_sample: int, output_path: str | None
) -> str | None:
    """Save a simple frequency-vs-time image for unknown SSTV modes."""
    if not output_path:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import signal

    chunk = data[start_sample:start_sample + int(sr * 5)]  # first 5 seconds
    freqs, times, Sxx = signal.spectrogram(chunk, fs=sr, nperseg=256, noverlap=192)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.pcolormesh(times, freqs, 10 * np.log10(Sxx + 1e-12), shading="auto", cmap="gray")
    ax.set_ylim(1000, 2500)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("SSTV Signal (Unknown Mode)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def format_result(result: dict[str, Any]) -> str:
    lines = []
    d = result["data"]
    if not result["detected"]:
        return f"No SSTV signal detected. {d.get('error', '')}"

    lines.append(f"SSTV mode detected:  {d['mode']}")
    lines.append(f"VIS code:            {d['vis_code']}")
    lines.append(f"Mode type:           {d['mode_type']}")
    if d.get("duration_sec"):
        lines.append(f"Expected duration:   {d['duration_sec']:.1f}s")
    lines.append(f"Decoder:             {d['decoder']}")
    lines.append(f"Confidence:          {result['confidence']:.1%}")
    if d.get("image_path"):
        lines.append(f"Image saved:         {d['image_path']}")
    return "\n".join(lines)
