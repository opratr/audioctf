"""Spectrogram generation and FFT frequency analysis."""

from typing import Any

import numpy as np
from scipy import signal


def analyze(
    data: np.ndarray,
    sr: int,
    output_path: str | None = None,
    window: str = "hann",
    overlap: float = 0.75,
    n_fft: int = 2048,
    top_n: int = 10,
) -> dict[str, Any]:
    """Generate a spectrogram and report dominant frequencies.

    Args:
        data: Mono float32 audio array.
        sr: Sample rate in Hz.
        output_path: If provided, save spectrogram PNG to this path.
        window: Window function name (hann, hamming, blackman, etc.).
        overlap: Overlap fraction between windows (0.0–0.99).
        n_fft: FFT size (number of samples per window).
        top_n: Number of top dominant frequencies to report.

    Returns:
        Result dict with detected, confidence, data (frequencies, image path).
    """
    hop = int(n_fft * (1 - overlap))
    freqs, times, Sxx = signal.spectrogram(
        data,
        fs=sr,
        window=window,
        nperseg=n_fft,
        noverlap=n_fft - hop,
        scaling="density",
    )

    Sxx_db = 10 * np.log10(Sxx + 1e-12)

    # Find dominant frequencies via global FFT
    dominant = _find_dominant_frequencies(data, sr, top_n)

    saved_path = None
    if output_path:
        saved_path = _save_spectrogram(freqs, times, Sxx_db, output_path, sr)

    return {
        "detected": True,
        "confidence": 1.0,
        "data": {
            "dominant_frequencies": dominant,
            "spectrogram_path": saved_path,
            "duration_sec": float(times[-1]) if len(times) > 0 else 0.0,
            "freq_resolution_hz": float(freqs[1] - freqs[0]) if len(freqs) > 1 else 0.0,
            "n_fft": n_fft,
            "window": window,
            "overlap": overlap,
        },
    }


def _find_dominant_frequencies(
    data: np.ndarray, sr: int, top_n: int
) -> list[dict[str, float]]:
    """Compute global FFT and return top N frequency peaks."""
    n = len(data)
    fft_vals = np.abs(np.fft.rfft(data * np.hanning(n), n=n))
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)

    # Simple peak picking: find local maxima
    from scipy.signal import find_peaks

    peaks, props = find_peaks(fft_vals, distance=max(1, int(n / sr * 10)), prominence=0)
    if len(peaks) == 0:
        # Fallback: just take argmax
        idx = np.argsort(fft_vals)[::-1][:top_n]
        peaks = idx

    # Sort by amplitude
    peak_amps = fft_vals[peaks]
    top_idx = np.argsort(peak_amps)[::-1][:top_n]
    top_peaks = peaks[top_idx]

    max_amp = fft_vals.max() if fft_vals.max() > 0 else 1.0
    result = []
    for p in top_peaks:
        result.append(
            {
                "frequency_hz": round(float(freqs[p]), 2),
                "amplitude": round(float(fft_vals[p]), 4),
                "relative_db": round(20 * np.log10(fft_vals[p] / max_amp + 1e-12), 2),
            }
        )
    return result


def _save_spectrogram(
    freqs: np.ndarray,
    times: np.ndarray,
    Sxx_db: np.ndarray,
    output_path: str,
    sr: int,
) -> str:
    """Save a labeled spectrogram PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 6))
    img = ax.pcolormesh(
        times, freqs, Sxx_db, shading="auto", cmap="inferno", vmin=-80, vmax=0
    )
    cbar = fig.colorbar(img, ax=ax)
    cbar.set_label("Power (dB)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Time (s)")
    ax.set_title("Spectrogram")
    ax.set_ylim(0, sr / 2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def format_result(result: dict[str, Any]) -> str:
    d = result["data"]
    lines = []
    if d.get("spectrogram_path"):
        lines.append(f"Spectrogram saved: {d['spectrogram_path']}")
    lines.append(f"Duration:          {d['duration_sec']:.3f}s")
    lines.append(f"Freq resolution:   {d['freq_resolution_hz']:.2f} Hz")
    lines.append(f"FFT size:          {d['n_fft']}")
    lines.append(f"Window:            {d['window']}")
    lines.append(f"Overlap:           {d['overlap']:.0%}")
    lines.append("")
    lines.append(f"Top {len(d['dominant_frequencies'])} dominant frequencies:")
    lines.append(f"  {'Frequency (Hz)':>15}  {'Amplitude':>12}  {'dB':>8}")
    lines.append(f"  {'-'*15}  {'-'*12}  {'-'*8}")
    for f in d["dominant_frequencies"]:
        lines.append(
            f"  {f['frequency_hz']:>15.2f}  {f['amplitude']:>12.4f}  {f['relative_db']:>8.2f}"
        )
    return "\n".join(lines)
