"""Click-based CLI entry point for the Audio CTF Analysis Tool."""

import sys
from typing import Optional

import click
import numpy as np

from audio_ctf.audio_loader import load_audio, to_mono
from audio_ctf.analyzers import (
    metadata as _metadata,
    spectral as _spectral,
    morse as _morse,
    dtmf as _dtmf,
    modem as _modem,
    lsb as _lsb,
    sstv as _sstv,
    baudot as _baudot,
)


# ─── Shared options ──────────────────────────────────────────────────────────

def _load(path: str, mono: bool = True) -> tuple[np.ndarray, int]:
    """Load audio file, printing an error and exiting on failure."""
    try:
        data, sr = load_audio(path)
        if mono:
            data = to_mono(data)
        return data, sr
    except FileNotFoundError:
        click.echo(f"Error: File not found: {path}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error loading file: {e}", err=True)
        sys.exit(1)


# ─── CLI group ───────────────────────────────────────────────────────────────

@click.group()
@click.version_option(version="0.1.0", prog_name="audioctf")
def main():
    """Audio CTF Analysis Tool — detect and decode audio steganography.

    \b
    Supported techniques:
      Morse code, DTMF, FSK modems (Bell 103/202, AFSK),
      LSB steganography, SSTV, Baudot/RTTY, spectrogram analysis.
    """


# ─── info ────────────────────────────────────────────────────────────────────

@main.command()
@click.argument("file", type=click.Path(exists=True))
def info(file: str):
    """Show file metadata, audio statistics, and embedded tag information."""
    data, sr = _load(file, mono=False)
    result = _metadata.analyze(file, data, sr)
    click.echo(_metadata.format_result(result))


# ─── spectrogram ─────────────────────────────────────────────────────────────

@main.command()
@click.argument("file", type=click.Path(exists=True))
@click.option("-o", "--output", default="spectrogram.png", show_default=True,
              help="Output PNG file path.")
@click.option("--window", default="hann", show_default=True,
              help="Window function (hann, hamming, blackman, …).")
@click.option("--overlap", default=0.75, show_default=True, type=float,
              help="Window overlap fraction (0.0–0.99).")
@click.option("--fft-size", default=2048, show_default=True, type=int,
              help="FFT window size in samples.")
@click.option("--top", default=10, show_default=True, type=int,
              help="Number of dominant frequencies to report.")
def spectrogram(file: str, output: str, window: str, overlap: float, fft_size: int, top: int):
    """Generate a spectrogram PNG and report dominant frequencies."""
    data, sr = _load(file)
    result = _spectral.analyze(
        data, sr,
        output_path=output,
        window=window,
        overlap=overlap,
        n_fft=fft_size,
        top_n=top,
    )
    click.echo(_spectral.format_result(result))


# ─── morse ───────────────────────────────────────────────────────────────────

@main.command()
@click.argument("file", type=click.Path(exists=True))
@click.option("--carrier", default=None, type=float,
              help="Carrier frequency in Hz (auto-detected if omitted).")
def morse(file: str, carrier: Optional[float]):
    """Detect and decode CW Morse code from audio."""
    data, sr = _load(file)
    result = _morse.analyze(data, sr, carrier_hz=carrier)
    click.echo(_morse.format_result(result))
    sys.exit(0 if result["detected"] else 1)


# ─── dtmf ────────────────────────────────────────────────────────────────────

@main.command()
@click.argument("file", type=click.Path(exists=True))
def dtmf(file: str):
    """Detect and decode DTMF telephone tones."""
    data, sr = _load(file)
    result = _dtmf.analyze(data, sr)
    click.echo(_dtmf.format_result(result))
    sys.exit(0 if result["detected"] else 1)


# ─── modem ───────────────────────────────────────────────────────────────────

@main.command()
@click.argument("file", type=click.Path(exists=True))
def modem(file: str):
    """Detect FSK modem signals and decode data (Bell 103/202, AFSK/AX.25)."""
    data, sr = _load(file)
    result = _modem.analyze(data, sr)
    click.echo(_modem.format_result(result))
    sys.exit(0 if result["detected"] else 1)


# ─── lsb ─────────────────────────────────────────────────────────────────────

@main.command()
@click.argument("file", type=click.Path(exists=True))
@click.option("--bits", default=1, show_default=True, type=click.IntRange(1, 3),
              help="Number of LSBs to extract per sample.")
@click.option("--output-dir", default=".", show_default=True,
              help="Directory to save any detected embedded files.")
def lsb(file: str, bits: int, output_dir: str):
    """Extract LSB steganography from audio samples.

    Checks both MSB-first and LSB-first bit orderings. Detects embedded
    files by magic bytes (PNG, ZIP, JPEG, PDF, etc.) and saves them.
    """
    data, sr = _load(file, mono=False)
    result = _lsb.analyze(data, sr, bits=bits, output_dir=output_dir)
    click.echo(_lsb.format_result(result))
    sys.exit(0 if result["detected"] else 1)


# ─── sstv ────────────────────────────────────────────────────────────────────

@main.command()
@click.argument("file", type=click.Path(exists=True))
@click.option("-o", "--output", default="sstv_decoded.png", show_default=True,
              help="Output PNG path for decoded image.")
def sstv(file: str, output: str):
    """Decode an SSTV (Slow Scan Television) image from audio."""
    data, sr = _load(file)
    result = _sstv.analyze(data, sr, output_path=output)
    click.echo(_sstv.format_result(result))
    sys.exit(0 if result["detected"] else 1)


# ─── rtty ────────────────────────────────────────────────────────────────────

@main.command()
@click.argument("file", type=click.Path(exists=True))
def rtty(file: str):
    """Decode Baudot/RTTY (radio teletype) from audio."""
    data, sr = _load(file)
    result = _baudot.analyze(data, sr)
    click.echo(_baudot.format_result(result))
    sys.exit(0 if result["detected"] else 1)


# ─── scan ────────────────────────────────────────────────────────────────────

@main.command()
@click.argument("file", type=click.Path(exists=True))
@click.option("--spectrogram-out", default=None,
              help="Save spectrogram PNG during scan (optional).")
@click.option("--lsb-bits", default=1, show_default=True, type=click.IntRange(1, 3),
              help="LSB bits to check during scan.")
@click.option("--output-dir", default=".", show_default=True,
              help="Directory for extracted files during scan.")
def scan(file: str, spectrogram_out: Optional[str], lsb_bits: int, output_dir: str):
    """Run ALL detectors and report findings with confidence scores.

    Exits 0 if any signal is detected, 1 if nothing is found.
    """
    click.echo(f"Scanning: {file}")
    click.echo("=" * 60)

    data_stereo, sr = _load(file, mono=False)
    data_mono = to_mono(data_stereo)

    jobs = [
        ("metadata",    lambda: _metadata.analyze(file, data_stereo, sr)),
        ("spectrogram", lambda: _spectral.analyze(data_mono, sr, output_path=spectrogram_out)),
        ("morse",       lambda: _morse.analyze(data_mono, sr)),
        ("dtmf",        lambda: _dtmf.analyze(data_mono, sr)),
        ("modem",       lambda: _modem.analyze(data_mono, sr)),
        ("lsb",         lambda: _lsb.analyze(data_stereo, sr, bits=lsb_bits, output_dir=output_dir)),
        ("sstv",        lambda: _sstv.analyze(data_mono, sr)),
        ("rtty",        lambda: _baudot.analyze(data_mono, sr)),
    ]

    results = {}
    for name, fn in jobs:
        click.echo(f"\n[{name}] ", nl=False)
        try:
            result = fn()
            results[name] = result
            status = "DETECTED" if result.get("detected") else "not detected"
            conf_str = f"(confidence: {result['confidence']:.0%})" if result.get("detected") else ""
            click.echo(f"{status} {conf_str}")
        except Exception as e:
            click.echo(f"ERROR: {e}")
            results[name] = {"detected": False, "confidence": 0.0, "data": {"error": str(e)}}

    # Summary table
    click.echo("\n" + "=" * 60)
    click.echo("SCAN SUMMARY")
    click.echo("=" * 60)

    detected_any = False
    formatters = {
        "metadata":    _metadata.format_result,
        "spectrogram": _spectral.format_result,
        "morse":       _morse.format_result,
        "dtmf":        _dtmf.format_result,
        "modem":       _modem.format_result,
        "lsb":         _lsb.format_result,
        "sstv":        _sstv.format_result,
        "rtty":        _baudot.format_result,
    }

    for name, result in results.items():
        detected = result.get("detected", False)
        confidence = result.get("confidence", 0.0)
        marker = "✓" if detected else "✗"
        conf_bar = _confidence_bar(confidence)
        click.echo(f"  {marker}  {name:<14} {conf_bar}  {confidence:.0%}")
        if detected:
            detected_any = True

    # Print full details for detected signals
    click.echo("")
    for name, result in results.items():
        if result.get("detected") and name not in ("metadata", "spectrogram"):
            click.echo(f"\n{'─'*60}")
            click.echo(f"  {name.upper()} DETAILS")
            click.echo(f"{'─'*60}")
            formatter = formatters.get(name)
            if formatter:
                click.echo(formatter(result))

    click.echo("")
    if detected_any:
        click.echo("Finding(s) detected. Review the details above.")
    else:
        click.echo("No signals detected.")

    sys.exit(0 if detected_any else 1)


def _confidence_bar(confidence: float, width: int = 10) -> str:
    filled = int(confidence * width)
    return "[" + "█" * filled + "░" * (width - filled) + "]"


if __name__ == "__main__":
    main()
