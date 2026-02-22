"""File metadata and audio statistics analyzer."""

import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


def analyze(path: str, data: np.ndarray, sr: int) -> dict[str, Any]:
    """Return metadata and audio statistics for a file.

    Args:
        path: Path to the audio file.
        data: Audio samples as float32 numpy array.
        sr: Sample rate in Hz.

    Returns:
        Result dict with keys: detected, confidence, data.
    """
    file_path = Path(path)
    file_size = file_path.stat().st_size

    # Compute hashes
    md5, sha256 = _hash_file(path)

    # Audio properties
    n_samples = data.shape[0] if data.ndim == 1 else data.shape[0]
    channels = 1 if data.ndim == 1 else data.shape[1]
    duration = n_samples / sr

    # Bit depth and format from soundfile
    bit_depth = None
    fmt = None
    subtype = None
    tags = {}
    try:
        info = sf.info(path)
        bit_depth = _subtype_to_bits(info.subtype)
        fmt = info.format
        subtype = info.subtype
        tags = dict(info.extra_info) if hasattr(info, "extra_info") else {}
    except Exception:
        pass

    # Audio statistics
    mono = data.mean(axis=1) if data.ndim == 2 else data
    rms = float(np.sqrt(np.mean(mono**2)))
    peak = float(np.max(np.abs(mono)))
    dc_offset = float(np.mean(mono))

    # Suspicious metadata check
    suspicious = []
    if tags:
        for key, val in tags.items():
            if len(str(val)) > 200:
                suspicious.append(f"Long tag '{key}' ({len(str(val))} chars)")

    result_data = {
        "file": {
            "path": str(file_path.resolve()),
            "size_bytes": file_size,
            "format": fmt or file_path.suffix.lstrip(".").upper(),
            "md5": md5,
            "sha256": sha256,
        },
        "audio": {
            "duration_sec": round(duration, 4),
            "sample_rate_hz": sr,
            "channels": channels,
            "n_samples": n_samples,
            "bit_depth": bit_depth,
            "subtype": subtype,
            "rms": round(rms, 6),
            "peak": round(peak, 6),
            "dc_offset": round(dc_offset, 6),
        },
        "tags": tags,
        "suspicious": suspicious,
    }

    return {
        "detected": True,
        "confidence": 1.0,
        "data": result_data,
    }


def _hash_file(path: str) -> tuple[str, str]:
    md5 = hashlib.md5()
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            md5.update(chunk)
            sha256.update(chunk)
    return md5.hexdigest(), sha256.hexdigest()


def _subtype_to_bits(subtype: str) -> int | None:
    if subtype is None:
        return None
    subtype = subtype.upper()
    mapping = {
        "PCM_8": 8,
        "PCM_16": 16,
        "PCM_24": 24,
        "PCM_32": 32,
        "FLOAT": 32,
        "DOUBLE": 64,
        "ULAW": 8,
        "ALAW": 8,
        "IMA_ADPCM": 4,
        "MS_ADPCM": 4,
        "VORBIS": None,
        "OPUS": None,
        "FLAC": None,
    }
    for k, v in mapping.items():
        if k in subtype:
            return v
    return None


def format_result(result: dict[str, Any]) -> str:
    """Format metadata result as human-readable text."""
    d = result["data"]
    lines = []

    f = d["file"]
    lines.append(f"File:        {f['path']}")
    lines.append(f"Size:        {f['size_bytes']:,} bytes")
    lines.append(f"Format:      {f['format']}")
    lines.append(f"MD5:         {f['md5']}")
    lines.append(f"SHA256:      {f['sha256']}")
    lines.append("")

    a = d["audio"]
    lines.append(f"Duration:    {a['duration_sec']:.3f}s")
    lines.append(f"Sample Rate: {a['sample_rate_hz']:,} Hz")
    lines.append(f"Channels:    {a['channels']}")
    lines.append(f"Samples:     {a['n_samples']:,}")
    if a["bit_depth"]:
        lines.append(f"Bit Depth:   {a['bit_depth']}-bit")
    if a["subtype"]:
        lines.append(f"Subtype:     {a['subtype']}")
    lines.append(f"RMS:         {a['rms']:.6f}")
    lines.append(f"Peak:        {a['peak']:.6f}")
    lines.append(f"DC Offset:   {a['dc_offset']:.6f}")

    if d["tags"]:
        lines.append("")
        lines.append("Tags:")
        for k, v in d["tags"].items():
            lines.append(f"  {k}: {v}")

    if d["suspicious"]:
        lines.append("")
        lines.append("Suspicious:")
        for s in d["suspicious"]:
            lines.append(f"  ! {s}")

    return "\n".join(lines)
