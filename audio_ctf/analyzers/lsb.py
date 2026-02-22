"""LSB (Least Significant Bit) steganography extractor for audio samples."""

import os
import struct
from typing import Any

import numpy as np

# Known file magic bytes for embedded file detection
FILE_MAGIC = {
    b"\x89PNG\r\n\x1a\n": ("PNG image", ".png"),
    b"PK\x03\x04": ("ZIP archive", ".zip"),
    b"\xff\xd8\xff": ("JPEG image", ".jpg"),
    b"%PDF": ("PDF document", ".pdf"),
    b"GIF87a": ("GIF image", ".gif"),
    b"GIF89a": ("GIF image", ".gif"),
    b"BM": ("BMP image", ".bmp"),
    b"RIFF": ("RIFF file (WAV/AVI)", ".wav"),
    b"\x1f\x8b": ("GZIP archive", ".gz"),
    b"7z\xbc\xaf'": ("7-Zip archive", ".7z"),
    b"\xca\xfe\xba\xbe": ("Mach-O binary", ".macho"),
    b"\x7fELF": ("ELF binary", ".elf"),
    b"MZ": ("Windows PE/DOS", ".exe"),
    b"OggS": ("OGG container", ".ogg"),
    b"fLaC": ("FLAC audio", ".flac"),
    b"ID3": ("MP3 (ID3 tag)", ".mp3"),
    b"\xff\xfb": ("MP3 audio", ".mp3"),
}


def analyze(
    data: np.ndarray,
    sr: int,
    bits: int = 1,
    output_dir: str | None = None,
) -> dict[str, Any]:
    """Extract LSB steganography from audio samples.

    Args:
        data: Mono float32 audio array (or multi-channel; will be flattened).
        sr: Sample rate (unused, kept for uniform API).
        bits: Number of LSBs to extract per sample (1, 2, or 3).
        output_dir: If set, write any detected embedded files here.

    Returns:
        Result dict with detected, confidence, data.
    """
    bits = max(1, min(3, bits))

    # Convert float32 to 16-bit integers for LSB extraction
    flat = data.flatten()
    samples_int16 = np.clip(flat * 32768, -32768, 32767).astype(np.int16)

    results = {}

    # Try both bit orderings
    for msb_first in (False, True):
        label = "msb_first" if msb_first else "lsb_first"
        extracted = _extract_lsb_bytes(samples_int16, bits, msb_first)
        entropy = _byte_entropy(extracted)
        text, text_ratio = _try_decode_text(extracted)
        magic, file_type, file_ext = _detect_file_magic(extracted)
        saved_path = None

        if magic and output_dir:
            saved_path = _save_embedded_file(extracted, file_ext, output_dir, label)

        results[label] = {
            "entropy": round(entropy, 4),
            "text_ratio": round(text_ratio, 4),
            "text_preview": text[:200] if text else None,
            "file_magic": magic.hex() if magic else None,
            "file_type": file_type,
            "saved_path": saved_path,
            "bytes_extracted": len(extracted),
        }

    # Determine detection confidence
    # Only file magic bytes or high text ratio are strong signals.
    # Low entropy alone is not sufficient — structured audio (sine waves, etc.)
    # naturally has low-entropy LSBs but contains no hidden data.
    confidences = []
    for label, r in results.items():
        if r["file_type"]:
            confidences.append(0.95)
        elif r["text_ratio"] > 0.7:
            confidences.append(0.8)
        elif r["entropy"] < 3.0:
            # Very low entropy (well below sine-wave ~4 bits/byte) is suspicious
            confidences.append(0.5)
        else:
            confidences.append(0.1)

    confidence = max(confidences)
    detected = confidence > 0.4

    return {
        "detected": detected,
        "confidence": confidence,
        "data": {
            "bits_extracted": bits,
            "total_samples": len(samples_int16),
            "results": results,
        },
    }


def _extract_lsb_bytes(
    samples: np.ndarray, n_bits: int, msb_first: bool
) -> bytes:
    """Extract n_bits LSBs from each sample and pack into bytes."""
    # Extract the n_bits LSBs from each sample
    mask = (1 << n_bits) - 1
    lsbs = (samples.astype(np.int32) & mask).astype(np.uint8)

    # Pack bits into bytes
    bit_buffer = []
    for val in lsbs:
        for bit_idx in range(n_bits):
            if msb_first:
                bit_buffer.append((val >> (n_bits - 1 - bit_idx)) & 1)
            else:
                bit_buffer.append((val >> bit_idx) & 1)

    # Group into bytes (8 bits each)
    result = bytearray()
    for i in range(0, len(bit_buffer) - 7, 8):
        byte_val = 0
        for j in range(8):
            byte_val = (byte_val << 1) | bit_buffer[i + j]
        result.append(byte_val)
    return bytes(result)


def _byte_entropy(data: bytes) -> float:
    """Compute Shannon entropy in bits/byte."""
    if not data:
        return 0.0
    counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
    probs = counts / len(data)
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def _try_decode_text(data: bytes) -> tuple[str, float]:
    """Try to decode bytes as UTF-8 or ASCII text. Returns (text, printable_ratio)."""
    if not data:
        return "", 0.0
    printable = sum(
        1 for c in data if 0x20 <= c <= 0x7E or c in (0x09, 0x0A, 0x0D)
    )
    ratio = printable / len(data)
    if ratio > 0.6:
        try:
            return data.decode("utf-8", errors="replace"), ratio
        except Exception:
            return data.decode("latin-1", errors="replace"), ratio
    return "", ratio


def _detect_file_magic(data: bytes) -> tuple[bytes | None, str | None, str | None]:
    """Check if extracted bytes start with a known file magic sequence."""
    for magic, (ftype, ext) in FILE_MAGIC.items():
        if data[:len(magic)] == magic:
            return magic, ftype, ext
    return None, None, None


def _save_embedded_file(
    data: bytes, ext: str, output_dir: str, label: str
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    filename = f"lsb_extracted_{label}{ext}"
    path = os.path.join(output_dir, filename)
    with open(path, "wb") as f:
        f.write(data)
    return path


def format_result(result: dict[str, Any]) -> str:
    lines = []
    d = result["data"]
    lines.append(f"Bits extracted per sample: {d['bits_extracted']}")
    lines.append(f"Total samples analyzed:    {d['total_samples']:,}")
    lines.append(f"Confidence:                {result['confidence']:.1%}")
    lines.append("")

    for label, r in d["results"].items():
        lines.append(f"  [{label}]")
        lines.append(f"    Bytes extracted: {r['bytes_extracted']:,}")
        lines.append(f"    Entropy:         {r['entropy']:.4f} bits/byte  (random ~8.0)")
        lines.append(f"    Printable ratio: {r['text_ratio']:.1%}")
        if r["file_type"]:
            lines.append(f"    Embedded file:   {r['file_type']}  (magic: {r['file_magic']})")
        if r["saved_path"]:
            lines.append(f"    Saved to:        {r['saved_path']}")
        if r["text_preview"]:
            lines.append(f"    Text preview:")
            for line in r["text_preview"].splitlines()[:5]:
                lines.append(f"      {line}")
        lines.append("")

    return "\n".join(lines).rstrip()
