# Audio CTF Analysis Tool

A CLI tool for analyzing audio files in CTF (Capture The Flag) competitions.
Detects and decodes common audio steganography and encoding techniques.

## Installation

```bash
pip install -e .
# For SSTV support:
pip install -e ".[sstv]"
```

Requires `ffmpeg` on PATH for MP3 support.

## Usage

```
audioctf COMMAND [OPTIONS] FILE
```

### Commands

| Command | Description |
|---------|-------------|
| `info` | Basic metadata, duration, sample rate, channels, bit depth |
| `spectrogram` | Generate spectrogram PNG |
| `morse` | Detect and decode CW Morse code |
| `dtmf` | Detect DTMF tones (phone keypad) |
| `modem` | Detect FSK modem signals and decode data (Bell 103/202, AFSK) |
| `lsb` | Extract LSB steganography from audio samples |
| `sstv` | Decode SSTV image from audio signal |
| `rtty` | Decode Baudot/RTTY (radio teletype) |
| `scan` | Run ALL detectors and report findings with confidence scores |

### Examples

```bash
# Show audio file metadata
audioctf info suspicious.wav

# Generate a spectrogram image
audioctf spectrogram audio.wav -o spec.png

# Decode Morse code
audioctf morse cw_signal.wav

# Detect DTMF tones
audioctf dtmf phone_recording.wav

# Demodulate FSK modem
audioctf modem bell202_signal.wav

# Extract LSB steganography
audioctf lsb stego_audio.wav --bits 2

# Decode SSTV image
audioctf sstv sstv_signal.wav -o image.png

# Decode Baudot/RTTY
audioctf rtty rtty_signal.wav

# Run all detectors
audioctf scan unknown_audio.wav
```

## Supported Formats

- WAV, FLAC, OGG (via soundfile/libsndfile)
- MP3 (via pydub + ffmpeg)

## Supported Protocols

- **Morse Code**: CW and AM-keyed signals, ITU international code
- **DTMF**: All 16 tones (0-9, *, #, A-D)
- **Bell 103**: 300 baud FSK modem
- **Bell 202**: 1200 baud FSK modem
- **AFSK/AX.25**: 1200 baud packet radio
- **LSB Steganography**: 1-3 bit extraction with file magic detection
- **SSTV**: Martin M1, Scottie S1, Robot 36 modes
- **Baudot/RTTY**: 45.45/50 baud, standard and UHF frequencies
