from setuptools import setup, find_packages

setup(
    name="audio-ctf",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "soundfile>=0.12.0",
        "pydub>=0.25.0",
        "matplotlib>=3.7.0",
        "click>=8.1.0",
        "Pillow>=10.0.0",
    ],
    extras_require={
        "sstv": ["pysstv>=0.4.1"],
    },
    entry_points={
        "console_scripts": [
            "audioctf=audio_ctf.cli:main",
        ],
    },
    python_requires=">=3.9",
)
