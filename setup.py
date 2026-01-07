"""Setup script for meeting-analysis-gpu-server."""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="meeting-analysis-gpu-server",
    version="0.1.0",
    author="Sean",
    description="GPU processing server for meeting audio analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/spiritnzsp/meeting-analysis-gpu-server",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "websockets>=12.0",
        "pyyaml>=6.0",
        "numpy>=1.24.0",
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "faster-whisper>=1.0.0",
        "pyannote.audio>=3.1.0",
        "argon2-cffi>=23.1.0",  # Secure password/API key hashing
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "gpu-server=gpu_server.__main__:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
