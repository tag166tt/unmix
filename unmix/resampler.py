"""Utilities to resample audio files."""

import logging
from pathlib import Path

import torchaudio  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


def resample_to_mono(input_path: Path, output_path: Path, sample_rate: int = 16000) -> Path:
    """
    Resample an audio file to mono and specified sample rate.

    Args:
        input_path: Path to the input audio file.
        output_path: Path to save the resampled audio file.
        sample_rate: Target sample rate for resampling. Default is 16,000 Hz.

    Returns:
        Path: Path to the resampled audio file.

    """
    waveform, sr = torchaudio.load(input_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
    torchaudio.save(output_path, waveform, sample_rate)
    logger.debug(f"Resampled → {output_path}")
    return output_path
