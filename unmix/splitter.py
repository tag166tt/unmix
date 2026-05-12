"""Module for separation of vocals and instruments."""

import logging
from pathlib import Path

from audio_separator.separator import Separator

logger = logging.getLogger(__name__)


def extract_vocals(
    audio_path: Path,
    output_dir: Path,
    output_format: str = "WAV",
    model: str = "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
) -> tuple[Path, Path]:
    """
    Extract vocals and instrumental from audio.

    Args:
        audio_path: Path to the input audio file.
        output_dir: Directory to save the separated audio files.
        output_format: Format for saving the separated audio files (default: "WAV").
        model: Path to the model checkpoint file (default: "model_bs_roformer_ep_317_sdr_12.9755.ckpt").

    Returns:
        Tuple of paths to the separated vocals and instrumental audio files.

    """
    separator = Separator(
        output_dir=output_dir,
        output_format=output_format,
    )
    separator.load_model(model)
    outputs = separator.separate(str(audio_path))

    vocals_filename = next(p for p in outputs if "Vocals" in p)
    instrumental_filename = next(p for p in outputs if "Instrumental" in p)
    vocals_path = output_dir / Path(vocals_filename).name
    instrumental_path = output_dir / Path(instrumental_filename).name
    logger.debug(f"Vocals extracted → {vocals_path}")
    logger.debug(f"Instrumental extracted → {instrumental_path}")
    return vocals_path, instrumental_path
