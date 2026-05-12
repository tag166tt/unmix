"""Voice separation module."""

import logging
from pathlib import Path
from typing import Literal

from pyannote.core import Annotation

from .resampler import resample_to_mono
from .splitter import extract_vocals
from .voice_separation import separate_overlapping_speakers, separate_speakers

logger = logging.getLogger(__name__)


__all__ = [
    "extract_vocals",
    "resample_to_mono",
    "separate_speakers",
    "separate_overlapping_speakers",
    "split_and_separate_voices_from_music",
]


def split_and_separate_voices_from_music(
    audio_path: Path,
    output_dir: Path = Path("./output"),
    overlapping_speakers: bool = False,
    token: str | None = None,
    hq_output: bool = True,
    overlap_strategy: Literal["include", "silence", "separate"] = "include",
    num_speakers: int | None = None,
) -> tuple[Annotation, dict[str, str]]:
    """
    Split and separate vocals from music.

    Args:
        audio_path: Path to the input audio file.
        output_dir: Directory to save the output files. Defaults to "./output".
        overlapping_speakers: Whether to handle overlapping speakers. Defaults to False.
        token: Optional token for authentication. Defaults to None.
        hq_output: Whether to generate high-quality output. Defaults to True.
        overlap_strategy: Strategy for handling overlapping speakers. Defaults to "include".
        num_speakers: Number of speakers to separate. Defaults to None.

    Returns:
        Tuple containing the diarization annotation and a dictionary of output file paths.

    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("\n[1/3] Extracting vocals...")
    vocals_path, instruments_path = extract_vocals(audio_path, output_dir)

    logger.info("\n[2/3] Resampling to mono 16kHz...")
    resampled_path = output_dir / "vocals_16k.wav"
    resample_to_mono(vocals_path, resampled_path)

    logger.info("\n[3/3] Diarizing and separating speakers...")
    if overlapping_speakers:
        diarization, output_files = separate_overlapping_speakers(
            vocals_16k_path=resampled_path,
            output_dir=output_dir,
            hq_vocals_path=vocals_path if hq_output else None,
            token=token,
            num_speakers=num_speakers,
        )
    else:
        diarization, output_files = separate_speakers(
            vocals_16k_path=resampled_path,
            output_dir=output_dir,
            hq_vocals_path=vocals_path if hq_output else None,
            token=token,
            overlap_strategy=overlap_strategy,
        )

    rttm_path = output_dir / f"{Path(audio_path).stem}.rttm"
    with open(rttm_path, "w") as f:
        diarization.write_rttm(f)
    logger.debug(f"\nTimeline saved → {rttm_path}")

    logger.debug("\n--- Timeline ---")
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        logger.debug(f"  [{turn.start:.2f}s → {turn.end:.2f}s]  {speaker}")

    return diarization, output_files
