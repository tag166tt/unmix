"""Test script."""

import os
from pathlib import Path

from unmix import split_and_separate_voices_from_music

if __name__ == "__main__":
    diarization, files = split_and_separate_voices_from_music(
        Path("ref_audio.wav"),
        token=os.environ.get("HF_TOKEN"),
        hq_output=True,
        overlapping_speakers=False,
    )
