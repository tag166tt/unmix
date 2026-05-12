# unmix

A Python library for extracting vocals from music and separating individual speakers from audio.

## Overview

`unmix` combines several state-of-the-art audio ML models into a single pipeline:

1. **Vocal extraction** — separates vocals from instrumental backing tracks
2. **Audio normalization** — resamples audio to mono 16 kHz
3. **Speaker diarization** — identifies who speaks when and writes each speaker to a separate file

The result is one WAV file per speaker, plus an RTTM file with speaker timings.

## Installation

> Requires Python 3.11–3.14 and a CUDA-capable GPU for reasonable performance.

The package is not published to PyPI.

## Quick Start

```python
from unmix import split_and_separate_voices_from_music

diarization, output_files = split_and_separate_voices_from_music(
    audio_path="recording.wav",
    output_dir="output/",
)

# output_files contains paths like:
# {"SPEAKER_00": "output/SPEAKER_00.wav", "SPEAKER_01": "output/SPEAKER_01.wav", ...}
print(diarization)
```

## API

### `split_and_separate_voices_from_music`

```python
split_and_separate_voices_from_music(
    audio_path: str | Path,
    output_dir: str | Path,
    hq: bool = False,
    overlap: Literal["include", "silence", "separate"] = "include",
    use_neural_separation: bool = False,
    num_speakers: int | None = None,
) -> tuple[Annotation, dict[str, Path]]
```

| Parameter | Description |
|-----------|-------------|
| `audio_path` | Input audio file (any format supported by `torchaudio`) |
| `output_dir` | Directory where output files are written |
| `hq` | Use the high-quality vocal track as a mask when doing neural separation |
| `overlap` | How to handle segments where multiple speakers talk simultaneously (see below) |
| `use_neural_separation` | Use neural source separation (`speech-separation-ami-1.0`) instead of standard diarization — better for overlapping speech |
| `num_speakers` | Hint to the diarization pipeline about the expected number of speakers |

**Overlap strategies:**

| Value | Behavior |
|-------|----------|
| `"include"` | Overlapping audio is included in every active speaker's track |
| `"silence"` | Overlapping regions are zeroed out in all tracks |
| `"separate"` | Overlapping regions are written to a separate `overlap.wav` file |

**Returns:** a tuple of (`pyannote.core.Annotation`, `dict[str, Path]`) — the diarization result and a mapping from speaker label to output file path.

## Models Used

| Task | Model |
|------|-------|
| Vocal/instrumental separation | `model_bs_roformer_ep_317_sdr_12.9755.ckpt` via [audio-separator](https://github.com/nomadkaraoke/python-audio-separator) |
| Speaker diarization | `pyannote/speaker-diarization-3.1` |
| Neural source separation | `pyannote/speech-separation-ami-1.0` |

## Output Files

| File | Description |
|------|-------------|
| `vocals_16k.wav` | Extracted vocals, mono 16 kHz |
| `SPEAKER_XX.wav` | Audio for each identified speaker |
| `<name>.rttm` | Speaker timeline in RTTM format |
| `overlap.wav` | Overlapping speech segments (only when `overlap="separate"`) |

## Development

```bash
# Install with uv
uv sync

# Run linters
uv run ruff check .
uv run mypy .

# Run tests
uv run pytest
```
