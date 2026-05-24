"""Voice separation module."""

import functools
import importlib.util
import inspect
from itertools import combinations
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torchaudio  # type: ignore[import-untyped]
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from pyannote.core import Annotation, Segment, Timeline

_orig_torch_load = torch.load


@functools.wraps(_orig_torch_load)
def _patched_torch_load(*args, **kwargs):  # type: ignore[no-untyped-def]
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)


torch.load = _patched_torch_load


def _patch_speechbrain_interfaces() -> None:
    spec = importlib.util.find_spec("speechbrain.inference.interfaces")
    if spec is None:
        return
    path = Path(spec.origin)  # type: ignore[arg-type]
    text = path.read_text()
    patches = [
        ('"cuda" in self.device', '"cuda" in str(self.device)'),
        ("self.device.split(", "str(self.device).split("),
        ('self.device = run_opts["device"]', 'self.device = str(run_opts["device"])'),
        ('self.device = run_opts.get("device"', 'self.device = str(run_opts.get("device"'),
        ('if self.device == "cpu":', 'if str(self.device) == "cpu":'),
        (
            'device_type=getattr(self, "device_type", str(self.device).split(":")[0]),',
            'device_type=getattr(self, "device_type", "cuda" if torch.cuda.is_available() else "cpu"),',
        ),
    ]
    changed = False
    for old, new in patches:
        if old in text:
            text = text.replace(old, new)
            changed = True
            print(f"[patch] Applied: {old!r}")
    if changed:
        path.write_text(text)
        cache_path = Path(importlib.util.cache_from_source(str(path)))
        if cache_path.exists():
            cache_path.unlink()
            print("[patch] Removed stale .pyc cache")


_patch_speechbrain_interfaces()

from speechbrain.inference.interfaces import Pretrained

_orig_pretrained_init = Pretrained.__init__
_valid_pretrained_params = set(inspect.signature(_orig_pretrained_init).parameters.keys())


@functools.wraps(_orig_pretrained_init)
def _patched_pretrained_init(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
    kwargs = {k: v for k, v in kwargs.items() if k in _valid_pretrained_params}
    if "device" in kwargs and isinstance(kwargs["device"], torch.device):
        kwargs["device"] = str(kwargs["device"])
    _orig_pretrained_init(self, *args, **kwargs)


Pretrained.__init__ = _patched_pretrained_init


def _compute_overlaps(diarization: Annotation) -> Timeline:
    """Return merged list of segments where more than one speaker is active."""
    turns = [(seg, spk) for seg, _, spk in diarization.itertracks(yield_label=True)]
    raw = []
    for (seg1, spk1), (seg2, spk2) in combinations(turns, 2):
        if spk1 == spk2:
            continue
        start = max(seg1.start, seg2.start)
        end = min(seg1.end, seg2.end)
        if start < end:
            raw.append(Segment(start, end))
    # Merge overlapping overlap-segments
    return Timeline(raw).support()


def separate_speakers(
    vocals_16k_path: Path,
    output_dir: Path,
    hq_vocals_path: Path | None = None,
    token: str | None = None,
    overlap_strategy: Literal["include", "silence", "separate"] = "include",
) -> tuple[Annotation, dict[str, str]]:
    """
    Separate vocals into individual speakers using PyAnnote's speaker diarization pipeline.

    Args:
        vocals_16k_path: Path to the vocals audio file in 16kHz format
        output_dir: Directory where separated tracks will be saved
        hq_vocals_path: Optional path to high-quality vocals audio file
        token: Optional authentication token for accessing the model
        overlap_strategy: Strategy for handling overlapping audio regions
                          'include' – overlapping audio copied to every active speaker's track
                          'silence' – overlapping regions zeroed out in all tracks
                          'separate' – same as 'include' but also writes an overlap.wav

    Returns:
        Tuple containing the diarization result and the separated speaker tracks

    """
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=token,
    )
    pipeline.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))  # type: ignore[union-attr]

    with ProgressHook() as hook:
        diarization = pipeline(vocals_16k_path, hook=hook)  # type: ignore[misc]

    # pyannote >= 4.0 returns DiarizeOutput; extract the Annotation
    if hasattr(diarization, "speaker_diarization"):
        diarization = diarization.speaker_diarization

    source_path = hq_vocals_path or vocals_16k_path
    waveform, sr = torchaudio.load(source_path)

    overlap_segments = _compute_overlaps(diarization)
    overlap_mask = torch.zeros(waveform.shape[1], dtype=torch.bool)
    for seg in overlap_segments:
        s, e = int(seg.start * sr), int(seg.end * sr)
        overlap_mask[s:e] = True

    total_samples = waveform.shape[1]
    overlap_samples = overlap_mask.sum().item()
    overlap_pct = 100 * overlap_samples / total_samples
    print(f"Overlap: {overlap_pct:.1f}% of audio ({len(overlap_segments)} regions)")

    speaker_audio = {spk: torch.zeros_like(waveform) for spk in diarization.labels()}

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        s = int(turn.start * sr)
        e = int(turn.end * sr)
        speaker_audio[speaker][:, s:e] = waveform[:, s:e]

    # Apply overlap strategy
    if overlap_strategy == "silence":
        for audio in speaker_audio.values():
            audio[:, overlap_mask] = 0.0

    output_dir = Path(output_dir)
    output_files = {}

    for speaker, audio in speaker_audio.items():
        out_path = output_dir / f"{speaker}.wav"
        torchaudio.save(str(out_path), audio, sr)
        print(f"Saved {speaker} → {out_path}")
        output_files[speaker] = str(out_path)

    if overlap_strategy == "separate" and overlap_samples > 0:
        overlap_audio = torch.zeros_like(waveform)
        overlap_audio[:, overlap_mask] = waveform[:, overlap_mask]
        out_path = output_dir / "overlap.wav"
        torchaudio.save(str(out_path), overlap_audio, sr)
        print(f"Saved overlap → {out_path}")
        output_files["overlap"] = str(out_path)

    return diarization, output_files


def separate_overlapping_speakers(
    vocals_16k_path: Path,
    output_dir: Path,
    hq_vocals_path: Path | None = None,
    token: str | None = None,
    num_speakers: int | None = None,
) -> tuple[Annotation, dict[str, str]]:
    """
    Separate overlapping vocals into individual speakers using PyAnnote's neural source separation pipeline.

    Args:
        vocals_16k_path: Path to the 16kHz vocals audio file.
        output_dir: Directory to save the separated speaker audio files.
        hq_vocals_path: Optional path to the full-quality vocals audio file for slicing.
        token: Optional authentication token for PyAnnote's model.
        num_speakers: Optional number of speakers to separate.

    Returns:
        Tuple containing the diarization result and a dictionary of output file paths.

    """
    pipeline = Pipeline.from_pretrained(
        "pyannote/speech-separation-ami-1.0",
        token=token,
    )
    pipeline.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))  # type: ignore[union-attr]

    inference_kwargs = {}
    if num_speakers is not None:
        inference_kwargs["num_speakers"] = num_speakers

    with ProgressHook() as hook:
        diarization, sources = pipeline(  # type: ignore[misc]
            vocals_16k_path,
            hook=hook,
            **inference_kwargs,  # type: ignore[arg-type]
        )

    # sources.data is (num_samples, num_speakers), float32, at 16kHz
    # speakers are ordered by first appearance in diarization.labels()
    output_dir = Path(output_dir)
    output_files = {}

    if hq_vocals_path:
        # Use the 16kHz separation as a soft mask to guide slicing of HQ audio.
        # For each speaker, we take HQ audio only where that speaker dominates
        # (their separated energy exceeds all others at each sample).
        waveform_hq, sr_hq = torchaudio.load(hq_vocals_path)
        waveform_16k, _ = torchaudio.load(vocals_16k_path)
        # num_speakers_found = sources.data.shape[1]

        # Build dominance mask at 16kHz, then upsample to HQ sample rate
        source_tensor = torch.from_numpy(sources.data).T  # (num_speakers, num_samples)
        energy = source_tensor.abs()
        dominant = energy.argmax(dim=0)  # (num_samples,) — index of loudest speaker

        for s, speaker in enumerate(diarization.labels()):
            # Create a binary mask at 16kHz where this speaker dominates
            mask_16k = (dominant == s).float().unsqueeze(0).unsqueeze(0)  # (1,1,N)
            # Upsample mask to HQ sample rate
            target_len = waveform_hq.shape[1]
            mask_hq = torch.nn.functional.interpolate(mask_16k, size=target_len, mode="nearest").squeeze(
                0
            )  # (1, target_len)

            speaker_audio = waveform_hq * mask_hq
            out_path = output_dir / f"{speaker}_hq.wav"
            torchaudio.save(str(out_path), speaker_audio, sr_hq)
            print(f"Saved {speaker} (HQ) → {out_path}")
            output_files[speaker] = str(out_path)

    else:
        # Save separated tracks directly at 16kHz
        for s, speaker in enumerate(diarization.labels()):
            audio_data = sources.data[:, s]
            # Normalise to prevent clipping
            peak = np.max(np.abs(audio_data))
            if peak > 0:
                audio_data = audio_data / peak
            audio_tensor = torch.from_numpy(audio_data).unsqueeze(0)
            out_path = output_dir / f"{speaker}.wav"
            torchaudio.save(str(out_path), audio_tensor, 16000)
            print(f"Saved {speaker} → {out_path}")
            output_files[speaker] = str(out_path)

    return diarization, output_files
