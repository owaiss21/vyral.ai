from __future__ import annotations
from pathlib import Path
from typing import Optional
import os

def extract_audio(video_path: str, out_dir: str, profile: str = "speech_wav") -> str:
    """Extract audio to WAV/OPUS based on profile. Returns audio path."""
    vp = Path(video_path)
    if not vp.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    op = Path(out_dir); op.mkdir(parents=True, exist_ok=True)
    stem = vp.stem

    # Prefer moviepy for portability
    try:
        from moviepy.editor import VideoFileClip
    except Exception as e:
        raise RuntimeError("moviepy is required for audio extraction. Install with: pip install moviepy") from e

    out_path = op / f"{stem}.wav"
    with VideoFileClip(str(vp)) as clip:
        if clip.audio is None:
            raise RuntimeError("No audio track found in the video.")
        clip.audio.write_audiofile(str(out_path), fps=16000, nbytes=2, codec="pcm_s16le", verbose=False, logger=None)
    return str(out_path.resolve())
