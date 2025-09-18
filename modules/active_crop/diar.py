from __future__ import annotations
from typing import List, Tuple
import subprocess, tempfile, os
import numpy as np
import torch
from scipy.io import wavfile

def _ffmpeg_extract_wav_16k_mono(video_path: str, t0: float, t1: float) -> str:
    dur = max(0.01, float(t1) - float(t0))
    tmp_wav = tempfile.NamedTemporaryFile(prefix="vad_", suffix=".wav", delete=False)
    tmp_wav.close()
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{t0:.3f}", "-t", f"{dur:.3f}",
        "-i", video_path,
        "-ac", "1", "-ar", "16000", "-vn",
        tmp_wav.name,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return tmp_wav.name

def speech_regions(video_path: str, t0: float, t1: float) -> List[Tuple[float, float]]:
    """
    Return a list of (start, end) seconds with active speech using Silero VAD.
    Raises RuntimeError if Silero cannot be loaded.
    """
    try:
        model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', trust_repo=True)
        (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
    except Exception as e:
        raise RuntimeError("silero-vad not available via torch.hub; ensure internet access or cached hub") from e

    wav_path = _ffmpeg_extract_wav_16k_mono(video_path, float(t0), float(t1))
    try:
        sr, data = wavfile.read(wav_path)
        if sr != 16000:
            raise RuntimeError(f"unexpected sample rate {sr}")
        if data.dtype.kind == 'i':
            audio = torch.from_numpy(data.astype(np.float32) / 32768.0)
        elif data.dtype.kind == 'f':
            audio = torch.from_numpy(data.astype(np.float32))
        else:
            raise RuntimeError(f"unsupported dtype: {data.dtype}")
        if audio.dim() > 1:
            audio = audio.mean(dim=1)
        model.eval()
        ts = get_speech_timestamps(audio, model, sampling_rate=16000)
        out: List[Tuple[float, float]] = []
        for seg in ts:
            s = float(seg['start']) / 16000.0 + float(t0)
            e = float(seg['end']) / 16000.0 + float(t0)
            if e > s:
                out.append((s, e))
        return out
    finally:
        try: os.remove(wav_path)
        except Exception: pass
