from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Tuple
import json

from modules.utils import ensure_dir
from modules.downloader import download_youtube_video
from modules.extract_audio import extract_audio
from modules.transcribe import transcribe_audio
from modules.transliteration import transliterate_transcript
from modules.llm_clipper import propose_plan_file
from modules.video_editing import render_from_paths

try:
    from config import DEFAULT as CFG
except Exception:
    class _CFG:
        whisper_model = "medium"
        device = "auto"
        word_timestamps = True
        enable_conditional_force_non_en = True
        force_non_english_to = "ur"
        english_codes = ("en",)
        num_clips = 5
        target_duration = 35
        openai_model = "gpt-4o-mini"
        audio_profile = "speech_wav"
        verbose = True
    CFG = _CFG()

def run_full_pipeline(url_or_path: str, workdir: str = "workdir", quality: int = 1080) -> List[str]:
    wd = Path(workdir)
    vids = ensure_dir(wd / "raw_videos")
    auds = ensure_dir(wd / "raw_audios")
    trans = ensure_dir(wd / "transcriptions")
    translit = ensure_dir(wd / "transliteration")
    plans = ensure_dir(wd / "llm_shorts")
    shorts = ensure_dir(wd / "finalized_shorts")

    # 1) Video
    if url_or_path.lower().startswith(("http://","https://")):
        video_path = download_youtube_video(url_or_path, str(vids), quality=quality)
    else:
        vp = Path(url_or_path)
        if not vp.exists():
            raise FileNotFoundError(f"Video not found: {vp}")
        video_path = str(vp.resolve())

    # 2) Audio
    audio_path = extract_audio(video_path, str(auds), profile=getattr(CFG, "audio_profile", "speech_wav"))

    # 3) Transcribe
    transcript_path, _ = transcribe_audio(
        audio_path, str(trans),
        model_size=getattr(CFG, "whisper_model", "medium"),
        device=getattr(CFG, "device", "auto"),
        word_timestamps=getattr(CFG, "word_timestamps", True),
        force_language=getattr(CFG, "force_language", None),
        conditional_force_non_en=(getattr(CFG, "force_non_english_to", None) if getattr(CFG, "enable_conditional_force_non_en", True) else None),
        english_codes=getattr(CFG, "english_codes", ("en",)),
        verbose=getattr(CFG, "verbose", True),
    )

    # 4) Transliterate
    translit_path = transliterate_transcript(transcript_path, out_dir=str(translit), model=getattr(CFG, "openai_model", "gpt-4o-mini"))

    # 5) Plan
    plan_json = propose_plan_file(
        translit_path, str(plans),
        num_clips=getattr(CFG, "num_clips", 5),
        target_duration=getattr(CFG, "target_duration", 35),
        model=getattr(CFG, "openai_model", "gpt-4o-mini"),
    )

    # 6) Render
    out_paths = render_from_paths(video_path, translit_path, plan_json, str(shorts), verbose=getattr(CFG, "verbose", True))
    return out_paths

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="YouTube URL or local video path")
    ap.add_argument("--workdir", default="workdir")
    ap.add_argument("--quality", type=int, default=1080)
    args = ap.parse_args()
    for p in (run_full_pipeline(args.src, workdir=args.workdir, quality=args.quality) or []):
        print(p)
