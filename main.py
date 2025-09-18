from __future__ import annotations
import argparse, os
from pathlib import Path
from typing import List, Optional, Tuple

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

STAGES = ("download","extract","transcribe","transliterate","plan","render")

def _assert_file(p: Path, label: str) -> None:
    if not p.exists() or p.stat().st_size == 0:
        raise FileNotFoundError(f"{label} not found or empty: {p}")

def _log(msg: str) -> None:
    if getattr(CFG, "verbose", True):
        print(msg)

def run_pipeline(*, start: str, workdir: str,
                 url: Optional[str] = None,
                 video_path: Optional[str] = None,
                 quality: int = 1080,
                 device: Optional[str] = None,
                 force_non_english_to: Optional[str] = None,
                 num_clips: int = 5,
                 llm_model: Optional[str] = None,
                 transcript: Optional[str] = None,
                 plan: Optional[str] = None,
                 audio: Optional[str] = None) -> List[str]:
    device = device or getattr(CFG, "device", "auto")
    model_name = llm_model or getattr(CFG, "openai_model", "gpt-4o-mini")
    conditional_force = force_non_english_to if getattr(CFG, "enable_conditional_force_non_en", True) else None

    wd = Path(workdir)
    vids_dir = ensure_dir(wd / "raw_videos")
    audio_dir = ensure_dir(wd / "raw_audios")
    tran_dir = ensure_dir(wd / "transcriptions")
    translit_dir = ensure_dir(wd / "transliteration")
    plans_dir = ensure_dir(wd / "llm_shorts")
    shorts_dir = ensure_dir(wd / "finalized_shorts")

    if start not in STAGES:
        raise ValueError(f"Unknown start stage: {start}")

    # Stage: download (produces video_path)
    if start == "download":
        _log("[PIPE] Stage: download")
        if url and url.lower().startswith(("http://","https://")):
            video_path = download_youtube_video(url, str(vids_dir), quality=quality)
        elif video_path:
            vp = Path(video_path)
            if not vp.exists():
                raise FileNotFoundError(f"Video not found: {video_path}")
            video_path = str(vp.resolve())
        else:
            raise ValueError("Provide --url or --video for start=download")

    # If later stages need video, validate/provide
    if start in ("extract","transcribe","render"):
        if not video_path:
            raise ValueError(f"--video is required for start={start}")
        vp = Path(video_path)
        if not vp.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        video_path = str(vp.resolve())
    video_stem = Path(video_path).stem if video_path else None

    # Stage: extract (produces audio_path)
    audio_path = None
    if start in ("download","extract"):
        _log("[PIPE] Stage: extract")
        if not video_path:
            raise FileNotFoundError("Video path missing for extract stage.")
        audio_path = extract_audio(video_path, str(audio_dir), profile=getattr(CFG, "audio_profile", "speech_wav"))
        _assert_file(Path(audio_path), "Audio")
    elif start == "transcribe":
        if audio and Path(audio).exists():
            audio_path = str(Path(audio).resolve())
        else:
            if not video_stem:
                raise FileNotFoundError("No --audio provided and cannot infer from --video.")
            cand = audio_dir / f"{video_stem}.wav"
            if cand.exists():
                audio_path = str(cand.resolve())
            else:
                raise FileNotFoundError("Audio path missing. Run with start=extract (or download) first to create the WAV, or provide --audio explicitly.")
        _assert_file(Path(audio_path), "Audio")

    # Stage: transcribe (produces transcript_path)
    transcript_path = None
    if start in ("download","extract","transcribe"):
        _log("[PIPE] Stage: transcribe")
        if not audio_path:
            raise FileNotFoundError("Audio path missing for transcription.")
        transcript_path, _ = transcribe_audio(
            audio_path, str(tran_dir),
            model_size=getattr(CFG, "whisper_model", "medium"),
            device=device,
            word_timestamps=getattr(CFG, "word_timestamps", True),
            force_language=None,
            conditional_force_non_en=conditional_force,
            english_codes=getattr(CFG, "english_codes", ("en",)),
            verbose=getattr(CFG, "verbose", True),
        )
        _assert_file(Path(transcript_path), "Transcript JSON")
    else:
        if transcript and Path(transcript).exists():
            transcript_path = str(Path(transcript).resolve())
        else:
            if video_stem:
                cand = tran_dir / f"{video_stem}.json"
                if cand.exists():
                    transcript_path = str(cand.resolve())
        if not transcript_path:
            raise FileNotFoundError("Transcript JSON not found. Provide --transcript or start earlier.")

    # Stage: transliterate (produces translit_path)
    translit_path = None
    if start in ("download","extract","transcribe","transliterate"):
        _log("[PIPE] Stage: transliterate")
        translit_path = transliterate_transcript(transcript_path, out_dir=str(translit_dir), model=model_name)
        _assert_file(Path(translit_path), "Transliteration JSON")
    else:
        cand = Path(transcript_path).with_name(Path(transcript_path).stem + "_roman.json")
        if cand.exists():
            translit_path = str(cand.resolve())
        else:
            # derive by workdir
            if video_stem:
                cand2 = translit_dir / f"{video_stem}_roman.json"
                translit_path = str(cand2.resolve()) if cand2.exists() else None
        if not translit_path:
            raise FileNotFoundError("Transliteration JSON not found. Provide --transcript with *_roman.json or start earlier.")

    # Stage: plan (produces plan_json)
    plan_json = None
    if start in ("download","extract","transcribe","transliterate","plan"):
        _log("[PIPE] Stage: plan")
        plan_json = propose_plan_file(
            translit_path, str(plans_dir),
            num_clips=int(num_clips or getattr(CFG, "num_clips", 5)),
            target_duration=getattr(CFG, "target_duration", 35),
            model=model_name
        )
        _assert_file(Path(plan_json), "Plan JSON")
    else:
        if plan and Path(plan).exists():
            plan_json = str(Path(plan).resolve())
        else:
            if video_stem:
                cand = plans_dir / f"{video_stem}_roman_plan.json"
                if cand.exists():
                    plan_json = str(cand.resolve())
        if not plan_json:
            raise FileNotFoundError("Plan JSON not found. Provide --plan or start earlier.")

    # Stage: render (consumes video+translit+plan)
    _log("[PIPE] Stage: render")
    if not video_path:
        # Try to infer from translit/transcript stem or latest in raw_videos
        guess = None
        stems = []
        try:
            if translit_path:
                stems.append(Path(translit_path).stem.replace("_roman",""))
            if transcript_path:
                stems.append(Path(transcript_path).stem.replace("_roman",""))
        except Exception:
            pass
        for stem in stems:
            for ext in (".mp4",".mkv",".mov",".webm"):
                p = vids_dir / f"{stem}{ext}"
                if p.exists():
                    guess = str(p.resolve()); break
            if guess: break
        if not guess:
            vids = sorted([p for p in vids_dir.glob("*.*") if p.suffix.lower() in (".mp4",".mkv",".mov",".webm")],
                          key=lambda p: p.stat().st_mtime, reverse=True)
            if vids:
                guess = str(vids[0].resolve())
        if not guess:
            raise FileNotFoundError("--video is required for start=render (unable to infer a source video).")
        video_path = guess

    out_paths = render_from_paths(video_path, translit_path, plan_json, str(shorts_dir), verbose=getattr(CFG, "verbose", True))
    return out_paths

def _parse():
    ap = argparse.ArgumentParser(description="Run pipeline start-to-end or from any stage.")
    ap.add_argument("--start", choices=STAGES, default="download")
    ap.add_argument("--workdir", default="workdir")
    ap.add_argument("--url")
    ap.add_argument("--video")
    ap.add_argument("--quality", type=int, default=1080)
    ap.add_argument("--device", default=None)
    ap.add_argument("--force-non-en-to", default=None)
    ap.add_argument("--num-clips", type=int, default=getattr(CFG, "num_clips", 5))
    ap.add_argument("--llm-model", default=None)
    ap.add_argument("--transcript", help="Path to transcript JSON (pre- or post-transliteration)", default=None)
    ap.add_argument("--plan", help="Path to an existing plan JSON (skip planning)", default=None)
    ap.add_argument("--audio", help="Path to an existing WAV/Opus to skip extract stage for transcribe", default=None)
    return ap.parse_args()

if __name__ == "__main__":
    args = _parse()
    paths = run_pipeline(
        start=args.start, workdir=args.workdir,
        url=args.url, video_path=args.video, quality=args.quality,
        device=args.device, force_non_english_to=args.force_non_en_to,
        num_clips=args.num_clips, llm_model=args.llm_model,
        transcript=args.transcript, plan=args.plan, audio=args.audio,
    )
    for p in (paths or []):
        print(p)
