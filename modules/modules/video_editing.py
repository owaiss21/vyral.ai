from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Tuple
import json, re, shutil, subprocess, tempfile, os
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="ffmpeg")

# --------------------------- Config (fallbacks if config.DEFAULT missing) ---------------------------
try:
    from config import DEFAULT as CFG
except Exception:
    class _CFG:
        verbose = True
        ffmpeg_path = "ffmpeg"
        # Canvas/output geometry
        portrait_width = 1080
        portrait_height = 1920
        # Legacy blur canvas settings (used when active crop is False)
        enable_blur_bg = True
        bg_blur_sigma = 24.0
        color_contrast = 1.08
        color_saturation = 1.12
        color_gamma = 1.03
        unsharp_luma_msize = (5, 5)
        unsharp_luma_amount = 0.70
        unsharp_chroma_msize = (5, 5)
        unsharp_chroma_amount = 0.00
        # Captions
        caption_font = "Arial"
        caption_font_emph = "Arial Black"
        caption_base_fs = 92
        caption_emph_fs = 112
        caption_min_fs = 72
        caption_margin_lr = 80
        caption_margin_v = 180
        # Encoding
        vcodec = "libx264"
        preset = "slow"
        crf = 16
        pix_fmt = "yuv420p"
        acodec = "aac"
        audio_bitrate_kbps = 192
        # Active crop control
        enable_active_crop = False
        active_crop_margin = 1.6
        active_crop_headroom = 0.12
        active_crop_smooth_alpha = 0.6
    CFG = _CFG()

VERBOSE = bool(getattr(CFG, "verbose", True))

# --------------------------- Small utils ---------------------------
_DRIVE_COLON = re.compile(r"^([A-Za-z]):/")

def ensure_dir(p: os.PathLike | str) -> Path:
    pp = Path(p)
    pp.mkdir(parents=True, exist_ok=True)
    return pp

def safe_stem(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s or "")

def _ffmpeg_sub_path_arg(sub_path: Path) -> str:
    # Convert Windows "C:/..." to "C\:/..." for FFmpeg
    p = sub_path.resolve().as_posix()
    p = _DRIVE_COLON.sub(r"\1\:/", p)
    return f"subtitles=filename='{p}'"

# --------------------------- ASS subtitles (compact & robust) ---------------------------
import re as _re
_WORD_RE = _re.compile(r"[A-Za-z0-9']+")

def _ass_time(t: float) -> str:
    t = max(0.0, float(t))
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    cs = int(round((t - int(t)) * 100))
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

def _write_clip_ass(segments: List[Dict[str, Any]], clip_start: float, clip_end: float, out_path: Path) -> List[Tuple[float, float]]:
    PLAY_W = int(getattr(CFG, "portrait_width", 1080))
    PLAY_H = int(getattr(CFG, "portrait_height", 1920))
    MARGIN_LR = int(getattr(CFG, "caption_margin_lr", 80))
    MARGIN_V  = int(getattr(CFG, "caption_margin_v", 180))
    BASE_FS = int(getattr(CFG, "caption_base_fs", 92))
    EMPH_FS = int(getattr(CFG, "caption_emph_fs", 112))
    CHAR_W_FACTOR = 0.55
    STYLE_HEADER = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {PLAY_W}\nPlayResY: {PLAY_H}\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Caption,{getattr(CFG,'caption_font','Arial')},{BASE_FS},&H00FFFFFF,&H000000FF,&H00000000,&H64000000,0,0,0,0,100,100,0,0,1,4,0,2,{MARGIN_LR},{MARGIN_LR},{MARGIN_V},0\n"
        f"Style: Emph,{getattr(CFG,'caption_font_emph','Arial Black')},{EMPH_FS},&H00FFFFFF,&H000000FF,&H00000000,&H64000000,1,0,0,0,100,100,0,0,1,4,0,2,{MARGIN_LR},{MARGIN_LR},{MARGIN_V},0\n\n"
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    # Gather words in [clip_start, clip_end]
    tokens: List[str] = []
    for seg in segments:
        s = float(seg.get("start", 0.0))
        e = float(seg.get("end", 0.0))
        if e < clip_start or s > clip_end:
            continue
        if seg.get("words"):
            for w in seg["words"]:
                ws = float(w.get("start", s)); we = float(w.get("end", e))
                if we < clip_start or ws > clip_end:
                    continue
                tokens.append((w.get("word") or w.get("text") or "").strip())
        else:
            text = (seg.get("text") or "").strip()
            if text:
                tokens += _WORD_RE.findall(text)

    text = " ".join(t for t in tokens if t).upper()
    # Simple fit by max chars per line
    avg_char_w = CHAR_W_FACTOR * BASE_FS
    SAFE_W = PLAY_W - 2 * MARGIN_LR
    max_chars = max(12, int(SAFE_W / max(1.0, avg_char_w)))
    words = text.split()
    chunks: List[List[str]] = []
    cur: List[str] = []
    cur_len = 0
    for w in words:
        add = len(w) + (1 if cur else 0)
        if cur_len + add > max_chars:
            if cur: chunks.append(cur)
            cur = [w]; cur_len = len(w)
        else:
            cur.append(w); cur_len += add
    if cur: chunks.append(cur)

    t0, t1 = float(clip_start), float(clip_end)
    T = max(0.1, t1 - t0)
    total = sum(len(c) for c in chunks) or 1

    lines: List[str] = [STYLE_HEADER]
    punch_windows: List[Tuple[float, float]] = []
    acc = 0.0
    for ch in chunks:
        frac = len(ch) / total
        dur = frac * T
        s = t0 + acc; e = s + dur; acc += dur
        # emphasize the longest token in the line
        if ch:
            idx = max(range(len(ch)), key=lambda i: len(ch[i]))
            ch_fmt = []
            for i, tok in enumerate(ch):
                if i == idx:
                    ch_fmt.append("{\\b1}" + tok + "{\\b0}")
                else:
                    ch_fmt.append(tok)
            txt = " ".join(ch_fmt)
            lines.append(f"Dialogue: 0,{_ass_time(s)},{_ass_time(e)},Caption,,0000,0000,0000,,{txt}\n")
            punch_windows.append((s, e))

    out_path.write_text("".join(lines), encoding="utf-8")
    return punch_windows

# --------------------------- Legacy blur path (no escape warnings) ---------------------------
def _build_portrait_vf() -> str:
    """
    Build a classic blurred-canvas portrait composition using gblur + overlay.
    Avoids min(h\\,w) escaping entirely.
    """
    W = int(getattr(CFG, "portrait_width", 1080))
    H = int(getattr(CFG, "portrait_height", 1920))
    sgm = float(getattr(CFG, "bg_blur_sigma", 24.0))
    ccon = float(getattr(CFG, "color_contrast", 1.08))
    csat = float(getattr(CFG, "color_saturation", 1.12))
    cgam = float(getattr(CFG, "color_gamma", 1.03))
    uLx, uLy = getattr(CFG, "unsharp_luma_msize", (5, 5))
    uLa = float(getattr(CFG, "unsharp_luma_amount", 0.70))
    uCx, uCy = getattr(CFG, "unsharp_chroma_msize", (5, 5))
    uCa = float(getattr(CFG, "unsharp_chroma_amount", 0.00))

    # Two-branch graph: blurred bg + scaled fg, then overlay + mild color/sharpen
    vf = (
        f"[0:v]split[bg][fg];"
        f"[bg]scale={W}:{H}:force_original_aspect_ratio=increase,"
        f"crop={W}:{H},gblur=sigma={sgm}[bg2];"
        f"[fg]scale={W}:-2:flags=lanczos[fg2];"
        f"[bg2][fg2]overlay=(W-w)/2:(H-h)/2,"
        f"eq=contrast={ccon}:saturation={csat}:gamma={cgam},"
        f"unsharp={uLx}:{uLy}:{uLa}:{uCx}:{uCy}:{uCa}"
    )
    return vf

# --------------------------- Public API ---------------------------
def cut_portrait_with_subs(
    input_video: str,
    transcript_segments: List[Dict[str, Any]],
    clip: Dict[str, Any],
    out_dir: str
) -> str:
    """
    Backward-compatible entry point.
    Returns the path to the non-subtitled .mp4 (subtitled variant is also written with _sub suffix).
    """
    inp = Path(input_video)
    out_dir_p = ensure_dir(out_dir)

    start = float(clip.get("start", 0.0))
    end   = float(clip.get("end", 0.0))
    if end <= start:
        raise ValueError("Invalid clip range")
    duration = end - start

    title = (clip.get("title") or "").strip()
    caption = (clip.get("caption") or "").strip()

    tmp_dir = Path(tempfile.mkdtemp(prefix="shorts_"))
    try:
        # 1) Write ASS subs for [start, end]
        ass_path = tmp_dir / f"{safe_stem(inp.stem)}_{int(round(start))}_{int(round(end))}.ass"
        _ = _write_clip_ass(transcript_segments, start, end, ass_path)

        # 2) Choose video filter according to active-crop flag
        use_active = bool(getattr(CFG, "enable_active_crop", False))
        if use_active:
            # Explicitly orchestrate each Active-Crop step (faces -> asd -> diar -> fuse -> path -> render)
            try:
                from modules.active_crop import faces as ac_faces, asd as ac_asd, diar as ac_diar, fuser as ac_fuser, path as ac_path, render as ac_render
            except Exception as e:
                raise RuntimeError(f"[active-crop:init] import failed: {e}") from e

            play_w = int(getattr(CFG, "portrait_width", 1080))
            play_h = int(getattr(CFG, "portrait_height", 1920))
            margin = float(getattr(CFG, "active_crop_margin", 1.6))
            headroom = float(getattr(CFG, "active_crop_headroom", 0.12))
            smooth_alpha = float(getattr(CFG, "active_crop_smooth_alpha", 0.6))

        # faces
        try:
            tracks, meta = ac_faces.extract_tracks(str(inp), start, end)
        except Exception as e:
            raise RuntimeError(f"[active-crop:faces.extract_tracks] {e}") from e
        if not tracks:
            raise RuntimeError("[active-crop:faces.extract_tracks] no faces tracked")

        # --- sanitize meta to avoid int(None) surprises ---
        try:
            W = int(meta.get("width") or 0)
            H = int(meta.get("height") or 0)
            fps_meta = float(meta.get("fps") or 25.0)
        except Exception as e:
            raise RuntimeError(f"[active-crop:faces.meta] invalid meta: {meta!r} ({e})") from e
        if W <= 0 or H <= 0:
            raise RuntimeError(f"[active-crop:faces.meta] invalid video dimensions: width={W}, height={H}")

        # ASD
        try:
            asd_scores = ac_asd.score_lip_motion(str(inp), tracks, start, end, fps_hint=fps_meta)
        except Exception as e:
            raise RuntimeError(f"[active-crop:asd.score_lip_motion] {e}") from e

        # VAD / diarization
        try:
            vad = ac_diar.speech_regions(str(inp), start, end)
        except Exception as e:
            raise RuntimeError(f"[active-crop:diar.speech_regions] {e}") from e
        if not vad:
            raise RuntimeError("[active-crop:diar.speech_regions] no speech detected")

        # Fuse
        try:
            fused = ac_fuser.fuse_asd_and_vad(asd_scores, vad, enter_thr=0.55, exit_thr=0.45, min_hold_ms=300)
        except Exception as e:
            raise RuntimeError(f"[active-crop:fuser.fuse_asd_and_vad] {e}") from e
        if not fused:
            raise RuntimeError("[active-crop:fuser.fuse_asd_and_vad] empty active-speaker segments")

        # Schedule
        try:
            play_w = int(getattr(CFG, "portrait_width", 1080) or 1080)
            play_h = int(getattr(CFG, "portrait_height", 1920) or 1920)
            margin = float(getattr(CFG, "active_crop_margin", 1.6))
            headroom = float(getattr(CFG, "active_crop_headroom", 0.12))
            smooth_alpha = float(getattr(CFG, "active_crop_smooth_alpha", 0.6))
        except Exception as e:
            raise RuntimeError(f"[active-crop:cfg.params] invalid CFG values ({e})") from e

        try:
            schedule = ac_path.build_crop_schedule(
                fused=fused,
                tracks=tracks,
                W_in=W,
                H_in=H,
                play_w_in=play_w,
                play_h_in=play_h,
                margin_in=margin,
                headroom_in=headroom,
                smooth_alpha_in=smooth_alpha,
            )
        except Exception as e:
            raise RuntimeError(f"[active-crop:path.build_crop_schedule] {e}") from e
        if not schedule:
            raise RuntimeError("[active-crop:path.build_crop_schedule] empty crop schedule")


        # FFmpeg crop expr
        try:
            core_vf = ac_render.build_crop_vf(schedule, play_w, play_h)
        except Exception as e:
            raise RuntimeError(f"[active-crop:render.build_crop_vf] {e}") from e


            # VAD / diarization
            try:
                vad = ac_diar.speech_regions(str(inp), start, end)
            except Exception as e:
                raise RuntimeError(f"[active-crop:diar.speech_regions] {e}") from e
            if not vad:
                raise RuntimeError("[active-crop:diar.speech_regions] no speech detected")

            # Fuse
            try:
                fused = ac_fuser.fuse_asd_and_vad(asd_scores, vad, enter_thr=0.55, exit_thr=0.45, min_hold_ms=300)
            except Exception as e:
                raise RuntimeError(f"[active-crop:fuser.fuse_asd_and_vad] {e}") from e
            if not fused:
                raise RuntimeError("[active-crop:fuser.fuse_asd_and_vad] empty active-speaker segments")

            # Schedule
            try:
                schedule = ac_path.build_crop_schedule(fused, tracks, W, H, play_w, play_h, margin, headroom, smooth_alpha)
            except Exception as e:
                raise RuntimeError(f"[active-crop:path.build_crop_schedule] {e}") from e
            if not schedule:
                raise RuntimeError("[active-crop:path.build_crop_schedule] empty crop schedule")

            # FFmpeg crop expr
            try:
                core_vf = ac_render.build_crop_vf(schedule, play_w, play_h)
            except Exception as e:
                raise RuntimeError(f"[active-crop:render.build_crop_vf] {e}") from e
        else:
            core_vf = _build_portrait_vf()

        # 3) Compose subtitles
        vf_sub = f"{core_vf},{_ffmpeg_sub_path_arg(ass_path)}"

        # 4) Output paths / encode
        base_name = f"{safe_stem(title or inp.stem)}-{safe_stem(caption) if caption else f'{int(start)}_{int(end)}'}"
        out_path = out_dir_p / f"{base_name}.mp4"
        out_path_sub = out_dir_p / f"{base_name}_sub.mp4"

        ffmpeg_bin = str(getattr(CFG, "ffmpeg_path", "ffmpeg"))
        # clean version
        cmd_nosub = [
            ffmpeg_bin, "-y",
            "-ss", f"{start:.3f}", "-i", str(inp), "-t", f"{duration:.3f}",
            "-vf", core_vf,
            "-c:v", str(getattr(CFG, "vcodec", "libx264")),
            "-preset", str(getattr(CFG, "preset", "slow")),
            "-crf", str(getattr(CFG, "crf", 16)),
            "-pix_fmt", str(getattr(CFG, "pix_fmt", "yuv420p")),
            "-c:a", str(getattr(CFG, "acodec", "aac")),
            "-b:a", f"{int(getattr(CFG, 'audio_bitrate_kbps', 192))}k",
            "-movflags", "+faststart",
            str(out_path)
        ]
        subprocess.run(cmd_nosub, check=True)

        # subtitled version
        cmd_sub = [
            ffmpeg_bin, "-y",
            "-ss", f"{start:.3f}", "-i", str(inp), "-t", f"{duration:.3f}",
            "-vf", vf_sub,
            "-c:v", str(getattr(CFG, "vcodec", "libx264")),
            "-preset", str(getattr(CFG, "preset", "slow")),
            "-crf", str(getattr(CFG, "crf", 16)),
            "-pix_fmt", str(getattr(CFG, "pix_fmt", "yuv420p")),
            "-c:a", str(getattr(CFG, "acodec", "aac")),
            "-b:a", f"{int(getattr(CFG, 'audio_bitrate_kbps', 192))}k",
            "-movflags", "+faststart",
            str(out_path_sub)
        ]
        subprocess.run(cmd_sub, check=True)

        return str(out_path)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

def render_all_clips(input_video: str, transcript: Dict[str, Any], clips: List[Dict[str, Any]], out_dir: str, verbose: bool = True) -> List[str]:
    paths: List[str] = []
    segs: List[Dict[str, Any]] = transcript.get("segments") or []
    for clip in clips:
        start = float(clip.get("start", 0.0)); end = float(clip.get("end", 0.0))
        subset = [s for s in segs if not (float(s.get("end", 0.0)) < start or float(s.get("start", 0.0)) > end)]
        p = cut_portrait_with_subs(input_video, subset, clip, out_dir)
        paths.append(p)
    return paths

def render_from_paths(video_path: str, transcript_json_path: str, plan_json_path: str, out_dir: str, verbose: bool = True) -> List[str]:
    with open(transcript_json_path, "r", encoding="utf-8") as f:
        transcript = json.load(f)
    with open(plan_json_path, "r", encoding="utf-8") as f:
        plan = json.load(f)
    return render_all_clips(video_path, transcript, plan.get("clips", []), out_dir, verbose=verbose)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Render portrait shorts; active-crop optionally enabled")
    ap.add_argument("--video", required=True)
    ap.add_argument("--transcript", required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out", default="finalized_shorts")
    ap.add_argument("--active", action="store_true")
    args = ap.parse_args()
    if args.active:
        try: setattr(CFG, "enable_active_crop", True)
        except Exception: pass
    transcript = json.loads(Path(args.transcript).read_text(encoding="utf-8"))
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    for p in render_all_clips(args.video, transcript, plan.get("clips", []), args.out, verbose=VERBOSE):
        print("short=", p)
