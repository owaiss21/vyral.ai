from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Tuple
import json, re, shutil, subprocess, tempfile, time, os

# --------------------------- Config + utils ---------------------------
try:
    from config import DEFAULT as CFG
except Exception:
    class _CFG:
        verbose = True
        # Paths / tools
        ffmpeg_path = "ffmpeg"                  # override if ffmpeg is not on PATH
        # Canvas
        portrait_width = 1080
        portrait_height = 1920
        # Blur bg + color tweak
        enable_blur_bg = True
        bg_blur_sigma = 24.0
        color_contrast = 1.08
        color_saturation = 1.12
        color_gamma = 1.03
        unsharp_luma_msize = (5, 5)
        unsharp_luma_amount = 0.70
        unsharp_chroma_msize = (5, 5)
        unsharp_chroma_amount = 0.00
        # Subtitles (ASS) – font + layout
        caption_font = "Arial"
        caption_font_emph = "Arial Black"
        caption_base_fs = 92
        caption_emph_fs = 112
        caption_min_fs = 72
        caption_margin_lr = 80
        caption_margin_v = 180
        caption_border = 10
        caption_shadow = 6
        # Word chunking (kept for back-compat; max is still respected)
        chunk_min_words = 3
        chunk_max_words = 4
        # Punch (contrast/saturation boost) around emphasis (disabled now)
        enable_dynamic_punch = False
        punch_contrast = 1.22
        punch_saturation = 1.25
        punch_gamma = 1.01
        punch_unsharp = (7, 7, 1.00, 7, 7, 0.00)
        # Encoding
        vcodec = "libx264"
        preset = "slow"
        crf = 16
        pix_fmt = "yuv420p"
        acodec = "aac"
        audio_bitrate_kbps = 192
    CFG = _CFG()

try:
    from modules.utils import ensure_dir, safe_stem, step
except Exception:
    # Minimal fallbacks
    def ensure_dir(p: os.PathLike | str) -> Path:
        pp = Path(p); pp.mkdir(parents=True, exist_ok=True); return pp
    def safe_stem(s: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    def step(msg: str, verbose: bool = True) -> None:
        if verbose: print(msg)

VERBOSE = bool(getattr(CFG, "verbose", True))

# --------------------------- Tokenization helpers ---------------------------
_WORD_RE = re.compile(r"[A-Za-z0-9']+")

_STOPWORDS = {
    "the","a","an","and","or","but","so","to","of","in","on","for","with","at","by","from",
    "this","that","these","those","is","are","was","were","be","been","am","i","you","he",
    "she","it","we","they","me","him","her","them","my","your","his","its","our","their",
    "as","if","then","than","not","no","yes","do","does","did","done","have","has","had",
    "can","could","will","would","shall","should","may","might","about","into","over","under",
    "up","down","out","off","just","now","well","also","very","really","more","most","some",
    "any","much","many","there","here","what","which","when","where","who","whom","why","how"
}

def _tokenize(text: str) -> List[str]:
    return _WORD_RE.findall(text or "")

def _extract_words(seg: Dict[str, Any]) -> List[Tuple[str, float, float]]:
    """
    Return [(word, start, end)] for a segment.
    Prefer precise per-word timings in seg['words'].
    """
    s0, e0 = float(seg.get("start", 0.0)), float(seg.get("end", 0.0))
    if not seg.get("words"):
        # Fallback only if words missing (kept for backward compatibility).
        txt = (seg.get("text") or "").strip()
        words = _tokenize(txt)
        if not words or e0 <= s0:
            return []
        dur = e0 - s0
        step_dur = dur / len(words)
        return [(w, s0 + i * step_dur, s0 + (i + 1) * step_dur) for i, w in enumerate(words)]

    out: List[Tuple[str, float, float]] = []
    for w in seg["words"]:
        wtxt = (w.get("word") or w.get("text") or "").strip()
        if not wtxt:
            continue
        ws = float(w.get("start", s0)); we = float(w.get("end", s0))
        ws = max(s0, ws); we = min(e0, we)
        if we > ws:
            out.append((wtxt, ws, we))
    return out

def _split_into_chunks_gap_aware(
    words: List[Tuple[str, float, float]],
    min_sz: int,
    max_sz: int,
    gap_thr: float = 0.22  # seconds; start a new chunk when gap > this
) -> List[List[Tuple[str, float, float]]]:
    """
    Build chunks that never bridge across pauses (gaps) larger than gap_thr.
    Also respects max_sz to avoid overly long lines; min_sz is best-effort.
    """
    chunks: List[List[Tuple[str, float, float]]] = []
    if not words:
        return chunks

    cur: List[Tuple[str, float, float]] = []
    last_end = None
    for triple in words:
        w, ws, we = triple
        if last_end is None:
            cur = [triple]
        else:
            gap = ws - last_end
            if gap > gap_thr or (cur and len(cur) >= max_sz):
                chunks.append(cur)
                cur = [triple]
            else:
                cur.append(triple)
        last_end = we

    if cur:
        chunks.append(cur)

    # If any chunk is smaller than min_sz but got isolated by a big gap, keep it as-is
    return chunks

def _pick_emphasis_index(words_only: List[str]) -> int:
    best_idx, best_len = -1, -1
    for i, w in enumerate(words_only):
        if w.lower() in _STOPWORDS:
            continue
        L = len(w)
        if L > best_len:
            best_len = L
            best_idx = i
    if best_idx == -1:
        best_idx = max(0, len(words_only) - 1)
    return best_idx

# --------------------------- ASS subtitle generation ---------------------------
PLAY_W = int(getattr(CFG, "portrait_width", 1080))
PLAY_H = int(getattr(CFG, "portrait_height", 1920))
MARGIN_LR = int(getattr(CFG, "caption_margin_lr", 80))
MARGIN_V  = int(getattr(CFG, "caption_margin_v", 180))
SAFE_W = PLAY_W - 2 * MARGIN_LR

# Move captions upward by additional 10% of canvas height
MARGIN_V_EFF = MARGIN_V + int(0.10 * PLAY_H)

BASE_FS = int(getattr(CFG, "caption_base_fs", 92))
EMPH_FS = int(getattr(CFG, "caption_emph_fs", 112))
MIN_FS  = int(getattr(CFG, "caption_min_fs", 72))

CHAR_W_FACTOR = 0.55  # rough per-character width factor for Arial

def _estimate_px(words: List[str], fs_norm: float, fs_emph: float, emph_idx: int) -> float:
    px = 0.0
    for i, w in enumerate(words):
        fs = fs_emph if i == emph_idx else fs_norm
        px += (len(w) * CHAR_W_FACTOR + (0.33 if i != 0 else 0.0)) * fs
    return px

def _format_chunk_line(words: List[str], emphasize_idx: int) -> str:
    # ALL CAPS visually
    words_vis = [w.upper() for w in words]

    fs_norm, fs_emph = BASE_FS, EMPH_FS
    est = _estimate_px(words_vis, fs_norm, fs_emph, emphasize_idx)
    if est > SAFE_W:
        scale = max(SAFE_W / est, MIN_FS / fs_norm)
        fs_norm = int(max(MIN_FS, fs_norm * scale))
        fs_emph = int(max(MIN_FS + 10, fs_emph * scale))

    # re-check after scaling; if still too wide, split into 2 lines
    est2 = _estimate_px(words_vis, fs_norm, fs_emph, emphasize_idx)
    if est2 > SAFE_W * 1.05 and len(words_vis) >= 4:
        words_vis = [words_vis[0] + " " + words_vis[1], " ".join(words_vis[2:])]
        emphasize_idx = 0 if len(words_vis[0]) >= len(words_vis[1]) else 1

    font  = str(getattr(CFG, "caption_font", "Arial"))
    fontB = str(getattr(CFG, "caption_font_emph", "Arial Black"))
    bord  = int(getattr(CFG, "caption_border", 10))
    shad  = int(getattr(CFG, "caption_shadow", 6))

    base_open = rf"{{\an2\q2\bord{bord}\shad{shad}\fad(60,60)\fn{font}\fs{fs_norm}\1c&HFFFFFF&}}"
    emph_on   = rf"{{\b1\fn{fontB}\fs{fs_emph}\1c&H00FFFF&}}"   # yellow emphasis
    emph_off  = rf"{{\b0\fn{font}\fs{fs_norm}\1c&HFFFFFF&}}"

    parts: List[str] = [base_open]

    # If we split into two lines manually, join with \N
    if isinstance(words_vis, list) and len(words_vis) == 2 and " " in words_vis[0] and " " in words_vis[1]:
        halves = words_vis
        idx = 0 if len(halves[0]) >= len(halves[1]) else 1
        for j, phrase in enumerate(halves):
            if j == idx:
                parts.append(emph_on + phrase + emph_off)
            else:
                parts.append(phrase)
            if j == 0:
                parts.append(r"\N")
        return "".join(parts)

    for i, w in enumerate(words_vis):
        if i == emphasize_idx:
            parts.append(emph_on + w + emph_off)
        else:
            parts.append(w)
        if i != len(words_vis) - 1:
            parts.append(" ")
    return "".join(parts)

def _ass_header() -> str:
    font  = str(getattr(CFG, "caption_font", "Arial"))
    fontB = str(getattr(CFG, "caption_font_emph", "Arial Black"))
    return (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {PLAY_W}\n"
        f"PlayResY: {PLAY_H}\n"
        "ScaledBorderAndShadow: yes\n"
        "Collisions: Normal\n"
        "WrapStyle: 2\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Caption, {font}, {BASE_FS}, &H00FFFFFF, &H00FFFFFF, &H00101010, &H80202020, "
        "1,0,0,0,100,100,0,0,1,10,5,2, "
        f"{MARGIN_LR},{MARGIN_LR},{MARGIN_V_EFF},0\n"
        f"Style: Emph, {fontB}, {EMPH_FS}, &H0000FFFF, &H00FFFFFF, &H00101010, &H80202020, "
        "1,0,0,0,100,100,0,0,1,12,6,2, "
        f"{MARGIN_LR},{MARGIN_LR},{MARGIN_V_EFF},0\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

def _ass_time(t: float) -> str:
    if t < 0: t = 0
    h = int(t // 3600); m = int((t % 3600) // 60); s = int(t % 60)
    cs = int(round((t - int(t)) * 100))
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"

def _write_clip_ass(
    transcript_segments: List[Dict[str, Any]],
    clip_start: float,
    clip_end: float,
    ass_path: Path
) -> List[Tuple[float, float]]:
    """
    Create ASS captions aligned to word timing (if available).
    Chunks are split at pauses so text never lingers during silences.
    """
    all_words: List[Tuple[str, float, float]] = []
    for seg in transcript_segments:
        for (w, ws, we) in _extract_words(seg):
            if we <= clip_start or ws >= clip_end:
                continue
            all_words.append((w, max(ws, clip_start), min(we, clip_end)))

    lines = [_ass_header()]
    punch_windows: List[Tuple[float, float]] = []

    if not all_words:
        ass_path.write_text("".join(lines), encoding="utf-8")
        return punch_windows

    # Sort words by time and split by gaps
    all_words.sort(key=lambda x: x[1])
    chunks_raw = _split_into_chunks_gap_aware(
        all_words,
        int(getattr(CFG, "chunk_min_words", 3)),
        int(getattr(CFG, "chunk_max_words", 4)),
        gap_thr=0.22
    )

    for ch in chunks_raw:
        words = [w for (w, _, _) in ch]
        # Use exact window: from first word start to last word end
        t0 = ch[0][1] - clip_start
        t1 = ch[-1][2] - clip_start
        if t1 <= t0:
            continue

        emp_idx = _pick_emphasis_index(words)
        line_text = _format_chunk_line(words, emp_idx)

        start = _ass_time(max(0.0, t0))
        end   = _ass_time(t1)
        lines.append(f"Dialogue: 0,{start},{end},Caption,,0000,0000,0000,,{line_text}\n")

    ass_path.write_text("".join(lines), encoding="utf-8")
    return punch_windows

# --------------------------- FFmpeg filter helpers ---------------------------
_DRIVE_COLON = re.compile(r"^([A-Za-z]):/")

def _subtitles_filter_arg(sub_path: Path) -> str:
    p = sub_path.resolve().as_posix()
    p = _DRIVE_COLON.sub(r"\1\:/" , p)
    return f"subtitles=filename='{p}'"

def _build_portrait_vf() -> str:
    """
    1080x1920 canvas with blurred background and centered foreground; mild enhancement.
    """
    W = int(getattr(CFG, "portrait_width", 1080))
    H = int(getattr(CFG, "portrait_height", 1920))

    if not bool(getattr(CFG, "enable_blur_bg", True)):
        # Basic letterbox portrait without heavy blur
        return (
            f"scale={W}:-2:flags=lanczos,"
            f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:black,"
            f"eq=contrast={getattr(CFG,'color_contrast',1.08)}:"
            f"saturation={getattr(CFG,'color_saturation',1.12)}:"
            f"gamma={getattr(CFG,'color_gamma',1.03)}"
        )

    sgm = float(getattr(CFG, "bg_blur_sigma", 24.0))
    uLx, uLy = getattr(CFG, "unsharp_luma_msize", (5,5))
    uLa      = float(getattr(CFG, "unsharp_luma_amount", 0.70))
    uCx, uCy = getattr(CFG, "unsharp_chroma_msize", (5,5))
    uCa      = float(getattr(CFG, "unsharp_chroma_amount", 0.00))

    return (
        f"split[bg][fg];"
        f"[bg]scale={W}:{H}:force_original_aspect_ratio=increase,"
            f"crop={W}:{H},"
            f"gblur=sigma={sgm}[bg];"
        f"[fg]scale={W}:-2:flags=lanczos[fgs];"
        f"[bg][fgs]overlay=(W-w)/2:(H-h)/2,"
            f"eq=contrast={getattr(CFG,'color_contrast',1.08)}:"
            f"saturation={getattr(CFG,'color_saturation',1.12)}:"
            f"gamma={getattr(CFG,'color_gamma',1.03)},"
            f"unsharp={uLx}:{uLy}:{uLa}:{uCx}:{uCy}:{uCa}"
    )

def _append_punch(vf: str, punch_windows: List[Tuple[float, float]]) -> str:
    """
    Previously applied time-windowed contrast/saturation boosts for emphasized words.
    Now explicitly disabled per requirements.
    """
    return vf

# --------------------------- Public API ---------------------------
def cut_portrait_with_subs(
    input_video: str,
    transcript_segments: List[Dict[str, Any]],
    clip: Dict[str, Any],
    out_dir: str
) -> str:
    """
    Cut [start,end], convert to 1080x1920 with blurred background (no crop),
    burn dynamic-fit CAPS captions aligned to word timing, export high quality.

    NOW: Saves two outputs:
      - <base>.mp4            (no subtitles)
      - <base>_sub.mp4        (with subtitles)
    Returns the path to the non-subtitled file (no other behavior changed).
    """
    out_dir_p = ensure_dir(out_dir)
    inp = Path(input_video).resolve()

    start = float(clip["start"]); end = float(clip["end"])
    duration = max(0.1, end - start)

    # temp ASS
    tmp_dir = Path(tempfile.mkdtemp(prefix="subs_"))
    try:
        # ASS name (temp)
        ass_name = f"{safe_stem(inp.stem)}_{int(round(start))}_{int(round(end))}.ass"
        ass_path = tmp_dir / ass_name
        punch_windows = _write_clip_ass(transcript_segments, start, end, ass_path)

        core_vf = _build_portrait_vf()  # base portrait pipeline (no subs)
        vf_sub = f"{core_vf},{_subtitles_filter_arg(ass_path)}"
        vf_sub = _append_punch(vf_sub, punch_windows)  # no-op now

        # Final filename pattern: title_with_underscores-caption_with_underscores.mp4
        title = (clip.get("title") or "").strip()
        caption = (clip.get("caption") or "").strip()
        title_s = safe_stem(title.replace(" ", "_")) if title else safe_stem(inp.stem)
        caption_s = safe_stem(caption.replace(" ", "_")) if caption else f"{int(round(start))}_{int(round(end))}"
        base_name = f"{title_s}-{caption_s}"

        # Paths: no-subs (original name) and subs (with _sub suffix)
        out_path = out_dir_p / f"{base_name}.mp4"
        out_path_sub = out_dir_p / f"{base_name}_sub.mp4"

        ffmpeg_bin = str(getattr(CFG, "ffmpeg_path", "ffmpeg"))

        # 1) Render WITHOUT subtitles (unchanged filename)
        cmd_nosub = [
            ffmpeg_bin, "-y",
            "-ss", f"{start:.3f}",
            "-i", str(inp),
            "-t", f"{duration:.3f}",
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

        # 2) Render WITH subtitles (adds _sub suffix)
        cmd_sub = [
            ffmpeg_bin, "-y",
            "-ss", f"{start:.3f}",
            "-i", str(inp),
            "-t", f"{duration:.3f}",
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

        # Keep return value identical to previous behavior (no-subs path)
        return str(out_path)
    finally:
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass


def render_all_clips(video_path: str,
                     transcript: Dict[str, Any],
                     clips: List[Dict[str, Any]],
                     out_dir: str,
                     verbose: bool | None = None):
    """
    Render each planned clip; captions come from the (transliterated) transcript's word timings.
    """
    if verbose is None:
        verbose = bool(getattr(CFG, "verbose", True))

    os.makedirs(out_dir, exist_ok=True)

    total = len(clips)
    if total == 0:
        step("[RND] No clips to render.", verbose)
        return []

    step(f"[RND] Rendering {total} clips → {out_dir}", verbose)
    t0 = time.time()
    done_paths: List[str] = []

    segs = list(transcript.get("segments") or [])
    if not segs:
        raise RuntimeError("Transcript has no segments; cannot create word-timed subtitles.")

    for idx, clip in enumerate(clips, start=1):
        start = float(clip.get("start", 0.0))
        end   = float(clip.get("end", 0.0))
        title = (clip.get("title") or "")[:60]
        step(f"[RND] clip {idx}/{total}  {start:.2f}s → {end:.2f}s  title='{title}'", verbose)

        try:
            out_path = cut_portrait_with_subs(video_path, segs, clip, out_dir)
        except subprocess.CalledProcessError as e:
            step(f"[RND] ffmpeg failed on clip {idx}: {e}", True)
            continue

        done_paths.append(out_path)

        # coarse across-clips ETA
        elapsed_total = time.time() - t0
        avg = elapsed_total / idx
        eta = avg * (total - idx)
        step(f"[RND] progress {idx}/{total}  elapsed={elapsed_total:5.1f}s  eta={eta:5.1f}s", verbose)

    step(f"[RND] Completed {len(done_paths)}/{total} clips. Total time: {time.time()-t0:5.1f}s", verbose)
    return done_paths

# --------------------------- CLI ---------------------------

# === Added for coherent path-based orchestration ===
def render_from_paths(video_path: str,
                      transcript_json_path: str,
                      plan_json_path: str,
                      out_dir: str,
                      verbose: bool = True) -> list[str]:
    """Load transcript and plan from their paths, render all clips, and return output paths."""
    with open(transcript_json_path, "r", encoding="utf-8") as f:
        transcript = json.load(f)
    with open(plan_json_path, "r", encoding="utf-8") as f:
        pj = json.load(f)
    clips = pj.get("clips", [])
    return render_all_clips(video_path, transcript, clips, out_dir, verbose=verbose)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Render portrait shorts with dynamic-fit captions (word-timed) and blurred background")
    ap.add_argument("--video", required=True, help="Path to source video")
    ap.add_argument("--transcript", required=True, help="Path to transliterated transcript JSON (…_roman.json)")
    ap.add_argument("--plan", required=True, help="Path to clip plan JSON (expects 'title' and 'caption' per clip)")
    ap.add_argument("--out", default="finalized_shorts", help="Output folder for shorts")
    args = ap.parse_args()

    transcript = json.loads(Path(args.transcript).read_text(encoding="utf-8"))
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    clips = plan.get("clips", [])
    paths = render_all_clips(args.video, transcript, clips, args.out, verbose=VERBOSE)
    for p in paths:
        print(f"short={p}")
