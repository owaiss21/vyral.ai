from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import time, json, os, re

try:
    from faster_whisper import WhisperModel
except Exception:
    WhisperModel = None  # type: ignore

# Utilities
try:
    from modules.utils import write_json, ensure_dir, seconds_to_srt_time, progress, step
except Exception:
    from .utils import write_json, ensure_dir, seconds_to_srt_time, progress, step  # type: ignore

# Config
try:
    from config import DEFAULT as CFG
except Exception:
    class _CFG:
        whisper_model = "large-v3"
        device = "auto"
        word_timestamps = True
        verbose = True
    CFG = _CFG()

# =====================================================================================
# Whisper model singleton (prevents native destructor crash on return in some setups)
# =====================================================================================
_MODEL_SINGLETON: Optional[WhisperModel] = None # type: ignore
_MODEL_SIZE: Optional[str] = None
_MODEL_DEVICE: Optional[str] = None  # "cuda" or "cpu"

def _get_whisper_model(model_size: str, device: str | None) -> Tuple[WhisperModel, str]: # type: ignore
    """
    Returns a (singleton_model, effective_device) tuple.
    - If device == "auto", try CUDA first, then CPU.
    - compute_type = "float16" if CUDA else "int8".
    - Reuses an existing singleton if the same model_size is requested.
    """
    global _MODEL_SINGLETON, _MODEL_SIZE, _MODEL_DEVICE

    if WhisperModel is None:
        raise RuntimeError("faster-whisper is not available in this environment.")

    # If we already have a model of this size, reuse it regardless of requested device.
    if _MODEL_SINGLETON is not None and _MODEL_SIZE == model_size:
        return _MODEL_SINGLETON, (_MODEL_DEVICE or "cpu")

    requested = (device or "auto").lower()

    def _try_make(dev: str) -> WhisperModel: # type: ignore
        compute_type = "float16" if dev == "cuda" else "int8"  # <- required
        return WhisperModel(model_size, device=dev, compute_type=compute_type)

    # Try CUDA if asked, or when "auto"
    if requested in ("auto", "cuda"):
        try:
            m = _try_make("cuda")
            _MODEL_SINGLETON, _MODEL_SIZE, _MODEL_DEVICE = m, model_size, "cuda"
            step("[ASR] Using CUDA", True)
            return m, "cuda"
        except Exception as e:
            msg = str(e)
            step(f"[ASR] CUDA init failed, falling back to CPU: {msg}", True)

    # Fallback to CPU
    m = _try_make("cpu")
    _MODEL_SINGLETON, _MODEL_SIZE, _MODEL_DEVICE = m, model_size, "cpu"
    step("[ASR] Using CPU (int8)", True)
    return m, "cpu"

# =====================================================================================
# Atomic JSON I/O
# =====================================================================================

def _write_json_atomic(path: Path, data: Dict[str, Any], *, retries: int = 25, delay: float = 0.08) -> None:
    """
    Atomically write JSON with Windows-friendly retries.
    Avoid fsync on Windows (can hang with AV/indexers), and retry os.replace()
    to handle transient file locks.
    """
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)

    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
        f.flush()

    last_err: Optional[BaseException] = None
    for _ in range(retries):
        try:
            os.replace(tmp, path)  # atomic on Windows & POSIX
            return
        except (PermissionError, OSError) as e:
            last_err = e
            time.sleep(delay)

    if last_err:
        step(f"[ASR] WARNING: final replace retry due to: {last_err}", True)
    os.replace(tmp, path)

def _verify_transcript_json(path: Path) -> Dict[str, Any]:
    """Ensure final JSON exists, parses, and contains non-empty 'segments'."""
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        raise RuntimeError(f"Transcription JSON not found or empty: {p}")
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to read transcription JSON {p}: {e}") from e
    segs = data.get("segments", [])
    if not isinstance(segs, list) or len(segs) == 0:
        raise RuntimeError(f"Transcription has no segments: {p}")
    return data

# =====================================================================================
# Streaming writer for intermediate pass files
# =====================================================================================

class _JsonStreamWriter:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = self.path.open("w", encoding="utf-8")
        self._first = True
        self._open = False

    def begin(self, audio: Path, duration: Optional[float], language: Optional[str]) -> None:
        self._f.write('{\n')
        self._f.write(f'  "audio": {json.dumps(str(audio), ensure_ascii=False)},\n')
        self._f.write(f'  "duration": {json.dumps(duration)},\n')
        self._f.write(f'  "language": {json.dumps(language, ensure_ascii=False)},\n')
        self._f.write('  "segments": [\n')
        self._open = True
        self._f.flush()

    def write_segment(self, seg: Dict[str, Any]) -> None:
        if not self._open:
            return
        if not self._first:
            self._f.write(",\n")
        else:
            self._first = False
        self._f.write("    " + json.dumps(seg, ensure_ascii=False))
        self._f.flush()

    def end(self) -> None:
        if not self._open:
            return
        self._f.write('\n  ]\n}\n')
        self._f.flush()
        self._f.close()
        self._open = False

# =====================================================================================
# Segment helpers
# =====================================================================================

def _segment_to_dict(seg) -> Dict[str, Any]:
    d: Dict[str, Any] = {
        "id": getattr(seg, "id", None),
        "start": float(getattr(seg, "start", 0.0)),
        "end": float(getattr(seg, "end", 0.0)),
        "text": str(getattr(seg, "text", "")).strip(),
    }
    d["text_native"] = d["text"]
    words = []
    for w in getattr(seg, "words", []) or []:
        try:
            words.append({
                "start": float(getattr(w, "start", d["start"])),
                "end": float(getattr(w, "end", d["start"])),
                "word": str(getattr(w, "word", "") or getattr(w, "text", "")).strip(),
            })
        except Exception:
            continue
    if words:
        d["words"] = words
    return d

def _is_latin_char(ch: str) -> bool:
    cp = ord(ch)
    return (0x0041 <= cp <= 0x005A) or (0x0061 <= cp <= 0x007A)

def _is_letter(ch: str) -> bool:
    return ch.isalpha()

def _english_ratio(text: str) -> float:
    letters = [c for c in text if _is_letter(c)]
    if not letters:
        return 0.0
    latin = sum(1 for c in letters if _is_latin_char(c))
    return latin / len(letters)

def _is_english_text(text: str, threshold: float = 0.6) -> bool:
    return _english_ratio(text) >= threshold

def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))

def _segments_coverage_ratio(segments: List[Dict[str, Any]], total_duration: Optional[float]) -> float:
    if not segments or not total_duration or total_duration <= 0:
        return 0.0
    cov = 0.0
    for s in segments:
        try:
            st = float(s.get("start", 0.0))
            en = float(s.get("end", 0.0))
            cov += max(0.0, en - st)
        except Exception:
            continue
    return min(1.0, cov / total_duration)

def _iou(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    a0, a1 = float(a.get("start", 0.0)), float(a.get("end", 0.0))
    b0, b1 = float(b.get("start", 0.0)), float(b.get("end", 0.0))
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return (inter / union) if union > 0 else 0.0

def _union_nonoverlapping(
    base: List[Dict[str, Any]],
    extras: List[Dict[str, Any]],
    iou_threshold: float = 0.25
) -> List[Dict[str, Any]]:
    out = list(base)
    for e in extras:
        try:
            if all(_iou(e, b) < iou_threshold for b in out):
                out.append(e)
        except Exception:
            continue
    out.sort(key=lambda s: float(s.get("start", 0.0)))
    return out

def _merge_segments(pass1: List[Dict[str, Any]], pass2: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    for s1 in pass1:
        t1 = s1.get("text", "") or ""
        if _is_english_text(t1):
            merged.append(s1)
            continue
        s1_start, s1_end = float(s1.get("start", 0.0)), float(s1.get("end", 0.0))
        overlaps = [s2 for s2 in pass2 if _overlap(s1_start, s1_end, float(s2.get("start", 0.0)), float(s2.get("end", 0.0))) > 0.05]
        if overlaps:
            overlaps_sorted = sorted(overlaps, key=lambda x: float(x.get("start", 0.0)))
            new_text = " ".join((s.get("text") or "").strip() for s in overlaps_sorted).strip()
            new_words = []
            for s in overlaps_sorted:
                if isinstance(s.get("words"), list):
                    new_words.extend(s["words"])
            s_new = dict(s1)
            if new_text:
                s_new["text"] = new_text
                s_new["text_native"] = new_text
                if new_words:
                    s_new["words"] = new_words
                else:
                    s_new.pop("words", None)
            merged.append(s_new)
        else:
            merged.append(s1)
    return merged

def _consume_with_progress(
    seg_iter,
    total_dur: Optional[float],
    verbose: bool,
    json_stream: Optional[_JsonStreamWriter] = None,
) -> List[Dict[str, Any]]:
    start_ts = time.time()
    segs: List[Dict[str, Any]] = []
    last_end = 0.0
    for s in seg_iter:
        d = _segment_to_dict(s)
        segs.append(d)
        if json_stream is not None:
            json_stream.write_segment(d)
        last_end = float(d.get("end", last_end))
        if total_dur and total_dur > 0:
            progress(min(last_end, total_dur), total_dur, start_ts, prefix="[ASR]", verbose=verbose)
    if total_dur and total_dur > 0:
        progress(total_dur, total_dur, start_ts, prefix="[ASR]", verbose=verbose)
    return segs

# =====================================================================================
# Main entry
# =====================================================================================

def transcribe_audio(
    audio_path: str,
    out_dir: str = "transcriptions",
    model_size: str = "large-v3",
    device: str = "auto",
    word_timestamps: bool = True,
    force_language: Optional[str] = None,             # kept for backward-compat (ignored)
    conditional_force_non_en: Optional[str] = None,   # kept for backward-compat (ignored)
    english_codes: tuple[str, ...] = ("en",),         # kept for backward-compat (ignored)
    verbose: bool = CFG.verbose,
) -> Tuple[str, str]:

    audio = Path(audio_path).resolve()
    out_dir_p = ensure_dir(out_dir)
    stem = audio.stem

    json_out       = out_dir_p / (stem + ".json")            # final
    json_out_auto  = out_dir_p / (stem + "_auto.json")       # stream for auto pass
    srt_out        = out_dir_p / (stem + ".srt")             # placeholder

    # Get (or create) the singleton model
    model, effective_device = _get_whisper_model(model_size, device)

    # VAD profile: choose SENSITIVE to avoid missing speech
    VAD_SENSITIVE = dict(
        threshold=0.2,
        min_speech_duration_ms=150,
        min_silence_duration_ms=150,
        speech_pad_ms=200,
    )

    def _run(lang: Optional[str], use_vad: bool):
        return model.transcribe(
            str(audio),
            vad_filter=use_vad,
            vad_parameters=VAD_SENSITIVE if use_vad else None,
            word_timestamps=word_timestamps,
            language=lang,             # we will pass None for auto
            task="transcribe",
        )

    step("[ASR] Transcribing (single auto-language pass)…", verbose)

    # --- Single pass: AUTO language with VAD ---
    seg_iter, info = _run(None, use_vad=True)
    duration = float(getattr(info, "duration", 0.0)) if info is not None else None
    detected = getattr(info, "language", None)

    jsw = _JsonStreamWriter(json_out_auto)
    try:
        jsw.begin(audio, duration, detected)
        segs = _consume_with_progress(seg_iter, duration, verbose, json_stream=jsw)
    finally:
        jsw.end()

    # Coverage rescue if too low (e.g., missed voices): re-run without VAD and union
    cov = _segments_coverage_ratio(segs, duration)
    if cov < 0.60:
        step(f"[ASR] Coverage {cov:.2%} is low with VAD; running no-VAD rescue…", verbose)
        seg_iter2, info2 = _run(None, use_vad=False)
        segs2 = _consume_with_progress(
            seg_iter2,
            float(getattr(info2, "duration", duration) or 0.0),
            verbose,
            json_stream=None
        )
        segs = _union_nonoverlapping(segs, segs2, iou_threshold=0.25)

    transcript = {
        "audio": str(audio),
        "duration": duration,
        "language": detected,
        "segments": segs,
    }
    _write_json_atomic(json_out, transcript)

    # cleanup intermediates
    try:
        if json_out_auto.exists():
            json_out_auto.unlink()
    except Exception:
        pass

    _ = _verify_transcript_json(json_out)
    abs_json = str(Path(json_out).resolve())
    abs_srt  = str(Path(srt_out).resolve())
    print(f"[ASR] transcript_ready={abs_json}", flush=True)
    return abs_json, abs_srt


# =====================================================================================
# CLI
# =====================================================================================

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Transcribe audio using Faster-Whisper")
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out", default="transcriptions")
    try:
        from config import DEFAULT as CFG
    except Exception:
        class _CFG:
            whisper_model='large-v3'
            device='auto'
            word_timestamps=True
            verbose=True
        CFG=_CFG()
    ap.add_argument("--whisper-model", default=CFG.whisper_model)
    ap.add_argument("--device", default=CFG.device, choices=["auto", "cpu", "cuda"])
    ap.add_argument("--word-timestamps", default="on", choices=["on", "off"])
    args = ap.parse_args()

    j, s = transcribe_audio(
        args.audio,
        args.out,
        model_size=args.whisper_model,
        device=args.device,
        word_timestamps=(args.word_timestamps == "on"),
        verbose=CFG.verbose,
    )
    print(j)
    print(s)