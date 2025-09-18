from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple
import argparse
import json
import os
import re
import pathlib
from math import ceil
from datetime import datetime

# ============================== Config & log =============================
try:
    from config import DEFAULT as CFG
except Exception:
    class _CFG:
        # Core (kept for compatibility; OPENAI_MODEL takes precedence)
        llm_model = "gpt-4o-mini"

        # Planner knobs
        num_clips = 5
        target_duration = 35  # advisory; min/max are enforced
        min_duration = 30
        max_duration = 40

        # Prompt sizing
        planner_segment_text_max_chars = 1800
        planner_prompt_text_max_chars = 120_000  # total text budget for CONTEXT

        # LLM transport/timeouts (kept for signature compat)
        llm_timeout_connect_s = 30
        llm_timeout_read_s = 900
        llm_keep_alive = "30m"

        # Verbosity
        verbose = True

    CFG = _CFG()

def _cfg(name: str, default, *legacy_names: str):
    if hasattr(CFG, name):
        return getattr(CFG, name)
    for n in legacy_names:
        if hasattr(CFG, n):
            return getattr(CFG, n)
    return default

VERBOSE = bool(_cfg("verbose", True))

def step(msg: str, verbose: bool = True) -> None:
    if verbose:
        print(msg)


# ============================= .env + OpenAI =============================
try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None  # optional

try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore

_ENV_LOADED = False
_OAI_CLIENT: Optional[OpenAI] = None  # type: ignore

def _load_env_once() -> None:
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    if load_dotenv is not None:
        here = pathlib.Path(__file__).resolve()
        candidates = [
            pathlib.Path.cwd() / ".env",
            here.parent / ".env",
            here.parent.parent / ".env",
        ]
        for p in candidates:
            try:
                if p.exists():
                    load_dotenv(p, override=True)
                    break
            except Exception:
                pass
        try:
            load_dotenv(override=True)
        except Exception:
            pass
    _ENV_LOADED = True

def _resolve_model(model_arg: Optional[str]) -> str:
    env_model = os.getenv("OPENAI_MODEL")
    if env_model:
        return env_model
    if model_arg:
        return model_arg
    return "gpt-4.1-nano"

def _get_openai_client_and_model(model_arg: Optional[str]):
    global _OAI_CLIENT
    _load_env_once()
    if OpenAI is None:
        raise RuntimeError("openai package not installed. Run: pip install openai python-dotenv")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Put it in your environment or .env file.")
    base_url = os.getenv("OPENAI_BASE_URL")  # optional
    org = os.getenv("OPENAI_ORG")  # optional
    if _OAI_CLIENT is None:
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        if org:
            kwargs["organization"] = org
        _OAI_CLIENT = OpenAI(**kwargs)
    return _OAI_CLIENT, _resolve_model(model_arg)


# ============================== Debug dumps =============================
DEBUG_DUMP_DIR = pathlib.Path("workdir/debug")
DEBUG_DUMP_DIR.mkdir(parents=True, exist_ok=True)

def _dump_debug(name: str, content: str) -> None:
    try:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        p = DEBUG_DUMP_DIR / f"{ts}_{name}"
        p.write_text(content or "", encoding="utf-8", errors="ignore")
    except Exception:
        pass


# ============================ Pre-processing ============================
def _reduce_segments(segments: List[Dict[str, Any]],
                     max_chars_per_segment: int,
                     total_max_chars: int) -> List[Dict[str, Any]]:
    """
    Keep {id,start,end,text}. Text may be trimmed to keep prompt size sane.
    """
    lean: List[Dict[str, Any]] = []
    budget = total_max_chars or 0
    for s in segments:
        text = (s.get("text") or s.get("text_native") or "").strip()
        if not text:
            continue
        if max_chars_per_segment and len(text) > max_chars_per_segment:
            limit = max_chars_per_segment
            cut = text.rfind(" ", 0, limit)
            if cut <= 0 or cut < int(limit * 0.6):
                cut = limit
            text = text[:cut].strip()
        item = {
            "id": s.get("id"),
            "start": float(s.get("start", 0.0)),
            "end": float(s.get("end", 0.0)),
            "text": text
        }
        cost = len(text)
        if total_max_chars:
            if cost > budget and budget <= 0:
                break
            budget -= cost
        lean.append(item)
    return lean


# ========================== Windows: time-based =========================
def _build_windows_time_based(lean: List[Dict[str, Any]], min_d: float, max_d: float) -> List[Dict[str, Any]]:
    """
    Build windows by concatenating consecutive segments until duration falls within [min_d, max_d].
    """
    windows: List[Dict[str, Any]] = []
    n = len(lean)
    for i in range(n):
        start = lean[i]["start"]
        text_buf = []
        end = start
        for j in range(i, n):
            end = lean[j]["end"]
            text_buf.append(lean[j]["text"])
            dur = end - start
            if dur >= min_d:
                if dur <= max_d:
                    windows.append({
                        "win_id": len(windows),
                        "start": round(start, 2),
                        "end": round(end, 2),
                        "text": " ".join(text_buf).strip(),
                    })
                break
    return windows


# ============================= Partitions ===============================
def _video_bounds(lean: List[Dict[str, Any]]) -> Tuple[float, float]:
    if not lean:
        return (0.0, 0.0)
    start = float(lean[0]["start"])
    end = float(lean[-1]["end"])
    return (start, end)

def _partition_by_time(lean: List[Dict[str, Any]],
                       windows: List[Dict[str, Any]],
                       parts: int) -> List[Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]]:
    """
    Partition lean+windows into 'parts' equal time bins.
    """
    if parts <= 1:
        return [(lean, windows)]
    v0, v1 = _video_bounds(lean)
    if v1 <= v0:
        return [(lean, windows)]
    span = v1 - v0
    bins = []
    for k in range(parts):
        b0 = v0 + (span * k) / parts
        b1 = v0 + (span * (k + 1)) / parts
        lean_k = [s for s in lean if not (s["end"] <= b0 or s["start"] >= b1)]
        wins_k = [w for w in windows if not (w["end"] <= b0 or w["start"] >= b1)]
        if lean_k and wins_k:
            bins.append((lean_k, wins_k))
    return bins if bins else [(lean, windows)]


# ============================ OpenAI ask (JSON) =========================
def _post_chat_blocking(messages: List[Dict[str, str]],
                        model: Optional[str],
                        api_base: Optional[str],
                        connect_timeout: int,
                        read_timeout: int,
                        keep_alive: str) -> str:
    """
    Backward-compatible shim: same signature as OpenAI version, but uses OpenAI
    Chat Completions + function calling to FORCE JSON. Returns a JSON string.
    """
    client, effective_model = _get_openai_client_and_model(model)

    tools = [{
        "type": "function",
        "function": {
            "name": "emit_clips",
            "description": "Return selected short-form clips as JSON",
            "parameters": {
                "type": "object",
                "properties": {
                    "clips": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "win_id": {"type": "integer"},
                                "start": {"type": "number"},
                                "end": {"type": "number"},
                                "score": {"type": "integer"},
                                "title": {"type": "string"},
                                "reason": {"type": "string"},
                                "caption": {"type": "string"}
                            },
                            "required": ["start", "end", "score", "title", "reason"],
                            "additionalProperties": False
                        },
                        "minItems": 1
                    }
                },
                "required": ["clips"],
                "additionalProperties": False
            }
        }
    }]

    resp = client.chat.completions.create(
        model=effective_model,
        messages=messages,
        tools=tools,
        tool_choice={"type": "function", "function": {"name": "emit_clips"}},
        top_p=0.9,
    )
    m = resp.choices[0].message
    if m.tool_calls and m.tool_calls[0].function:
        args_raw = m.tool_calls[0].function.arguments
        try:
            obj = json.loads(args_raw)
        except Exception:
            return args_raw  # let caller try to parse
        return json.dumps(obj, ensure_ascii=False)
    # Fallback
    return m.content or ""


# ============================ Parsing helpers ===========================
_SMART_QUOTES = str.maketrans({
    "“": "\"",
    "”": "\"",
    "„": "\"",
    "‟": "\"",
    "‘": "'",
    "’": "'",
    "‚": "'",
    "‛": "'",
})

def _strip_code_fences(s: str) -> str:
    if not s:
        return s
    m = re.search(r"```[a-zA-Z]*\s*(.*?)\s*```", s, flags=re.DOTALL)
    return m.group(1) if m else s

def _sanitize_jsonish(s: str) -> str:
    if not s:
        return s
    s = s.translate(_SMART_QUOTES)
    s = _strip_code_fences(s)
    s = re.sub(r'\bNaN\b', 'null', s)
    s = re.sub(r'\bInfinity\b', '1e9999', s)
    s = re.sub(r'\b-Infinity\b', '-1e9999', s)
    return s

def _parse_possible_clips(raw: str) -> List[Dict[str, Any]]:
    """
    Accept either:
    - a JSON array of clip objects, or
    - a JSON object with key 'clips' -> array.
    """
    if not raw or not raw.strip():
        raise ValueError("Empty response.")
    s = _sanitize_jsonish(raw.strip())
    # Direct parse attempts
    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict) and isinstance(obj.get("clips"), list):
            return obj["clips"]
    except Exception:
        pass
    # Try extracting the first JSON array
    first, last = s.find("["), s.rfind("]")
    if first != -1 and last != -1 and last > first:
        snippet = s[first:last + 1]
        obj = json.loads(snippet)
        if isinstance(obj, list):
            return obj
    # Regex-based last resort
    m = re.search(r"\[.*\]", s, flags=re.DOTALL)
    if m:
        obj = json.loads(m.group(0))
        if isinstance(obj, list):
            return obj
    raise ValueError("Model did not return a JSON array or {\"clips\": [...] }.")


# ========== Validation / snapping to candidate windows + non-overlap ==========
def _build_index_by_times(windows: List[Dict[str, Any]], eps: float = 0.25) -> Dict[Tuple[int, int], Dict[str, Any]]:
    idx: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for w in windows:
        st = float(w["start"]); en = float(w["end"])
        k = (round(st / eps), round(en / eps))
        idx[k] = w
    return idx

def _snap_to_window(start: float, end: float,
                    index: Dict[Tuple[int, int], Dict[str, Any]],
                    eps: float = 0.25) -> Optional[Dict[str, Any]]:
    k = (round(start / eps), round(end / eps))
    return index.get(k)

def _normalize_model_clips(raw_clips: List[Dict[str, Any]],
                           windows: List[Dict[str, Any]],
                           min_d: float,
                           max_d: float) -> List[Dict[str, Any]]:
    by_id = {w["win_id"]: w for w in windows}
    idx_by_time = _build_index_by_times(windows)
    out: List[Dict[str, Any]] = []
    for c in raw_clips:
        w = None
        if "win_id" in c:
            try:
                wid = int(c["win_id"])
                w = by_id.get(wid)
            except Exception:
                w = None
        else:
            try:
                st = float(c.get("start", -1))
                en = float(c.get("end", -1))
                if en > st > -1:
                    w = _snap_to_window(st, en, idx_by_time)
            except Exception:
                w = None
        if not w:
            continue
        st, en = float(w["start"]), float(w["end"])
        dur = en - st
        if dur < min_d or dur > max_d:
            continue
        title = (c.get("title") or "").strip()
        reason = (c.get("reason") or "").strip()
        score = int(c.get("score", 50))
        caption = (c.get("caption") or "").strip()
        out.append({
            "start": round(st, 2),
            "end": round(en, 2),
            "score": score,
            "title": title[:55] if title else f"Clip {st:.0f}-{en:.0f}s",
            "reason": reason[:200] if reason else "",
            "caption": caption[:220] if caption else "",  # NEW: carry caption if present
        })
    return out

def _pick_non_overlapping(candidates: List[Dict[str, Any]], want: int, min_gap: float = 0.0) -> List[Dict[str, Any]]:
    if not candidates:
        return []
    cands = sorted(candidates, key=lambda x: (-int(x.get("score", 50)), float(x["start"])))
    picked: List[Dict[str, Any]] = []
    occupied: List[Tuple[float, float]] = []

    def overlaps(st: float, en: float) -> bool:
        for (a, b) in occupied:
            if not (en <= a + min_gap or st >= b - min_gap):
                return True
        return False

    for c in cands:
        st, en = float(c["start"]), float(c["end"])
        if overlaps(st, en):
            continue
        picked.append(c)
        occupied.append((st, en))
        if len(picked) >= want:
            break
    return picked


# ================================ Prompts ===============================
SYSTEM = (
    "You are an expert short-form video editor for Shorts/Reels/TikTok.\n"
    "You will be given FULL context (lean segments with {id,start,end,text}) and a list of PRECOMPUTED candidate windows (start/end only).\n"
    "Each window ALREADY satisfies the duration bounds; pick from THESE windows only.\n"
    "Return ONLY JSON. Do not include markdown, code fences, or explanations."
)

USER_TEMPLATE = (
    "CONTEXT_SEGMENTS (compact list; text may be trimmed):\n"
    "{context}\n\n"
    "CANDIDATE_WINDOWS (ALL within {min_d}-{max_d}s; choose only from these):\n"
    "{windows}\n\n"
    "TASK: Pick exactly {num} NON-OVERLAPPING windows that will perform best as short-form content.\n"
    "Rules:\n"
    "- Choose from CANDIDATE_WINDOWS ONLY. Do NOT invent new times.\n"
    "- Output either {{\"win_id\": <int>, ...}} OR {{\"start\": <float>, \"end\": <float>, ...}}.\n"
    "- Hook fast: questions, surprising claims, strong opinions, clear how-tos, numbered advice, high emotion, conflict, concise stories.\n"
    "- Spread across the whole talk when possible; avoid clustering.\n"
    "- Titles must be specific and non-empty (<=55 chars, no emojis).\n"
    "- Add a concise social caption (<=220 chars) optimized for attention; avoid hashtags/emojis.\n"
    "- Return ONLY JSON with this shape:\n"
    "{{\"clips\": [{{\"win_id\": <int> (or start/end), \"score\": 0-100, \"title\": \"engaging hook\", \"reason\": \"why\", \"caption\": \"concise social caption\"}}, ...]}}"
)

def _make_messages(lean: List[Dict[str, Any]],
                   windows: List[Dict[str, Any]],
                   n: int,
                   min_d: float,
                   max_d: float) -> List[Dict[str, str]]:
    user = USER_TEMPLATE.format(
        context=json.dumps(lean, ensure_ascii=False),
        windows=json.dumps(
            [{"win_id": w["win_id"], "start": w["start"], "end": w["end"]} for w in windows],
            ensure_ascii=False
        ),
        num=n,
        min_d=min_d,
        max_d=max_d
    )
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user}
    ]


# ======================= Single pass ask + normalize + pick =======================
def _ask_and_pick(lean: List[Dict[str, Any]],
                  windows: List[Dict[str, Any]],
                  n: int,
                  model: Optional[str],
                  api_base: Optional[str],
                  connect_to: int,
                  read_to: int,
                  keep_alive: str,
                  min_d: float,
                  max_d: float,
                  attempt_tag: str) -> List[Dict[str, Any]]:
    messages = _make_messages(lean, windows, n, min_d, max_d)
    _dump_debug(f"prompt_{attempt_tag}.json", messages[1]["content"])
    raw = _post_chat_blocking(messages, model, api_base, connect_to, read_to, keep_alive)
    _dump_debug(f"plan_raw_{attempt_tag}.json", raw)
    raw_clips = _parse_possible_clips(raw)
    normalized = _normalize_model_clips(raw_clips, windows, min_d, max_d)
    picked = _pick_non_overlapping(normalized, n, min_gap=0.0)
    return picked


# ================================ Public API ================================
def propose_clips(  # name kept for backward compatibility
    segments: List[Dict[str, Any]],
    num_clips: Optional[int] = None,
    target_duration: Optional[int] = None,
    model: Optional[str] = None,
    api_base: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Returns a list of clip dicts: [{start,end,score,title,reason,caption}, ...]
    Uses ONLY precomputed windows (durations guaranteed).
    Hierarchical fallback: 1 pass → if empty → split in 2 → if still empty → split in 4.
    """
    n = int(num_clips or _cfg("num_clips", 5))
    min_d = float(_cfg("min_duration", 30))
    max_d = float(_cfg("max_duration", 40))

    # 1) Lean transcript
    step("[PLAN] Building lean segments…", VERBOSE)
    lean = _reduce_segments(
        segments,
        int(_cfg("planner_segment_text_max_chars", 1800, "plan_max_chars_per_segment")),
        int(_cfg("planner_prompt_text_max_chars", 120_000, "plan_total_max_chars")),
    )
    step(f"[PLAN] Lean segments: {len(lean)} (chars≈{sum(len(s['text']) for s in lean)})", VERBOSE)
    if not lean:
        raise RuntimeError("No usable segments for planning.")

    # 2) Pure time-based windows within [min_d, max_d]
    step(f"[PLAN] Composing windows in [{min_d:.0f},{max_d:.0f}]s…", VERBOSE)
    windows = _build_windows_time_based(lean, min_d, max_d)
    step(f"[PLAN] Windows built (ALL considered): {len(windows)}", VERBOSE)
    if not windows:
        raise RuntimeError("No candidate windows could be constructed.")

    connect_to = int(_cfg("llm_timeout_connect_s", 30))
    read_to = int(_cfg("llm_timeout_read_s", 900))
    keep_alive = str(_cfg("llm_keep_alive", "30m"))

    # Pass A: single shot, full context
    attempts = 2
    for t in range(1, attempts + 1):
        step(f"[PLAN] Requesting plan (full context) attempt {t}/{attempts}…", VERBOSE)
        try:
            picks = _ask_and_pick(
                lean, windows, n,
                model, api_base,
                connect_to, read_to + 120*(t-1), keep_alive,
                min_d, max_d,
                f"full_{t}"
            )
            if picks:
                step(f"[PLAN] Planned {len(picks)} clips (non-overlapping).", VERBOSE)
                return picks[:n]
            step("[PLAN] Model returned empty after normalization; will retry…", VERBOSE)
        except Exception as e:
            step(f"[PLAN] Parse/validation failed on full context: {e}", VERBOSE)

    # Pass B: split into 2 parts
    step("[PLAN] Full context failed → splitting into 2 parts…", VERBOSE)
    parts2 = _partition_by_time(lean, windows, 2)
    collected: List[Dict[str, Any]] = []
    remaining = n
    for pi, (lean_p, wins_p) in enumerate(parts2, start=1):
        ask_n = max(remaining, 1)
        step(f"[PLAN] Part {pi}/2 → windows={len(wins_p)}; asking for up to {ask_n}", VERBOSE)
        try:
            picks = _ask_and_pick(
                lean_p, wins_p, ask_n,
                model, api_base,
                connect_to, read_to, keep_alive,
                min_d, max_d,
                f"p2_{pi}"
            )
            collected.extend(picks)
            collected = _pick_non_overlapping(collected, n, min_gap=0.0)
            remaining = n - len(collected)
            if remaining <= 0:
                break
        except Exception as e:
            step(f"[PLAN] Part {pi}/2 failed: {e}", VERBOSE)
    if collected:
        step(f"[PLAN] Planned {len(collected)} clips after 2-part split.", VERBOSE)
        return collected[:n]

    # Pass C: split into 4 parts
    step("[PLAN] 2-part split yielded nothing → splitting into 4 parts…", VERBOSE)
    parts4 = _partition_by_time(lean, windows, 4)
    collected4: List[Dict[str, Any]] = []
    for pi, (lean_p, wins_p) in enumerate(parts4, start=1):
        ask_n = max(ceil(n / len(parts4)), 1)
        step(f"[PLAN] Part {pi}/4 → windows={len(wins_p)}; asking for up to {ask_n}", VERBOSE)
        try:
            picks = _ask_and_pick(
                lean_p, wins_p, ask_n,
                model, api_base,
                connect_to, read_to, keep_alive,
                min_d, max_d,
                f"p4_{pi}"
            )
            collected4.extend(picks)
        except Exception as e:
            step(f"[PLAN] Part {pi}/4 failed: {e}", VERBOSE)

    if collected4:
        final_picks = _pick_non_overlapping(collected4, n, min_gap=0.0)
        if final_picks:
            step(f"[PLAN] Planned {len(final_picks)} clips after 4-part split.", VERBOSE)
            return final_picks[:n]

    raise RuntimeError("Planning failed: model returned no usable clips after full, 2-part, and 4-part passes.")

def propose_clips_out(
    segments: List[Dict[str, Any]],
    num_clips: Optional[int] = None,
    target_duration: Optional[int] = None,
    model: Optional[str] = None,
    api_base: Optional[str] = None
) -> Dict[str, Any]:
    """Legacy/alternate API: returns {'clips': [...]}."""
    return {"clips": propose_clips(segments, num_clips, target_duration, model, api_base)}


# ============================ File helpers & CLI ============================
def _save_plan_json(transcript_path: str,
                    clips: List[Dict[str, Any]],
                    out_dir: str,
                    verbose: bool = True) -> str:
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(transcript_path))[0]
    out_path = os.path.join(out_dir, f"{stem}_plan.json")
    data = {"clips": clips}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    step(f"[PLAN] Saved plan: {out_path}", verbose)
    return out_path

def _load_segments(transcript_path: str) -> List[Dict[str, Any]]:
    with open(transcript_path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    return doc.get("segments") or []

def propose_plan_file(transcript_json_path: str,
                      out_dir: str,
                      num_clips: int,
                      target_duration: int,
                      model: str,
                      api_base: str,
                      verbose: bool = True) -> str:
    """Load transcript from path, run clip planner, save plan JSON, and return plan path."""
    with open(transcript_json_path, "r", encoding="utf-8") as f:
        tx = json.load(f)
    segments = tx.get("segments", [])
    if not isinstance(segments, list) or not segments:
        raise ValueError("Transcript JSON has no 'segments' to plan from.")
    clips = propose_clips(
        segments,
        num_clips=num_clips,
        target_duration=target_duration,
        model=model,
        api_base=api_base,
    )
    return _save_plan_json(transcript_json_path, clips, out_dir, verbose)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Plan short clips from a transliterated transcript (Roman Urdu).")
    ap.add_argument("--transcript", required=True, help="Path to *_roman.json (with segments).")
    ap.add_argument("--outdir", default="workdir/llm_shorts", help="Directory to save the *_plan.json")
    ap.add_argument("--num-clips", type=int, default=None)
    ap.add_argument("--target-duration", type=int, default=None)
    ap.add_argument("--llm-model", default=None)
    ap.add_argument("--openai-url", default=None, help="Ignored (compatibility)")
    args = ap.parse_args()

    model = args.llm_model or _cfg("llm_model", "gpt-4o-mini")
    api_base = args.openai_url  # ignored
    segs = _load_segments(args.transcript)
    clips = propose_clips(
        segs,
        num_clips=args.num_clips,
        target_duration=args.target_duration,
        model=model,
        api_base=api_base,
    )
    plan_path = _save_plan_json(args.transcript, clips, args.outdir, VERBOSE)
    print(f"plan={plan_path}")
