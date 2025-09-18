# modules/transliteration.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Optional
import json, argparse, time, re, os, sys, codecs

# ============================== Config ==================================
try:
    from config import DEFAULT as CFG
except Exception:
    class _CFG:
        llm_model = "gpt-4o-mini"          # kept for CLI/back-compat; ignored if OPENAI_MODEL set
        transliteration_batch_size = 12
        transliteration_qa_passes = 1
        verbose = True
    CFG = _CFG()

BATCH_SIZE = int(getattr(CFG, "transliteration_batch_size", 12))
QA_PASSES  = int(getattr(CFG, "transliteration_qa_passes", 1))
VERBOSE    = bool(getattr(CFG, "verbose", True))

# =========================== .env + OpenAI ===============================
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
    """Load .env once from common locations."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    if load_dotenv is not None:
        here = Path(__file__).resolve()
        candidates = [
            Path.cwd() / ".env",
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
        # Also try default discovery
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
    return "gpt-4o-mini"


def _get_openai_client_and_model(model_arg: Optional[str]):
    """Return (OpenAI client, effective_model)."""
    global _OAI_CLIENT
    _load_env_once()
    if OpenAI is None:
        raise RuntimeError("openai package not installed. Run: pip install openai python-dotenv")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Put it in your environment or .env file.")
    base_url = os.getenv("OPENAI_BASE_URL")  # optional (proxy/gateway)
    org = os.getenv("OPENAI_ORG")            # optional

    if _OAI_CLIENT is None:
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        if org:
            kwargs["organization"] = org
        _OAI_CLIENT = OpenAI(**kwargs)

    return _OAI_CLIENT, _resolve_model(model_arg)


# ============================== Utilities ================================
_ARABIC_RE  = re.compile(r"[\u0600-\u06FF]")
_DEVNAG_RE  = re.compile(r"[\u0900-\u097F]")

def contains_native(s: str) -> bool:
    """True if the string contains Arabic (Urdu) or Devanagari (Hindi)."""
    return bool(_ARABIC_RE.search(s) or _DEVNAG_RE.search(s))

def step(msg: str, verbose: bool = True) -> None:
    if verbose:
        print(msg, flush=True)

def _progress_bar(done: int, total: int, t0: float, prefix="[TRL]", width=30, verbose=True) -> None:
    if not verbose or total <= 0:
        return
    done = max(0, min(done, total))
    elapsed = time.time() - t0
    pct = done / total if total else 0.0
    eta = (elapsed / pct) * (1 - pct) if pct > 0 else 0.0
    filled = int(width * pct)
    bar = "#" * filled + "-" * (width - filled)
    print(f"\r{prefix} [{bar}] {pct*100:5.1f}%  elapsed={elapsed:5.1f}s  eta={eta:5.1f}s", end="", flush=True)
    if done == total:
        print("", flush=True)

def _atomic_save_json(obj: Dict[str, Any], out_path: Path) -> None:
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(out_path)

def _read_json_any_encoding(path: Path) -> Dict[str, Any]:
    """Robust reader for JSON produced on Windows/macOS/Linux (UTF-8/UTF-8-BOM/UTF-16)."""
    data = path.read_bytes()
    if data.startswith(codecs.BOM_UTF8):
        text = data.decode("utf-8-sig")
    elif data.startswith(codecs.BOM_UTF16_LE) or data.startswith(codecs.BOM_UTF16_BE):
        text = data.decode("utf-16")
    else:
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            enc = (sys.getfilesystemencoding() or "cp1252")
            try:
                text = data.decode(enc)
            except UnicodeDecodeError:
                text = data.decode("utf-16")
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        snippet = text[:200].replace("\n", " ").replace("\r", " ")
        raise ValueError(f"Input file is not valid JSON: {path}. Starts with: {snippet!r}") from e


# ============================== Prompts =================================
SYSTEM = (
    "You transliterate Urdu (Nastaliq/Arabic) and Hindi (Devanagari) to Roman Urdu.\n"
    "Do NOT paraphrase or summarize. Output phonetic Roman forms only.\n"
    "If a token is already in Latin script, keep it EXACTLY as-is.\n"
    "When native text clearly targets an English word, name, brand, team, or acronym, output its canonical Latin form.\n"
    "Use context to disambiguate and avoid inserting extra vowels or letters that distort known spellings.\n"
    "Acronyms should remain uppercase. Never output any native-script characters.\n"
    "Return ONLY JSON; no code fences or explanations."
)

STYLE_GUIDE = (
    "Guidelines:\n"
    " • Preserve punctuation and spacing feel.\n"
    " • Keep existing Latin tokens unchanged.\n"
    " • For well-known names and acronyms, prefer the standard Latin spelling.\n"
    " • Do not invent vowels; keep concise transliterations.\n"
    " • Output only Latin letters, digits, spaces, and punctuation.\n"
)

BATCH_INSTR = (
    "TASK:\n"
    "Given JSON of segments, transliterate each to Roman Urdu.\n"
    "INPUT JSON shape:\n"
    "{\"segments\":[{\"i\":<int>,\"text\":\"<line possibly in Urdu/Devanagari or Latin>\",\"words\":[\"<token>\", ...]}, ...]}\n"
    "Return ONLY a JSON array:\n"
    "[{\"i\":<int>,\"text_roman\":\"<roman line>\",\"words_roman\":[\"<roman token>\", ...]}, ...]\n"
    "Constraints:\n"
    " • Do NOT translate/paraphrase. Correct obvious native typos only.\n"
    " • If a token was Latin, keep it unchanged in words_roman and text_roman.\n"
    " • If the native text encodes an English word/name/acronym, output the canonical Latin form.\n"
    " • words_roman length MUST equal input words length (1:1 alignment).\n"
    " • Output MUST contain only Latin characters plus digits/punctuation; no native scripts.\n"
    " • JSON array ONLY. No extra keys/comments.\n"
)


# =========================== OpenAI call (batch) ========================
def _openai_chat_transliterate(batch_items: List[Dict[str, Any]], model: Optional[str]) -> List[Dict[str, Any]]:
    """
    Ask OpenAI (Chat Completions + function calling) to produce an array of:
      [{"i": int, "text_roman": str, "words_roman": [str, ...]}, ...]
    """
    client, effective_model = _get_openai_client_and_model(model)

    tools = [{
        "type": "function",
        "function": {
            "name": "return_transliterations",
            "description": "Return Roman Urdu transliterations for the provided items.",
            "parameters": {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "i": {"type": "integer"},
                                "text_roman": {"type": "string"},
                                "words_roman": {"type": "array", "items": {"type": "string"}}
                            },
                            "required": ["i", "text_roman", "words_roman"],
                            "additionalProperties": False
                        },
                        "minItems": 1
                    }
                },
                "required": ["items"],
                "additionalProperties": False
            }
        }
    }]

    user = (
        STYLE_GUIDE
        + "\n\n"
        + BATCH_INSTR
        + "\nINPUT:\n"
        + json.dumps({"segments": batch_items}, ensure_ascii=False)
    )

    resp = client.chat.completions.create(
        model=effective_model,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
        ],
        tools=tools,
        tool_choice={"type": "function", "function": {"name": "return_transliterations"}},
        top_p=0.9,
    )
    msg = resp.choices[0].message

    if msg.tool_calls:
        args_raw = msg.tool_calls[0].function.arguments
        try:
            obj = json.loads(args_raw)
            arr = obj.get("items", [])
            if isinstance(arr, list):
                return arr
        except Exception:
            pass

    # Fallback: try raw content as JSON array
    raw = (msg.content or "").strip()
    try:
        arr2 = json.loads(raw)
        if isinstance(arr2, list):
            return arr2
    except Exception:
        pass

    raise RuntimeError("Model did not return valid transliteration array.")


# ============================== Repair Helpers ==========================
LATIN_SAFE_RE = re.compile(r"[^A-Za-z0-9'’\-.,!?:;()&+/ ]+")

def _ensure_latin_only(s: str) -> str:
    """Strip any non-Latin characters while keeping punctuation/spaces."""
    return LATIN_SAFE_RE.sub("", s or "").strip()

def _repair_words_with_llm(original_words: List[str], context_text_roman: str, model: Optional[str]) -> List[str]:
    """
    Ask the LLM to produce a same-length, Latin-only list aligned 1:1 with original_words.
    Existing Latin tokens must be preserved; native tokens must be romanized.
    """
    client, effective_model = _get_openai_client_and_model(model)

    system = (
        "You fix a per-token Romanization list. Return a JSON object only.\n"
        "Rules:\n"
        " • Preserve tokens that are already Latin (copy verbatim).\n"
        " • For native-script tokens (Urdu/Devanagari), output a concise Roman Urdu token.\n"
        " • Use context from the full romanized line to infer intended English names/acronyms/teams.\n"
        " • Acronyms stay uppercase. No native script in output.\n"
        " • The output list MUST have the SAME LENGTH and order as the input tokens.\n"
        "Return ONLY: {\"words_roman\": [\"...\", ...]}\n"
    )
    user = json.dumps({
        "context_text_roman": context_text_roman,
        "original_words": original_words,
    }, ensure_ascii=False)

    tools = [{
        "type": "function",
        "function": {
            "name": "return_words_roman",
            "description": "Return same-length list of Latin-only tokens aligned 1:1 to the input original_words.",
            "parameters": {
                "type": "object",
                "properties": {
                    "words_roman": {
                        "type": "array",
                        "items": {"type": "string"},
                    }
                },
                "required": ["words_roman"],
                "additionalProperties": False
            }
        }
    }]

    resp = client.chat.completions.create(
        model=effective_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        tools=tools,
        tool_choice={"type": "function", "function": {"name": "return_words_roman"}},
        top_p=0.9,
    )
    msg = resp.choices[0].message

    words_out: List[str] = []
    if msg.tool_calls:
        args_raw = msg.tool_calls[0].function.arguments
        try:
            obj = json.loads(args_raw)
            arr = obj.get("words_roman", [])
            if isinstance(arr, list):
                words_out = [_ensure_latin_only(x) for x in arr]
        except Exception:
            pass

    # Fallback: try content as JSON
    if not words_out:
        raw = (msg.content or "").strip()
        try:
            obj2 = json.loads(raw)
            arr2 = obj2.get("words_roman", [])
            if isinstance(arr2, list):
                words_out = [_ensure_latin_only(x) for x in arr2]
        except Exception:
            pass

    # Final guardrails: enforce same length + Latin-only
    if len(words_out) != len(original_words):
        tokens = [t for t in re.split(r"\s+", context_text_roman.strip()) if t]
        if len(tokens) >= len(original_words):
            words_out = tokens[:len(original_words)]
        else:
            words_out = tokens + [""] * (len(original_words) - len(tokens))

    return [_ensure_latin_only(w) for w in words_out]


# ============================== QA Helpers ==============================
def _segment_has_native(seg: Dict[str, Any]) -> bool:
    if contains_native(seg.get("text") or ""):
        return True
    for w in (seg.get("words") or []):
        if contains_native((w.get("word") or w.get("native") or "")):
            return True
    return False

def _qa_fix_pass(out_obj: Dict[str, Any], *, model: str, batch_size: int, verbose: bool) -> int:
    """
    One QA pass:
      - Find segments that still contain native script.
      - Re-send ONLY those segments for transliteration using their original 'text_native' (or current text).
      - Update out_obj in place. Returns number of segments fixed.
    """
    segments = out_obj.get("segments") or []
    bad_idx: List[int] = [i for i, s in enumerate(segments) if _segment_has_native(s)]
    if not bad_idx:
        return 0

    step(f"[QA] Found {len(bad_idx)} segments needing re-transliteration.", verbose)
    t0 = time.time()
    fixed = 0
    for bstart in range(0, len(bad_idx), batch_size):
        bend = min(bstart + batch_size, len(bad_idx))
        batch_ids = bad_idx[bstart:bend]

        items: List[Dict[str, Any]] = []
        for i in batch_ids:
            s = segments[i]
            native_text = s.get("text_native") or s.get("text") or ""
            native_words = [(w.get("native") or w.get("word") or "").strip() for w in (s.get("words") or [])]
            items.append({"i": i, "text": native_text, "words": native_words})

        try:
            arr = _openai_chat_transliterate(items, model=model)
        except Exception as e:
            step(f"[QA] Batch failed; leaving these items unchanged. Reason: {e}", True)
            arr = []

        results: Dict[int, Dict[str, Any]] = {}
        for it in arr:
            if not isinstance(it, dict):
                continue
            idx = it.get("i")
            if idx is None:
                continue
            results[int(idx)] = {
                "text_roman": (it.get("text_roman") or "").strip(),
                "words_roman": list(it.get("words_roman") or []),
            }

        for i in batch_ids:
            if i not in results:
                continue
            out_seg = segments[i]
            upd = results[i]

            # Update line text
            out_seg["text"] = upd.get("text_roman") or out_seg.get("text") or ""

            # Update words (no leaking of native tokens)
            if "words" in out_seg and out_seg["words"]:
                wr = upd.get("words_roman") or []
                if len(wr) != len(out_seg["words"]) or any(contains_native(tok) for tok in wr):
                    original_words = [(w.get("native") or w.get("word") or "") for w in out_seg["words"]]
                    wr = _repair_words_with_llm(original_words, out_seg["text"], model)
                wr = [_ensure_latin_only(t) for t in wr]
                if len(wr) != len(out_seg["words"]):
                    wr = (wr + [""] * len(out_seg["words"]))[:len(out_seg["words"])]
                for j, w in enumerate(out_seg["words"]):
                    w["word"] = wr[j] or _ensure_latin_only(w.get("word") or w.get("native") or "")
            fixed += 1

        _progress_bar(bend, len(bad_idx), t0, prefix="[QA]", verbose=verbose)

    return fixed


# ============================== Main API ================================
def transliterate_transcript(  # name kept for backward compatibility
    transcript_json_path: str,
    out_dir: str = "transliteration",
    model: str = getattr(CFG, "llm_model", "gpt-4o-mini"),
    api_base: str = "",  # legacy/ignored; kept to avoid CLI breakage
    batch_size: int = BATCH_SIZE,
    qa_passes: int = QA_PASSES,
) -> str:
    """
    Single-pass transliteration using OpenAI + QA loop.
    Writes <stem>_roman.json. Returns absolute path to the output file.
    """
    verbose = VERBOSE

    in_path = Path(transcript_json_path).resolve()
    out_dir_p = Path(out_dir); out_dir_p.mkdir(parents=True, exist_ok=True)
    out_path = out_dir_p / (in_path.stem + "_roman.json")

    # Load source (robust)
    src = _read_json_any_encoding(in_path)
    src_segments: List[Dict[str, Any]] = src.get("segments") or []

    # Prepare output skeleton and save immediately
    out_obj: Dict[str, Any] = {
        "audio": src.get("audio"),
        "duration": src.get("duration"),
        "language": src.get("language"),
        "segments": [],
    }
    for seg in src_segments:
        seg_out: Dict[str, Any] = {
            "id": seg.get("id"),
            "start": float(seg.get("start", 0.0)),
            "end": float(seg.get("end", 0.0)),
            "text_native": seg.get("text_native") or (seg.get("text") or ""),
            "text": seg.get("text") or "",   # will be replaced with Roman
        }
        if seg.get("words"):
            words_out: List[Dict[str, Any]] = []
            for w in seg["words"]:
                orig = (w.get("word") or w.get("text") or "").strip()
                words_out.append({
                    "start": float(w.get("start", seg_out["start"])),
                    "end": float(w.get("end", seg_out["start"])),
                    "word": orig,      # will become Roman
                    "native": w.get("native") or orig,
                })
            seg_out["words"] = words_out
        out_obj["segments"].append(seg_out)

    _atomic_save_json(out_obj, out_path)
    step(f"[TRL] Output JSON created: {out_path}", verbose)
    step("[TRL] Transliteration started (single contextual pass)…", verbose)

    total = len(src_segments)
    t0 = time.time()

    # Process in batches by segment id order for stable progress
    order = sorted(range(total), key=lambda i: (src_segments[i].get("id"), i))
    for bstart in range(0, total, batch_size):
        bend = min(bstart + batch_size, total)
        batch_idx = order[bstart:bend]

        # Build payload for this batch
        items: List[Dict[str, Any]] = []
        for i in batch_idx:
            seg = src_segments[i]
            words = [(w.get("word") or w.get("text") or "").strip() for w in (seg.get("words") or [])]
            items.append({
                "i": i,
                "text": (seg.get("text") or seg.get("text_native") or ""),
                "words": words
            })

        # Call OpenAI once per batch
        try:
            arr = _openai_chat_transliterate(items, model=model)
        except Exception as e:
            step(f"[TRL] Batch failed; falling back to identity for this batch. Reason: {e}", True)
            arr = [{"i": it["i"], "text_roman": it["text"], "words_roman": it["words"]} for it in items]

        # Index results by i
        results: Dict[int, Dict[str, Any]] = {}
        for it in arr:
            if not isinstance(it, dict):
                continue
            idx = it.get("i")
            if idx is None:
                continue
            results[int(idx)] = {
                "text_roman": (it.get("text_roman") or "").strip(),
                "words_roman": list(it.get("words_roman") or []),
            }

        # Apply to output
        for i in batch_idx:
            src_seg = src_segments[i]
            src_words = [(w.get("word") or w.get("text") or "").strip() for w in (src_seg.get("words") or [])]
            item = results.get(i, {
                "text_roman": (src_seg.get("text") or src_seg.get("text_native") or ""),
                "words_roman": src_words[:],
            })

            out_seg = out_obj["segments"][i]

            # Update line text first
            out_seg["text"] = item.get("text_roman") or (src_seg.get("text") or src_seg.get("text_native") or "")

            # Update words with strict Latin-only policy (no padding with originals)
            if "words" in out_seg and out_seg["words"]:
                wr = item.get("words_roman") or []
                # If mismatch or any token still contains native script, repair via LLM using roman line.
                if len(wr) != len(out_seg["words"]) or any(contains_native(tok) for tok in wr):
                    original_words = [(w.get("native") or w.get("word") or "") for w in out_seg["words"]]
                    wr = _repair_words_with_llm(original_words, out_seg["text"], model)
                # Latin-only guardrail and exact length
                wr = [_ensure_latin_only(t) for t in wr]
                if len(wr) != len(out_seg["words"]):
                    wr = (wr + [""] * len(out_seg["words"]))[:len(out_seg["words"])]
                for j, w in enumerate(out_seg["words"]):
                    w["word"] = wr[j] or _ensure_latin_only(w.get("word") or w.get("native") or "")

        # Persist after each batch
        _atomic_save_json(out_obj, out_path)
        _progress_bar(bend, total, t0, prefix="[TRL]", verbose=verbose)

    # ========================= QA loop =========================
    if qa_passes > 0:
        for pidx in range(qa_passes):
            step(f"[QA] Pass {pidx+1}…", verbose)
            fixed = _qa_fix_pass(out_obj, model=model, batch_size=batch_size, verbose=verbose)
            if fixed <= 0:
                step("[QA] No issues found.", verbose)
                break
            _atomic_save_json(out_obj, out_path)

    step(f"[TRL] Saved JSON: {out_path}", verbose)
    return str(out_path.resolve())


# --------------------------------- CLI ----------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Transliterate transcript JSON to Roman Urdu via OpenAI with QA")
    ap.add_argument("--transcript", required=True)
    ap.add_argument("--out", default="transliteration")
    ap.add_argument("--llm-model", default=getattr(CFG, "llm_model", "gpt-4o-mini"))
    ap.add_argument("--openai-url", default="", help="(unused; legacy compatibility)")
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--qa-passes", type=int, default=QA_PASSES, help="Number of QA passes to re-fix native segments")
    args = ap.parse_args()
    print(transliterate_transcript(
        args.transcript,
        out_dir=args.out,
        model=args.llm_model,
        api_base=args.openai_url,          # ignored; kept for CLI compatibility
        batch_size=args.batch_size,
        qa_passes=args.qa_passes,
    ))
