from __future__ import annotations
import json
from pathlib import Path
import subprocess
import shlex
import re

def ensure_dir(p: Path | str) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def read_json(path: Path | str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json(path: Path | str, data) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def safe_stem(name: str) -> str:
    # file-system friendly name
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("_")

def run_ffprobe_get_dims(video_path: str) -> tuple[int, int]:
    """Return (width, height) using ffprobe; raises on failure."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0",
        str(video_path)
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    w_str, h_str = out.split(",")
    return int(w_str), int(h_str)

def ffmpeg_escape_path(p: Path | str) -> str:
    """Escape path for ffmpeg filter inputs on Windows/macOS/Linux."""
    s = str(Path(p).resolve())
    # Replace backslashes
    s = s.replace('\\', '\\\\')
    # Escape special characters for filtergraph
    for ch in [':', ',', ';', '[', ']', "'"]:
        s = s.replace(ch, '\\' + ch)
    return s

def seconds_to_srt_time(t: float) -> str:
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int(round((t - int(t)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

if __name__ == "__main__":
    import argparse, sys
    ap = argparse.ArgumentParser(description="Utility helpers")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("probe", help="Probe video width/height via ffprobe")
    p1.add_argument("--video", required=True, help="Path to a video")

    p2 = sub.add_parser("srt-time", help="Print SRT timestamp for given seconds (float)")
    p2.add_argument("--seconds", type=float, required=True)

    args = ap.parse_args()
    if args.cmd == "probe":
        w,h = run_ffprobe_get_dims(args.video)
        print(f"{w}x{h}")
    elif args.cmd == "srt-time":
        print(seconds_to_srt_time(args.seconds))


# ──────────────────────────────────────────────────────────────────────
# Lightweight progress logger with ETA
# ──────────────────────────────────────────────────────────────────────
import time as _time

def progress(current: float, total: float, start_ts: float, prefix: str = "", verbose: bool = True) -> None:
    """
    Simple line-by-line progress (no carriage return). Safe if total==0.
    """
    if not verbose or total is None or total <= 0:
        return
    try:
        current = float(current); total = float(total)
    except Exception:
        return
    current = max(0.0, min(current, total))
    elapsed = _time.time() - start_ts
    pct = (current / total) if total else 0.0
    eta = (elapsed / pct) * (1 - pct) if pct > 0 else 0.0
    print(f"{prefix} {current:.1f}/{total:.1f} ({pct*100:.1f}%)  elapsed={elapsed:.1f}s  eta={eta:.1f}s")

def step(msg: str, verbose: bool = True) -> None:
    """
    One-off log line controlled by verbosity.
    """
    if verbose:
        print(msg)
