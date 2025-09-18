from __future__ import annotations
from typing import Dict, List, Tuple, Any
import math

def _as_int(x, name: str) -> int:
    if x is None:
        raise ValueError(f"{name} is None")
    try:
        xi = int(x)
    except Exception as e:
        raise ValueError(f"{name} not int-compatible: {x!r} ({e})")
    return xi

def _as_float(x, name: str) -> float:
    if x is None:
        raise ValueError(f"{name} is None")
    try:
        xf = float(x)
    except Exception as e:
        raise ValueError(f"{name} not float-compatible: {x!r} ({e})")
    if math.isnan(xf) or math.isinf(xf):
        raise ValueError(f"{name} invalid (NaN/Inf): {x!r}")
    return xf

def build_crop_schedule(
    fused: List[Tuple[float, float, int]],
    tracks: List[Dict[str, Any]],
    W_in: int | float | None,
    H_in: int | float | None,
    play_w_in: int | float | None,
    play_h_in: int | float | None,
    margin_in: float | None = 1.6,         # kept for API compatibility (unused in portrait-full-height mode)
    headroom_in: float | None = 0.12,      # kept for API compatibility (unused here)
    smooth_alpha_in: float | None = 0.6,   # kept for API compatibility (unused here)
) -> List[Tuple[float, float, float, float, float, float]]:
    """
    Build a crop schedule that ALWAYS produces 9:16 portrait windows using the FULL source height.
    Logic:
      - h = H (full height)
      - w = floor(H * 9 / 16)
      - For each interval [s,e] and track id, compute the face box nearest to time s (or last before s).
      - Center the crop horizontally at the face center x; clamp to [0, W - w]; set y = 0.
      - If fused is empty, create one interval covering the min..max time seen in tracks, centered on the
        most supported track (largest number of boxes). This keeps active-crop semantics without blur fallback.

    Returns: List[(t0, t1, x, y, w, h)]
    """
    # Source / target geometry
    W  = _as_int(W_in, "source width W")
    H  = _as_int(H_in, "source height H")
    _  = _as_int(play_w_in, "portrait width play_w")   # only validated; not used to compute crop
    __ = _as_int(play_h_in, "portrait height play_h")  # only validated; not used to compute crop
    if W <= 0 or H <= 0:
        raise ValueError(f"invalid source dimensions W={W}, H={H}")

    # Portrait crop using full height
    crop_h = float(H)
    crop_w = float(max(2, min(W, int(round(H * 9.0 / 16.0)))))  # clamp width to source width
    # If the source is already narrower than 9:16, keep crop_w = W (rare for landscape inputs)

    # Build track map (sorted by time, cleaned)
    tmap: Dict[int, List[Tuple[float, float, float, float, float]]] = {}
    for tr in tracks or []:
        tid = int(tr.get("id"))
        cleaned: List[Tuple[float, float, float, float, float]] = []
        for box in tr.get("boxes") or []:
            if box is None or len(box) != 5:
                continue
            tb, x, y, w, h = box
            try:
                tb = _as_float(tb, "box.t")
                x  = _as_float(x,  "box.x")
                y  = _as_float(y,  "box.y")
                w  = _as_float(w,  "box.w")
                h  = _as_float(h,  "box.h")
            except Exception:
                continue
            if w <= 1 or h <= 1:
                continue
            cleaned.append((tb, x, y, w, h))
        if cleaned:
            cleaned.sort(key=lambda t: t[0])
            tmap[tid] = cleaned

    def last_box_before(tid: int, t: float):
        arr = tmap.get(tid) or []
        last = None
        for (tb, x, y, w, h) in arr:
            if tb <= t:
                last = (x, y, w, h)
            else:
                break
        return last

    def track_support_count(tid: int) -> int:
        return len(tmap.get(tid) or [])

    # If fused is empty, synthesize a single interval from tracks' time span
    fused_intervals: List[Tuple[float, float, int]] = []
    if fused:
        # Validate fused and keep only sane segments
        for seg in fused:
            if seg is None or len(seg) != 3:
                continue
            s_raw, e_raw, tid_raw = seg
            try:
                s = _as_float(s_raw, "fused.start")
                e = _as_float(e_raw, "fused.end")
                tid = int(tid_raw)
            except Exception:
                continue
            if e <= s:
                continue
            fused_intervals.append((s, e, tid))
    else:
        # Derive a reasonable default interval and speaker id from tracks
        all_times: List[float] = []
        for arr in tmap.values():
            all_times.extend([tb for (tb, *_rest) in arr])
        if all_times:
            s_def = min(all_times)
            e_def = max(all_times)
            # choose the most supported track
            cand = sorted((tid for tid in tmap.keys()), key=lambda t: track_support_count(t), reverse=True)
            if cand:
                fused_intervals = [(s_def, e_def, cand[0])]

    schedule: List[Tuple[float, float, float, float, float, float]] = []
    if not fused_intervals:
        # If still nothing, return a single center-crop over the entire visible time window (or time 0..0.001)
        # This keeps active-crop alive but clearly indicates upstream detectors were too sparse.
        # We avoid raising to keep your run flowing for batches; you can detect this case by seeing a single interval.
        s_fallback = 0.0
        e_fallback = 0.001
        cx = W / 2.0
        x = max(0.0, min(cx - crop_w / 2.0, W - crop_w))
        schedule.append((s_fallback, e_fallback, float(x), 0.0, float(crop_w), float(crop_h)))
        return schedule

    for (s, e, tid) in fused_intervals:
        # Pick the last box before 's' for that track; if none, try the earliest box for that track
        box = last_box_before(tid, s)
        if box is None:
            arr = tmap.get(tid) or []
            if arr:
                _tb, x0, y0, w0, h0 = arr[0]
                box = (x0, y0, w0, h0)

        # If still None (no boxes for this tid), try the most supported track overall
        if box is None:
            cand = sorted((tid2 for tid2 in tmap.keys()), key=lambda t: track_support_count(t), reverse=True)
            if cand:
                arr = tmap.get(cand[0]) or []
                if arr:
                    _tb, x0, y0, w0, h0 = arr[0]
                    box = (x0, y0, w0, h0)

        # If we genuinely have no boxes at all, center in the frame to keep schedule valid
        if box is None:
            cx = W / 2.0
        else:
            x0, y0, w0, h0 = box
            cx = x0 + w0 / 2.0

        x = max(0.0, min(cx - crop_w / 2.0, W - crop_w))
        schedule.append((float(s), float(e), float(x), 0.0, float(crop_w), float(crop_h)))

    return schedule
