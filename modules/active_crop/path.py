from __future__ import annotations
from typing import Dict, List, Tuple, Any
import math

def _as_int(x, name: str) -> int:
    if x is None:
        raise ValueError(f"{name} is None")
    try:
        xi = int(x)
    except Exception as e:
        raise ValueError(f"{name} not an int-compatible value: {x!r} ({e})")
    return xi

def _as_float(x, name: str) -> float:
    if x is None:
        raise ValueError(f"{name} is None")
    try:
        xf = float(x)
    except Exception as e:
        raise ValueError(f"{name} not a float-compatible value: {x!r} ({e})")
    if math.isnan(xf) or math.isinf(xf):
        raise ValueError(f"{name} invalid (NaN/Inf): {x!r}")
    return xf

def _aspect_fit_box(
    x: float, y: float, w: float, h: float,
    W: int, H: int, ar_w: int, ar_h: int
) -> tuple[float, float, float, float]:
    """Expand box to match target aspect (ar_w:ar_h), clamp inside frame."""
    target = ar_w / float(ar_h)
    cx, cy = x + w / 2.0, y + h / 2.0
    bw, bh = float(max(1.0, w)), float(max(1.0, h))
    cur = bw / bh
    if cur > target:    # too wide -> increase height
        bh = bw / target
    else:               # too tall -> increase width
        bw = bh * target
    nx = max(0.0, min(cx - bw / 2.0, W - bw))
    ny = max(0.0, min(cy - bh / 2.0, H - bh))
    bw = min(bw, W)
    bh = min(bh, H)
    return nx, ny, bw, bh

def build_crop_schedule(
    fused: List[Tuple[float, float, int]],
    tracks: List[Dict[str, Any]],
    W_in: int | float | None,
    H_in: int | float | None,
    play_w_in: int | float | None,
    play_h_in: int | float | None,
    margin_in: float | None = 1.6,
    headroom_in: float | None = 0.12,
    smooth_alpha_in: float | None = 0.6,
) -> List[Tuple[float, float, float, float, float, float]]:
    """
    Build a piecewise-constant crop schedule in source coords.
    Returns List[(t0, t1, x, y, w, h)] where (w,h) already matches the portrait aspect.
    """

    # --- Validate & coerce scalar params early (so errors are clear and close to source) ---
    W = _as_int(W_in, "source width W")
    H = _as_int(H_in, "source height H")
    play_w = _as_int(play_w_in, "portrait width play_w")
    play_h = _as_int(play_h_in, "portrait height play_h")
    if W <= 0 or H <= 0:
        raise ValueError(f"invalid source dimensions W={W}, H={H}")
    if play_w <= 0 or play_h <= 0:
        raise ValueError(f"invalid portrait dimensions play_w={play_w}, play_h={play_h}")

    margin = _as_float(margin_in, "margin")
    headroom = _as_float(headroom_in, "headroom")
    smooth_alpha = _as_float(smooth_alpha_in, "smooth_alpha")
    smooth_alpha = min(0.99, max(0.0, smooth_alpha))

    # --- Build track map; drop invalid boxes (None/NaN or non-positive size) ---
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
                x = _as_float(x, "box.x")
                y = _as_float(y, "box.y")
                w = _as_float(w, "box.w")
                h = _as_float(h, "box.h")
            except Exception:
                continue
            if w <= 1 or h <= 1:
                continue
            if x < 0 or y < 0 or x >= W or y >= H:
                # we still accept it; will be clamped later by _aspect_fit_box
                pass
            cleaned.append((tb, x, y, w, h))
        if cleaned:
            # ensure chronological order
            cleaned.sort(key=lambda t: t[0])
            tmap[tid] = cleaned

    if not tmap:
        raise ValueError("no valid boxes in tracks after cleaning")

    def last_box_before(tid: int, t: float):
        arr = tmap.get(tid) or []
        last = None
        for (tb, x, y, w, h) in arr:
            if tb <= t:
                last = (x, y, w, h)
            else:
                break
        return last

    schedule: List[Tuple[float, float, float, float, float, float]] = []
    prev_cxcy: tuple[float, float] | None = None

    for seg in fused or []:
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

        box = last_box_before(tid, s)
        if box is None:
            # No box before 's' for this tid—skip segment
            continue

        x, y, w, h = box
        # expand around face
        cx = x + w / 2.0
        cy = y + h / 2.0
        w2 = max(2.0, w * margin)
        h2 = max(2.0, h * margin)
        x2 = cx - w2 / 2.0
        y2 = cy - h2 / 2.0

        # fit portrait aspect
        x3, y3, w3, h3 = _aspect_fit_box(x2, y2, w2, h2, W, H, play_w, play_h)
        # headroom shift up
        y3 = max(0.0, y3 - headroom * h3)

        # smooth center across segments
        cx3, cy3 = x3 + w3 / 2.0, y3 + h3 / 2.0
        if prev_cxcy is None:
            cxf, cyf = cx3, cy3
        else:
            cxf = smooth_alpha * prev_cxcy[0] + (1.0 - smooth_alpha) * cx3
            cyf = smooth_alpha * prev_cxcy[1] + (1.0 - smooth_alpha) * cy3
        prev_cxcy = (cxf, cyf)

        xf = max(0.0, min(cxf - w3 / 2.0, W - w3))
        yf = max(0.0, min(cyf - h3 / 2.0, H - h3))
        schedule.append((float(s), float(e), float(xf), float(yf), float(w3), float(h3)))

    return schedule
