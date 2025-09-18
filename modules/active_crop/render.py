from __future__ import annotations
from typing import List, Tuple

def _between_expr(s: float, e: float) -> str:
    # Escape commas so FFmpeg doesn't split the filter graph
    return f"between(t\\,{s:.3f}\\,{e:.3f})"

def _sum_of_products(pieces: List[Tuple[float, float, float]], default_val: float) -> str:
    """
    Returns: (v1*between(...) + v2*between(...) + ... + default_val)
    """
    terms = [f"({v:.3f})*{_between_expr(s, e)}" for (s, e, v) in pieces]
    terms.append(f"({default_val:.3f})")
    return "(" + "+".join(terms) + ")"

def _min(a: str, b: str) -> str:
    return f"min({a}\\,{b})"

def _max(a: str, b: str) -> str:
    return f"max({a}\\,{b})"

def build_crop_vf(
    schedule: List[Tuple[float, float, float, float, float, float]],
    play_w: int,
    play_h: int,
    src_w: int,
    src_h: int,
) -> str:
    """
    Build a safe crop expression that:
      - selects (x,y,w,h) by time with sum-of-products gates,
      - clamps w,h to [2 .. src_w/src_h],
      - clamps x,y to keep the crop fully inside the source,
      - always valid even when no schedule piece matches at some t.

    Returns a single -vf string ending with scale=play_w:play_h.
    """
    if not schedule:
        raise RuntimeError("empty crop schedule")

    x_p: List[Tuple[float, float, float]] = []
    y_p: List[Tuple[float, float, float]] = []
    w_p: List[Tuple[float, float, float]] = []
    h_p: List[Tuple[float, float, float]] = []

    for (s, e, x, y, w, h) in schedule:
        x_p.append((s, e, float(x)))
        y_p.append((s, e, float(y)))
        w_p.append((s, e, float(w)))
        h_p.append((s, e, float(h)))

    # Defaults when no interval is active at t
    def_w = float(min(play_w, src_w))
    def_h = float(min(play_h, src_h))
    def_x = 0.0
    def_y = 0.0

    # Raw time-gated expressions
    x_raw = _sum_of_products(x_p, default_val=def_x)
    y_raw = _sum_of_products(y_p, default_val=def_y)
    w_raw = _sum_of_products(w_p, default_val=def_w)
    h_raw = _sum_of_products(h_p, default_val=def_h)

    # Clamp widths/heights to be inside the source and >= 2
    w_clamped = _min(str(src_w), _max("2", w_raw))
    h_clamped = _min(str(src_h), _max("2", h_raw))

    # Clamp positions so the crop stays fully inside the frame
    # x_clamped = max(0, min(x_raw, src_w - w_clamped))
    # y_clamped = max(0, min(y_raw, src_h - h_clamped))
    x_inner = _min(x_raw, f"{src_w}-{w_clamped}")
    x_clamped = _max("0", x_inner)
    y_inner = _min(y_raw, f"{src_h}-{h_clamped}")
    y_clamped = _max("0", y_inner)

    return (
        f"crop=floor({w_clamped}):floor({h_clamped}):floor({x_clamped}):floor({y_clamped}),"
        f"scale={int(play_w)}:{int(play_h)}"
    )
