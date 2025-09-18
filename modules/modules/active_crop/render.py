from __future__ import annotations
from typing import List, Tuple

def _expr_piecewise(pieces: List[Tuple[float, float, float]]) -> str:
    """
    Build a nested 'if(between(t,s,e),v,if(...))' returning a number.
    We keep one decimal to shorten the expression.
    """
    expr = None
    for (s, e, v) in pieces:
        term = f"if(between(t,{s:.3f},{e:.3f}),{v:.1f},"
        expr = term + (expr if expr is not None else "")
    if expr is None:
        return "0"
    return expr + "0" + (")" * len(pieces))

def build_crop_vf(
    schedule: List[Tuple[float, float, float, float, float, float]],
    play_w: int,
    play_h: int
) -> str:
    """
    Convert schedule into a single FFmpeg -vf fragment:
      crop=w(t):h(t):x(t):y(t),scale=play_w:play_h
    where w,x,y,h are piecewise-constant expressions of time.
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

    x_expr = _expr_piecewise(x_p)
    y_expr = _expr_piecewise(y_p)
    w_expr = _expr_piecewise(w_p)
    h_expr = _expr_piecewise(h_p)

    # crop requires integers; floor() to be safe
    vf = f"crop=floor({w_expr}):floor({h_expr}):floor({x_expr}):floor({y_expr}),scale={int(play_w)}:{int(play_h)}"
    return vf
