from __future__ import annotations
from typing import List, Tuple

# schedule items are tuples: (start, end, x, y, w, h)

def _fmtf(v: float) -> str:
    # stable float formatting for ffmpeg expressions
    return f"{float(v):.6f}"

def _between(s: float, e: float) -> str:
    # Escape commas for ffmpeg filter arg lists
    return f"between(t\\,{_fmtf(s)}\\,{_fmtf(e)})"

def _sum_gated(pieces: List[Tuple[float, float, float]], default_val: float) -> str:
    """
    Build a time-gated sum: sum(v_i * between(t, s_i, e_i)), with fallback default if none active.
    Returns an expression that equals default_val when no piece is active.
    """
    if not pieces:
        return _fmtf(default_val)

    terms = [f"({_fmtf(v)})*{_between(s, e)}" for (s, e, v) in pieces]
    mask_terms = [f"{_between(s, e)}" for (s, e, _) in pieces]
    sum_terms = "(" + "+".join(terms) + ")"
    sum_masks = "(" + "+".join(mask_terms) + ")"
    # default when none active: default_val * (1 - gt(sum_masks, 0))
    return f"({sum_terms}) + ({_fmtf(default_val)})*(1-gt({sum_masks}\\,0))"

def build_crop_vf(
    schedule: List[Tuple[float, float, float, float, float, float]],
    play_w: int,
    play_h: int,
    src_w: int,
    src_h: int,
) -> str:
    """
    Build a robust -vf chain that:
      1) Crops to the schedule-proposed window (time varying, inside the source; no padding).
      2) setsar=1 to avoid display squeeze.
      3) Scales to COVER a 9:16 canvas without changing AR.
      4) Crops to exactly play_w x play_h (no bars, no blur).

    The result is a true crop-only portrait with zero stretch.

    schedule: list of (start, end, x, y, w, h)
    play_w, play_h: desired portrait frame (e.g., 1080x1920)
    src_w, src_h: source frame dimensions
    """

    # target aspect
    A = float(play_w) / float(play_h)  # e.g., 9/16

    if not schedule:
        # Max 9:16 inside source, centered
        return (
            "crop="
            f"w='if(gte(iw/ih,{A:.9f}),ih*{A:.9f},iw)':"
            f"h='if(gte(iw/ih,{A:.9f}),ih,iw/{A:.9f})':"
            "x='(iw-w)/2':"
            "y='(ih-h)/2',"
            "setsar=1,"
            f"scale='if(gte(a\\,{A:.9f}),-2,{play_w})':'if(gte(a\\,{A:.9f}),{play_h},-2)':flags=lanczos,"
            f"crop={play_w}:{play_h}"
        )

    # Build time-gated raw x, y, w, h from schedule
    xs = [(s, e, x) for (s, e, x, y, w, h) in schedule]
    ys = [(s, e, y) for (s, e, x, y, w, h) in schedule]
    ws = [(s, e, w) for (s, e, x, y, w, h) in schedule]
    hs = [(s, e, h) for (s, e, x, y, w, h) in schedule]

    # Defaults if none active at a time: center a maximal portrait box inside the source
    default_h = min(float(src_h), float(src_w) / A)
    default_w = default_h * A
    default_x = (float(src_w) - default_w) * 0.5
    default_y = (float(src_h) - default_h) * 0.5

    x_raw = _sum_gated(xs, default_x)
    y_raw = _sum_gated(ys, default_y)
    w_raw = _sum_gated(ws, default_w)
    h_raw = _sum_gated(hs, default_h)

    # Clamp w/h to source size
    w_clamped = f"min({w_raw}\\,{src_w})"
    h_clamped = f"min({h_raw}\\,{src_h})"

    # Clamp x/y so the crop rectangle stays inside the source
    x_right_limit = f"({src_w})-({w_clamped})"
    y_bottom_limit = f"({src_h})-({h_clamped})"
    x_clamped = f"max(0\\,min(({x_raw})\\,{x_right_limit}))"
    y_clamped = f"max(0\\,min(({y_raw})\\,{y_bottom_limit}))"

    # Final chain:
    #  1) Crop to scheduled window (no padding)
    #  2) setsar=1
    #  3) Scale to cover target AR without stretching (branch on current a vs A)
    #  4) Crop to exact play_w x play_h
    vf = (
        "crop="
        f"floor({w_clamped}):"
        f"floor({h_clamped}):"
        f"floor({x_clamped}):"
        f"floor({y_clamped}),"
        "setsar=1,"
        f"scale='if(gte(a\\,{A:.9f}),-2,{play_w})':'if(gte(a\\,{A:.9f}),{play_h},-2)':flags=lanczos,"
        f"crop={play_w}:{play_h}"
    )

    return vf