from __future__ import annotations
from typing import Dict, List, Tuple, Any, Optional
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
    margin_in: float | None = 1.6,         # kept for API compatibility (unused)
    headroom_in: float | None = 0.12,      # kept for API compatibility (unused)
    smooth_alpha_in: float | None = 0.6,   # EMA smoothing for cx
) -> List[Tuple[float, float, float, float, float, float]]:
    """
    Build a dynamic crop schedule that is "throttled" to prevent
    overly long FFmpeg filter strings.
    
    Logic (Throttled EMA):
      - h = H (full height), w = floor(H * 9 / 16)
      - Create a timeline of events (speaker changes).
      - Add new time points every 100ms (MIN_STEP_MS) during speech.
      - At each time point, find the nearest face (cx) for the active speaker.
      - Apply EMA smoothing to this cx value.
      - This creates a schedule that is smooth but sparse, fixing WinError 206.

    Returns: List[(t0, t1, x, y, w, h)]
    """
    # Source / target geometry
    W  = _as_int(W_in, "source width W")
    H  = _as_int(H_in, "source height H")
    _  = _as_int(play_w_in, "portrait width play_w")
    __ = _as_int(play_h_in, "portrait height play_h")
    if W <= 0 or H <= 0:
        raise ValueError(f"invalid source dimensions W={W}, H={H}")

    # Portrait crop using full height
    crop_h = float(H)
    crop_w = float(max(2, min(W, int(round(H * 9.0 / 16.0)))))
    default_cx = W / 2.0
    
    alpha = max(0.0, min(0.98, float(smooth_alpha_in or 0.6)))

    # Build track map (raw face boxes, sorted by time)
    tmap: Dict[int, List[Tuple[float, float, float, float, float]]] = {}
    for tr in tracks or []:
        tid = int(tr.get("id"))
        cleaned: List[Tuple[float, float, float, float, float]] = []
        for box in tr.get("boxes") or []:
            if box is None or len(box) != 5: continue
            try:
                tb, x, y, w, h = [_as_float(v, "box.val") for v in box]
            except Exception:
                continue
            if w <= 1 or h <= 1: continue
            cleaned.append((tb, x, y, w, h))
        if cleaned:
            cleaned.sort(key=lambda t: t[0])
            tmap[tid] = cleaned

    # Validate fused segments
    fused_intervals: List[Tuple[float, float, Optional[int]]] = []
    if fused:
        for seg in fused:
            if seg is None or len(seg) != 3: continue
            s_raw, e_raw, tid_raw = seg
            try:
                s, e = _as_float(s_raw, "f.s"), _as_float(e_raw, "f.e")
                tid: Optional[int] = int(tid_raw) if tid_raw is not None else None
            except Exception:
                continue
            if e <= s: continue
            fused_intervals.append((s, e, tid))

    # Helper to find active speaker at a specific time
    def get_tid_at(t: float) -> Optional[int]:
        for (s, e, tid) in fused_intervals:
            if s <= t < e:
                return tid
        # Check the last segment's end time explicitly
        if fused_intervals:
             last_s, last_e, last_tid = fused_intervals[-1]
             if t == last_e:
                 return last_tid
        return None # No active speaker (silence)

    # Helper to find the nearest face box (cx) for a track at a specific time
    def get_nearest_cx(tid: int, t: float) -> Optional[float]:
        arr = tmap.get(tid) or []
        if not arr:
            return None
        # Find box with minimum time difference
        best_box = min(arr, key=lambda box: abs(box[0] - t))
        return best_box[1] + best_box[3] / 2.0 # cx = x + w/2

    # Find the "most visible" track as a fallback
    most_supported_tid: Optional[int] = None
    if tmap:
        most_supported_tid = max(tmap.keys(), key=lambda tid: len(tmap[tid]))

    # --- Fallback ---
    if not fused_intervals:
        x = max(0.0, min(default_cx - crop_w / 2.0, W - crop_w))
        return [(0.0, 0.001, float(x), 0.0, float(crop_w), float(crop_h))]

    # --- Build Throttled Timeline ---
    all_times = set()
    fused_min_t = min(s for s,e,_ in fused_intervals)
    fused_max_t = max(e for s,e,_ in fused_intervals)

    # Add segment boundaries
    for (s, e, _) in fused_intervals:
        all_times.add(s)
        all_times.add(e)

    # --- START: [WinError 206] FIX ---
    # Add throttled time points (e.g., every 100ms) ONLY during speech
    MIN_STEP_MS = 100.0 / 1000.0 # 100ms
    
    for (s_seg, e_seg, tid) in fused_intervals:
        if tid is not None: # Only add steps during active speech
            t = s_seg
            while t < e_seg:
                all_times.add(t)
                t += MIN_STEP_MS
    # --- END: [WinError 206] FIX ---
            
    sorted_times = sorted(list(all_times))
    
    # --- Build Stair-Step Schedule ---
    schedule = []
    last_cx = default_cx # Start centered

    for i in range(len(sorted_times)):
        s = sorted_times[i]
        if i + 1 < len(sorted_times):
            e = sorted_times[i+1]
        else:
            e = max(s + 0.04, fused_max_t) # Hold last frame
            
        if e - s < 0.001: continue # Skip empty segments
        if s > fused_max_t or e < fused_min_t: continue # Skip segments outside range

        t_mid = (s + e) / 2.0
        tid = get_tid_at(t_mid)

        target_cx: Optional[float] = None
        if tid is not None:
            target_cx = get_nearest_cx(tid, t_mid)
        
        # If active speaker has no box, use fallback track
        if target_cx is None and most_supported_tid is not None:
            target_cx = get_nearest_cx(most_supported_tid, t_mid)
            
        # If no faces at all, use default center
        if target_cx is None:
            target_cx = default_cx

        # Apply EMA smoothing
        smoothed_cx = alpha * last_cx + (1.0 - alpha) * target_cx
        last_cx = smoothed_cx # Update for next iteration

        # Calculate final x, clamped to boundaries
        x = max(0.0, min(smoothed_cx - crop_w / 2.0, W - crop_w))
        
        schedule.append((float(s), float(e), float(x), 0.0, float(crop_w), float(crop_h)))
    
    # --- Merge consecutive segments with the same crop ---
    if not schedule:
         x = max(0.0, min(default_cx - crop_w / 2.0, W - crop_w))
         return [(fused_min_t, fused_max_t, float(x), 0.0, float(crop_w), float(crop_h))]

    merged_schedule = []
    s_start, e_end, x_last, y, w, h = schedule[0]
    
    for i in range(1, len(schedule)):
        s_next, e_next, x_next, y_next, w_next, h_next = schedule[i]
        
        # Check if crop x is very close and segments are contiguous
        if abs(x_next - x_last) < 1.0 and abs(s_next - e_end) < 0.01:
            # Merge: extend the end time
            e_end = e_next
        else:
            # Different crop or gap in time: save previous segment and start new one
            if e_end > s_start:
                merged_schedule.append((s_start, e_end, x_last, y, w, h))
            s_start, e_end, x_last = s_next, e_next, x_next
            
    # Add the last segment
    if e_end > s_start:
        merged_schedule.append((s_start, e_end, x_last, y, w, h))

    return merged_schedule