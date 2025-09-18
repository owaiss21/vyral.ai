from __future__ import annotations
from typing import Dict, List, Tuple, Optional

def fuse_asd_and_vad(
    asd_scores: Dict[int, List[Tuple[float, float]]],
    speech_regions: List[Tuple[float, float]],
    *,
    enter_thr: float = 0.55,
    exit_thr: float = 0.45,
    min_hold_ms: int = 300
) -> List[Tuple[float, float, Optional[int]]]:
    """
    Returns a list of segments [(t0, t1, track_id_or_None)] picking the active
    track when speech is present. Hysteresis avoids flicker.
    """
    # Build unified time grid from ASD samples
    times = sorted(set(tt for v in asd_scores.values() for (tt, _) in v))
    if not times:
        return [(0.0, 0.0, None)]
    # Map tid -> time->score dict
    S: Dict[int, Dict[float, float]] = {tid: {tt: sc for (tt, sc) in series} for tid, series in asd_scores.items()}

    ms_hold = float(min_hold_ms) / 1000.0
    active: Optional[int] = None
    segs: List[Tuple[float, float, Optional[int]]] = []
    seg_start = times[0]
    last_switch_t = times[0]

    def speech_on(t: float) -> bool:
        for s, e in speech_regions:
            if s <= t <= e:
                return True
        return False

    for i, t in enumerate(times):
        if not speech_on(t):
            # No speech: no active track
            next_active = None
        else:
            # Pick best track by ASD score
            best_tid, best_sc = None, -1.0
            for tid in S.keys():
                sc = S[tid].get(t, 0.0)
                if sc > best_sc:
                    best_sc, best_tid = sc, tid

            if active is None:
                next_active = best_tid if (best_sc >= enter_thr) else None
                if next_active is not None:
                    last_switch_t = t
            else:
                # keep if still above exit_thr or within hold
                curr_sc = S[active].get(t, 0.0)
                if curr_sc >= exit_thr or (t - last_switch_t) < ms_hold:
                    next_active = active
                else:
                    # switch if competitor is clearly stronger
                    if best_sc >= enter_thr and best_tid != active:
                        next_active = best_tid
                        last_switch_t = t
                    else:
                        next_active = None

        # segment bookkeeping
        if i == 0:
            active = next_active
        else:
            if next_active != active:
                segs.append((seg_start, t, active))
                seg_start = t
                active = next_active

    segs.append((seg_start, times[-1], active))
    return segs
