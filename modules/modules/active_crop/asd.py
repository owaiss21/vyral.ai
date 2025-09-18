# modules/active_crop/asd.py
from __future__ import annotations
from typing import Dict, List, Any, Tuple
import numpy as np
import cv2
import torch
import face_alignment


# --- Robustly choose the "2D landmarks" enum across face-alignment versions ---
def _landmarks_2d_enum():
    """
    Return the 2D landmarks enum member from face_alignment.LandmarksType,
    handling naming differences across versions.
    """
    LT = getattr(face_alignment, "LandmarksType", None)
    if LT is None:
        raise RuntimeError("face_alignment.LandmarksType not found")

    # Try common names first
    for name in ("_2D", "TWO_D", "FACE_ALIGNMENT_2D", "LANDMARKS_2D"):
        if hasattr(LT, name):
            return getattr(LT, name)

    # Fallback: iterate enum members and pick the one whose name contains "2D"
    try:
        for m in LT:  # Enum is iterable in some versions
            nm = getattr(m, "name", str(m))
            if "2D" in nm.replace("-", "").upper():
                return m
    except TypeError:
        pass

    raise RuntimeError("Could not find a 2D landmarks enum in face_alignment.LandmarksType")


# 68-point mouth landmarks (iBUG)
MOUTH_OUTER = list(range(48, 60))  # 48..59


def _mouth_opening_ratio(pts: np.ndarray) -> float:
    # pts: (68, 2)
    top = pts[51]
    bottom = pts[57]
    left = pts[48]
    right = pts[54]
    vert = float(np.linalg.norm(top - bottom))
    width = float(np.linalg.norm(right - left)) + 1e-6
    return vert / width


def _nearest_box_at(tr_boxes: List[Tuple[float, int, int, int, int]], t: float) -> Tuple[int, int, int, int] | None:
    """
    Boxes are [(tb, x, y, w, h)] sorted by time; return the last box with tb <= t.
    """
    chosen = None
    for (tb, x, y, w, h) in reversed(tr_boxes):
        if tb <= t:
            chosen = (int(x), int(y), int(w), int(h))
            break
    return chosen


def score_lip_motion(
    video_path: str,
    tracks: List[Dict[str, Any]],
    start_sec: float,
    end_sec: float,
    fps_hint: float | None = None,
    smoothing: float = 0.8,
) -> Dict[int, List[Tuple[float, float]]]:
    """
    Compute a simple active-speech score per tracked face based on normalized mouth opening.
    Returns: {track_id: [(t, score), ...]}
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    landmarks_type = _landmarks_2d_enum()
    fa = face_alignment.FaceAlignment(landmarks_type, device=device, flip_input=False)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")

    fps = float(fps_hint or (cap.get(cv2.CAP_PROP_FPS) or 25.0))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if W <= 0 or H <= 0:
        cap.release()
        raise RuntimeError("invalid video dimensions")

    start_f = max(0, int(round(start_sec * fps)))
    end_f = max(start_f + 1, int(round(end_sec * fps)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

    per_track: Dict[int, List[Tuple[float, float]]] = {int(tr["id"]): [] for tr in tracks}
    last_score: Dict[int, float] = {int(tr["id"]): 0.0 for tr in tracks}

    f = start_f
    while f < end_f:
        ok, frame = cap.read()
        if not ok:
            break
        t = f / fps

        for tr in tracks:
            tid = int(tr["id"])
            boxes = tr.get("boxes") or []
            bx = _nearest_box_at(boxes, t)
            if bx is None:
                per_track[tid].append((t, last_score[tid]))
                continue

            x, y, w, h = bx
            x2, y2 = min(x + w, W), min(y + h, H)
            if x2 <= x or y2 <= y:
                per_track[tid].append((t, last_score[tid]))
                continue

            crop = frame[y:y2, x:x2]
            if crop.size == 0:
                per_track[tid].append((t, last_score[tid]))
                continue

            # face_alignment expects RGB
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            preds = fa.get_landmarks(rgb)
            if not preds:
                per_track[tid].append((t, last_score[tid]))
                continue

            pts = preds[0]  # (68, 2)
            mor = _mouth_opening_ratio(pts)
            s_now = smoothing * last_score[tid] + (1.0 - smoothing) * float(mor)
            last_score[tid] = float(s_now)
            per_track[tid].append((t, float(s_now)))

        f += 1

    cap.release()
    return per_track
