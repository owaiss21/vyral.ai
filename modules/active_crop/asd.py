from __future__ import annotations
from typing import Dict, List, Any, Tuple
import warnings
import numpy as np
import cv2
import torch
import face_alignment

def _landmarks_2d_enum():
    LT = getattr(face_alignment, "LandmarksType", None)
    if LT is None:
        raise RuntimeError("face_alignment.LandmarksType not found")
    for name in ("_2D", "TWO_D", "FACE_ALIGNMENT_2D", "LANDMARKS_2D"):
        if hasattr(LT, name):
            return getattr(LT, name)
    try:
        for m in LT:
            nm = getattr(m, "name", str(m))
            if "2D" in nm.replace("-", "").upper():
                return m
    except TypeError:
        pass
    raise RuntimeError("Could not find a 2D landmarks enum in face_alignment.LandmarksType")

def _expand_box(x, y, w, h, factor, W, H):
    cx = x + w / 2.0
    cy = y + h / 2.0
    w2 = max(2.0, w * factor)
    h2 = max(2.0, h * factor)
    x2 = max(0.0, min(cx - w2 / 2.0, W - w2))
    y2 = max(0.0, min(cy - h2 / 2.0, H - h2))
    return int(x2), int(y2), int(min(w2, W)), int(min(h2, H))

def _mouth_opening_ratio(pts: np.ndarray) -> float:
    top = pts[51]; bottom = pts[57]; left = pts[48]; right = pts[54]
    vert = float(np.linalg.norm(top - bottom))
    width = float(np.linalg.norm(right - left)) + 1e-6
    return vert / width

def _nearest_box_at(tr_boxes: List[Tuple[float, int, int, int, int]], t: float) -> Tuple[int,int,int,int] | None:
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
    expand_factor: float = 1.6,
    motion_proxy: bool = True
) -> Dict[int, List[Tuple[float, float]]]:
    """
    Returns per-track series: {track_id: [(t, score), ...]}.
    Per-frame 'No faces were detected.' warnings are suppressed. After the interval,
    we emit one warning if *no* landmarks were found at all (scores still come from motion proxy).
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
    end_f   = max(start_f + 1, int(round(end_sec * fps)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

    per_track: Dict[int, List[Tuple[float, float]]] = {int(tr["id"]): [] for tr in tracks}
    last_score: Dict[int, float] = {int(tr["id"]): 0.0 for tr in tracks}
    last_crop_gray: Dict[int, np.ndarray] = {}
    any_landmarks_for_track: Dict[int, bool] = {int(tr["id"]): False for tr in tracks}

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
                per_track[tid].append((t, last_score[tid])); continue

            x, y, w, h = bx
            xe, ye, we, he = _expand_box(x, y, w, h, expand_factor, W, H)
            x2, y2 = min(xe + we, W), min(ye + he, H)
            if x2 <= xe or y2 <= ye:
                per_track[tid].append((t, last_score[tid])); continue

            crop = frame[ye:y2, xe:x2]
            if crop.size == 0:
                per_track[tid].append((t, last_score[tid])); continue

            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_LINEAR)

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="No faces were detected.", category=UserWarning)
                preds = fa.get_landmarks(rgb)

            score: float | None = None
            if preds:
                any_landmarks_for_track[tid] = True
                pts = preds[0]
                score = float(_mouth_opening_ratio(pts))
            elif motion_proxy:
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                h_half = gray.shape[0] // 2
                roi = gray[h_half:, :]
                prev = last_crop_gray.get(tid)
                if prev is not None and prev.shape == gray.shape:
                    diff = cv2.absdiff(prev[h_half:, :], roi)
                    score = float(np.mean(diff) / 255.0)
                last_crop_gray[tid] = gray

            if score is None:
                s_now = last_score[tid]
            else:
                s_now = smoothing * last_score[tid] + (1.0 - smoothing) * float(score)
            last_score[tid] = float(s_now)
            per_track[tid].append((t, float(s_now)))

        f += 1

    cap.release()

    if not any(any_landmarks_for_track.values()):
        warnings.warn("ASD: No facial landmarks detected in the interval; using motion proxy only.", UserWarning)

    return per_track
