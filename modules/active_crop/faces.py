from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any
import cv2
import numpy as np
import torch
from ultralytics import YOLO

@dataclass
class Track:
    id: int
    boxes: List[Tuple[float, int, int, int, int]]  # (t, x, y, w, h)

_MODEL: YOLO | None = None

def _get_model() -> YOLO:
    global _MODEL
    if _MODEL is None:
        _MODEL = YOLO("yolov8n-face.pt")
    return _MODEL

def _iou(a: Tuple[int,int,int,int], b: Tuple[int,int,int,int]) -> float:
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0: return 0.0
    union = aw * ah + bw * bh - inter
    return inter / (union + 1e-6)

def _clip_box(x: int, y: int, w: int, h: int, W: int, H: int) -> Tuple[int,int,int,int]:
    x = max(0, min(x, W-1)); y = max(0, min(y, H-1))
    w = max(1, min(w, W - x)); h = max(1, min(h, H - y))
    return x, y, w, h

def extract_tracks(video_path: str, start_sec: float, end_sec: float):
    """
    YOLOv8-face detection per frame + IoU tracking.
    Returns (tracks, meta) where:
      - tracks: List[{'id': int, 'boxes': [(t, x, y, w, h), ...]}]
      - meta  : {'width': int, 'height': int, 'fps': float}
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if W <= 0 or H <= 0:
        cap.release()
        raise RuntimeError("invalid video dimensions")

    start_f = max(0, int(round(start_sec * fps)))
    end_f   = max(start_f + 1, int(round(end_sec * fps)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

    model = _get_model()
    device = 0 if torch.cuda.is_available() else "cpu"

    tracks: List[Track] = []
    next_id = 1
    iou_thr = 0.35
    conf_thr = 0.25
    min_face_area = max(900, int(0.0004 * W * H))  # ignore tiny faces

    f = start_f
    while f < end_f:
        ok, frame = cap.read()
        if not ok:
            break
        t = f / fps

        res = model.predict(frame, verbose=False, conf=conf_thr, imgsz=896, device=device)
        dets: List[Tuple[int,int,int,int]] = []
        if res and len(res) > 0:
            r0 = res[0]
            if r0.boxes is not None and r0.boxes.xyxy is not None:
                xyxy = r0.boxes.xyxy.cpu().numpy()
                for x1, y1, x2, y2 in xyxy:
                    x1 = int(max(0, x1)); y1 = int(max(0, y1))
                    x2 = int(min(W-1, x2)); y2 = int(min(H-1, y2))
                    w = max(1, x2 - x1); h = max(1, y2 - y1)
                    if w * h < min_face_area:
                        continue
                    x, y, w, h = _clip_box(x1, y1, w, h, W, H)
                    dets.append((x, y, w, h))

        # associate by IoU
        for det in dets:
            best_iou, best_idx = 0.0, -1
            for idx, tr in enumerate(tracks):
                if not tr.boxes:
                    continue
                _, x0, y0, w0, h0 = tr.boxes[-1]
                iou = _iou((x0, y0, w0, h0), det)
                if iou > best_iou:
                    best_iou, best_idx = iou, idx
            x, y, w, h = det
            if best_iou >= iou_thr:
                tracks[best_idx].boxes.append((t, x, y, w, h))
            else:
                tracks.append(Track(id=next_id, boxes=[(t, x, y, w, h)]))
                next_id += 1

        f += 1

    cap.release()

    # sanitize meta
    fps = float(fps if fps and fps > 0 else 25.0)
    W = int(W if W and W > 0 else 0)
    H = int(H if H and H > 0 else 0)

    tracks_out: List[Dict[str, Any]] = [{"id": tr.id, "boxes": tr.boxes} for tr in tracks]
    meta = {"width": W, "height": H, "fps": fps}
    return tracks_out, meta
