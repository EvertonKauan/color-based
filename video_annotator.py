"""
Video annotation for judo: colored boxes from detected athlete kimono and belt colors.
Filters black-gi referees and background when configured; keeps identified athletes.
"""

import os
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from ultralytics import YOLO
from identify_colors import (
    is_black_kimono,
    belt_scores_from_pixels,
    kimono_scores_from_pixels,
    pick_kimono_label_from_scores,
)

YOLO_CLASS_REFEREE = 0
YOLO_CLASS_ATHLETE = 1


def _safe_crop(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> Optional[np.ndarray]:
    h, w = frame.shape[:2]
    x1 = int(max(0, min(int(x1), w - 1)))
    y1 = int(max(0, min(int(y1), h - 1)))
    x2 = int(max(0, min(int(x2), w - 1)))
    y2 = int(max(0, min(int(y2), h - 1)))
    if x2 <= x1 or y2 <= y1:
        return None
    region = frame[y1:y2, x1:x2]
    return region if region is not None and region.size > 0 else None


def _normalize_roi_for_color(bgr: np.ndarray) -> np.ndarray:
    """Light illumination normalization (reduces shadow flicker); CLAHE on L in Lab."""
    if bgr is None or bgr.size == 0:
        return bgr
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)


def _sample_pixels_stride(rgb_img: np.ndarray, max_samples: int = 25000) -> np.ndarray:
    """Deterministic stride sampling to limit flicker (no np.random). Returns RGB (N,3)."""
    if rgb_img is None or rgb_img.size == 0:
        return np.empty((0, 3), dtype=np.uint8)
    flat = rgb_img.reshape(-1, 3)
    n = int(flat.shape[0])
    if n <= max_samples:
        return flat
    step = max(1, n // int(max_samples))
    return flat[::step]


def _belt_scores_identify_colors_from_region(bgr_region: Optional[np.ndarray]) -> Dict[str, float]:
    """
    Belt path aligned with identify_colors: CLAHE, RGB, stride sample,
    then ``belt_scores_from_pixels`` (``white`` / ``red`` / ``black`` fractions)
    mapped to ``red`` / ``white`` / ``black`` / ``unknown``.
    """
    out = {"red": 0.0, "white": 0.0, "black": 0.0, "unknown": 1.0}
    if bgr_region is None or bgr_region.size == 0:
        return out

    bgr = _normalize_roi_for_color(bgr_region)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Central crop like identify_colors (reduces judogi edge / mat bleed)
    h, w = rgb.shape[:2]
    mx = int(w * 0.08)
    my = int(h * 0.12)
    roi = rgb[my:h - my, mx:w - mx] if (h > 2 * my and w > 2 * mx) else rgb
    flat = roi.reshape(-1, 3)
    if flat.size == 0:
        return out

    # Deterministic sampling to reduce flicker
    px = _sample_pixels_stride(flat.reshape(-1, 1, 3), max_samples=40000).reshape(-1, 3)
    if px.size == 0:
        return out

    scores_ic = belt_scores_from_pixels(px)
    red = float(scores_ic.get("red", 0.0))
    white = float(scores_ic.get("white", 0.0))
    black = float(scores_ic.get("black", 0.0))
    out = {
        "red": red,
        "white": white,
        "black": black,
        "unknown": float(max(0.0, 1.0 - max(red, white, black))),
    }
    return out


def _kimono_scores_identify_colors_from_region(bgr_region: Optional[np.ndarray]) -> Dict[str, float]:
    """
    Kimono path aligned with identify_colors: CLAHE, RGB, stride sample,
    then ``kimono_scores_from_pixels`` (``white`` / ``blue`` / ``black``)
    mapped to ``white`` / ``blue`` / ``black`` / ``unknown``.
    """
    out = {"white": 0.0, "blue": 0.0, "black": 0.0, "unknown": 1.0}
    if bgr_region is None or bgr_region.size == 0:
        return out

    bgr = _normalize_roi_for_color(bgr_region)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Central crop (reduces mat / noise), same idea as identify_colors
    h, w = rgb.shape[:2]
    mx = int(w * 0.10)
    my = int(h * 0.10)
    roi = rgb[my:h - my, mx:w - mx] if (h > 2 * my and w > 2 * mx) else rgb
    flat = roi.reshape(-1, 3)
    if flat.size == 0:
        return out

    # Deterministic sampling to reduce flicker
    px = _sample_pixels_stride(flat.reshape(-1, 1, 3), max_samples=50000).reshape(-1, 3)
    if px.size == 0:
        return out

    scores_ic = kimono_scores_from_pixels(px)
    white = float(scores_ic.get("white", 0.0))
    blue = float(scores_ic.get("blue", 0.0))
    black = float(scores_ic.get("black", 0.0))
    out = {
        "white": white,
        "blue": blue,
        "black": black,
        "unknown": float(max(0.0, 1.0 - max(white, blue, black))),
    }
    return out


def classify_kimono_color_from_scores_identify_colors(kim_scores: Dict[str, float]) -> str:
    """Same decision as ``pick_kimono_label_from_scores``; returns Title-case English."""
    if not kim_scores:
        return "Unknown"
    scores_ic = {
        "white": float(kim_scores.get("white", 0.0)),
        "blue": float(kim_scores.get("blue", 0.0)),
        "black": float(kim_scores.get("black", 0.0)),
    }
    label = pick_kimono_label_from_scores(scores_ic)
    if label == "blue":
        return "Blue"
    if label == "white":
        return "White"
    return "Unknown"

def classify_belt_color_from_scores_identify_colors(belt_scores: Dict[str, float]) -> str:
    """Same red-vs-white heuristic as identify_colors, then max-score belt label."""
    if not belt_scores:
        return "Unknown"
    red_sc = float(belt_scores.get("red", 0.0))
    white_sc = float(belt_scores.get("white", 0.0))
    black_sc = float(belt_scores.get("black", 0.0))

    # identify_colors belt heuristic
    if red_sc >= 0.07 and red_sc >= white_sc * 0.45:
        return "Red"

    # pick max
    best = max(("white", "red", "black"), key=lambda k: float(belt_scores.get(k, 0.0)))
    if float(belt_scores.get(best, 0.0)) <= 0.0:
        return "Unknown"
    if best == "white":
        return "White"
    if best == "red":
        return "Red"
    return "Black"


def _region_color_scores(bgr_region: Optional[np.ndarray]) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Approximate color fractions (HSV + Lab pipeline via identify_colors helpers).

    Returns:
      kimono_scores: {"white","blue","black","unknown"}
      belt_scores:   {"red","white","black","unknown"}
    """
    kim = {"white": 0.0, "blue": 0.0, "black": 0.0, "unknown": 1.0}
    belt = {"red": 0.0, "white": 0.0, "black": 0.0, "unknown": 1.0}
    if bgr_region is None or bgr_region.size == 0:
        return kim, belt

    bgr = _normalize_roi_for_color(bgr_region)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    px = _sample_pixels_stride(rgb, max_samples=25000)
    if px.size == 0:
        return kim, belt

    kim = _kimono_scores_identify_colors_from_region(bgr_region)
    belt = _belt_scores_identify_colors_from_region(bgr_region)
    return kim, belt


def _extract_torso_and_belt_from_keypoints(
    frame: np.ndarray,
    keypoints_xy_i: Optional[np.ndarray],
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Use COCO keypoints for cleaner ROIs:
      - torso (kimono): shoulder/hip anchors (5,6,11,12)
      - belt: horizontal band around hip line
    Fallback: bbox crop helpers.
    """
    fallback_k = extract_kimono_region(frame, x1, y1, x2, y2)
    fallback_b = extract_belt_region(frame, x1, y1, x2, y2)

    if keypoints_xy_i is None or not isinstance(keypoints_xy_i, np.ndarray) or keypoints_xy_i.size == 0:
        return fallback_k, fallback_b
    if keypoints_xy_i.ndim != 2 or keypoints_xy_i.shape[1] != 2 or keypoints_xy_i.shape[0] <= 12:
        return fallback_k, fallback_b

    needed = [5, 6, 11, 12]
    pts = keypoints_xy_i[needed].astype(np.float32)
    if not np.all(np.isfinite(pts)):
        return fallback_k, fallback_b
    if np.any((pts[:, 0] <= 1.0) & (pts[:, 1] <= 1.0)):
        return fallback_k, fallback_b

    xs = pts[:, 0]
    ys = pts[:, 1]
    minx, maxx = float(xs.min()), float(xs.max())
    miny, maxy = float(ys.min()), float(ys.max())

    box_w = max(2, int(x2 - x1))
    box_h = max(2, int(y2 - y1))
    pad_x = int(0.08 * box_w)
    pad_y = int(0.10 * box_h)

    torso_x1 = max(x1, int(minx) - pad_x)
    torso_x2 = min(x2, int(maxx) + pad_x)
    torso_y1 = max(y1, int(miny) - pad_y)
    torso_y2 = min(y2, int(maxy) + pad_y)

    kimono = _safe_crop(frame, torso_x1, torso_y1, torso_x2, torso_y2)
    if kimono is None or kimono.size == 0:
        kimono = fallback_k

    hip_y = float((keypoints_xy_i[11, 1] + keypoints_xy_i[12, 1]) / 2.0)
    belt_h = int(max(2, 0.16 * box_h))
    belt_y1 = max(y1, int(hip_y - belt_h * 0.55))
    belt_y2 = min(y2, int(hip_y + belt_h * 0.45))

    # Belt ROI is usually centered on the torso.
    # Shrink X to reduce white judogi spill on the sides.
    torso_w = max(2, int(torso_x2 - torso_x1))
    # More aggressive crop to reduce mat/occlusion bleed (same-class case).
    shrink = int(0.18 * torso_w)
    belt_x1 = min(torso_x2 - 2, torso_x1 + shrink)
    belt_x2 = max(torso_x1 + 2, torso_x2 - shrink)
    belt = _safe_crop(frame, belt_x1, belt_y1, belt_x2, belt_y2)
    if belt is None or belt.size == 0:
        belt = fallback_b

    return kimono, belt


def _ema_update(prev: Dict[str, float], cur: Dict[str, float], alpha: float) -> Dict[str, float]:
    a = float(alpha)
    keys = set(prev.keys()) | set(cur.keys())
    out: Dict[str, float] = {}
    for k in keys:
        out[k] = float((1 - a) * float(prev.get(k, 0.0)) + a * float(cur.get(k, 0.0)))
    return out


def _norm_expected_color(label: str) -> str:
    s = str(label or "").strip().lower()
    if s in ("white",):
        return "white"
    if s in ("blue",):
        return "blue"
    if s in ("red",):
        return "red"
    if s in ("black",):
        return "black"
    if s in ("undefined", "unknown", ""):
        return "unknown"
    return s


def _label_from_scores(scores: Dict[str, float], candidates: List[str], min_conf: float, out_case: str = "Title") -> str:
    best = None
    bestv = -1.0
    for k in candidates:
        v = float(scores.get(k, 0.0))
        if v > bestv:
            bestv = v
            best = k
    if best is None or bestv < float(min_conf):
        return "Unknown"
    if best == "black":
        return "Black"
    if out_case == "Title":
        return best.capitalize()
    return best


def _expected_combo_for_athlete(aid: str, expected_athletes: Dict[str, Dict[str, str]]) -> Tuple[str, str]:
    """Return (Kimono, Belt) display strings for the athlete (normalized casing)."""
    exp = expected_athletes.get(str(aid), {}) or {}
    k = _norm_expected_color(exp.get("Kimono"))
    b = _norm_expected_color(exp.get("Belt"))
    k_out = "Unknown" if k == "unknown" else k.capitalize()
    b_out = "Unknown" if b == "unknown" else b.capitalize()
    return k_out, b_out


def extract_kimono_region(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    """
    Extract kimono region (upper body, excluding head and belt).
    """
    h, w = frame.shape[:2]
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))
    
    person_h = y2 - y1
    person_w = x2 - x1
    
    if person_h <= 2 or person_w <= 2:
        return None

    # Grappling / fall: if the person is lying (wider-than-tall bbox),
    # a fixed vertical slice tends to grab mat/occluder; sample a central block.
    if person_w > person_h:
        cx = x1 + person_w // 2
        cy = y1 + person_h // 2
        sample_w = max(2, int(person_w * 0.35))
        sample_h = max(2, int(person_h * 0.35))
        kx1 = max(x1, cx - sample_w // 2)
        kx2 = min(x2, cx + sample_w // 2)
        ky1 = max(y1, cy - sample_h // 2)
        ky2 = min(y2, cy + sample_h // 2)
        kimono_region = frame[ky1:ky2, kx1:kx2]
        return kimono_region if kimono_region is not None and kimono_region.size > 0 else None
    
    # Kimono band: from top (~12% skip for head) to waist (~56% height).
    kimono_top_ratio = 0.12
    belt_center_ratio = 0.56
    
    kimono_y1 = y1 + int(person_h * kimono_top_ratio)
    kimono_y2 = y1 + int(person_h * belt_center_ratio)
    
    if kimono_y2 <= kimono_y1:
        kimono_y2 = y1 + int(person_h * 0.7)  # fallback
    
    kimono_region = frame[kimono_y1:kimono_y2, x1:x2]
    return kimono_region


def extract_belt_region(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    """
    Extract belt (waist) region from a detection.
    """
    h, w = frame.shape[:2]
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))
    
    person_h = y2 - y1
    person_w = x2 - x1
    
    if person_h <= 2 or person_w <= 2:
        return None

    # Horizontal bbox: belt may be hidden; sample a smaller central block.
    # to avoid mat/occluder.
    if person_w > person_h:
        cx = x1 + person_w // 2
        cy = y1 + person_h // 2
        sample_w = max(2, int(person_w * 0.30))
        sample_h = max(2, int(person_h * 0.25))
        bx1 = max(x1, cx - sample_w // 2)
        bx2 = min(x2, cx + sample_w // 2)
        by1 = max(y1, cy - sample_h // 2)
        by2 = min(y2, cy + sample_h // 2)
        belt_region = frame[by1:by2, bx1:bx2]
        return belt_region if belt_region is not None and belt_region.size > 0 else None
    
    # Belt band: ~18% height centered at ~56% of person height.
    belt_height_ratio = 0.18
    belt_center_ratio = 0.56
    
    belt_h = max(2, int(person_h * belt_height_ratio))
    belt_center_y = y1 + int(person_h * belt_center_ratio)
    belt_y1 = belt_center_y - belt_h // 2
    belt_y2 = belt_y1 + belt_h
    
    belt_y1 = max(y1, min(belt_y1, h - 1))
    belt_y2 = max(y1, min(belt_y2, h - 1))
    
    if belt_y2 <= belt_y1:
        return None
    
    belt_region = frame[belt_y1:belt_y2, x1:x2]
    return belt_region


def classify_kimono_color_from_region(kimono_region: np.ndarray) -> str:
    """
    Classify kimono color from a cropped region.
    Returns: "White", "Blue", or "Black"
    """
    if kimono_region is None or kimono_region.size == 0:
        return "Unknown"
    
    # Black filter (referee / crowd).
    if is_black_kimono(kimono_region):
        return "Black"

    kim_scores, _ = _region_color_scores(kimono_region)
    # Preto muito forte -> Black
    if float(kim_scores.get("black", 0.0)) >= 0.45:
        return "Black"

    # Same logic as identify_colors.py.
    return classify_kimono_color_from_scores_identify_colors(kim_scores)


def classify_belt_color_from_region(belt_region: np.ndarray) -> str:
    """
    Classify belt color from a cropped region.
    Returns: "Red", "White", "Black", or "Unknown"
    """
    if belt_region is None or belt_region.size == 0:
        return "Unknown"

    _, belt_scores = _region_color_scores(belt_region)
    return classify_belt_color_from_scores_identify_colors(belt_scores)


def classify_belt_color_from_scores(belt_scores: Dict[str, float]) -> str:
    # Backward-compatible wrapper.
    return classify_belt_color_from_scores_identify_colors(belt_scores)


def validate_unique_combinations(expected_athletes: Dict[str, Dict[str, str]]) -> bool:
    """Return True if each athlete has a unique (Kimono, Belt) pair."""
    if len(expected_athletes) < 2:
        return True
    
    combinations = []
    for athlete_id, colors in expected_athletes.items():
        kimono = colors.get("Kimono", "").lower()
        belt = colors.get("Belt", "").lower()

        if kimono == "undefined":
            kimono = "unknown"
        if belt == "undefined":
            belt = "unknown"

        combo = (kimono, belt)
        if combo in combinations:
            return False
        combinations.append(combo)
    
    return True


def match_athlete_colors(
    detected_kimono: str,
    detected_belt: str,
    expected_athletes: Dict[str, Dict[str, str]]
) -> Optional[str]:
    """
    Match detected colors to expected athletes.
    Assumes unique kimono+belt combinations in the config.
    Returns athlete id ("1" or "2") on a unique match, else None.
    """
    if not expected_athletes:
        return None
    
    # Validate expected combinations are unique.
    if not validate_unique_combinations(expected_athletes):
        # Duplicate expected colors: skip reliable matching.
        return None
    
    detected_kimono_lower = detected_kimono.lower()
    detected_belt_lower = detected_belt.lower()
    
    # Both unknown: no match.
    if detected_kimono_lower == "unknown" and detected_belt_lower == "unknown":
        return None
    
    # Build detected combo for uniqueness checks.
    detected_combo = (detected_kimono_lower, detected_belt_lower)
    
    matches = []
    
    for athlete_id, colors in expected_athletes.items():
        expected_kimono = colors.get("Kimono", "").lower()
        expected_belt = colors.get("Belt", "").lower()
        
        if expected_kimono == "undefined":
            expected_kimono = "unknown"
        if expected_belt == "undefined":
            expected_belt = "unknown"
        
        expected_combo = (expected_kimono, expected_belt)
        
        # Exact kimono+belt match.
        if detected_combo == expected_combo:
            matches.append((athlete_id, 10))  # High score for exact match.
        # Partial: kimono exact and belt compatible.
        elif detected_kimono_lower == expected_kimono and detected_kimono_lower != "unknown":
            if detected_belt_lower == "unknown" or expected_belt == "unknown":
                matches.append((athlete_id, 5))  # Medium score for partial match.
        # Weak: kimono-only partial match.
        elif detected_kimono_lower != "unknown" and expected_kimono != "unknown":
            if detected_kimono_lower == expected_kimono:
                matches.append((athlete_id, 2))  # Low score.
    
    # Multiple matches: need stronger signal.
    if len(matches) == 0:
        return None
    
    # Sort by score descending.
    matches.sort(key=lambda x: x[1], reverse=True)
    
    # Strong unique best match: return it.
    if matches[0][1] >= 5:
        # Tie on best score?
        best_score = matches[0][1]
        best_matches = [m for m in matches if m[1] == best_score]
        
        if len(best_matches) == 1:
            return best_matches[0][0]
        # Ambiguous tie: return None.
        return None
    
    # Low scores: return only if unique.
    if len(matches) == 1:
        return matches[0][0]
    
    return None


def get_athlete_color(athlete_id: str, expected_athletes: Dict[str, Dict[str, str]]) -> Tuple[int, int, int]:
    """BGR color for the athlete bounding box (distinct tints per id)."""
    color_map = {
        "1": (255, 0, 0),    # blue in BGR
        "2": (0, 0, 255),    # red in BGR
    }
    return color_map.get(athlete_id, (0, 255, 0))  # green fallback


def get_belt_draw_color(belt_label: str) -> Tuple[int, int, int]:
    """
    BGR color for drawing from belt label.
    - Red   -> red
    - White -> white
    - Black -> black
    - Unknown -> green (fallback)
    """
    s = str(belt_label or "").strip().lower()
    if s == "red":
        return (0, 0, 255)
    if s == "white":
        return (255, 255, 255)
    if s == "black":
        return (0, 0, 0)
    return (0, 255, 0)


def get_kimono_draw_color(kimono_label: str) -> Tuple[int, int, int]:
    """
    BGR color for drawing from kimono label.
    - Blue  -> blue
    - White -> white
    - Black -> black
    - Unknown -> green (fallback)
    """
    s = str(kimono_label or "").strip().lower()
    if s == "blue":
        return (255, 0, 0)
    if s == "white":
        return (255, 255, 255)
    if s == "black":
        return (0, 0, 0)
    return (0, 255, 0)


def _text_color_for_bg(bgr: Tuple[int, int, int]) -> Tuple[int, int, int]:
    # Black text on light bg; white on dark.
    b, g, r = [int(x) for x in bgr]
    y = 0.114 * b + 0.587 * g + 0.299 * r
    return (0, 0, 0) if y >= 150 else (255, 255, 255)

def _yolo_line_from_bbox(
    class_id: int,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    width: int,
    height: int,
) -> str:
    # YOLO line format: class cx cy w h normalized 0..1
    bw = max(0, x2 - x1)
    bh = max(0, y2 - y1)
    cx = x1 + (bw / 2.0)
    cy = y1 + (bh / 2.0)
    return (
        f"{int(class_id)} "
        f"{(cx / float(width)):.6f} "
        f"{(cy / float(height)):.6f} "
        f"{(bw / float(width)):.6f} "
        f"{(bh / float(height)):.6f}"
    )


def _extract_strict_waist_region(
    frame: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> Optional[np.ndarray]:
    """
    Tight waist crop to reduce false red.
    Narrow horizontal band on the torso center.
    """
    h, w = frame.shape[:2]
    x1 = int(max(0, min(x1, w - 1)))
    y1 = int(max(0, min(y1, h - 1)))
    x2 = int(max(0, min(x2, w - 1)))
    y2 = int(max(0, min(y2, h - 1)))
    if x2 <= x1 or y2 <= y1:
        return None

    bw = x2 - x1
    bh = y2 - y1
    if bw <= 2 or bh <= 2:
        return None

    # Waist center (~56% height), narrow band (~12%).
    waist_center_y = y1 + int(0.56 * bh)
    waist_h = max(2, int(0.12 * bh))
    wy1 = max(y1, waist_center_y - waist_h // 2)
    wy2 = min(y2, wy1 + waist_h)

    # Keep horizontal center strip to avoid sleeves/mat.
    margin_x = int(0.22 * bw)
    wx1 = min(x2 - 2, x1 + margin_x)
    wx2 = max(x1 + 2, x2 - margin_x)
    if wx2 <= wx1 or wy2 <= wy1:
        return None
    roi = frame[wy1:wy2, wx1:wx2]
    return roi if roi is not None and roi.size > 0 else None


def _is_ref_from_black_majority(region: Optional[np.ndarray]) -> bool:
    """
    Mark referee when dark/black pixels dominate.
    """
    if region is None or region.size == 0:
        return False
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    total = float(gray.size)
    if total <= 0:
        return False
    dark_ratio = float((gray < 60).sum()) / total
    very_dark_ratio = float((gray < 40).sum()) / total
    return (dark_ratio >= 0.62) or (very_dark_ratio >= 0.45)


def _is_ref_from_white_shirt_dark_pants(region: Optional[np.ndarray]) -> bool:
    """
    Heuristic for white-shirt referee:
    - upper body (shirt) lighter with high white content
    - lower body (pants) darker with many dark pixels
    """
    if region is None or region.size == 0:
        return False
    h, w = region.shape[:2]
    if h < 20 or w < 10:
        return False

    top = region[: int(0.45 * h), :]
    bottom = region[int(0.55 * h) :, :]
    if top.size == 0 or bottom.size == 0:
        return False

    top_gray = cv2.cvtColor(top, cv2.COLOR_BGR2GRAY)
    bottom_gray = cv2.cvtColor(bottom, cv2.COLOR_BGR2GRAY)
    top_luma = float(np.mean(top_gray))
    bottom_luma = float(np.mean(bottom_gray))

    top_hsv = cv2.cvtColor(top, cv2.COLOR_BGR2HSV)
    bottom_hsv = cv2.cvtColor(bottom, cv2.COLOR_BGR2HSV)
    top_white = float(cv2.countNonZero(cv2.inRange(top_hsv, (0, 0, 175), (180, 70, 255)))) / float(top_hsv.shape[0] * top_hsv.shape[1])
    bottom_dark = float(cv2.countNonZero(cv2.inRange(bottom_hsv, (0, 0, 0), (180, 255, 80)))) / float(bottom_hsv.shape[0] * bottom_hsv.shape[1])

    # Contraste vertical marcado: topo claro + base escura.
    return (
        top_luma >= 135.0
        and bottom_luma <= 100.0
        and (top_luma - bottom_luma) >= 30.0
        and top_white >= 0.20
        and bottom_dark >= 0.28
    )


def _blue_evidence_from_region(region: Optional[np.ndarray]) -> float:
    """
    Whole-bbox blue evidence (0..1), robust to dark blue.
    """
    if region is None or region.size == 0:
        return 0.0
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0].astype(np.int32)
    s = hsv[:, :, 1].astype(np.float32) / 255.0
    v = hsv[:, :, 2].astype(np.float32) / 255.0

    # Wide blue hue range with shadow tolerance.
    blue_mask = (h >= 78) & (h <= 165) & (s >= 0.12) & (v >= 0.08)
    blue_frac_hsv = float(blue_mask.mean())

    # Boost when B channel dominates in RGB (dark blue).
    bgr = region.astype(np.float32)
    b = bgr[:, :, 0]
    g = bgr[:, :, 1]
    r = bgr[:, :, 2]
    blue_dom = (b >= g + 18.0) & (b >= r + 18.0) & (b >= 35.0)
    blue_frac_rgb = float(blue_dom.mean())

    return max(blue_frac_hsv, blue_frac_rgb)


def _classify_person_for_dataset(
    kimono_region: np.ndarray,
    belt_region: Optional[np.ndarray] = None,
) -> Tuple[Optional[int], str, float]:
    """
    Classify a detection as:
      0 = referee (ref)
      1 = athlete (blue/white/red only in display_label / visualization)
    """
    if kimono_region is None or kimono_region.size == 0:
        return None, "unknown", 0.0

    kim_scores, _ = _region_color_scores(kimono_region)
    blue_sc = float(kim_scores.get("blue", 0.0))
    white_sc = float(kim_scores.get("white", 0.0))
    black_sc = float(kim_scores.get("black", 0.0))
    blue_ev = _blue_evidence_from_region(kimono_region)
    ref_white_pattern = _is_ref_from_white_shirt_dark_pants(kimono_region)

    # Prefer white referee when pattern is strong and blue is weak.
    if ref_white_pattern and white_sc >= 0.24 and blue_sc < 0.22 and blue_ev < 0.24:
        return YOLO_CLASS_REFEREE, "ref", 0.0

    # Guard against false ref on dark blue gi.
    if (blue_sc >= max(0.14, black_sc * 0.56) and blue_sc >= white_sc * 0.85) or blue_ev >= 0.24:
        return YOLO_CLASS_ATHLETE, "blue", 0.0

    # Referee: strong black, little blue/white evidence.
    if (
        (is_black_kimono(kimono_region) or _is_ref_from_black_majority(kimono_region))
        and blue_sc < 0.10
        and blue_ev < 0.14
        and white_sc < 0.22
    ) or (black_sc >= 0.62 and blue_sc < 0.10 and white_sc < 0.20):
        return YOLO_CLASS_REFEREE, "ref", 0.0

    # Extra: white shirt + dark pants (white referee).
    # Only if blue evidence is not strong.
    if blue_sc < 0.12 and blue_ev < 0.15 and white_sc >= 0.22 and ref_white_pattern:
        return YOLO_CLASS_REFEREE, "ref", 0.0

    # Branco por score global.
    if white_sc >= max(0.10, blue_sc):
        if belt_region is not None and belt_region.size > 0:
            _, belt_scores = _region_color_scores(belt_region)
            red_score = float(belt_scores.get("red", 0.0))
            white_belt = float(belt_scores.get("white", 0.0))
            black_belt = float(belt_scores.get("black", 0.0))
            belt_label = classify_belt_color_from_scores(belt_scores)
            # Stricter red rule: avoid calling white as red.
            if (
                belt_label == "Red"
                and white_sc >= 0.18
                and red_score >= 0.16
                and red_score >= (white_belt + 0.04)
                and red_score >= (black_belt + 0.04)
            ):
                return YOLO_CLASS_ATHLETE, "red", red_score
        return YOLO_CLASS_ATHLETE, "white", 0.0

    # Fallback from original label.
    kimono_label = classify_kimono_color_from_region(kimono_region)
    if kimono_label == "Blue":
        return YOLO_CLASS_ATHLETE, "blue", 0.0
    if kimono_label == "White":
        return YOLO_CLASS_ATHLETE, "white", 0.0
    if kimono_label == "Black":
        return YOLO_CLASS_REFEREE, "ref", 0.0

    # Uncertain kimono: do not force red (limits false positives).
    return None, "unknown", 0.0


def _class_visual_info(class_id: int, display_label: str = "") -> Tuple[Tuple[int, int, int], str]:
    s = str(display_label or "").strip().lower()
    if class_id == YOLO_CLASS_REFEREE:
        return (0, 0, 0), "ref"
    if class_id == YOLO_CLASS_ATHLETE:
        if s == "red":
            return (0, 0, 255), "red"
        if s == "blue":
            return (255, 0, 0), "blue"
        return (255, 255, 255), "white"
    if s == "red":
        return (0, 0, 255), "red"
    if s == "blue":
        return (255, 0, 0), "blue"
    return (255, 255, 255), "white"


def _export_yolo_class_and_visual(
    base_class_id: int,
    display_label: str,
    export_schema: str,
) -> Tuple[int, Tuple[int, int, int], str]:
    """
    Map internal ref/athlete+color to YOLO class and display label.

    export_schema:
      - \"default\": classes 0=juiz, 1=atleta; cores por white/blue/red.
      - \"2_c\": 0=referee, 1=athlete (no color split in class id).
      - \"3_c\": 0=referee, 1=athlete_blue, 2=athlete_white (faixa vermelha → athlete_white).
    """
    s = str(display_label or "").strip().lower()
    if export_schema == "default":
        color, lab = _class_visual_info(int(base_class_id), display_label)
        return int(base_class_id), color, lab
    if export_schema == "2_c":
        if int(base_class_id) == YOLO_CLASS_REFEREE:
            return 0, (0, 0, 0), "referee"
        return 1, (0, 165, 255), "athlete"
    if export_schema == "3_c":
        if int(base_class_id) == YOLO_CLASS_REFEREE:
            return 0, (0, 0, 0), "referee"
        if s == "blue":
            return 1, (255, 0, 0), "athlete_blue"
        return 2, (255, 255, 255), "athlete_white"
    raise ValueError(f"Invalid export_schema: {export_schema!r} (use default, 2_c, or 3_c)")


def _annotate_frame_color_heuristic(
    frame: np.ndarray,
    model: YOLO,
    *,
    conf_threshold: float,
    iou_threshold: float,
    max_det: int,
    imgsz: int,
    export_schema: str,
) -> Tuple[np.ndarray, List[str]]:
    """
    One BGR frame: pose + color heuristic + drawing (same as annotate_frames_from_dir).
    Return annotated frame and normalized YOLO lines for the requested schema.
    """
    h, w = frame.shape[:2]
    annotated = frame.copy()
    yolo_lines: List[str] = []

    results = model.predict(
        source=frame,
        classes=[0],
        conf=conf_threshold,
        iou=iou_threshold,
        max_det=max_det,
        verbose=False,
        imgsz=imgsz,
    )
    if results and len(results) > 0:
        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy().astype(int)
            confidences = boxes.conf.cpu().numpy() if hasattr(boxes, "conf") and boxes.conf is not None else None
            kps = getattr(result, "keypoints", None)
            keypoints_xy = None
            try:
                if kps is not None and hasattr(kps, "xy") and kps.xy is not None:
                    keypoints_xy = kps.xy.cpu().numpy()
            except Exception:
                keypoints_xy = None

            candidates_all: List[Dict[str, float]] = []
            for idx, (x1, y1, x2, y2) in enumerate(xyxy):
                x1 = int(max(0, min(x1, w - 1)))
                y1 = int(max(0, min(y1, h - 1)))
                x2 = int(max(0, min(x2, w - 1)))
                y2 = int(max(0, min(y2, h - 1)))
                if x2 <= x1 or y2 <= y1:
                    continue

                kp_xy_i = keypoints_xy[idx] if keypoints_xy is not None and idx < len(keypoints_xy) else None
                kimono_region, belt_region = _extract_torso_and_belt_from_keypoints(frame, kp_xy_i, x1, y1, x2, y2)
                strict_waist = _extract_strict_waist_region(frame, x1, y1, x2, y2)
                belt_for_label = strict_waist if strict_waist is not None else belt_region
                class_id, display_label, red_score = _classify_person_for_dataset(
                    kimono_region,
                    belt_for_label,
                )
                if class_id is None:
                    continue

                conf = float(confidences[idx]) if confidences is not None and idx < len(confidences) else 0.0
                area = float(max(1, (x2 - x1) * (y2 - y1)))
                rank_score = (conf * 1_000_000.0) + area

                candidates_all.append({
                    "class_id": float(class_id),
                    "display_label": display_label,
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                    "rank_score": float(rank_score),
                    "area": float(area),
                    "red_score": float(red_score),
                })

            candidates_considered = candidates_all
            candidates_considered.sort(key=lambda d: float(d.get("area", 0.0)), reverse=True)
            nearest_three = candidates_considered[:3]

            referee_candidates = [d for d in nearest_three if int(d["class_id"]) == YOLO_CLASS_REFEREE]
            athlete_candidates = [d for d in nearest_three if int(d["class_id"]) != YOLO_CLASS_REFEREE]
            athlete_candidates.sort(key=lambda d: float(d.get("rank_score", 0.0)), reverse=True)

            selected_dets: List[Dict[str, float]] = []
            if referee_candidates:
                selected_dets.append(max(referee_candidates, key=lambda d: float(d.get("rank_score", 0.0))))

            selected_athletes = athlete_candidates[:2]
            if selected_dets:
                ref_det = selected_dets[0]
                selected_athletes = [
                    d for d in selected_athletes
                    if not (
                        int(d.get("x1", -1)) == int(ref_det.get("x1", -2))
                        and int(d.get("y1", -1)) == int(ref_det.get("y1", -2))
                        and int(d.get("x2", -1)) == int(ref_det.get("x2", -2))
                        and int(d.get("y2", -1)) == int(ref_det.get("y2", -2))
                    )
                ]
                selected_athletes = selected_athletes[:2]
            red_idx = [i for i, d in enumerate(selected_athletes) if str(d.get("display_label", "")).lower() == "red"]
            if len(red_idx) >= 2:
                keep = max(red_idx, key=lambda i: float(selected_athletes[i].get("red_score", 0.0)))
                for i in red_idx:
                    if i != keep:
                        selected_athletes[i]["display_label"] = "white"
                        selected_athletes[i]["class_id"] = float(YOLO_CLASS_ATHLETE)

            blue_idx = [i for i, d in enumerate(selected_athletes) if str(d.get("display_label", "")).lower() == "blue"]
            if len(blue_idx) >= 2:
                keep_blue = max(blue_idx, key=lambda i: float(selected_athletes[i].get("rank_score", 0.0)))
                for i in blue_idx:
                    if i != keep_blue:
                        selected_athletes[i]["class_id"] = float(YOLO_CLASS_ATHLETE)
                        selected_athletes[i]["display_label"] = "white"

            selected_dets.extend(selected_athletes)

            for det in selected_dets:
                class_id = int(det["class_id"])
                if det is None:
                    continue
                x1 = int(det["x1"])
                y1 = int(det["y1"])
                x2 = int(det["x2"])
                y2 = int(det["y2"])
                yolo_cls, color, label = _export_yolo_class_and_visual(
                    class_id,
                    str(det.get("display_label", "")),
                    export_schema,
                )

                yolo_lines.append(_yolo_line_from_bbox(yolo_cls, x1, y1, x2, y2, w, h))
                _box_t, _font_sc, _font_th = 2, 0.55, 1
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, _box_t)
                text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, _font_sc, _font_th)
                text_y = max(y1 - 6, text_size[1] + 6)
                _pad_v, _pad_h = 4, 3
                cv2.rectangle(
                    annotated,
                    (x1, text_y - text_size[1] - _pad_v),
                    (x1 + text_size[0] + 2 * _pad_h, text_y + _pad_h),
                    color,
                    -1,
                )
                cv2.putText(
                    annotated,
                    label,
                    (x1 + _pad_h, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    _font_sc,
                    _text_color_for_bg(color),
                    _font_th,
                    cv2.LINE_AA,
                )

    return annotated, yolo_lines


def annotate_frames_from_dir(
    input_dir: str,
    output_annotated_dir: str,
    output_labels_dir: str,
    model_path: str = "yolo11n-pose.pt",
    conf_threshold: float = 0.20,
    iou_threshold: float = 0.85,
    max_det: int = 30,
    imgsz: int = 1280,
    export_schema: str = "default",
) -> None:
    """
    Process static frames (e.g. img folder) and emit:
    1) imagem com boxes desenhadas
    2) arquivo .txt no formato YOLO para cada imagem

    export_schema:
      - \"default\": 2 classes YOLO (0 juiz, 1 atleta) com subtipo visual white/blue/red.
      - \"2_c\": 2 classes YOLO (0 referee, 1 athlete).
      - \"3_c\": 3 classes YOLO (0 referee, 1 athlete_blue, 2 athlete_white); faixa vermelha → athlete_white.
    """
    if not os.path.isabs(model_path):
        candidate = os.path.join(os.path.dirname(__file__), model_path)
        if os.path.exists(candidate):
            model_path = candidate

    os.makedirs(output_annotated_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    model = YOLO(model_path)

    image_files = [
        f for f in sorted(os.listdir(input_dir))
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
    ]
    if not image_files:
        raise ValueError(f"No images found in: {input_dir}")

    for image_name in image_files:
        image_path = os.path.join(input_dir, image_name)
        frame = cv2.imread(image_path)
        if frame is None:
            continue

        annotated, yolo_lines = _annotate_frame_color_heuristic(
            frame,
            model,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            max_det=max_det,
            imgsz=imgsz,
            export_schema=export_schema,
        )

        base_name, ext = os.path.splitext(image_name)
        out_img_path = os.path.join(output_annotated_dir, f"{base_name}{ext}")
        out_txt_path = os.path.join(output_labels_dir, f"{base_name}.txt")

        cv2.imwrite(out_img_path, annotated)
        with open(out_txt_path, "w", encoding="utf-8") as f:
            if yolo_lines:
                f.write("\n".join(yolo_lines) + "\n")

    print(f"Annotated frames saved to: {output_annotated_dir}")
    print(f"YOLO labels saved to: {output_labels_dir}")


def annotate_video_color_heuristic(
    video_path: str,
    output_path: str,
    model_path: str = "yolo11n-pose.pt",
    conf_threshold: float = 0.20,
    iou_threshold: float = 0.85,
    max_det: int = 30,
    imgsz: int = 1280,
    export_schema: str = "3_c",
    max_frames: Optional[int] = None,
) -> None:
    """
    Full video using the same color-heuristic pipeline as `annotate_frames_from_dir`
    (pose + kimono/belt rules), writing an annotated MP4.
    max_frames: if set, only process the first N frames (debug).
    """
    if not os.path.isabs(model_path):
        candidate = os.path.join(os.path.dirname(__file__), model_path)
        if os.path.exists(candidate):
            model_path = candidate

    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    try:
        if os.path.exists(output_path):
            os.remove(output_path)
    except Exception:
        base, ext = os.path.splitext(output_path)
        i = 2
        while True:
            candidate = f"{base}_v{i}{ext or '.mp4'}"
            if not os.path.exists(candidate):
                output_path = candidate
                break
            i += 1

    final_path = output_path
    tmp_path = f"{final_path}.writing.mp4"
    try:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    except Exception:
        pass

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(tmp_path, fourcc, fps, (width, height))
    if not out.isOpened():
        cap.release()
        raise ValueError(f"Could not create output video: {tmp_path}")

    frame_count = 0
    written = 0
    print(f"[based_color/{export_schema}] Input: {video_path}", flush=True)
    print(f"Output: {final_path} | ~{total_frames} frames @ {fps:.2f} fps", flush=True)

    ok_write = False
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count % 60 == 0 and total_frames > 0:
                print(f"  frame {frame_count}/{total_frames}", flush=True)

            annotated, _yolo = _annotate_frame_color_heuristic(
                frame,
                model,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                max_det=max_det,
                imgsz=imgsz,
                export_schema=export_schema,
            )
            out.write(annotated)
            written += 1
            if max_frames is not None and written >= int(max_frames):
                break
        ok_write = True
    finally:
        cap.release()
        out.release()

    if ok_write and written > 0 and os.path.isfile(tmp_path):
        try:
            if os.path.isfile(final_path):
                os.remove(final_path)
            os.replace(tmp_path, final_path)
        except OSError:
            print(f"WARNING: could not rename {tmp_path} -> {final_path}; partial output in .writing.mp4", flush=True)
            raise
    else:
        try:
            if os.path.isfile(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass

    print(f"Wrote {written} frames to: {final_path}", flush=True)


def annotate_video_with_colors(
    video_path: str,
    output_path: str,
    expected_athletes: Dict[str, Dict[str, str]],
    model_path: str = "yolo11n-pose.pt",
    conf_threshold: float = 0.15,
    iou_threshold: float = 0.85,
    max_det: int = 20,
    imgsz: int = 1280,
    min_area_ratio: float = 0.003,
    min_height_ratio: float = 0.15,
    max_missing_frames: int = 10,
    max_center_jump_ratio: float = 0.20,
    ghost_fusion_distance_ratio: float = 0.08,
    collect_metrics: bool = False,
) -> None:
    """
    Process video frame-by-frame; detect athletes by color; draw colored boxes.
    
    Args:
        video_path: Input video path
        output_path: Output video path
        expected_athletes: Dict of expected athlete colors
                          {"1": {"Kimono": "White", "Belt": "Red"}, ...}
                          Each kimono+belt combination must be unique.
        model_path: Path to YOLO weights
        conf_threshold: Detection confidence threshold
        iou_threshold: NMS IoU threshold (raise to keep very close athletes)
        max_det: Max detections per frame (balance noise vs recall)
        imgsz: Inference resolution (larger helps heavy occlusion)
        min_area_ratio: Minimum relative bbox area
        min_height_ratio: Minimum relative bbox height
        max_missing_frames: Frames to reuse last bbox when detector drops an athlete
        max_center_jump_ratio: Max center jump as fraction of max frame side between frames
        ghost_fusion_distance_ratio: If an athlete disappears but history says they were glued to the other,
                                     draw a ghost box aligned to the visible athlete.
        collect_metrics: If True, simple belt/kimono hit metrics per athlete_id (proxy)
    """
    # Validate unique combinations
    if not validate_unique_combinations(expected_athletes):
        raise ValueError(
            "Kimono+belt combinations must be unique across athletes! "
            "Each athlete must have a different kimono and belt pair."
        )
    
    # Resolve model_path relative to this file (stable cwd)
    if not os.path.isabs(model_path):
        candidate = os.path.join(os.path.dirname(__file__), model_path)
        if os.path.exists(candidate):
            model_path = candidate
    
    model = YOLO(model_path)

    # Open input video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Read video properties
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Define codec e cria VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # Ensure output dir exists; allow overwrite (broken locks on VideoWriter)
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    try:
        if os.path.exists(output_path):
            os.remove(output_path)
    except Exception:
        # If delete fails (locked file), try alternate output name
        base, ext = os.path.splitext(output_path)
        i = 2
        while True:
            candidate = f"{base}_v{i}{ext or '.mp4'}"
            if not os.path.exists(candidate):
                output_path = candidate
                break
            i += 1
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not out.isOpened():
        raise ValueError(f"Could not create output video: {output_path}")
    
    frame_count = 0
    written_frames = 0
    min_area_abs = min_area_ratio * (width * height)
    min_height_abs = min_height_ratio * height
    
    # Per-frame athlete tracking state
    # track_id -> {athlete_id, last_seen_frame, box_area, ...}
    athlete_tracking = {}

    # Temporal smoothing per track_id (EMA on color scores)
    # track_id -> {"kim": {...}, "belt": {...}, "last_seen_frame": int}
    track_color_state: Dict[int, dict] = {}
    
    # Last bbox per athlete when detector merges close boxes
    # athlete_id -> {x1,y1,x2,y2,last_seen_frame,detected_kimono,detected_belt}
    last_bbox_by_athlete: Dict[str, dict] = {}

    def _center_from_bbox(x1: int, y1: int, x2: int, y2: int) -> Tuple[float, float]:
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    
    def _dist(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))
    
    print(f"Processing video: {video_path}")
    print(f"Total frames: {total_frames}")
    
    # Proxy metrics for same-class (e.g. white-white) where belt picks the id.
    # Use raw_detected_* before forcing expected combo to avoid inflated scores.
    metric = {
        "frames": 0,
        "frames_with_any_det": 0,
        "a1_ok_belt": 0,
        "a2_ok_belt": 0,
        "both_ok_belt": 0,
        "a1_seen": 0,
        "a2_seen": 0,
        # Per-track metrics to pick red belt by aggregated evidence
        "track": {},  # tid -> {"n":int,"sum_red":float,"sum_white":float,"sum_black":float,"as_1":int,"as_2":int}
    }

    try:
        # Precompute same-kimono case and which athlete expects red belt.
        athlete_ids_expected = [str(a) for a in expected_athletes.keys()]
        expected_kimonos_norm = [_norm_expected_color((expected_athletes.get(a) or {}).get("Kimono")) for a in athlete_ids_expected]
        same_kimono_expected = (len(athlete_ids_expected) == 2) and (len(set(expected_kimonos_norm)) == 1)

        red_aid = None
        other_aid = None
        if len(athlete_ids_expected) == 2:
            a0, a1 = athlete_ids_expected[0], athlete_ids_expected[1]
            b0 = _norm_expected_color((expected_athletes.get(a0) or {}).get("Belt"))
            b1 = _norm_expected_color((expected_athletes.get(a1) or {}).get("Belt"))
            if b0 == "red" and b1 != "red":
                red_aid, other_aid = a0, a1
            elif b1 == "red" and b0 != "red":
                red_aid, other_aid = a1, a0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            if frame_count % 30 == 0:
                if total_frames > 0:
                    print(f"Processing frame {frame_count}/{total_frames} ({100*frame_count/total_frames:.1f}%)")
                else:
                    print(f"Processing frame {frame_count}/? (FPS={fps:.2f})")
            
            # YOLO detect + integrated tracker
            results = model.track(
                source=frame,
                classes=[0],  # 0 = 'person' no COCO
                conf=conf_threshold,
                iou=iou_threshold,
                max_det=max_det,
                verbose=False,
                imgsz=imgsz,
                persist=True,  # keep tracks across frames
                tracker="bytetrack.yaml",  # Ultralytics ByteTrack default
            )
            
            if not results or len(results) == 0:
                out.write(frame)
                written_frames += 1
                continue
            
            result = results[0]
            boxes = getattr(result, "boxes", None)
            
            if boxes is None or len(boxes) == 0:
                out.write(frame)
                written_frames += 1
                continue
            
            # Process each detection
            xyxy = boxes.xyxy.cpu().numpy().astype(int)
            confidences = boxes.conf.cpu().numpy() if hasattr(boxes, "conf") else None

            # Keypoints when using a pose model
            kps = getattr(result, "keypoints", None)
            keypoints_xy = None
            keypoints_conf = None
            try:
                if kps is not None and hasattr(kps, "xy") and kps.xy is not None:
                    keypoints_xy = kps.xy.cpu().numpy()  # (N, K, 2)
                if kps is not None and hasattr(kps, "conf") and kps.conf is not None:
                    keypoints_conf = kps.conf.cpu().numpy()  # (N, K)
            except Exception:
                keypoints_xy = None
                keypoints_conf = None
            
            # Read tracking ids when present
            track_ids = None
            if hasattr(boxes, "id") and boxes.id is not None:
                track_ids = boxes.id.cpu().numpy().astype(int)
            
            annotated_frame = frame.copy()

            # ROI central (evita bordas/plateia)
            roi_xmin = int(0.20 * width)
            roi_xmax = int(0.80 * width)
            roi_ymin = int(0.25 * height)
            roi_ymax = int(0.65 * height)
            
            # Collect valid athlete detections
            valid_athlete_detections = []

            # Local: continuous match score vs expected colors
            def _athlete_match_score(aid: str, kim_scores: Dict[str, float], belt_scores: Dict[str, float]) -> float:
                exp = expected_athletes.get(str(aid), {}) or {}
                exp_k = _norm_expected_color(exp.get("Kimono"))
                exp_b = _norm_expected_color(exp.get("Belt"))
                score = 0.0
                if exp_k in ("white", "blue", "black"):
                    score += float(kim_scores.get(exp_k, 0.0))
                # Se os kimonos esperados forem diferentes, a faixa pouco importa:
                # kimono-only emphasis (user-tuned behavior).
                if not same_kimono_expected:
                    return float(score)

                if exp_b in ("red", "white", "black"):
                    score += 0.70 * float(belt_scores.get(exp_b, 0.0))
                return float(score)
            
            for idx, (x1, y1, x2, y2) in enumerate(xyxy):
                # track_id when available
                track_id = int(track_ids[idx]) if track_ids is not None and idx < len(track_ids) else None
                # Filtra por tamanho
                box_w = x2 - x1
                box_h = y2 - y1
                box_area = box_w * box_h
                
                if box_area < min_area_abs or box_h < min_height_abs:
                    continue
                
                # Central ROI filter (edges, crowd, bottom-frame people)
                cx = x1 + box_w // 2
                cy = y1 + box_h // 2
                if not (roi_xmin <= cx <= roi_xmax and roi_ymin <= cy <= roi_ymax):
                    continue
                
                # Skip detections hugging the bottom (crowd)
                # Only keep people whose bbox center is in the upper part of the frame
                if cy > height * 0.65:
                    continue
                
                # Extract kimono/belt ROIs (prefer pose keypoints when available)
                kp_xy_i = keypoints_xy[idx] if keypoints_xy is not None and idx < len(keypoints_xy) else None
                kimono_region, belt_region = _extract_torso_and_belt_from_keypoints(frame, kp_xy_i, x1, y1, x2, y2)
                
                # Validate ROIs
                if kimono_region is None or kimono_region.size == 0:
                    continue
                
                # Black check before color classify (stricter crowd filter)
                # Stricter filter: region mostly black?
                if is_black_kimono(kimono_region):
                    continue
                
                # Extra: too much black in ROI (even with small white details)
                kimono_gray = cv2.cvtColor(kimono_region, cv2.COLOR_BGR2GRAY)
                dark_pixels = (kimono_gray < 60).sum()
                total_pixels = kimono_gray.size
                dark_ratio = dark_pixels / total_pixels if total_pixels > 0 else 0
                
                # If >60% dark pixels, treat as black gi / crowd / referee
                if dark_ratio > 0.60:
                    continue

                # Deterministic color scores + per-track_id EMA smoothing
                kim_scores_raw, _ = _region_color_scores(kimono_region)
                _, belt_scores_raw = _region_color_scores(belt_region) if belt_region is not None else ({}, {"unknown": 1.0})

                kim_scores = kim_scores_raw
                belt_scores = belt_scores_raw
                if track_id is not None:
                    prev_state = track_color_state.get(track_id)
                    if prev_state is None:
                        track_color_state[track_id] = {
                            "kim": dict(kim_scores_raw),
                            "belt": dict(belt_scores_raw),
                            "last_seen_frame": int(frame_count),
                        }
                    else:
                        track_color_state[track_id] = {
                            "kim": _ema_update(prev_state.get("kim", {}), kim_scores_raw, alpha=0.25),
                            "belt": _ema_update(prev_state.get("belt", {}), belt_scores_raw, alpha=0.25),
                            "last_seen_frame": int(frame_count),
                        }
                    kim_scores = track_color_state[track_id]["kim"]
                    belt_scores = track_color_state[track_id]["belt"]

                detected_kimono = classify_kimono_color_from_region(kimono_region)  # usa scores internos
                # Para faixa, usa scores (com EMA quando houver track_id) para reduzir flicker/erro em classe igual.
                detected_belt = classify_belt_color_from_scores(belt_scores) if belt_region is not None else "Unknown"
                
                # Skip black (ref/crowd) or fully unknown colors
                if detected_kimono == "Black":
                    continue
                
                # Unknown kimono and no belt signal: likely not an athlete
                if detected_kimono == "Unknown" and detected_belt == "Unknown":
                    continue
                
                # Continuous score vs expected colors (more stable than label-only)
                athlete_ids = [str(a) for a in expected_athletes.keys()]
                match_scores = {aid: _athlete_match_score(aid, kim_scores, belt_scores) for aid in athlete_ids}

                athlete_id = None
                if match_scores:
                    ordered = sorted(match_scores.items(), key=lambda kv: kv[1], reverse=True)
                    best_aid, best_score = ordered[0]
                    second = ordered[1][1] if len(ordered) > 1 else -1.0
                    # Min confidence/margin to allow swaps
                    if best_score >= 0.25 and (best_score - second) >= 0.08:
                        athlete_id = best_aid

                # Fallback: label match (legacy compatibility)
                if athlete_id is None:
                    athlete_id = match_athlete_colors(detected_kimono, detected_belt, expected_athletes)
                
                # MELHORIA: Se temos 2 atletas esperados e apenas kimono foi detectado (faixa unknown),
                # if expected belts differ, use belt to split even at low confidence
                if athlete_id is None and len(athlete_ids) == 2 and detected_kimono != "Unknown" and detected_belt == "Unknown":
                    # Try belt-based split even when detection is uncertain
                    # Usa os scores brutos da faixa para fazer match
                    for aid in athlete_ids:
                        exp = expected_athletes.get(str(aid), {}) or {}
                        exp_b = _norm_expected_color(exp.get("Belt"))
                        if exp_b in ("red", "white", "black") and exp_b != "unknown":
                            belt_score = float(belt_scores.get(exp_b, 0.0))
                            # Se a faixa esperada tem algum score (mesmo baixo), considera
                            if belt_score > 0.05:  # very low threshold, split-only
                                # Check kimono agreement too
                                exp_k = _norm_expected_color(exp.get("Kimono"))
                                if detected_kimono.lower() == exp_k:
                                    # Match parcial mas com faixa diferenciada
                                    if athlete_id is None:
                                        athlete_id = aid
                                    else:
                                        # If multiple matches, compare scores
                                        other_exp = expected_athletes.get(str(athlete_id), {}) or {}
                                        other_b = _norm_expected_color(other_exp.get("Belt"))
                                        other_belt_score = float(belt_scores.get(other_b, 0.0))
                                        if belt_score > other_belt_score:
                                            athlete_id = aid
                
                # ADJ 2: Two expected athletes and one already identified earlier,
                # this unidentified detection is assigned the other athlete
                if len(athlete_ids) == 2 and athlete_id is None:
                    # Check if we already have a resolved detection this frame
                    # (handled later; history can help)
                    # Leave to downstream logic for now
                    pass
                
                # Known track_id: use history for stability
                if track_id is not None and track_id in athlete_tracking:
                    previous_info = athlete_tracking[track_id]
                    prev_aid = previous_info.get('athlete_id')
                    # On failed match, fall back to history
                    if athlete_id is None:
                        athlete_id = prev_aid
                    # Label changes only with strong score support (reduce ID-switch)
                    elif prev_aid is not None and str(prev_aid) != str(athlete_id):
                        try:
                            s_prev = float(match_scores.get(str(prev_aid), 0.0))
                            s_new = float(match_scores.get(str(athlete_id), 0.0))
                            if (s_new - s_prev) < 0.18:
                                athlete_id = str(prev_aid)
                        except Exception:
                            athlete_id = str(prev_aid)
                    # After athlete_id is fixed, snap to expected kimono+belt (no cross-mix).
                    if athlete_id:
                        expected_kimono, expected_belt = _expected_combo_for_athlete(str(athlete_id), expected_athletes)
                        if expected_kimono != "Unknown":
                            detected_kimono = expected_kimono
                        if expected_belt != "Unknown":
                            detected_belt = expected_belt

                    # Update history
                    athlete_tracking[track_id] = {
                        'athlete_id': athlete_id,
                        'last_seen_frame': frame_count,
                        'box_area': box_area,
                        'detected_kimono': detected_kimono,
                        'detected_belt': detected_belt,
                        'match_scores': match_scores,
                    }
                elif track_id is not None and athlete_id is not None:
                    expected_kimono, expected_belt = _expected_combo_for_athlete(str(athlete_id), expected_athletes)
                    if expected_kimono != "Unknown":
                        detected_kimono = expected_kimono
                    if expected_belt != "Unknown":
                        detected_belt = expected_belt

                    # New track_id with valid match: store for later
                    athlete_tracking[track_id] = {
                        'athlete_id': athlete_id,
                        'last_seen_frame': frame_count,
                        'box_area': box_area,
                        'detected_kimono': detected_kimono,
                        'detected_belt': detected_belt,
                        'match_scores': match_scores,
                    }
                
                if athlete_id:
                    # Avoid impossible mixes (one athlete's kimono + other's belt).
                    # After athlete_id, always render the JSON expected combo.
                    raw_detected_kimono = detected_kimono
                    raw_detected_belt = detected_belt
                    expected_kimono, expected_belt = _expected_combo_for_athlete(str(athlete_id), expected_athletes)
                    if expected_kimono != "Unknown":
                        detected_kimono = expected_kimono
                    if expected_belt != "Unknown":
                        detected_belt = expected_belt

                    # Store valid detection for drawing
                    confidence = confidences[idx] if confidences is not None else 0.5
                    valid_athlete_detections.append({
                        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                        'box_area': box_area,
                        'confidence': confidence,
                        'athlete_id': athlete_id,
                        'track_id': track_id,
                        'detected_kimono': detected_kimono,
                        'detected_belt': detected_belt,
                        'match_scores': match_scores,
                        'raw_detected_kimono': raw_detected_kimono,
                        'raw_detected_belt': raw_detected_belt,
                        'belt_scores': dict(belt_scores) if isinstance(belt_scores, dict) else {},
                        'kim_scores': dict(kim_scores) if isinstance(kim_scores, dict) else {},
                    })

            # AJUSTE 2: Garantir que se temos 2 atletas esperados, eles sempre tenham IDs diferentes
            # (JSON combos are unique)
            athlete_ids = [str(a) for a in expected_athletes.keys()]
            if len(athlete_ids) == 2 and len(valid_athlete_detections) > 0:
                # Keep top-2 detections (never assign >2 athletes)
                if len(valid_athlete_detections) > 2:
                    valid_athlete_detections.sort(key=lambda d: float(d.get("box_area", 0.0)), reverse=True)
                    valid_athlete_detections[:] = valid_athlete_detections[:2]

                detected_ids = {d['athlete_id'] for d in valid_athlete_detections if d.get('athlete_id')}
                
                # Case 1: two boxes, only one identified
                if len(detected_ids) == 1 and len(valid_athlete_detections) == 2:
                    # The second box becomes the other athlete
                    detected_id = next(iter(detected_ids))
                    other_id = athlete_ids[1] if athlete_ids[0] == detected_id else athlete_ids[0]
                    
                    # Assign the missing athlete_id
                    for det in valid_athlete_detections:
                        if not det.get('athlete_id') or det['athlete_id'] != detected_id:
                            det['athlete_id'] = other_id
                            # Atualiza as cores detectadas para as esperadas do outro atleta
                            expected_kimono, expected_belt = _expected_combo_for_athlete(other_id, expected_athletes)
                            det['detected_kimono'] = expected_kimono
                            det['detected_belt'] = expected_belt
                            break
                
                # Case 2: duplicate athlete_id (should not happen)
                elif len(valid_athlete_detections) == 2 and len(detected_ids) == 1:
                    # Force the second detection to the other id
                    detected_id = next(iter(detected_ids))
                    other_id = athlete_ids[1] if athlete_ids[0] == detected_id else athlete_ids[0]
                    
                    # Find second occurrence and flip id
                    count = 0
                    for det in valid_athlete_detections:
                        if det.get('athlete_id') == detected_id:
                            count += 1
                            if count == 2:  # second occurrence of same id
                                det['athlete_id'] = other_id
                                # Atualiza as cores detectadas para as esperadas do outro atleta
                                expected_kimono, expected_belt = _expected_combo_for_athlete(other_id, expected_athletes)
                                det['detected_kimono'] = expected_kimono
                                det['detected_belt'] = expected_belt
                                break

            # REGRA (kimonos diferentes): as cores dos 2 atletas NUNCA devem ser iguais.
            # Blue vs White expected: pick best kimono-score assignment,
            # e o outro atleta recebe automaticamente a cor restante.
            if len(athlete_ids_expected) == 2 and (not same_kimono_expected) and len(valid_athlete_detections) == 2:
                aid1, aid2 = athlete_ids_expected[0], athlete_ids_expected[1]
                exp_k1 = _norm_expected_color((expected_athletes.get(aid1) or {}).get("Kimono"))
                exp_k2 = _norm_expected_color((expected_athletes.get(aid2) or {}).get("Kimono"))

                d0, d1 = valid_athlete_detections[0], valid_athlete_detections[1]
                s0 = d0.get("kim_scores") or {}
                s1 = d1.get("kim_scores") or {}

                def _kscore(sc: dict, key: str) -> float:
                    try:
                        return float(sc.get(key, 0.0))
                    except Exception:
                        return 0.0

                # Total score se d0->aid1 e d1->aid2 vs swap
                direct = _kscore(s0, exp_k1) + _kscore(s1, exp_k2)
                swap = _kscore(s0, exp_k2) + _kscore(s1, exp_k1)

                if swap > direct:
                    # troca
                    d0["athlete_id"], d1["athlete_id"] = aid2, aid1
                    ek, eb = _expected_combo_for_athlete(aid2, expected_athletes)
                    d0["detected_kimono"], d0["detected_belt"] = ek, eb
                    ek, eb = _expected_combo_for_athlete(aid1, expected_athletes)
                    d1["detected_kimono"], d1["detected_belt"] = ek, eb
                else:
                    d0["athlete_id"], d1["athlete_id"] = aid1, aid2
                    ek, eb = _expected_combo_for_athlete(aid1, expected_athletes)
                    d0["detected_kimono"], d0["detected_belt"] = ek, eb
                    ek, eb = _expected_combo_for_athlete(aid2, expected_athletes)
                    d1["detected_kimono"], d1["detected_belt"] = ek, eb

            # SPECIAL (same kimono): if both expect same kimono and one expects red belt,
            # assign by red evidence on belt ROI (reduces swaps when white belt is noisy).
            if same_kimono_expected and red_aid and other_aid and len(valid_athlete_detections) == 2:
                def _red_score(det: dict) -> float:
                    sc = det.get("belt_scores") or {}
                    try:
                        return float(sc.get("red", 0.0))
                    except Exception:
                        return 0.0

                d0, d1 = valid_athlete_detections[0], valid_athlete_detections[1]
                r0, r1 = _red_score(d0), _red_score(d1)

                # Only force if margin is enough; else keep tracker assignment.
                if abs(r0 - r1) >= 0.03:
                    red_det, white_det = (d0, d1) if r0 >= r1 else (d1, d0)
                    red_det["athlete_id"] = red_aid
                    white_det["athlete_id"] = other_aid

                    ek, eb = _expected_combo_for_athlete(red_aid, expected_athletes)
                    red_det["detected_kimono"], red_det["detected_belt"] = ek, eb
                    ek, eb = _expected_combo_for_athlete(other_aid, expected_athletes)
                    white_det["detected_kimono"], white_det["detected_belt"] = ek, eb

                # Never leave BOTH raw belts Red when the expected scenario is
                # (White/Red) + (White/White): if both raw Red, demote weaker red evidence to White.
                rb0 = str(d0.get("raw_detected_belt") or "Unknown")
                rb1 = str(d1.get("raw_detected_belt") or "Unknown")
                if rb0 == "Red" and rb1 == "Red":
                    # compare red evidence via score
                    if r0 >= r1:
                        d1["raw_detected_belt"] = "White"
                    else:
                        d0["raw_detected_belt"] = "White"

            # Metrics (pre-draw) using raw_detected_* only.
            if collect_metrics:
                metric["frames"] += 1
                if valid_athlete_detections:
                    metric["frames_with_any_det"] += 1

                exp1 = _expected_combo_for_athlete("1", expected_athletes)[1]  # Belt
                exp2 = _expected_combo_for_athlete("2", expected_athletes)[1]  # Belt
                a1_ok = False
                a2_ok = False
                for det in valid_athlete_detections:
                    aid = str(det.get("athlete_id") or "")
                    raw_b = str(det.get("raw_detected_belt") or "Unknown")
                    tid = det.get("track_id")
                    # aggregate per-track red evidence (proxy without per-frame ground truth)
                    if tid is not None:
                        tstat = metric["track"].setdefault(int(tid), {"n": 0, "sum_red": 0.0, "sum_white": 0.0, "sum_black": 0.0, "as_1": 0, "as_2": 0})
                        sc = det.get("belt_scores") or {}
                        tstat["n"] += 1
                        tstat["sum_red"] += float(sc.get("red", 0.0))
                        tstat["sum_white"] += float(sc.get("white", 0.0))
                        tstat["sum_black"] += float(sc.get("black", 0.0))
                        if aid == "1":
                            tstat["as_1"] += 1
                        elif aid == "2":
                            tstat["as_2"] += 1
                    if aid == "1":
                        metric["a1_seen"] += 1
                        if raw_b == exp1:
                            a1_ok = True
                    elif aid == "2":
                        metric["a2_seen"] += 1
                        if raw_b == exp2:
                            a2_ok = True
                if a1_ok:
                    metric["a1_ok_belt"] += 1
                if a2_ok:
                    metric["a2_ok_belt"] += 1
                if a1_ok and a2_ok:
                    metric["both_ok_belt"] += 1
                
                # Case 3: final check — two detections must have different ids
                if len(valid_athlete_detections) == 2:
                    final_ids = [d.get('athlete_id') for d in valid_athlete_detections if d.get('athlete_id')]
                    if len(final_ids) == 2 and final_ids[0] == final_ids[1]:
                        # Duplicate ids — flip the second
                        detected_id = final_ids[0]
                        other_id = athlete_ids[1] if athlete_ids[0] == detected_id else athlete_ids[0]
                        valid_athlete_detections[1]['athlete_id'] = other_id
                        expected_kimono, expected_belt = _expected_combo_for_athlete(other_id, expected_athletes)
                        valid_athlete_detections[1]['detected_kimono'] = expected_kimono
                        valid_athlete_detections[1]['detected_belt'] = expected_belt
            
            # Cap at two athletes: keep the two largest boxes.
            # (Do not prune by kimono+belt combo; occlusion/grappling causes false drops.)
            if len(valid_athlete_detections) > 0:
                valid_athlete_detections.sort(key=lambda d: d['box_area'], reverse=True)
                if len(valid_athlete_detections) > 2:
                    valid_athlete_detections = valid_athlete_detections[:2]
            
            # Proximity association (mitigate ID swap under heavy occlusion):
            # Athlete 1 near X should stay near X next frame.
            # Isso reduz trocas de atleta_id quando cores/track confundirem.
            max_jump = max(width, height) * float(max_center_jump_ratio)
            recent_last = {}
            for aid, info in last_bbox_by_athlete.items():
                if frame_count - int(info.get("last_seen_frame", -10**9)) <= max_missing_frames:
                    recent_last[aid] = info
            
            # With recent history for both athletes and two boxes, solve a 2x2 assignment.
            athlete_ids = [str(a) for a in expected_athletes.keys()]
            if len(athlete_ids) == 2 and len(valid_athlete_detections) == 2:
                a1, a2 = athlete_ids[0], athlete_ids[1]
                d0 = valid_athlete_detections[0]
                d1 = valid_athlete_detections[1]
                p0 = _center_from_bbox(d0["x1"], d0["y1"], d0["x2"], d0["y2"])
                p1 = _center_from_bbox(d1["x1"], d1["y1"], d1["x2"], d1["y2"])

                def _ms(det: dict, aid: str) -> float:
                    return float((det.get("match_scores") or {}).get(str(aid), 0.0))

                score_keep = _ms(d0, a1) + _ms(d1, a2)
                score_swap = _ms(d0, a2) + _ms(d1, a1)

                # Distance term only with fresh history; else color score only
                if a1 in recent_last and a2 in recent_last:
                    c1 = _center_from_bbox(recent_last[a1]["x1"], recent_last[a1]["y1"], recent_last[a1]["x2"], recent_last[a1]["y2"])
                    c2 = _center_from_bbox(recent_last[a2]["x1"], recent_last[a2]["y1"], recent_last[a2]["x2"], recent_last[a2]["y2"])
                    dist_keep_1 = _dist(c1, p0)
                    dist_keep_2 = _dist(c2, p1)
                    dist_swap_1 = _dist(c1, p1)
                    dist_swap_2 = _dist(c2, p0)

                    # Hybrid cost: distance - (score * scale)
                    scale = float(max_jump) * 0.80
                    cost_keep = (dist_keep_1 + dist_keep_2) - scale * score_keep
                    cost_swap = (dist_swap_1 + dist_swap_2) - scale * score_swap

                    ok_keep = (dist_keep_1 <= max_jump) and (dist_keep_2 <= max_jump)
                    ok_swap = (dist_swap_1 <= max_jump) and (dist_swap_2 <= max_jump)

                    if ok_keep and (not ok_swap or cost_keep <= cost_swap):
                        valid_athlete_detections[0]["athlete_id"] = a1
                        valid_athlete_detections[1]["athlete_id"] = a2
                    elif ok_swap:
                        valid_athlete_detections[0]["athlete_id"] = a2
                        valid_athlete_detections[1]["athlete_id"] = a1
                else:
                    # No reliable history: maximize color score
                    if score_swap > score_keep + 0.10:
                        valid_athlete_detections[0]["athlete_id"] = a2
                        valid_athlete_detections[1]["athlete_id"] = a1
                    else:
                        valid_athlete_detections[0]["athlete_id"] = a1
                        valid_athlete_detections[1]["athlete_id"] = a2
            
            # FINAL CHECK: two athlete boxes must never share the same id
            # (Never two athletes with the same kimono+belt combo)
            athlete_ids = [str(a) for a in expected_athletes.keys()]
            if len(athlete_ids) == 2 and len(valid_athlete_detections) == 2:
                id0 = valid_athlete_detections[0].get('athlete_id')
                id1 = valid_athlete_detections[1].get('athlete_id')
                
                # Fix duplicate ids or None mismatch
                if id0 and id1 and id0 == id1:
                    # Force the second detection to the other id
                    other_id = athlete_ids[1] if athlete_ids[0] == id0 else athlete_ids[0]
                    valid_athlete_detections[1]['athlete_id'] = other_id
                    expected_kimono, expected_belt = _expected_combo_for_athlete(other_id, expected_athletes)
                    valid_athlete_detections[1]['detected_kimono'] = expected_kimono
                    valid_athlete_detections[1]['detected_belt'] = expected_belt
                elif id0 and not id1:
                    # First resolved, second missing — assign the other id
                    other_id = athlete_ids[1] if athlete_ids[0] == id0 else athlete_ids[0]
                    valid_athlete_detections[1]['athlete_id'] = other_id
                    expected_kimono, expected_belt = _expected_combo_for_athlete(other_id, expected_athletes)
                    valid_athlete_detections[1]['detected_kimono'] = expected_kimono
                    valid_athlete_detections[1]['detected_belt'] = expected_belt
                elif id1 and not id0:
                    # Second resolved, first missing — assign the other id
                    other_id = athlete_ids[1] if athlete_ids[0] == id1 else athlete_ids[0]
                    valid_athlete_detections[0]['athlete_id'] = other_id
                    expected_kimono, expected_belt = _expected_combo_for_athlete(other_id, expected_athletes)
                    valid_athlete_detections[0]['detected_kimono'] = expected_kimono
                    valid_athlete_detections[0]['detected_belt'] = expected_belt
            
            # One detection, two histories: pick nearest within max jump.
            if len(athlete_ids) == 2 and len(valid_athlete_detections) == 1:
                d0 = valid_athlete_detections[0]
                p0 = _center_from_bbox(d0["x1"], d0["y1"], d0["x2"], d0["y2"])
                candidates = []
                for aid in athlete_ids:
                    if aid in recent_last:
                        c = _center_from_bbox(recent_last[aid]["x1"], recent_last[aid]["y1"], recent_last[aid]["x2"], recent_last[aid]["y2"])
                        dist = _dist(c, p0)
                        ms = float((d0.get("match_scores") or {}).get(str(aid), 0.0))
                        # same hybrid cost as above
                        cost = dist - (float(max_jump) * 0.80) * ms
                        candidates.append((aid, dist, cost))
                if candidates:
                    candidates.sort(key=lambda x: x[2])
                    best_aid, best_dist, _ = candidates[0]
                    if best_dist <= max_jump:
                        d0["athlete_id"] = best_aid
            
            # Prune stale tracks (not seen >30 frames)
            if frame_count % 30 == 0:
                tracks_to_remove = [
                    tid for tid, info in athlete_tracking.items()
                    if frame_count - info['last_seen_frame'] > 30
                ]
                for tid in tracks_to_remove:
                    del athlete_tracking[tid]

                # Prune old color EMA state too
                color_to_remove = [
                    tid for tid, info in track_color_state.items()
                    if frame_count - int(info.get("last_seen_frame", -10**9)) > 60
                ]
                for tid in color_to_remove:
                    del track_color_state[tid]

            # ------------------------------------------------------------
            # GARANTIA FINAL (hard): nunca permitir cores repetidas
            # - se kimonos esperados forem iguais: 1 deve ser Red e o outro White (quando esperado)
            # - se kimonos esperados forem diferentes: 1 deve ser Blue e o outro White (quando esperado)
            # After athlete_id, rewrite detected_* to the expected JSON combo,
            # porque algumas rotinas acima trocam athlete_id sem atualizar detected_kimono/detected_belt.
            # ------------------------------------------------------------
            if len(athlete_ids_expected) == 2 and len(valid_athlete_detections) == 2:
                d0, d1 = valid_athlete_detections[0], valid_athlete_detections[1]

                # When expected kimonos differ, align assignment by kimono scores
                # (prevents two "Blue" or two "White" labels on screen).
                if not same_kimono_expected:
                    aid1, aid2 = athlete_ids_expected[0], athlete_ids_expected[1]
                    exp_k1 = _norm_expected_color((expected_athletes.get(aid1) or {}).get("Kimono"))
                    exp_k2 = _norm_expected_color((expected_athletes.get(aid2) or {}).get("Kimono"))

                    s0 = d0.get("kim_scores") or {}
                    s1 = d1.get("kim_scores") or {}

                    def _ks(sc: dict, key: str) -> float:
                        try:
                            return float(sc.get(key, 0.0))
                        except Exception:
                            return 0.0

                    direct = _ks(s0, exp_k1) + _ks(s1, exp_k2)
                    swap = _ks(s0, exp_k2) + _ks(s1, exp_k1)
                    if swap > direct:
                        d0["athlete_id"], d1["athlete_id"] = str(aid2), str(aid1)
                    else:
                        d0["athlete_id"], d1["athlete_id"] = str(aid1), str(aid2)

                # Rewrite detected_* from final athlete_id (unique coherent combo),
                # evitando casos onde trocamos athlete_id mas o label/cor ficou antigo.
                for det in (d0, d1):
                    aid = det.get("athlete_id")
                    if aid:
                        ek, eb = _expected_combo_for_athlete(str(aid), expected_athletes)
                        det["detected_kimono"] = ek
                        det["detected_belt"] = eb
            
            # Desenha boxes apenas para os atletas selecionados
            athletes_drawn = set()
            drawn_bboxes: Dict[str, Tuple[int, int, int, int]] = {}
            for detection in valid_athlete_detections:
                x1 = detection['x1']
                y1 = detection['y1']
                x2 = detection['x2']
                y2 = detection['y2']
                athlete_id = detection['athlete_id']
                detected_kimono = detection['detected_kimono']
                detected_belt = detection['detected_belt']

                # DISPLAY GUARANTEE: text/colors mirror expected athlete combo (no duplicates).
                # Isso evita casos em que etapas anteriores trocam athlete_id mas deixam detected_* "antigo".
                exp_k, exp_b = _expected_combo_for_athlete(str(athlete_id), expected_athletes)
                if exp_k != "Unknown":
                    detected_kimono = exp_k
                if exp_b != "Unknown":
                    detected_belt = exp_b
                
                # Motion gate: block teleport swaps from mis-id.
                prev = last_bbox_by_athlete.get(str(athlete_id))
                if prev and (frame_count - int(prev.get("last_seen_frame", -10**9)) <= max_missing_frames):
                    prev_c = _center_from_bbox(prev["x1"], prev["y1"], prev["x2"], prev["y2"])
                    cur_c = _center_from_bbox(int(x1), int(y1), int(x2), int(y2))
                    if _dist(prev_c, cur_c) > max_jump:
                        # Reject detection for this athlete_id; draw previous bbox instead.
                        continue
                
                # Draw box: different expected kimonos -> kimono color; same kimono -> belt color tie-break.
                color = get_kimono_draw_color(detected_kimono) if not same_kimono_expected else get_belt_draw_color(detected_belt)
                thickness = 2

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)

                label = f"Athlete {athlete_id} ({detected_kimono}/{detected_belt})"
                _fsc, _fth = 0.52, 1
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, _fsc, _fth)
                label_y = max(y1 - 8, label_size[1] + 8)
                
                cv2.rectangle(
                    annotated_frame,
                    (x1, label_y - label_size[1] - 4),
                    (x1 + label_size[0] + 4, label_y + 4),
                    color,
                    -1
                )
                
                txt_color = _text_color_for_bg(color)
                cv2.putText(
                    annotated_frame,
                    label,
                    (x1 + 2, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    _fsc,
                    txt_color,
                    _fth,
                    cv2.LINE_AA,
                )
                
                athletes_drawn.add(str(athlete_id))
                drawn_bboxes[str(athlete_id)] = (int(x1), int(y1), int(x2), int(y2))
                last_bbox_by_athlete[str(athlete_id)] = {
                    "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
                    "last_seen_frame": int(frame_count),
                    "detected_kimono": detected_kimono,
                    "detected_belt": detected_belt,
                }
            
            # Fallback: brief miss under heavy occlusion -> reuse last bbox for continuity.
            # When the detector merges both athletes into one box, if only one was drawn but the
            # other was very close in recent history, draw a ghost box aligned to the visible athlete.
            ghost_bbox_for_missing: Dict[str, Tuple[int, int, int, int]] = {}
            if len(athletes_drawn) == 1 and len(expected_athletes) == 2:
                present_id = next(iter(athletes_drawn))
                present_bbox = drawn_bboxes.get(present_id)
                if present_bbox is not None:
                    px1, py1, px2, py2 = present_bbox
                    pc = _center_from_bbox(px1, py1, px2, py2)
                    fusion_dist = max(width, height) * float(ghost_fusion_distance_ratio)
                    for aid in expected_athletes.keys():
                        aid = str(aid)
                        if aid == present_id:
                            continue
                        info = last_bbox_by_athlete.get(aid)
                        if not info:
                            continue
                        if frame_count - int(info.get("last_seen_frame", -10**9)) > max_missing_frames:
                            continue
                        mc = _center_from_bbox(int(info["x1"]), int(info["y1"]), int(info["x2"]), int(info["y2"]))
                        if _dist(pc, mc) <= fusion_dist:
                            ghost_bbox_for_missing[aid] = present_bbox

            for athlete_id in expected_athletes.keys():
                athlete_id = str(athlete_id)
                if athlete_id in athletes_drawn:
                    continue
                info = last_bbox_by_athlete.get(athlete_id)
                if not info:
                    continue
                if frame_count - info["last_seen_frame"] > max_missing_frames:
                    continue
                
                if athlete_id in ghost_bbox_for_missing:
                    x1, y1, x2, y2 = ghost_bbox_for_missing[athlete_id]
                else:
                    x1 = info["x1"]
                    y1 = info["y1"]
                    x2 = info["x2"]
                    y2 = info["y2"]
                detected_kimono = info.get("detected_kimono", "Unknown")
                detected_belt = info.get("detected_belt", "Unknown")

                # Display guarantee on fallback path too
                exp_k, exp_b = _expected_combo_for_athlete(str(athlete_id), expected_athletes)
                if exp_k != "Unknown":
                    detected_kimono = exp_k
                if exp_b != "Unknown":
                    detected_belt = exp_b
                
                color = get_kimono_draw_color(detected_kimono) if not same_kimono_expected else get_belt_draw_color(detected_belt)
                thickness = 1
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
                
                if athlete_id in ghost_bbox_for_missing:
                    label = f"Athlete {athlete_id} ({detected_kimono}/{detected_belt}) [ghost]"
                else:
                    label = f"Athlete {athlete_id} ({detected_kimono}/{detected_belt}) [fallback]"
                _fsc_fb, _fth_fb = 0.5, 1
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, _fsc_fb, _fth_fb)
                label_y = max(y1 - 8, label_size[1] + 8)
                cv2.rectangle(
                    annotated_frame,
                    (x1, label_y - label_size[1] - 4),
                    (x1 + label_size[0] + 4, label_y + 4),
                    color,
                    -1
                )
                txt_color = _text_color_for_bg(color)
                cv2.putText(
                    annotated_frame,
                    label,
                    (x1 + 2, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    _fsc_fb,
                    txt_color,
                    _fth_fb,
                    cv2.LINE_AA,
                )
            
            out.write(annotated_frame)
            written_frames += 1
    
    finally:
        # Release resources even on error
        cap.release()
        out.release()
        print(f"Frames read: {frame_count} | Frames written: {written_frames}")
        print(f"Annotated video saved to: {output_path}")

        if collect_metrics:
            n = float(max(1, metric["frames"]))
            print("\n[Metrics - proxy]")
            print(f"Frames evaluated: {metric['frames']}")
            print(f"Frames with any valid detection: {metric['frames_with_any_det']}")
            print(f"A1 belt ok (por frame): {metric['a1_ok_belt']}/{metric['frames']} = {metric['a1_ok_belt']/n:.3f}")
            print(f"A2 belt ok (por frame): {metric['a2_ok_belt']}/{metric['frames']} = {metric['a2_ok_belt']/n:.3f}")
            print(f"Ambos belts ok (por frame): {metric['both_ok_belt']}/{metric['frames']} = {metric['both_ok_belt']/n:.3f}")

            # Proxy ID accuracy (same-class): pick the track with highest mean red evidence as
            # the "true red belt" and count frames labeled as athlete "1".
            if metric.get("track"):
                # Focus on dominant tracks (largest support) to ignore one-frame outliers.
                tracks_sorted = sorted(metric["track"].items(), key=lambda kv: int(kv[1].get("n", 0)), reverse=True)
                top = tracks_sorted[:4]
                min_n = max(25, int(0.10 * float(max(1, metric["frames"]))))
                top = [(tid, st) for (tid, st) in top if int(st.get("n", 0)) >= min_n] or tracks_sorted[:2]

                best_tid = None
                best_red = -1.0
                for tid, st in top:
                    nn = float(max(1, st.get("n", 0)))
                    red_mean = float(st.get("sum_red", 0.0)) / nn
                    if red_mean > best_red:
                        best_red = red_mean
                        best_tid = tid

                if best_tid is not None and best_tid in metric["track"]:
                    st = metric["track"][best_tid]
                    nn = float(max(1, st.get("n", 0)))
                    id_acc = float(st.get("as_1", 0)) / nn
                    print(f"ID proxy (reddest track labeled as '1'): {st.get('as_1',0)}/{st.get('n',0)} = {id_acc:.3f}")


def annotate_video_from_json(
    video_path: str,
    json_data: dict,
    output_dir: str,
    use_start_video_colors: bool = True,
) -> str:
    """
    Annotate a video using colors from JSON (e.g. produced by main.py).

    Args:
        video_path: Input video path.
        json_data: Color identification payload.
        output_dir: Directory for the annotated output video.
        use_start_video_colors: If True, use colors from the start segment; else first highlight.

    Returns:
        Path to the generated annotated video.
    """
    import os
    
    # Pick which identification block to use
    if use_start_video_colors:
        athletes_colors = json_data.get("athletes", {}).get("identification_start_video", {})
    else:
        athletes_colors = json_data.get("athletes", {}).get("identification_first_highlight", {})
    
    if not athletes_colors:
        raise ValueError("No color identification found in JSON")
    
    # Output filename
    video_basename = os.path.basename(video_path)
    video_name = os.path.splitext(video_basename)[0]
    output_filename = f"{video_name}_annotated.mp4"
    output_path = os.path.join(output_dir, output_filename)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Never overwrite: pick a unique filename if needed
    if os.path.exists(output_path):
        i = 2
        while True:
            candidate = os.path.join(output_dir, f"{video_name}_annotated_v{i}.mp4")
            if not os.path.exists(candidate):
                output_path = candidate
                break
            i += 1
    
    # Run annotation
    annotate_video_with_colors(
        video_path=video_path,
        output_path=output_path,
        expected_athletes=athletes_colors,
    )
    
    return output_path
