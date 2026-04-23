"""Kimono and belt color heuristics from cropped athlete images (K-means + pixel scores)."""

from __future__ import annotations

import re
from glob import glob

import colorsys
import cv2
import numpy as np

# --- BGR normalization (illumination / white balance) -----------------------------------------


def _normalize_bgr_for_color(bgr: np.ndarray) -> np.ndarray:
    """Light CLAHE on L channel in Lab space; leaves chroma unchanged."""
    if bgr is None or bgr.size == 0:
        return bgr
    try:
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l2 = clahe.apply(l_chan)
        lab2 = cv2.merge([l2, a_chan, b_chan])
        return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    except Exception:
        return bgr


def is_black_kimono(bgr_img: np.ndarray) -> bool:
    """True if the crop looks like a black gi (referee/spectator), not dark blue."""
    if bgr_img is None or bgr_img.size == 0:
        return False

    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    margin = int(min(h, w) * 0.10)
    roi_gray = gray[margin : h - margin, margin : w - margin] if h > 2 * margin and w > 2 * margin else gray

    v95 = float(np.percentile(roi_gray, 95))
    frac_dark = float((roi_gray < 45).mean())
    frac_very_dark = float((roi_gray < 25).mean())
    dark_candidate = (v95 < 65) or (frac_dark > 0.70) or (frac_very_dark > 0.30)
    if not dark_candidate:
        return False

    roi_bgr = (
        bgr_img[margin : h - margin, margin : w - margin]
        if (bgr_img.shape[0] > 2 * margin and bgr_img.shape[1] > 2 * margin)
        else bgr_img
    )
    try:
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        hh = hsv[:, :, 0].astype(np.int32)
        ss = hsv[:, :, 1].astype(np.float32) / 255.0
        vv = hsv[:, :, 2].astype(np.float32) / 255.0

        blue_like = (hh >= 75) & (hh <= 165) & (ss >= 0.12) & (vv >= 0.08)
        blue_frac = float(blue_like.mean())

        dark_mask = vv <= 0.25
        med_s_dark = float(np.median(ss[dark_mask])) if np.any(dark_mask) else float(np.median(ss))

        lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        a_lab = lab[:, :, 1] - 128.0
        b_lab = lab[:, :, 2] - 128.0
        chroma = np.sqrt(a_lab * a_lab + b_lab * b_lab)
        med_chroma = float(np.median(chroma))

        if blue_frac >= 0.06:
            return False

        return (med_s_dark <= 0.18) and (med_chroma <= 20.0)
    except Exception:
        return dark_candidate


# --- File collection per person id -------------------------------------------------------------


def collect_people(video_dir: str) -> dict[str, dict[str, str]]:
    """
    Build ``{ person_id: {"kimono": path, "belt": path } }`` from ``pessoa_<id>_{kimono,belt}.jpg``.
    Drops entries with missing kimono or black-gi filter.
    """
    raw: dict[str, dict[str, str]] = {}
    pattern = f"{video_dir}/pessoa_*_*.jpg"

    for path in glob(pattern):
        match = re.search(r"pessoa_(\d+)_(kimono|belt)\.jpg$", path)
        if not match:
            continue
        pid, kind = match.group(1), match.group(2)
        raw.setdefault(pid, {})[kind] = path

    people: dict[str, dict[str, str]] = {}
    for pid, parts in raw.items():
        kimono_path = parts.get("kimono")
        if not kimono_path:
            continue
        kimono_bgr = cv2.imread(kimono_path)
        if kimono_bgr is None or is_black_kimono(kimono_bgr):
            continue
        people[pid] = parts

    return people


# --- Pixel sampling and K-means ----------------------------------------------------------------


def sample_pixels(paths: list[str], max_pixels_total: int = 500_000, rng: np.random.Generator | None = None):
    if not paths:
        return None
    rng = rng or np.random.default_rng(0)
    per_image = max(1000, max_pixels_total // len(paths))
    chunks: list[np.ndarray] = []
    for path in paths:
        img = cv2.imread(path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        if w > 200:
            scale = 200.0 / w
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        flat = img.reshape(-1, 3)
        n = min(per_image, flat.shape[0])
        idx = rng.choice(flat.shape[0], size=n, replace=False)
        chunks.append(flat[idx])
    return np.vstack(chunks).astype(np.float32) if chunks else None


def kmeans_dominant_colors(pixels: np.ndarray | None, k: int = 5) -> list[dict]:
    if pixels is None or len(pixels) == 0:
        return []
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    centers = np.clip(np.rint(centers), 0, 255).astype(np.uint8)
    counts = np.bincount(labels.flatten(), minlength=k)
    order = counts.argsort()[::-1]
    total = counts.sum()
    out: list[dict] = []
    for i in order:
        out.append(
            {
                "cluster_index": int(i),
                "color": centers[i].astype(float).tolist(),
                "color_percentage": float(counts[i]) / float(total),
            }
        )
    return out


# --- Color helpers -----------------------------------------------------------------------------


def _rgb_luminance(rgb) -> float:
    r, g, b = [float(x) for x in rgb]
    r /= 255.0
    g /= 255.0
    b /= 255.0
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def is_white_tone(rgb) -> bool:
    r, g, b = [float(x) / 255.0 for x in rgb]
    v = max(r, g, b)
    minc = min(r, g, b)
    chroma = v - minc
    s = 0.0 if v == 0 else chroma / v
    return (v >= 0.72) and (minc >= 0.50) and ((s <= 0.45) or (chroma <= 0.18))


def is_blue_tone_kimono(rgb) -> bool:
    try:
        px = np.asarray(rgb, dtype=np.float32).reshape(1, 1, 3)
        px = np.clip(px, 0, 255).astype(np.uint8)
        hsv = cv2.cvtColor(px, cv2.COLOR_RGB2HSV).reshape(3)
        hue = int(hsv[0])
        sat = float(hsv[1]) / 255.0
        val = float(hsv[2]) / 255.0
        return (80 <= hue <= 155) and (sat >= 0.12) and (val >= 0.10)
    except Exception:
        r, g, b = [float(x) for x in rgb]
        maxc = max(r, g, b)
        if maxc == 0:
            return False
        return (b == maxc) and (b > 70) and (b - max(r, g)) > 15


def classify_kimono_colors(dom_kim: list[dict]) -> tuple[str, str]:
    if not dom_kim:
        return "undefined", "undefined"

    has_white = False
    has_blue = False

    for entry in dom_kim:
        rgb = entry["color"]
        perc = entry["color_percentage"]
        if perc < 0.02:
            continue
        if is_white_tone(rgb):
            has_white = True
        if is_blue_tone_kimono(rgb):
            has_blue = True

    if has_white and has_blue:
        return "blue", "white"
    if has_white and not has_blue:
        return "white", "white"
    if has_blue and not has_white:
        return "blue", "blue"
    return "undefined", "undefined"


def is_red_tone(rgb) -> bool:
    r, g, b = [float(x) / 255.0 for x in rgb]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    if v < 0.18 or s < 0.28:
        return False
    if not ((h < 0.08) or (h > 0.92)):
        return False
    if not (r >= g + 0.08 and r >= b + 0.08):
        return False
    return True


def is_black_tone(rgb) -> bool:
    return _rgb_luminance(rgb) < 0.20


def kimono_scores_from_pixels(rgb_pixels: np.ndarray) -> dict[str, float]:
    scores = {"white": 0.0, "blue": 0.0, "black": 0.0}
    if rgb_pixels is None or len(rgb_pixels) == 0:
        return scores

    px = np.asarray(rgb_pixels, dtype=np.float32)
    px = np.clip(px, 0, 255).astype(np.uint8)

    hsv = cv2.cvtColor(px.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    h = hsv[:, 0].astype(np.int32)
    s = hsv[:, 1].astype(np.float32) / 255.0
    v = hsv[:, 2].astype(np.float32) / 255.0

    lab = cv2.cvtColor(px.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    l = lab[:, 0]
    a_lab = lab[:, 1] - 128.0
    b_lab = lab[:, 2] - 128.0
    chroma = np.sqrt(a_lab * a_lab + b_lab * b_lab)

    r = px[:, 0].astype(np.float32) / 255.0
    g = px[:, 1].astype(np.float32) / 255.0
    b = px[:, 2].astype(np.float32) / 255.0

    black = (v <= 0.20) | (l <= 55.0)

    white_lab = (l >= 170.0) & (chroma <= 18.0)
    white_hsv = (v >= 0.70) & (s <= 0.25)
    white_rgb = (r >= 0.75) & (g >= 0.75) & (b >= 0.75) & (np.abs(r - g) <= 0.15) & (np.abs(g - b) <= 0.15)
    white = (white_lab | white_hsv | white_rgb) & ~((b > r + 0.15) & (b > g + 0.15))

    blue_hsv = (h >= 80) & (h <= 155) & (s >= 0.12) & (v >= 0.10)
    blue_rgb = (b > r + 0.10) & (b > g + 0.10) & (b >= 0.25)
    blue = blue_hsv | (blue_rgb & (h >= 70) & (h <= 165))

    n = float(len(px))
    scores["black"] = float(black.mean()) if n > 0 else 0.0
    scores["white"] = float(white.mean()) if n > 0 else 0.0
    scores["blue"] = float(blue.mean()) if n > 0 else 0.0
    return scores


def pick_kimono_label_from_scores(scores: dict) -> str:
    if not scores:
        return "undefined"

    white = float(scores.get("white", 0.0))
    blue = float(scores.get("blue", 0.0))

    if blue >= 0.08 and blue >= white * 0.55:
        return "blue"

    best = "white" if white >= blue else "blue"
    best_val = max(white, blue)
    if best_val < 0.12:
        return "undefined"
    return best


def _normalize_kimono_label(value: str) -> str:
    s = str(value or "").strip().lower()
    if s in ("white",):
        return "white"
    if s in ("blue",):
        return "blue"
    return "undefined"


def _normalize_belt_label(value: str) -> str:
    s = str(value or "").strip().lower()
    if s in ("white",):
        return "white"
    if s in ("red",):
        return "red"
    if s in ("black",):
        return "black"
    return "undefined"


def _ensure_unique_two_athletes(athlete_infos: list) -> list:
    if not athlete_infos or len(athlete_infos) < 2:
        return athlete_infos

    a1, a2 = athlete_infos[0], athlete_infos[1]

    a1["kimono_color"] = _normalize_kimono_label(a1.get("kimono_color"))
    a2["kimono_color"] = _normalize_kimono_label(a2.get("kimono_color"))
    a1["belt_color"] = _normalize_belt_label(a1.get("belt_color"))
    a2["belt_color"] = _normalize_belt_label(a2.get("belt_color"))

    combo1 = (a1["kimono_color"], a1["belt_color"])
    combo2 = (a2["kimono_color"], a2["belt_color"])
    if combo1 == combo2:
        if (
            a1["kimono_color"] == "white"
            and a2["kimono_color"] == "white"
            and a1["belt_color"] == "red"
            and a2["belt_color"] == "red"
        ):
            s1 = a1.get("belt_scores") or {}
            s2 = a2.get("belt_scores") or {}
            s1_red = float(s1.get("red", 0.0))
            s2_red = float(s2.get("red", 0.0))
            if s1_red >= s2_red:
                a2["belt_color"] = "white"
            else:
                a1["belt_color"] = "white"
        elif a1["belt_color"] == "red":
            a2["belt_color"] = "white"
        elif a1["belt_color"] == "white":
            a2["belt_color"] = "red"
        else:
            a2["belt_color"] = "white" if a1["belt_color"] != "white" else "red"

    combo1 = (a1["kimono_color"], a1["belt_color"])
    combo2 = (a2["kimono_color"], a2["belt_color"])
    if combo1 == combo2 and a1["kimono_color"] == a2["kimono_color"]:
        s1 = a1.get("kimono_scores") or {}
        s2 = a2.get("kimono_scores") or {}
        d1 = float(s1.get("blue", 0.0)) - float(s1.get("white", 0.0))
        d2 = float(s2.get("blue", 0.0)) - float(s2.get("white", 0.0))
        if d2 > d1:
            a2["kimono_color"] = "blue"
            a1["kimono_color"] = "white"
        else:
            a1["kimono_color"] = "blue"
            a2["kimono_color"] = "white"

    athlete_infos[0], athlete_infos[1] = a1, a2
    return athlete_infos


def belt_scores_from_dom(dom_belt: list[dict]) -> dict[str, float]:
    scores = {"white": 0.0, "red": 0.0, "black": 0.0}
    if not dom_belt:
        return scores

    for entry in dom_belt:
        rgb = entry["color"]
        perc = entry["color_percentage"]
        if perc < 0.02:
            continue
        if is_white_tone(rgb):
            scores["white"] += perc
        if is_red_tone(rgb):
            scores["red"] += perc
        if is_black_tone(rgb):
            scores["black"] += perc

    return scores


def belt_scores_from_pixels(rgb_pixels: np.ndarray) -> dict[str, float]:
    scores = {"white": 0.0, "red": 0.0, "black": 0.0}
    if rgb_pixels is None or len(rgb_pixels) == 0:
        return scores

    px = np.asarray(rgb_pixels, dtype=np.float32)
    px = np.clip(px, 0, 255).astype(np.uint8)

    hsv = cv2.cvtColor(px.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    h = hsv[:, 0].astype(np.int32)
    s = hsv[:, 1].astype(np.float32) / 255.0
    v = hsv[:, 2].astype(np.float32) / 255.0

    lab = cv2.cvtColor(px.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    l = lab[:, 0]
    a_lab = lab[:, 1] - 128.0
    b_lab = lab[:, 2] - 128.0
    chroma = np.sqrt(a_lab * a_lab + b_lab * b_lab)

    r = px[:, 0].astype(np.float32) / 255.0
    g = px[:, 1].astype(np.float32) / 255.0
    b = px[:, 2].astype(np.float32) / 255.0

    black = (v <= 0.20) | (l <= 55.0)

    white_lab = (l >= 175.0) & (chroma <= 18.0)
    white_hsv = (v >= 0.72) & (s <= 0.22)
    white = white_lab | white_hsv

    red_hsv = ((h <= 18) | (h >= 162)) & (s >= 0.14) & (v >= 0.10)
    red_rgb = (r >= g + 0.06) & (r >= b + 0.06) & (r >= 0.20)
    red_lab = (a_lab >= 10.0) & (chroma >= 12.0)
    red = red_hsv & (red_rgb | red_lab)

    n = float(len(px))
    scores["black"] = float(black.mean()) if n > 0 else 0.0
    scores["white"] = float(white.mean()) if n > 0 else 0.0
    scores["red"] = float(red.mean()) if n > 0 else 0.0
    return scores


def pick_belt_label_from_scores(scores: dict) -> str:
    label = max(scores, key=scores.get)
    if scores[label] == 0:
        return "undefined"
    return label


def second_color_after_white(scores: dict) -> str:
    items = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    for color, val in items:
        if color == "white":
            continue
        if val > 0:
            return color
    return "undefined"


def identify_colors_main(video_dir: str) -> list[dict] | None:
    people = collect_people(video_dir)

    if not people:
        print("No valid people found (check path or black-gi filter).")
        return None

    rng = np.random.default_rng(0)

    kimono_paths_all = [d["kimono"] for d in people.values() if "kimono" in d]
    belt_paths_all = [d["belt"] for d in people.values() if "belt" in d]

    kim_pixels_all = sample_pixels(kimono_paths_all, rng=rng)
    belt_pixels_all = sample_pixels(belt_paths_all, rng=rng)

    dom_kim_global = kmeans_dominant_colors(kim_pixels_all, K=10)
    _dom_belt_global = kmeans_dominant_colors(belt_pixels_all, K=10)

    global_kimono_1, global_kimono_2 = classify_kimono_colors(dom_kim_global)

    ids_sorted_all = sorted(people.keys(), key=lambda x: int(x))

    precomputed_kimono: dict[str, dict] = {}
    candidates: list[tuple[str, str, dict, float]] = []
    for pid_str in ids_sorted_all:
        parts = people.get(pid_str) or {}
        k_path = parts.get("kimono")
        if not k_path:
            continue
        k_img_bgr = cv2.imread(k_path)
        if k_img_bgr is None:
            continue
        k_img_bgr = _normalize_bgr_for_color(k_img_bgr)
        k_img_rgb = cv2.cvtColor(k_img_bgr, cv2.COLOR_BGR2RGB)

        h, w = k_img_rgb.shape[:2]
        mx = int(w * 0.10)
        my = int(h * 0.10)
        roi = k_img_rgb[my : h - my, mx : w - mx] if (h > 2 * my and w > 2 * mx) else k_img_rgb
        flat = roi.reshape(-1, 3)
        if len(flat) == 0:
            continue
        n = min(50_000, len(flat))
        sel = rng.choice(len(flat), size=n, replace=False)
        sampled = flat[sel]
        scores_k = kimono_scores_from_pixels(sampled)
        label_k = pick_kimono_label_from_scores(scores_k)

        quality = float(
            max(scores_k.get("white", 0.0), scores_k.get("blue", 0.0))
            - 0.40 * float(scores_k.get("black", 0.0))
        )
        precomputed_kimono[pid_str] = {"label": label_k, "scores": scores_k, "quality": quality}

        if parts.get("belt"):
            quality += 0.03
        candidates.append((pid_str, label_k, scores_k, quality))

    if not candidates:
        ordered_ids = ids_sorted_all[:2]
    else:
        confident = [(pid, lab, sc, q) for (pid, lab, sc, q) in candidates if q >= 0.12]
        pool = confident if confident else candidates

        blue_pool = sorted([c for c in pool if c[1] == "blue"], key=lambda x: x[3], reverse=True)
        white_pool = sorted([c for c in pool if c[1] == "white"], key=lambda x: x[3], reverse=True)
        any_pool = sorted(pool, key=lambda x: x[3], reverse=True)

        chosen: list[str | None] = []
        if blue_pool and white_pool:
            chosen = [blue_pool[0][0], white_pool[0][0]]
        else:
            chosen = [x[0] for x in any_pool[:2]]

        chosen = [c for c in chosen if c is not None]
        if len(chosen) < 2:
            for pid, _, _, _ in any_pool:
                if pid not in chosen:
                    chosen.append(pid)
                if len(chosen) >= 2:
                    break

        ordered_ids = chosen[:2]

    if len(ordered_ids) == 0:
        print("\nNo athletes with kimono after filters.")
        return None

    global_kimono_colors = [global_kimono_1, global_kimono_2]

    athlete_infos: list[dict] = []

    for idx, pid_str in enumerate(ordered_ids):
        pid = int(pid_str)
        parts = people[pid_str]

        kimono_color = "undefined"
        kimono_scores = None
        if pid_str in precomputed_kimono:
            kimono_color = precomputed_kimono[pid_str]["label"]
            kimono_scores = precomputed_kimono[pid_str]["scores"]
        else:
            k_path = parts.get("kimono")
            if k_path:
                k_img_bgr = cv2.imread(k_path)
                if k_img_bgr is not None:
                    k_img_bgr = _normalize_bgr_for_color(k_img_bgr)
                    k_img_rgb = cv2.cvtColor(k_img_bgr, cv2.COLOR_BGR2RGB)
                    h, w = k_img_rgb.shape[:2]

                    mx = int(w * 0.10)
                    my = int(h * 0.10)
                    roi = k_img_rgb[my : h - my, mx : w - mx] if (h > 2 * my and w > 2 * mx) else k_img_rgb

                    flat = roi.reshape(-1, 3)
                    if len(flat) > 0:
                        n = min(50_000, len(flat))
                        sel = rng.choice(len(flat), size=n, replace=False)
                        sampled = flat[sel]
                        scores_k = kimono_scores_from_pixels(sampled)
                        kimono_scores = scores_k
                        kimono_color = pick_kimono_label_from_scores(scores_k)

        if kimono_color == "undefined":
            kimono_color = global_kimono_colors[idx] if idx < len(global_kimono_colors) else "undefined"
            if kimono_scores is None:
                kimono_scores = {"white": 0.0, "blue": 0.0, "black": 0.0}

        scores_belt = {"white": 0.0, "red": 0.0, "black": 0.0}
        b_path = parts.get("belt")
        if b_path:
            b_img_bgr = cv2.imread(b_path)
            if b_img_bgr is not None:
                b_img_bgr = _normalize_bgr_for_color(b_img_bgr)
                b_img_rgb = cv2.cvtColor(b_img_bgr, cv2.COLOR_BGR2RGB)
                bh, bw = b_img_rgb.shape[:2]
                mx = int(bw * 0.08)
                my = int(bh * 0.12)
                broi = b_img_rgb[my : bh - my, mx : bw - mx] if (bh > 2 * my and bw > 2 * mx) else b_img_rgb
                flat_b = broi.reshape(-1, 3)
                if len(flat_b) > 0:
                    n = min(40_000, len(flat_b))
                    sel = rng.choice(len(flat_b), size=n, replace=False)
                    scores_belt = belt_scores_from_pixels(flat_b[sel])

        red_sc = float(scores_belt.get("red", 0.0))
        white_sc = float(scores_belt.get("white", 0.0))
        if red_sc >= 0.07 and red_sc >= white_sc * 0.45:
            belt_color_initial = "red"
        else:
            belt_color_initial = pick_belt_label_from_scores(scores_belt)

        athlete_infos.append(
            {
                "id": pid,
                "kimono_color": kimono_color,
                "kimono_scores": kimono_scores,
                "belt_scores": scores_belt,
                "belt_color": belt_color_initial,
            }
        )

    if len(athlete_infos) == 2:
        a1, a2 = athlete_infos[0], athlete_infos[1]

        same_kimono = a1["kimono_color"] == a2["kimono_color"]

        if same_kimono and a1["belt_color"] != "red" and a2["belt_color"] != "red":

            s1_red = a1["belt_scores"]["red"]
            s2_red = a2["belt_scores"]["red"]

            if s2_red > s1_red:
                a2["belt_color"] = "red"
            elif s1_red > s2_red:
                a1["belt_color"] = "red"
            else:
                a2["belt_color"] = "red"

        if same_kimono:
            if a1["belt_color"] == "red" and a2["belt_color"] == "red":
                s1_red = a1["belt_scores"]["red"]
                s2_red = a2["belt_scores"]["red"]
                if s1_red >= s2_red:
                    a2["belt_color"] = "white"
                else:
                    a1["belt_color"] = "white"
            elif a1["belt_color"] == "red" and a2["belt_color"] != "red":
                a2["belt_color"] = "white"
            elif a2["belt_color"] == "red" and a1["belt_color"] != "red":
                a1["belt_color"] = "white"

    if len(athlete_infos) == 2:
        a1, a2 = athlete_infos[0], athlete_infos[1]

        if a1["kimono_color"] != a2["kimono_color"]:
            final_colors = [a1["belt_color"], a2["belt_color"]]

            if final_colors[0] == "white" and final_colors[1] != "red":
                c2 = second_color_after_white(a1["belt_scores"])
                final_colors[0] = c2

            if final_colors[1] == "white" and final_colors[0] != "red":
                c2 = second_color_after_white(a2["belt_scores"])
                final_colors[1] = c2

            a1["belt_color"] = final_colors[0]
            a2["belt_color"] = final_colors[1]

    athlete_infos = _ensure_unique_two_athletes(athlete_infos)

    return athlete_infos


def identify_colors_for_json(video_dir: str) -> dict[str, dict[str, str]]:
    """Run ``identify_colors_main`` and map athlete list to JSON-friendly English labels."""
    athlete_infos = identify_colors_main(video_dir)
    if not athlete_infos:
        return {}

    display = {
        "white": "White",
        "blue": "Blue",
        "red": "Red",
        "black": "Black",
        "undefined": "Undefined",
    }

    def _to_json_color(label: str) -> str:
        if label is None:
            return "Undefined"
        key = str(label).strip().lower()
        return display.get(key, key.capitalize())

    result: dict[str, dict[str, str]] = {}
    for idx, athlete in enumerate(athlete_infos, start=1):
        kimono = athlete.get("kimono_color", "undefined")
        belt = athlete.get("belt_color", "undefined")

        result[str(idx)] = {
            "Kimono": _to_json_color(kimono),
            "Belt": _to_json_color(belt),
        }

    return result
