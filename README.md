# Color-based judo referee / athlete labeling (baseline)

This folder contains a **heuristic color baseline**: it does not train a domain-specific detector. It is meant as a simple comparator against a full YOLO model trained for judo (e.g. three classes: referee, athlete blue judogi, athlete white judogi).

## What it does

1. Run a **standard YOLO** model (pose variant by default) to detect **people** in each frame.
2. **Crop** each person and analyze pixels (dominant colors, HSV/Lab heuristics, belt regions).
3. Apply **rules** to assign a coarse role/color label (black gi → referee-like filter; white / blue gi → athletes;).

This pipeline is **intuitive** and can work when visuals are very standardized, but it is **less robust** than training YOLO directly on referee / athlete_blue / athlete_white.

## How to run (annotated videos)

Requirements: Python with `opencv-python`, `numpy`, `ultralytics`, and a YOLO weights file.

```bash
python run_based_color_videos.py --videos-dir videos --out-dir annotated_compare --export-schema 3_c
```

- **`--export-schema 3_c`**: draws three logical classes (referee, athlete blue, athlete white) on the output MP4.
- **`--model`**: optional path to different YOLO weights.
- **`--max-frames N`**: optional cap for quick tests.

Other modules:

- **`identify_colors.py`** — color logic from cropped stills (e.g. `pessoa_*_kimono.jpg` / `pessoa_*_belt.jpg`) for JSON-style athlete color hints.
- **`video_annotator.py`** — frame pipeline used by `run_based_color_videos.py`.

## Outputs

Annotated MP4s are written to the folder you pass as `--out-dir` (for example `annotated_compare/`), with filenames like `*_based_color_3_c.mp4`.
