"""
Run the color-based heuristic pipeline (pose + kimono/belt), same idea as ``annotate_frames_from_dir``,
over each video in a folder and write annotated MP4s (e.g. export_schema ``3_c``).
"""

from __future__ import annotations

import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from video_annotator import annotate_video_color_heuristic  # noqa: E402


def _list_videos(folder: str) -> list[str]:
    exts = (".mp4", ".MP4", ".mkv", ".MKV", ".webm", ".WEBM", ".avi", ".AVI")
    out: list[str] = []
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        if os.path.isfile(path) and name.lower().endswith(exts):
            out.append(path)
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description="Annotate videos with the based_color heuristic and YOLO export_schema labels."
    )
    p.add_argument(
        "--videos-dir",
        default=os.path.join(SCRIPT_DIR, "videos"),
        help="Folder containing videos (.mp4, etc.).",
    )
    p.add_argument(
        "--out-dir",
        default=os.path.join(SCRIPT_DIR, "saida_based_color", "3_c", "videos_anotados"),
        help="Output folder for annotated MP4s (default matches existing project layout).",
    )
    p.add_argument(
        "--export-schema",
        default="3_c",
        choices=("default", "2_c", "3_c"),
        help="Label layout on frames (3_c = referee / athlete_blue / athlete_white).",
    )
    p.add_argument("--model", default="yolo11n-pose.pt", help="YOLO pose weights (relative to project if present).")
    p.add_argument("--conf", type=float, default=0.20)
    p.add_argument("--iou", type=float, default=0.85)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Per video: 0 = all frames; >0 = only the first N frames (quick test).",
    )
    args = p.parse_args()

    vdir = os.path.abspath(args.videos_dir)
    if not os.path.isdir(vdir):
        raise SystemExit(f"Invalid folder: {vdir}")

    videos = _list_videos(vdir)
    if not videos:
        raise SystemExit(f"No videos in: {vdir}")

    out_root = os.path.abspath(args.out_dir)
    os.makedirs(out_root, exist_ok=True)

    for vp in videos:
        base = os.path.splitext(os.path.basename(vp))[0]
        out_path = os.path.join(out_root, f"{base}_based_color_{args.export_schema}.mp4")
        mf = int(args.max_frames)
        annotate_video_color_heuristic(
            video_path=vp,
            output_path=out_path,
            model_path=args.model,
            conf_threshold=float(args.conf),
            iou_threshold=float(args.iou),
            imgsz=int(args.imgsz),
            export_schema=str(args.export_schema),
            max_frames=(None if mf <= 0 else mf),
        )
        print("---")


if __name__ == "__main__":
    main()
