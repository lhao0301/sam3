#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Smooth fixed-size crops of annotated objects from an export record.

Reads an export directory produced by the annotation UI (``annotation.json``
+ ``masks/`` + ``annotated.mp4``), locates the matching original video under
a source directory, and writes one cropped clip per annotated object:

- The crop window has a FIXED size for the whole clip: the object's
  annotated box size (the ``--size-percentile`` percentile of per-frame
  widths/heights, max by default) plus a padding margin, clamped to the
  frame bounds and made even (H.264 requires even dimensions).
- The window center follows the object through a moving-average smoother so
  per-frame box jitter does not shake the crop. Frames without a box hold
  the previous window.
- ``--group 1+2`` crops several objects together: the per-frame window is
  derived from the UNION of the members' boxes ("obj1+2.mp4").
- ``--trim`` restricts the output to the annotated span (first..last frame
  with a box), dropping the undetected head/tail of the source video.
  ``--start``/``--end`` clamp that span manually (useful when the tail
  annotations linger through a source scene transition).

Outputs land in ``<export_dir>/crops/obj{N}.mp4`` together with
``crop_windows.json`` (the exact per-frame window, for reproducibility).

Usage:
    conda run -n sam3 python scripts/crop_smooth_from_export.py \
        --source-dir /tmp/sam3_uploads \
        "output/手机屏幕-素材-1_32b52eee" "output/需修复视频 (2)_5bb1b471"
"""

import argparse
import glob
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np


def probe_frame_count(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def frame_signature(path: Path, idx: int) -> np.ndarray:
    """Downscaled grayscale signature of one frame, for source matching."""
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read frame {idx} of {path}")
    gray = cv2.cvtColor(cv2.resize(frame, (96, 54)), cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.float32)
    return (gray - gray.mean()) / (gray.std() + 1e-6)


def find_source(source_dir: Path, num_frames: int, annotated: Path) -> Path:
    """Pick the upload whose frame count matches; break ties by content."""
    candidates = sorted(source_dir.glob("*.mp4"))
    same_len = [c for c in candidates if probe_frame_count(c) == num_frames]
    if not same_len:
        raise FileNotFoundError(
            f"No upload with {num_frames} frames under {source_dir}"
        )
    if len(same_len) == 1:
        return same_len[0]
    idxs = [num_frames // 4, num_frames // 2, 3 * num_frames // 4]
    ann_sigs = [frame_signature(annotated, i) for i in idxs]
    scores = {}
    for c in same_len:
        scores[c] = float(
            np.mean(
                [(frame_signature(c, i) * s).mean() for i, s in zip(idxs, ann_sigs)]
            )
        )
    return max(scores, key=scores.get)


def collect_boxes(annotation: dict, obj_id) -> list:
    """Per-frame bbox_xyxy (None where the object is absent)."""
    boxes = [None] * annotation["num_frames"]
    for fr in annotation["frames"]:
        for obj in fr.get("objects", []):
            if obj["id"] == obj_id and obj.get("bbox_xyxy") is not None:
                boxes[fr["frame_idx"]] = [float(v) for v in obj["bbox_xyxy"]]
    return boxes


def union_boxes(boxes_lists: list) -> list:
    """Per-frame union bbox across several objects (None if none present)."""
    n = len(boxes_lists[0])
    out = []
    for i in range(n):
        present = [b[i] for b in boxes_lists if b[i] is not None]
        if not present:
            out.append(None)
        else:
            out.append([
                min(b[0] for b in present), min(b[1] for b in present),
                max(b[2] for b in present), max(b[3] for b in present),
            ])
    return out


def trim_range(boxes: list):
    """(first, last) frame indices carrying a box, or None."""
    idx = [i for i, b in enumerate(boxes) if b is not None]
    return (idx[0], idx[-1]) if idx else None


def fixed_crop_size(
    boxes: list, frame_w: int, frame_h: int, pad_frac: float, size_percentile: float
):
    """Percentile box size + symmetric margin, clamped to the frame, even."""
    valid = [b for b in boxes if b is not None]
    if not valid:
        raise ValueError("Object has no annotated boxes")
    widths = np.array([b[2] - b[0] for b in valid])
    heights = np.array([b[3] - b[1] for b in valid])
    base_w = float(np.percentile(widths, size_percentile))
    base_h = float(np.percentile(heights, size_percentile))
    pad = max(base_w, base_h) * pad_frac
    w = min(base_w + 2 * pad, float(frame_w))
    h = min(base_h + 2 * pad, float(frame_h))
    return int(w) // 2 * 2, int(h) // 2 * 2


def smooth_track(centers: np.ndarray, valid: np.ndarray, window: int) -> np.ndarray:
    """Fill gaps by holding, then moving-average the center trajectory."""
    out = centers.copy()
    idx = np.where(valid)[0]
    first, last = idx[0], idx[-1]
    for i in range(len(out)):
        if valid[i]:
            prev = i
        elif first <= i <= last:
            prev = i - 1
        elif i < first:
            prev = first
        else:
            prev = last
        out[i] = out[prev]

    window = min(window, len(out) | 1)  # odd, at least 1
    if window < 3:
        return out
    kernel = np.ones(window) / window
    padded = np.pad(out, ((window // 2, window // 2), (0, 0)), mode="edge")
    return np.stack(
        [
            np.convolve(padded[:, 0], kernel, mode="valid"),
            np.convolve(padded[:, 1], kernel, mode="valid"),
        ],
        axis=1,
    )


def crop_windows(boxes: list, size, frame_w: int, frame_h: int, window: int):
    """Fixed-size windows (x0, y0) per frame, smoothed then clamped."""
    w, h = size
    valid = np.array([b is not None for b in boxes])
    centers = np.zeros((len(boxes), 2), dtype=np.float64)
    for i, b in enumerate(boxes):
        if b is not None:
            centers[i] = ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)
    centers = smooth_track(centers, valid, window)

    windows = []
    for cx, cy in centers:
        x0 = min(max(cx - w / 2.0, 0.0), frame_w - w)
        y0 = min(max(cy - h / 2.0, 0.0), frame_h - h)
        windows.append((int(round(x0)), int(round(y0))))
    return windows


def encode_clip(source: Path, windows, size, fps: float, out: Path, crf: int, start: int = 0):
    """Read the source once, crop frames [start, start+len(windows)) into an H.264 mp4."""
    w, h = size
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
        "-r", f"{fps:.6g}", "-i", "-",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
        "-pix_fmt", "yuv420p", str(out),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    cap = cv2.VideoCapture(str(source))
    n = 0
    idx = 0
    while idx < start + len(windows):
        ok, frame = cap.read()
        if not ok:
            break
        if idx >= start:
            x0, y0 = windows[n]
            proc.stdin.write(np.ascontiguousarray(frame[y0:y0 + h, x0:x0 + w]))
            n += 1
        idx += 1
    cap.release()
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed for {out}")
    return n


def process_export(export_dir: Path, source_dir: Path, args) -> None:
    annotation = json.loads((export_dir / "annotation.json").read_text())
    frame_w, frame_h = annotation["width"], annotation["height"]
    fps = float(annotation["fps"])
    annotated = export_dir / "annotated.mp4"
    source = find_source(source_dir, annotation["num_frames"], annotated)
    print(f"[{export_dir.name}] source: {source.name} ({annotation['num_frames']} frames @ {fps} fps)")

    crops_name = "crops" if not args.tag else f"crops_{args.tag}"
    crops_dir = export_dir / crops_name
    crops_dir.mkdir(parents=True, exist_ok=True)
    obj_ids = sorted({obj["id"] for obj in annotation["objects"]})
    if args.group:
        groups = []
        for spec in args.group:
            ids = [int(t) for t in spec.split("+")]
            unknown = [i for i in ids if i not in obj_ids]
            if unknown:
                raise ValueError(f"Unknown object ids {unknown} in export {export_dir.name}")
            groups.append(ids)
    else:
        groups = [[i] for i in obj_ids]

    for ids in groups:
        per_obj = [collect_boxes(annotation, i) for i in ids]
        boxes = union_boxes(per_obj) if len(ids) > 1 else per_obj[0]
        name = "obj" + "+".join(str(i) for i in ids)

        lo, hi = 0, len(boxes) - 1
        if args.trim:
            span = trim_range(boxes)
            if span is None:
                raise ValueError(f"{name}: no annotated frames")
            lo, hi = span
        if args.start is not None:
            lo = max(lo, args.start)
        if args.end is not None:
            hi = min(hi, args.end)
        if lo > hi:
            raise ValueError(f"{name}: empty frame range [{lo}, {hi}]")
        start, end = lo, hi
        boxes = boxes[start:end + 1]

        size = fixed_crop_size(boxes, frame_w, frame_h, args.pad, args.size_percentile)
        windows = crop_windows(boxes, size, frame_w, frame_h, args.smooth)
        out = crops_dir / f"{name}.mp4"
        n = encode_clip(source, windows, size, fps, out, args.crf, start=start)

        valid = [b for b in boxes if b is not None]
        drift = max(
            (windows[i][0] - windows[j][0]) ** 2 + (windows[i][1] - windows[j][1]) ** 2
            for i in range(len(windows)) for j in range(len(windows))
        ) ** 0.5
        span_txt = f", source frames {start}..{start + n - 1}" if args.trim else ""
        print(
            f"  {name}: {n} frames{span_txt}, crop {size[0]}x{size[1]}, "
            f"{len(valid)}/{len(boxes)} boxes annotated, "
            f"window drift {drift:.0f}px -> {out}"
        )
        (crops_dir / f"{name}_windows.json").write_text(json.dumps({
            "source": str(source),
            "source_start_frame": start,
            "objects": ids,
            "crop_size": list(size),
            "fps": fps,
            "windows": [
                {"frame_idx": start + i, "x0": x0, "y0": y0, "has_box": boxes[i] is not None}
                for i, (x0, y0) in enumerate(windows)
            ],
        }, ensure_ascii=False, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export_dirs", nargs="+", type=Path,
                        help="export directories containing annotation.json")
    parser.add_argument("--source-dir", type=Path, default=Path("/tmp/sam3_uploads"),
                        help="directory holding the original uploaded videos")
    parser.add_argument("--pad", type=float, default=0.3,
                        help="crop margin as a fraction of the largest box side")
    parser.add_argument("--size-percentile", type=float, default=100,
                        help="percentile of per-frame box w/h for the crop size "
                             "(100 = max, never clips the object; lower = tighter)")
    parser.add_argument("--tag", default=None,
                        help="output subdirectory suffix (crops_<tag> instead of crops)")
    parser.add_argument("--group", nargs="*", metavar="ID+ID",
                        help="crop several objects together, e.g. --group 1+2; "
                             "window derives from the union of their boxes")
    parser.add_argument("--trim", action="store_true",
                        help="output only the annotated span, dropping the "
                             "undetected head/tail of the source video")
    parser.add_argument("--start", type=int, default=None,
                        help="first source frame to keep (with --trim, clamps the "
                             "annotated span; without, crops this range)")
    parser.add_argument("--end", type=int, default=None,
                        help="last source frame to keep (same semantics as --start)")
    parser.add_argument("--smooth", type=int, default=13,
                        help="moving-average window (frames) for the crop center")
    parser.add_argument("--crf", type=int, default=16, help="x264 CRF quality")
    args = parser.parse_args()
    for export_dir in args.export_dirs:
        process_export(export_dir, args.source_dir, args)


if __name__ == "__main__":
    main()
