#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Render extra-stable fixed-size crops from an export record.

Goes beyond ``crop_smooth_from_export.py``'s moving-average follow with two
strategies chosen for perceived stability:

- ``--mode cine`` (default): zero-phase Savitzky-Golay smoothed window path
  rendered with SUBPIXEL ``warpAffine`` shifts. The frame edges glide at
  near-constant velocity (no integer stepping, no jitter) so the object
  stays nearly frozen inside the crop while the background pans smoothly.
- ``--shake-cancel`` (cine only): phones apply electronic stabilization as
  periodic 1-3px "re-anchor" snaps of the WHOLE frame; a smoothed window
  path cannot cancel them, so every snap stays visible inside the crop.
  This flag measures the scene's global translation per frame by phase
  correlation, extracts its high-frequency displacement (snap + shake) and
  adds it to the window path. The window then jumps WITH the content, so
  the rendered clip looks locked on a tripod. Safe for tight crops: the
  object box gets the same shift, so margins are unchanged.
- Static-hold (cine, default on): where the annotated box stops moving for
  a few frames, the smoothed path is held exactly constant (short blend at
  the run edges) instead of ringing around the stop, and the shake-cancel
  residual is gated to zero where the scene itself no longer moves, so
  locked-off heads/tails render perfectly still. ``--no-static-hold``
  restores the raw smoothed path.
- ``--mode locked``: camera-operator style. The window holds perfectly
  still (rounded int positions) and only glides at a capped speed when the
  object box nears the safe margin, then re-centers and locks again.
  Background is static most of the time; the object drifts within the
  frame between glides.

Both modes write ``<out-dir>/<name>.mp4`` plus ``<name>_windows.json``
(``source_start_frame`` + per-frame windows; floats for cine, ints for
locked) so downstream paste-back stays frame-aligned.

Usage:
    python scripts/render_stable_crop.py \
        --export output/<export_dir> --source <src.mp4> \
        --start 320 --end 608 --size 86x164 --mode cine \
        --out-dir output/<export_dir>/crops_start320_cine
"""

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np


def collect_boxes(annotation: dict, obj_id) -> list:
    boxes = [None] * annotation["num_frames"]
    for fr in annotation["frames"]:
        for obj in fr.get("objects", []):
            if obj["id"] == obj_id and obj.get("bbox_xyxy") is not None:
                boxes[fr["frame_idx"]] = [float(v) for v in obj["bbox_xyxy"]]
    return boxes


def clamp_box(b, W, H, FW, FH):
    """Window top-left that centers the box, clamped to the frame."""
    return (
        min(max((b[0] + b[2]) / 2 - W / 2, 0), FW - W),
        min(max((b[1] + b[3]) / 2 - H / 2, 0), FH - H),
    )


def box_margin(w, b, W, H):
    """Min distance from the box to the window edges (negative = outside)."""
    return min(b[0] - w[0], w[0] + W - b[2], b[1] - w[1], w[1] + H - b[3])


def _smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def static_run_mask(centres, radius=1.5, min_len=3):
    """Frames belonging to near-stationary runs of the box centres.

    A run is a maximal stretch where every centre stays within ``radius``
    px of the running mean (annotation jitter counts as static; a
    sustained drift larger than that does not).
    """
    n = len(centres)
    mask = np.zeros(n, dtype=bool)
    i = 0
    while i < n:
        j = i
        while j + 1 < n:
            seg = centres[i:j + 2]
            if np.max(np.linalg.norm(seg - seg.mean(0), axis=1)) > radius:
                break
            j += 1
        if j - i + 1 >= min_len:
            mask[i:j + 1] = True
            i = j + 1
        else:
            i += 1
    return mask


def hold_static_path(path, centres, mask, blend=3):
    """Freeze the smoothed path across near-stationary runs.

    Temporal smoothers ring at motion->rest boundaries: the polynomial fit
    keeps drifting for several frames after the box has stopped. Where
    ``mask`` marks a static run, replace the path by the run's constant
    centre so those frames render exactly still, ramping over up to
    ``blend`` frames inside the run edges that touch motion.
    """
    out = np.array(path, dtype=float, copy=True)
    n = len(out)
    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and mask[j + 1]:
            j += 1
        held = centres[i:j + 1].mean(0)
        length = j - i + 1
        a = min(blend, length) if i > 0 else 0
        b = min(blend, length - a) if j < n - 1 else 0
        for k in range(a):
            s = _smoothstep((k + 1) / (a + 1))
            out[i + k] = (1 - s) * path[i + k] + s * held
        for k in range(b):
            s = _smoothstep((k + 1) / (b + 1))
            out[j - k] = (1 - s) * path[j - k] + s * held
        out[i + a:j - b + 1] = held
        i = j + 1
    return out


def plan_cine(boxes, size, frame_wh, sg_window, sg_poly, static_hold=True):
    """Savitzky-Golay smoothed (zero-phase) float window path."""
    from scipy.signal import savgol_filter

    W, H = size
    FW, FH = frame_wh
    cs = np.array([[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for b in boxes])
    wlen = min(sg_window, len(cs) - (1 - len(cs) % 2))  # odd, <= len
    sx = savgol_filter(cs[:, 0], wlen, sg_poly)
    sy = savgol_filter(cs[:, 1], wlen, sg_poly)
    path = np.stack([sx, sy], axis=1)
    if static_hold:
        path = hold_static_path(path, cs, static_run_mask(cs))
    x0 = np.clip(path[:, 0] - W / 2, 0, FW - W)
    y0 = np.clip(path[:, 1] - H / 2, 0, FH - H)
    return np.stack([x0, y0], axis=1)


def plan_locked(boxes, size, frame_wh, trig, rel, speed):
    """Hold still; glide (capped speed) only when the margin gets tight."""
    W, H = size
    FW, FH = frame_wh
    cur = list(clamp_box(boxes[0], W, H, FW, FH))
    mode = "hold"
    wins = []
    for b in boxes:
        if mode == "hold" and box_margin(cur, b, W, H) < trig:
            mode = "pursue"
        if mode == "pursue":
            tgt = clamp_box(b, W, H, FW, FH)
            dx, dy = tgt[0] - cur[0], tgt[1] - cur[1]
            dist = max(abs(dx), abs(dy))
            if dist > 0:
                v = min(speed, dist) if dist >= 6 else max(0.8, dist / 4)
                cur = [cur[0] + dx / dist * v, cur[1] + dy / dist * v]
            if box_margin(cur, b, W, H) >= rel:
                mode = "hold"
        wins.append((round(cur[0]), round(cur[1])))
    return wins


def _ffmpeg_pipe(size, fps, out, crf):
    w, h = size
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
        "-r", f"{fps:.6g}", "-i", "-",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
        "-pix_fmt", "yuv420p", str(out),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def _phase_motion(source: Path, start: int, end: int):
    """Raw per-frame global translation (dx, dy) via phase correlation."""
    cap = cv2.VideoCapture(str(source))
    grays = []
    idx = 0
    while idx <= end:
        ok, frame = cap.read()
        if not ok:
            break
        if idx >= start:
            small = cv2.resize(frame, None, fx=0.5, fy=0.5)
            g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
            grays.append(cv2.GaussianBlur(g, (5, 5), 1.0))
        idx += 1
    cap.release()
    win = cv2.createHanningWindow((grays[0].shape[1], grays[0].shape[0]), cv2.CV_32F)
    mot = [(0.0, 0.0)]
    for i in range(1, len(grays)):
        (dx, dy), _ = cv2.phaseCorrelate(grays[i - 1], grays[i], win)
        mot.append((dx * 2.0, dy * 2.0))  # back to full-resolution units
    return np.clip(np.asarray(mot, dtype=np.float64), -8, 8)


def measure_shake(source: Path, start: int, end: int, hp_window: int = 81,
                  gate=(0.2, 0.6)):
    """High-frequency scene displacement (EIS snaps + handshake) per frame.

    Phase-correlates consecutive frames (half resolution, blurred),
    integrates the translation and subtracts its Savitzky-Golay smooth
    part, leaving only the high-frequency displacement the window must
    follow to freeze the content.

    The high-pass leaks the camera's deceleration transient into frames
    where nothing moves any more, so the residual is scaled to zero while
    the raw inter-frame motion stays under ``gate[0]`` px (smoothstep up to
    ``gate[1]`` px). Pass ``gate=None`` for the ungated residual.
    """
    from scipy.signal import savgol_filter

    mot = _phase_motion(source, start, end)
    cum = np.cumsum(mot, axis=0)
    wlen = min(hp_window, len(cum) - (1 - len(cum) % 2))
    smooth = np.stack(
        [savgol_filter(cum[:, 0], wlen, 3), savgol_filter(cum[:, 1], wlen, 3)],
        axis=1,
    )
    hf = np.clip(cum - smooth, -6, 6)
    if gate is not None:
        mag = np.linalg.norm(mot, axis=1)
        t = np.clip((mag - gate[0]) / (gate[1] - gate[0]), 0.0, 1.0)
        hf = hf * _smoothstep(t)[:, None]
    return hf


def encode_cine(source, windows, size, fps, out, crf, start):
    """Subpixel translation crop via warpAffine (float window positions)."""
    w, h = size
    proc = _ffmpeg_pipe(size, fps, out, crf)
    cap = cv2.VideoCapture(str(source))
    n = 0
    idx = 0
    while idx < start + len(windows):
        ok, frame = cap.read()
        if not ok:
            break
        if idx >= start:
            x0, y0 = windows[n]
            mat = np.float32([[1, 0, -x0], [0, 1, -y0]])
            crop = cv2.warpAffine(
                frame, mat, (w, h),
                flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
            )
            proc.stdin.write(np.ascontiguousarray(crop))
            n += 1
        idx += 1
    cap.release()
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed for {out}")
    return n


def encode_int(source, windows, size, fps, out, crf, start):
    """Integer crop via array slicing (locked mode: exact static holds)."""
    w, h = size
    proc = _ffmpeg_pipe(size, fps, out, crf)
    cap = cv2.VideoCapture(str(source))
    n = 0
    idx = 0
    while idx < start + len(windows):
        ok, frame = cap.read()
        if not ok:
            break
        if idx >= start:
            x0, y0 = windows[n]
            proc.stdin.write(
                np.ascontiguousarray(frame[y0:y0 + h, x0:x0 + w])
            )
            n += 1
        idx += 1
    cap.release()
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed for {out}")
    return n


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export", type=Path, required=True,
                    help="export directory containing annotation.json")
    ap.add_argument("--source", type=Path, required=True,
                    help="original video to crop from")
    ap.add_argument("--ids", default="1", help="comma-separated object ids")
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, required=True)
    ap.add_argument("--size", default=None,
                    help="crop WxH (default: max box + 12%% margin, even)")
    ap.add_argument("--mode", choices=["cine", "locked"], default="cine")
    ap.add_argument("--sg-window", type=int, default=81,
                    help="cine: Savitzky-Golay window length (odd)")
    ap.add_argument("--sg-poly", type=int, default=3,
                    help="cine: Savitzky-Golay poly order")
    ap.add_argument("--shake-cancel", action="store_true",
                    help="cine: follow the scene's high-frequency displacement "
                         "(phone EIS snaps / handshake) so the content renders "
                         "as if shot on a tripod")
    ap.add_argument("--no-static-hold", action="store_true",
                    help="cine: disable the exact hold across static/near-"
                         "static box runs and the shake-residual motion gate")
    ap.add_argument("--trig", type=float, default=5.0,
                    help="locked: start gliding when margin < this")
    ap.add_argument("--rel", type=float, default=11.0,
                    help="locked: stop gliding when margin >= this")
    ap.add_argument("--speed", type=float, default=3.0,
                    help="locked: glide speed cap (px/frame)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--crf", type=int, default=16)
    args = ap.parse_args()

    annotation = json.loads((args.export / "annotation.json").read_text())
    FW, FH = annotation["width"], annotation["height"]
    fps = float(annotation["fps"])

    ids = [int(t) for t in args.ids.split(",")]
    per_obj = [collect_boxes(annotation, i) for i in ids]
    if len(ids) > 1:
        boxes = [
            None if all(b[i] is None for b in per_obj)
            else [min(b[i][0] for b in per_obj if b[i] is not None),
                  min(b[i][1] for b in per_obj if b[i] is not None),
                  max(b[i][2] for b in per_obj if b[i] is not None),
                  max(b[i][3] for b in per_obj if b[i] is not None)]
            for i in range(annotation["num_frames"])
        ]
    else:
        boxes = per_obj[0]
    boxes = boxes[args.start:args.end + 1]
    if any(b is None for b in boxes):
        raise ValueError("span contains frames without a box; narrow --start/--end")

    if args.size:
        w, h = (int(t) for t in args.size.lower().split("x"))
        size = (w // 2 * 2, h // 2 * 2)
    else:
        bw = max(b[2] - b[0] for b in boxes)
        bh = max(b[3] - b[1] for b in boxes)
        pad = max(bw, bh) * 0.12
        size = (min(int(bw + 2 * pad), FW) // 2 * 2,
                min(int(bh + 2 * pad), FH) // 2 * 2)

    name = "obj" if len(ids) == 1 else "obj" + "+".join(map(str, ids))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "cine":
        windows = plan_cine(boxes, size, (FW, FH), args.sg_window, args.sg_poly,
                            static_hold=not args.no_static_hold)
        strategy = f"cine savgol w{args.sg_window}/p{args.sg_poly} subpixel"
        if not args.no_static_hold:
            strategy += " + static-hold"
        if args.shake_cancel:
            shake = measure_shake(args.source, args.start, args.end)
            windows = windows + shake
            windows[:, 0] = np.clip(windows[:, 0], 0, FW - size[0])
            windows[:, 1] = np.clip(windows[:, 1], 0, FH - size[1])
            strategy += " + shake-cancel"
        windows_out = [
            {"frame_idx": args.start + i, "x0": round(float(x0), 3),
             "y0": round(float(y0), 3), "has_box": True}
            for i, (x0, y0) in enumerate(windows)
        ]
        n = encode_cine(args.source, windows, size, fps,
                        args.out_dir / f"{name}.mp4", args.crf, args.start)
    else:
        windows = plan_locked(boxes, size, (FW, FH), args.trig, args.rel, args.speed)
        windows_out = [
            {"frame_idx": args.start + i, "x0": int(x0), "y0": int(y0),
             "has_box": True}
            for i, (x0, y0) in enumerate(windows)
        ]
        strategy = f"locked trig{args.trig} rel{args.rel} @{args.speed}"
        n = encode_int(args.source, windows, size, fps,
                       args.out_dir / f"{name}.mp4", args.crf, args.start)

    (args.out_dir / f"{name}_windows.json").write_text(json.dumps({
        "source": str(args.source),
        "source_start_frame": args.start,
        "objects": ids,
        "crop_size": list(size),
        "fps": fps,
        "strategy": strategy,
        "windows": windows_out,
    }, ensure_ascii=False, indent=1))

    # stability report
    W, H = size
    marg = np.array([box_margin(w, b, W, H) for w, b in zip(windows, boxes)])
    ws = np.array(windows, dtype=float)
    step = np.abs(np.diff(ws, axis=0)).max(1)
    extra = f", static frames {100 * (step == 0).mean():.0f}%"
    print(f"[{args.mode}] {name}: {n} frames (src {args.start}..{args.start + n - 1}), "
          f"crop {W}x{H}, min-margin {marg.min():.1f}px, "
          f"window max|step| {step.max():.2f}px{extra} "
          f"-> {args.out_dir / (name + '.mp4')}")


if __name__ == "__main__":
    main()
