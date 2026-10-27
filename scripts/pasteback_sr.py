#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Paste a super-resolved crop clip back into the original video.

Takes the original (base) video, a super-resolved clip that was produced
from one of the ``crop_smooth_from_export`` crops, that crop's window json
and the export mask sequence, and composites the SR content back into the
full-size frames:

- The SR frames map 1:1 onto base frames ``[source_start_frame,
  source_start_frame + n)`` (recorded in the crop window json); base frames
  outside that range pass through untouched. ``--sr``/``--windows`` may be
  repeated for multi-segment paste-back (e.g. SR clips tiled over the full
  timeline); each segment is placed at its own window json's positions.
- The union of the given object ids' masks decides WHICH pixels are
  replaced, so the background around the object stays original.
- The mask is eroded a little and Gaussian-blurred (feather), then blended
  as alpha: ``out = base * (1 - a) + SR * a``. Because the SR canvas equals
  the base outside the crop window, pixels outside the window are always
  the untouched original and no rectangular seam can appear.
- Per-frame SR models flicker on textured regions ("boiling").
  ``--temporal R`` applies a binomial low-pass of radius R over the
  downscaled SR frames (R=1 -> [1,2,1]/4) before compositing to calm that;
  the cost is slight motion softening on fast pans.
- ``--upscale N`` delivers an Nx canvas (e.g. 2 = 4K from 1080p): base
  frames are Lanczos-upscaled, windows/masks scale by N, and the SR clip
  (produced at a higher factor) is downscaled straight onto the Nx window,
  so the object region keeps more genuine SR detail than the Lanczos
  background. Feather args (--erode/--sigma) must then be given at the
  upscaled scale (roughly xN).
- When a span of the SR clip is unusable (e.g. defocused source frames
  that make the SR output blocky), ``--fade-in START:END`` / ``--fade-out
  START:END`` ramp the paste alpha smoothly (cosine smoothstep) between 0
  and 1 across those SR-clip frames, so the composite transitions
  temporally between SR and original content instead of popping. Indices
  are 0-based SR-clip frame indices (crop-clip index j, NOT the source
  frame index). Repeatable; multipliers compose multiplicatively.
  A ramp may be restricted to one segment by suffixing the segment's
  ``source_start_frame``: ``START:END@53`` (no suffix = every segment).
- ``--dip CENTER:RADIUS[@SEG]`` carves a temporal feather INSIDE a
  segment: the paste alpha dips smoothly (cosine smoothstep) to 0 at
  SR-clip frame CENTER and back to 1 RADIUS frames on either side, e.g.
  to let a defocused base stretch show through instead of a too-sharp
  repair patch. Repeatable; multiplies with the fade ramps.
- ``--freeze START:PICK[@SEG]`` handles a static tail: from SOURCE frame
  START to the end of that segment, every output frame is the FULL
  composited result of SOURCE frame PICK (PICK's base frame, SR content,
  mask and window position), so the whole picture is exactly as still as
  frame PICK itself instead of boiling. Fade ramps are evaluated at PICK.
- Window jsons from cine-mode crops carry float subpixel coordinates;
  they are rounded to the nearest integer (after ``--upscale`` scaling)
  before compositing.

Usage:
    conda run -n sam3 python scripts/pasteback_sr.py \
        --base output/sr_pasteback/src/need_fix_1.mp4 \
        --sr output/sr_pasteback/src/eye_crop1_8x.ts \
        --windows "output/需修复视频 (1)_46e6aedb/crops_groupend/obj1+2_windows.json" \
        --masks "output/需修复视频 (1)_46e6aedb/masks" \
        --ids 1 2 --scale 8 \
        --out output/sr_pasteback/眼药水-crop-1__8x_pasteback.mp4
"""

import argparse
import json
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np


def feather_alpha(mask_path: Path, ids, erode_iter: int, sigma: float,
                  upscale: int = 1) -> np.ndarray:
    """Full-frame alpha in [0, 1]: eroded union mask, Gaussian-feathered."""
    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"Cannot read mask {mask_path}")
    binary = np.isin(m, ids).astype(np.uint8)
    if upscale > 1:
        binary = cv2.resize(
            binary,
            (binary.shape[1] * upscale, binary.shape[0] * upscale),
            interpolation=cv2.INTER_NEAREST,
        )
    if not binary.any():
        return np.zeros(binary.shape, dtype=np.float32)
    if erode_iter > 0:
        binary = cv2.erode(binary, np.ones((3, 3), np.uint8), iterations=erode_iter)
        if not binary.any():
            return np.zeros(binary.shape, dtype=np.float32)
    alpha = cv2.GaussianBlur(binary.astype(np.float32), (0, 0), sigma)
    return np.clip(alpha, 0.0, 1.0)


def parse_ramp(spec: str):
    """Parse 'START:END[@SEG]' into (start, end, seg_or_None).

    The optional ``@SEG`` suffix restricts the ramp to the segment whose
    ``source_start_frame`` equals SEG (multi-segment paste-backs need
    per-segment feather zones).
    """
    seg = None
    if "@" in spec:
        spec, _, seg_part = spec.partition("@")
        try:
            seg = int(seg_part)
        except ValueError:
            raise ValueError(
                f"fade segment filter must be an int, got {spec!r}"
            ) from None
    try:
        start, end = (int(v) for v in spec.split(":"))
    except ValueError:
        raise ValueError(
            f"fade spec must be START:END[@SEG] (integers), got {spec!r}"
        ) from None
    if end < start:
        raise ValueError(f"fade END must be >= START, got {spec!r}")
    return start, end, seg


def ramp_multiplier(j: int, ramps, fade_in: bool, seg_start=None) -> float:
    """Smoothstep alpha multiplier for SR frame index j over 'START:END' ramps.

    fade_in: 0 before START, ramps up to 1 at END, 1 after.
    fade_out: the mirror image (1 before START, ramps down to 0 at END).
    Ramps carrying a ``@SEG`` filter apply only to that segment.
    """
    mult = 1.0
    for start, end, seg in ramps:
        if seg is not None and seg_start is not None and seg != seg_start:
            continue
        if fade_in:
            t = 0.0 if j < start else 1.0 if j >= end else (j - start) / (end - start)
        else:
            t = 1.0 if j < start else 0.0 if j >= end else 1.0 - (j - start) / (end - start)
        if start < end:
            t = 0.5 - 0.5 * math.cos(math.pi * t)
        mult *= t
    return mult


def parse_dip(spec: str):
    """Parse 'CENTER:RADIUS[@SEG]' into (center, radius, seg_or_None)."""
    seg = None
    if "@" in spec:
        spec, _, seg_part = spec.partition("@")
        try:
            seg = int(seg_part)
        except ValueError:
            raise ValueError(
                f"dip segment filter must be an int, got {seg_part!r}"
            ) from None
    try:
        center, radius = (int(v) for v in spec.split(":"))
    except ValueError:
        raise ValueError(
            f"dip spec must be CENTER:RADIUS[@SEG] (integers), got {spec!r}"
        ) from None
    if radius <= 0:
        raise ValueError(f"dip RADIUS must be > 0, got {spec!r}")
    return center, radius, seg


def dip_multiplier(j: int, dips, seg_start=None) -> float:
    """Smoothstep V-dip: alpha 1 outside, 0 at CENTER, over RADIUS frames.

    Lets a mid-clip stretch fall back to the base content (e.g. where the
    base is defocused and a sharp repair patch would read as a
    depth-of-field jump).
    """
    mult = 1.0
    for center, radius, seg in dips:
        if seg is not None and seg_start is not None and seg != seg_start:
            continue
        d = abs(j - center)
        if d < radius:
            mult *= 0.5 - 0.5 * math.cos(math.pi * d / radius)
    return mult


def parse_freeze(spec: str):
    """Parse 'START:PICK[@SEG]' (SOURCE frame indices) into (start, pick, seg)."""
    seg = None
    if "@" in spec:
        spec, _, seg_part = spec.partition("@")
        try:
            seg = int(seg_part)
        except ValueError:
            raise ValueError(
                f"freeze segment filter must be an int, got {seg_part!r}"
            ) from None
    try:
        start, pick = (int(v) for v in spec.split(":"))
    except ValueError:
        raise ValueError(
            f"freeze spec must be START:PICK[@SEG] (integers), got {spec!r}"
        ) from None
    if pick < start:
        raise ValueError(f"freeze PICK must be >= START, got {spec!r}")
    return start, pick, seg


def temporal_smooth(frames, radius: int):
    """Binomial low-pass along time (truncated and renormalized at ends)."""
    if radius <= 1 and radius != 1:
        return frames
    if len(frames) < 2:
        return frames
    kernel = np.array(
        [math.comb(2 * radius, k) for k in range(2 * radius + 1)],
        dtype=np.float64,
    )
    kernel /= kernel.sum()
    out = []
    n = len(frames)
    for i in range(n):
        acc = np.zeros_like(frames[0], dtype=np.float32)
        wsum = 0.0
        for k, wgt in enumerate(kernel):
            j = i - radius + k
            if 0 <= j < n:
                acc += frames[j] * wgt
                wsum += wgt
        out.append((acc / wsum).astype(frames[0].dtype))
    return out


def pasteback(args) -> None:
    fade_in = [parse_ramp(s) for s in args.fade_in]
    fade_out = [parse_ramp(s) for s in args.fade_out]
    dips = [parse_dip(s) for s in args.dip]
    freezes = [parse_freeze(s) for s in args.freeze]
    if len(args.sr) != len(args.windows):
        raise ValueError("--sr and --windows must be given the same number of times")
    # coordinate scale: --upscale also Lanczos-upscales the base frames,
    # --canvas-scale only scales windows/masks (base already at that size)
    cscale = args.canvas_scale if args.canvas_scale else args.upscale
    segments = []
    for sr_path, wj in zip(args.sr, args.windows):
        meta = json.loads(Path(wj).read_text())
        segments.append({
            "sr": sr_path,
            "w": meta["crop_size"][0],
            "h": meta["crop_size"][1],
            "start": meta.get("source_start_frame", 0),
            "windows": meta["windows"],
        })
    segments.sort(key=lambda s: s["start"])
    # scale windows and crop sizes to the (optional) upscaled canvas;
    # cine-mode crop jsons carry float subpixel window coordinates, while
    # the paste-back canvas is integer, so round AFTER the upscale scaling
    for seg in segments:
        seg["w"] *= cscale
        seg["h"] *= cscale
        for win in seg["windows"]:
            win["x0"] = int(round(win["x0"] * cscale))
            win["y0"] = int(round(win["y0"] * cscale))
    w, h = segments[0]["w"], segments[0]["h"]

    base_cap = cv2.VideoCapture(str(args.base))
    if not base_cap.isOpened():
        raise RuntimeError(f"Cannot open base video {args.base}")
    fps = base_cap.get(cv2.CAP_PROP_FPS) or 24.0
    width = int(base_cap.get(cv2.CAP_PROP_FRAME_WIDTH)) * args.upscale
    height = int(base_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) * args.upscale
    if args.canvas_scale and (args.canvas_scale != 1):
        # base frames are already at the target canvas size
        width = int(base_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(base_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    for seg in segments:
        cap = cv2.VideoCapture(str(seg["sr"]))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open SR clip {seg['sr']}")
        frames = []
        while True:
            ok, sr_frame = cap.read()
            if not ok:
                break
            frames.append(
                cv2.resize(sr_frame, (seg["w"], seg["h"]), interpolation=cv2.INTER_AREA)
            )
        cap.release()
        if len(frames) < len(seg["windows"]):
            raise RuntimeError(
                f"SR clip {seg['sr']} has {len(frames)} frames, "
                f"windows json expects {len(seg['windows'])}"
            )
        if args.temporal > 0:
            frames = temporal_smooth(frames, args.temporal)
        for start, pick, fseg in freezes:
            if fseg is not None and fseg != seg["start"]:
                continue
            j_pick = pick - seg["start"]
            if not 0 <= j_pick < len(frames):
                raise RuntimeError(
                    f"--freeze pick frame {pick} maps outside SR clip {seg['sr']} "
                    f"({len(frames)} frames from {seg['start']})"
                )
            if seg.get("freeze") is not None:
                raise RuntimeError(
                    f"multiple --freeze specs match segment {seg['start']}"
                )
            seg["freeze"] = (max(0, start - seg["start"]), j_pick)
        seg["frames"] = frames

    # Whole-frame freeze needs each PICK frame's BASE frame; grab them in
    # one extra sequential pass (this container cannot be seeked).
    needed = {
        seg["start"] + seg["freeze"][1]
        for seg in segments if seg.get("freeze") is not None
    }
    if needed:
        pick_base = {}
        pre = cv2.VideoCapture(str(args.base))
        if not pre.isOpened():
            raise RuntimeError(f"Cannot open base video {args.base}")
        i2 = 0
        while needed:
            ok, bfr = pre.read()
            if not ok:
                raise RuntimeError(
                    f"--freeze pick frame(s) {sorted(needed)} beyond base "
                    f"video ({i2} frames)")
            if i2 in needed:
                if args.upscale > 1:
                    bfr = cv2.resize(
                        bfr, (width, height), interpolation=cv2.INTER_LANCZOS4
                    )
                pick_base[i2] = bfr
                needed.discard(i2)
            i2 += 1
        pre.release()
        for seg in segments:
            if seg.get("freeze") is not None:
                seg["freeze_base"] = pick_base[seg["start"] + seg["freeze"][1]]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
        "-r", f"{fps:.6g}", "-i", "-",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", str(args.crf),
        "-pix_fmt", "yuv420p", str(args.out),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    n_pasted = n_total = 0
    i = 0
    seg_idx = 0
    while True:
        ok, frame = base_cap.read()
        if not ok:
            break
        if args.upscale > 1:
            frame = cv2.resize(
                frame, (width, height), interpolation=cv2.INTER_LANCZOS4
            )
        out_frame = frame
        # advance to the segment covering frame i (segments may gap)
        while seg_idx < len(segments) and i >= segments[seg_idx]["start"] + len(segments[seg_idx]["frames"]):
            seg_idx += 1
        if seg_idx < len(segments):
            seg = segments[seg_idx]
            if i >= seg["start"]:
                j = i - seg["start"]
                frz = seg.get("freeze")
                if frz is not None and j >= frz[0]:
                    # whole-frame freeze: emit PICK's composited result as-is
                    j = frz[1]
                    frame = seg["freeze_base"]
                sr_small = seg["frames"][j]
                sw, sh = seg["w"], seg["h"]
                alpha = feather_alpha(
                    Path(args.masks) / f"{seg['start'] + j:05d}.png", args.ids,
                    args.erode, args.sigma, cscale,
                )
                mult = ramp_multiplier(j, fade_in, True, seg["start"]) * ramp_multiplier(
                    j, fade_out, False, seg["start"]
                ) * dip_multiplier(j, dips, seg["start"])
                if mult < 1.0:
                    alpha = alpha * np.float32(mult)
                if alpha.any():
                    x0, y0 = seg["windows"][j]["x0"], seg["windows"][j]["y0"]
                    canvas = frame.copy()
                    canvas[y0:y0 + sh, x0:x0 + sw] = sr_small
                    a = alpha[..., None]
                    out_frame = (
                        frame.astype(np.float32) * (1.0 - a)
                        + canvas.astype(np.float32) * a
                    ).astype(np.uint8)
                n_pasted += 1
        proc.stdin.write(np.ascontiguousarray(out_frame))
        n_total += 1
        i += 1

    base_cap.release()
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed for {args.out}")
    print(
        f"{args.out}: {n_total} base frames, SR pasted on {n_pasted} frames "
        f"across {len(segments)} segment(s) "
        f"({', '.join(str(s['start']) + '..' + str(s['start'] + len(s['frames']) - 1) for s in segments)}), "
        f"crop {w}x{h} from {args.scale}x SR, feather sigma={args.sigma} "
        f"erode={args.erode} temporal={args.temporal}"
        + (f", dips={dips}" if dips else "")
        + (f", freezes={freezes}" if freezes else "")
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="original full video")
    parser.add_argument("--sr", action="append", required=True,
                        help="super-resolved crop clip (.ts/.mp4); repeat with "
                             "--windows for multi-segment paste-back")
    parser.add_argument("--windows", action="append", required=True,
                        help="crop window json from crop_smooth_from_export; "
                             "repeat once per --sr")
    parser.add_argument("--masks", required=True, help="export masks/ directory (full-frame label PNGs)")
    parser.add_argument("--ids", nargs="+", type=int, required=True,
                        help="object ids whose union mask limits the paste")
    parser.add_argument("--scale", type=float, default=8.0,
                        help="SR upscale factor of the clip (downscaled back before paste)")
    parser.add_argument("--erode", type=int, default=1,
                        help="mask erosion iterations before feathering")
    parser.add_argument("--sigma", type=float, default=4.0,
                        help="Gaussian sigma of the feather")
    parser.add_argument("--temporal", type=int, default=0,
                        help="binomial temporal low-pass radius over the SR "
                             "frames before compositing (0 = off, 1 = [1,2,1]/4)")
    parser.add_argument("--fade-in", action="append", default=[], metavar="START:END[@SEG]",
                        help="ramp alpha 0->1 across SR frames [START, END) "
                             "(smoothstep; SR-clip frame indices); repeatable; "
                             "optional @SEG applies it only to that segment")
    parser.add_argument("--fade-out", action="append", default=[], metavar="START:END[@SEG]",
                        help="ramp alpha 1->0 across SR frames [START, END) "
                             "(smoothstep; SR-clip frame indices); repeatable; "
                             "optional @SEG applies it only to that segment")
    parser.add_argument("--dip", action="append", default=[], metavar="CENTER:RADIUS[@SEG]",
                        help="temporal V-dip inside a segment: paste alpha dips "
                             "smoothly to 0 at SR-clip frame CENTER and back to 1 "
                             "RADIUS frames either side (e.g. let a defocused base "
                             "stretch show through); repeatable; optional @SEG")
    parser.add_argument("--freeze", action="append", default=[], metavar="START:PICK[@SEG]",
                        help="static-tail freeze: from SOURCE frame START to the "
                             "segment end every output frame equals the FULL "
                             "composited result of SOURCE frame PICK (its base "
                             "frame, SR content, mask and window position; kills "
                             "all residual motion on frozen footage); repeatable; "
                             "optional @SEG")
    parser.add_argument("--canvas-scale", type=int, default=None,
                        help="scale windows/masks by N while using the base "
                             "video AS-IS at its native (already-upscaled, "
                             "e.g. 4K) resolution; unlike --upscale the base "
                             "frames are not resampled")
    parser.add_argument("--upscale", type=int, default=1,
                        help="integer canvas upscale factor N: base frames are "
                             "Lanczos-upscaled Nx and windows/masks scaled to "
                             "match before compositing (e.g. 2 = 4K delivery)")
    parser.add_argument("--crf", type=int, default=16, help="x264 CRF quality")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    pasteback(args)


if __name__ == "__main__":
    main()
