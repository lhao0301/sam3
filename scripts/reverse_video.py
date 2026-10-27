#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Reverse the playback order of a video's frames (e.g. a ``_rev`` SR clip
back to normal chronological order).

Frames are read sequentially from the input (MPEG-TS containers cannot be
seeked, so no random access is attempted) and written out in reversed
order, either as a lossless PNG sequence, an H.264 MP4, or both. The PNG
sequence is the bit-exact form used as paste-back input; the MP4 is a
convenient review copy.

Usage:
    python scripts/reverse_video.py \
        --input output/sr_pasteback/obj1_1021/src/obj1_rev__12x.ema6500.zu4ref.ts \
        --out-png-dir output/sr_pasteback/obj1_1021/transpose_normal \
        --out-video output/sr_pasteback/obj1_1021/obj1_norm__12x.ema6500.zu4ref.mp4
"""

import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True,
                        help="input video (.ts/.mp4); always read sequentially")
    parser.add_argument("--out-png-dir", type=Path,
                        help="optional directory for the reversed lossless "
                             "PNG sequence (frame_00000.png, ...)")
    parser.add_argument("--out-video", type=Path,
                        help="optional reversed H.264 MP4 review copy")
    parser.add_argument("--crf", type=int, default=12,
                        help="x264 CRF quality for --out-video")
    args = parser.parse_args()

    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open input video {args.input}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from {args.input}")
    h, w = frames[0].shape[:2]
    frames.reverse()
    print(f"{args.input}: {len(frames)} frames {w}x{h} @ {fps:.6g} fps, reversed")

    if args.out_png_dir:
        args.out_png_dir.mkdir(parents=True, exist_ok=True)
        for i, frame in enumerate(frames):
            cv2.imwrite(str(args.out_png_dir / f"{i:05d}.png"), frame)
        print(f"  wrote {len(frames)} PNGs to {args.out_png_dir}")

    if args.out_video:
        args.out_video.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
            "-r", f"{fps:.6g}", "-i", "-",
            "-c:v", "libx264", "-preset", "medium", "-crf", str(args.crf),
            "-pix_fmt", "yuv420p", str(args.out_video),
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        for frame in frames:
            proc.stdin.write(np.ascontiguousarray(frame))
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed for {args.out_video}")
        print(f"  wrote {args.out_video}")


if __name__ == "__main__":
    main()
