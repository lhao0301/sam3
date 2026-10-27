#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Fill mid-sequence mask gaps in a SAM3 export using SAM3 video tracking.

A SAM3 annotation export can contain frames whose mask is empty even though
the SAME object has masks both before and after the gap (tracking dropped on
frames where the object is still clearly visible). This script re-tracks
those frames with SAM3:

- For each object, empty frames strictly INSIDE its annotated span
  (annotated frame before AND after the run) are treated as fill targets;
  head/tail empties (object not yet visible / scene changed) stay untouched.
- A padded box prompt derived from the nearest annotated frame's mask is
  added at that frame, the video predictor propagates over the whole clip,
  and only the gap frames adopt the propagated mask.
- The result is written to a new export directory (masks/ + annotation.json,
  with bbox entries added for filled frames), leaving the original intact.

Usage:
    conda run -n sam3 python scripts/fill_mask_gaps_sam3.py \
        --export "output/新建项目_视频5·样片_bdcd3f7a" \
        --video output/maskfill_视频5/src/新建项目_视频5·样片.mp4 \
        --out "output/新建项目_视频5·样片_bdcd3f7a_filled" --gpu 0
"""

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


def object_mask(masks: dict, oid: int, i: int) -> np.ndarray:
    return (masks[i] == oid).astype(np.uint8)


def empty_runs(have_set: set, lo: int, hi: int):
    """Consecutive empty runs strictly inside [lo, hi]."""
    runs = []
    for i in range(lo, hi + 1):
        if i in have_set:
            continue
        if runs and i == runs[-1][1] + 1:
            runs[-1][1] = i
        else:
            runs.append([i, i])
    return runs


def prompt_box(mask: np.ndarray, pad_frac: float, w: int, h: int):
    """Normalized [xmin, ymin, w, h] box around the mask, padded."""
    ys, xs = np.where(mask > 0)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    side = max(x1 - x0, y1 - y0)
    px = int(round(side * pad_frac))
    x0 = max(0, x0 - px)
    y0 = max(0, y0 - px)
    x1 = min(w - 1, x1 + px)
    y1 = min(h - 1, y1 + px)
    return [
        x0 / w,
        y0 / h,
        (x1 - x0 + 1) / w,
        (y1 - y0 + 1) / h,
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--export", required=True, type=Path,
                    help="export directory containing annotation.json + masks/")
    ap.add_argument("--video", required=True, type=Path,
                    help="source video file matching annotation.json")
    ap.add_argument("--out", required=True, type=Path,
                    help="output export directory (created, must not exist)")
    ap.add_argument("--ids", nargs="+", type=int, default=None,
                    help="object ids to process (default: all in annotation)")
    ap.add_argument("--pad-box", type=float, default=0.5,
                    help="box prompt padding as a fraction of the object side")
    ap.add_argument("--prob-thresh", type=float, default=0.5,
                    help="SAM3 output probability threshold")
    ap.add_argument("--gpu", default="0", help="CUDA device id")
    args = ap.parse_args()

    meta = json.loads((args.export / "annotation.json").read_text())
    n = meta["num_frames"]
    w, h = meta["width"], meta["height"]
    ids = args.ids or [int(o["id"]) for o in meta["objects"]]

    masks = {
        i: cv2.imread(str(args.export / "masks" / f"{i:05d}.png"),
                      cv2.IMREAD_GRAYSCALE)
        for i in range(n)
    }
    if any(m is None for m in masks.values()):
        raise RuntimeError("missing mask PNGs in export")

    # 1) per-object fill targets: empty runs inside the annotated span
    targets = {}  # oid -> {"prompt_frame": int, "frames": [int]}
    for oid in ids:
        have = [i for i in range(n) if object_mask(masks, oid, i).any()]
        if len(have) < 2:
            continue
        lo, hi = min(have), max(have)
        have_set = set(have)
        gaps = [i for a, b in empty_runs(have_set, lo, hi)
                for i in range(a, b + 1)]
        if not gaps:
            continue
        # prompt on the annotated frame immediately before the first gap
        prompt_frame = max(i for i in have if i < gaps[0])
        targets[oid] = {"prompt_frame": prompt_frame, "frames": gaps}
    if not targets:
        print("no mid-sequence gaps found; nothing to fill")
        return
    for oid, t in targets.items():
        print(f"object {oid}: fill {len(t['frames'])} frames "
              f"{t['frames'][0]}..{t['frames'][-1]}, prompt at f{t['prompt_frame']}")

    # 2) SAM3 tracking
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    from sam3.model_builder import build_sam3_predictor

    predictor = build_sam3_predictor(
        version="sam3",
        checkpoint_path=os.environ.get(
            "SAM3_CHECKPOINT_PATH", "checkpoints/sam3/sam3.pt"),
        apply_temporal_disambiguation=False,
        async_loading_frames=False,
    )
    sid = f"fillgap_{args.out.name}"
    predictor.handle_request({
        "type": "start_session",
        "resource_path": str(args.video),
        "session_id": sid,
        "offload_video_to_cpu": True,
    })
    for oid, t in targets.items():
        box = prompt_box(
            object_mask(masks, oid, t["prompt_frame"]), args.pad_box, w, h
        )
        predictor.handle_request({
            "type": "add_prompt",
            "session_id": sid,
            "frame_index": t["prompt_frame"],
            "bounding_boxes": [box],
            "bounding_box_labels": [oid],
            "obj_id": oid,
            "rel_coordinates": True,
        })
        print(f"object {oid}: box prompt {['%.3f' % v for v in box]} "
              f"at f{t['prompt_frame']}")
    want = {oid for oid in targets}
    propagated = {}  # (oid, frame_idx) -> uint8 mask
    for out in predictor.handle_stream_request({
        "type": "propagate_in_video",
        "session_id": sid,
        "propagation_direction": "both",
        "output_prob_thresh": args.prob_thresh,
    }):
        fi = out["frame_index"]
        outputs = out.get("outputs") or {}
        oids = outputs.get("out_obj_ids", None)
        bmasks = outputs.get("out_binary_masks", None)
        if oids is None or bmasks is None or len(oids) == 0 or len(bmasks) == 0:
            continue
        for oid, bm in zip(oids, bmasks):
            oid = int(oid)
            if oid not in want:
                continue
            m = np.asarray(bm)
            if m.ndim == 3:
                m = m[0]
            if m.shape != (h, w):
                m = cv2.resize(
                    m.astype(np.float32), (w, h),
                    interpolation=cv2.INTER_NEAREST,
                )
            propagated[(oid, fi)] = (m > 0.5).astype(np.uint8)
    predictor.handle_request({"type": "close_session", "session_id": sid})

    # 3) write the filled export
    args.out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.export / "annotation.json", args.out / "annotation.json")
    meta_out = json.loads((args.out / "annotation.json").read_text())
    frames_by_idx = {f["frame_idx"]: f for f in meta_out["frames"]}
    (args.out / "masks").mkdir()
    n_filled = 0
    for i in range(n):
        src = masks[i].copy()
        changed = False
        for oid, t in targets.items():
            if i not in t["frames"]:
                continue
            m = propagated.get((oid, i))
            if m is None or not m.any():
                print(f"WARNING: object {oid} f{i}: SAM3 produced no mask; "
                      f"left empty")
                continue
            src[m > 0] = oid
            changed = True
            n_filled += 1
            ys, xs = np.where(src == oid)
            fo = frames_by_idx.setdefault(
                i, {"frame_idx": i, "objects": []})
            fo["objects"] = [o for o in fo.get("objects", [])
                             if int(o.get("id", -1)) != oid]
            fo["objects"].append({
                "id": oid,
                "bbox_xyxy": [int(xs.min()), int(ys.min()),
                              int(xs.max()), int(ys.max())],
            })
        cv2.imwrite(str(args.out / "masks" / f"{i:05d}.png"), src)
    meta_out["frames"] = [frames_by_idx[k] for k in sorted(frames_by_idx)]
    (args.out / "annotation.json").write_text(
        json.dumps(meta_out, ensure_ascii=False))
    print(f"filled {n_filled} mask frames -> {args.out}")

    # 4) area continuity report over each object's full annotated span
    areas = np.array([
        int((cv2.imread(str(args.out / "masks" / f"{i:05d}.png"),
                        cv2.IMREAD_GRAYSCALE) > 0).sum())
        for i in range(n)
    ])
    filled_set = {i for t in targets.values() for i in t["frames"]}
    jumps = []
    for oid, t in targets.items():
        have = [i for i in range(n)
                if (cv2.imread(str(args.export / "masks" / f"{i:05d}.png"),
                               cv2.IMREAD_GRAYSCALE) == oid).any()]
        for i in range(min(have), max(have)):
            a0, a1 = areas[i], areas[i + 1]
            if a0 > 0 and a1 > 0:
                jumps.append((abs(np.log(a1 / a0)), i, a0, a1))
    jumps.sort(reverse=True)
    print("top consecutive-area jumps inside span:")
    for lg, i, a0, a1 in jumps[:5]:
        mark = " (filled)" if i in filled_set or (i + 1) in filled_set else ""
        print(f"  f{i}->f{i+1}: {a0} -> {a1} (x{np.exp(lg):.2f}){mark}")


if __name__ == "__main__":
    main()
