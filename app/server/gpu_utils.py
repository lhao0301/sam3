# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""GPU auto-selection utilities.

Picks the GPU with the most free memory at service startup and pins it
via ``CUDA_VISIBLE_DEVICES`` so the model loads there — no manual
``CUDA_VISIBLE_DEVICES=<n>`` needed in the launch command.

Selection precedence:
  1. ``CUDA_VISIBLE_DEVICES`` already set by the launcher — respected as-is
  2. ``SAM3_GPU=<index or csv list>`` — pinned to exactly those GPUs
  3. ``SAM3_GPU=auto`` (default) — the GPU with the most free memory wins
     (lowest utilization breaks ties, then the lowest index)

Must run BEFORE the first CUDA context is created (i.e. before any
``.cuda()`` / ``torch.cuda.*`` call); ``import torch`` alone does not
initialize CUDA, so calling this from the app lifespan works.
"""

import logging
import os
import subprocess
from typing import List, Optional, Tuple

# Standard logging (NOT sam3.logger): importing sam3 triggers the
# model_builder import chain, which initializes CUDA — exactly what
# this module must run BEFORE. A dedicated stderr handler keeps these
# startup lines visible in the nohup log without touching root config.
logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("%(levelname)s %(asctime)s %(name)s: %(message)s")
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def query_gpus() -> List[dict]:
    """Query all physical GPUs via nvidia-smi.

    Returns a list of dicts {index, used_mb, total_mb, free_mb,
    util_pct}; empty when nvidia-smi is unavailable.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        )
    except Exception as e:
        logger.warning(f"nvidia-smi query failed: {e}")
        return []

    gpus: List[dict] = []
    for line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 4:
            continue
        try:
            idx, used, total, util = (
                int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
            )
        except ValueError:
            continue
        gpus.append(
            {
                "index": idx,
                "used_mb": used,
                "total_mb": total,
                "free_mb": total - used,
                "util_pct": util,
            }
        )
    return gpus


def select_free_gpu(gpus: List[dict]) -> Optional[dict]:
    """Pick the GPU with the most free memory.

    Ties are broken by lower utilization (a busy card's free memory may
    soon be claimed by its current workload), then by lower index.
    """
    if not gpus:
        return None
    return sorted(
        gpus, key=lambda g: (-g["free_mb"], g["util_pct"], g["index"])
    )[0]


def configure_gpu() -> Tuple[Optional[int], List[dict]]:
    """Choose the GPU to run on and pin it via CUDA_VISIBLE_DEVICES.

    Returns (selected physical gpu index or None, gpu snapshot list).
    The env var is set before returning, so any CUDA context created
    afterwards lands on the chosen card.
    """
    gpus = query_gpus()

    # 1. Launcher already pinned the devices — respect it verbatim.
    pinned = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if pinned:
        first = pinned.split(",")[0].strip()
        logger.info(f"Using GPUs pinned by launcher: CUDA_VISIBLE_DEVICES={pinned!r}")
        try:
            return int(first), gpus
        except ValueError:
            return None, gpus

    # 2. Explicit SAM3_GPU pin ("3" or "0,1").
    choice = os.environ.get("SAM3_GPU", "auto").strip()
    if choice and choice.lower() != "auto":
        os.environ["CUDA_VISIBLE_DEVICES"] = choice
        first = choice.split(",")[0].strip()
        logger.info(f"Using GPU(s) from SAM3_GPU={choice!r}")
        try:
            return int(first), gpus
        except ValueError:
            return None, gpus

    # 3. Auto: most free memory wins.
    best = select_free_gpu(gpus)
    if best is None:
        logger.warning(
            "No GPUs detected via nvidia-smi; leaving device selection to CUDA"
        )
        return None, gpus

    os.environ["CUDA_VISIBLE_DEVICES"] = str(best["index"])
    summary = ", ".join(
        f"GPU{g['index']}: {g['free_mb']:.0f}MB free / {g['util_pct']:.0f}% util"
        for g in gpus
    )
    logger.info(f"GPU snapshot: {summary}")
    logger.info(
        f"Auto-selected GPU {best['index']} "
        f"(free {best['free_mb']:.0f} MB of {best['total_mb']:.0f} MB, "
        f"util {best['util_pct']:.0f}%)"
    )
    return best["index"], gpus
