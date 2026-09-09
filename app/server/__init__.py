# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Backend package init.

Runs the GPU auto-selection BEFORE any module below imports torch or
the `sam3` package: `sam3/__init__.py` pulls in `model_builder`, whose
import chain initializes CUDA. Once a CUDA context exists it is pinned
to a device and later `CUDA_VISIBLE_DEVICES` writes are ignored — so
the choice has to happen here, in the first code executed when
`app.server.app` is imported.
"""

from .gpu_utils import configure_gpu

# Physical index of the GPU this process runs on (None if unknown).
SELECTED_GPU, GPU_SNAPSHOT = configure_gpu()
