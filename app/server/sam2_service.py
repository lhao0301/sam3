# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""SAM2-task-mode service layer.

Wraps ``Sam3TrackerPredictor`` — the SAM2-style tracker inside the SAM3
checkpoint (weights come from the same sam3.pt; no SAM2 checkpoint needed) —
with async-friendly session / prompt / propagation APIs.

Key differences from the v1 detection-path service:

- Prompts are *incremental*: every submission extends the tracker state
  (no reset, no replay). ``obj_id`` is always user-specified.
- Points and boxes may be combined in one submission (box first), and a
  pure point / pure box submission both work to start a new object.
- New objects must be registered before propagation starts; afterwards
  only refinement of existing objects is allowed until the tracking is
  reset (``reset_tracking`` clears results and replays the prompt log).
- "both" propagation = forward pass (earliest prompted frame -> end)
  followed by a backward pass (latest prompted frame -> frame 0) so
  objects prompted on late frames get backtracked to the video start.
"""

import asyncio
from contextlib import contextmanager
from typing import AsyncIterator, Dict, List, Optional

import numpy as np
import torch

from sam3.logger import get_logger
from sam3.model_builder import build_sam3_video_model

logger = get_logger(__name__)


@contextmanager
def _bf16():
    """Tracker inference context (bf16 autocast).

    Sam3TrackerPredictor's __init__ globally enters a bf16 autocast on its
    loading thread, but autocast is thread-local: calls dispatched via
    asyncio.to_thread run on worker threads without it and hit dtype
    mismatches against the bf16 buffers created at load time. Wrap every
    tracker call explicitly (nesting is harmless).
    """
    if torch.cuda.is_available():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            yield
    else:
        yield


def _masks_to_boxes_xywh(masks: np.ndarray, height: int, width: int) -> List[List[float]]:
    """Tight bounding boxes (normalized xywh) around binary masks."""
    boxes = []
    for m in masks:
        ys, xs = np.nonzero(m)
        if len(ys) == 0:
            boxes.append([0.0, 0.0, 0.0, 0.0])
            continue
        x0 = float(xs.min()) / width
        x1 = float(xs.max() + 1) / width
        y0 = float(ys.min()) / height
        y1 = float(ys.max() + 1) / height
        boxes.append([x0, y0, x1 - x0, y1 - y0])
    return boxes


class Sam2TrackerService:
    """Singleton wrapper around the SAM2-task tracker (SAM3 weights).

    All sessions share the same model. An asyncio.Semaphore limits
    concurrent GPU inference to prevent OOM.
    """

    _instance: Optional["Sam2TrackerService"] = None
    _tracker = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        max_concurrent_inference: int = 1,
        checkpoint_path: Optional[str] = None,
    ):
        if self._tracker is None:
            logger.info("Loading SAM3 model for SAM2-task tracker...")
            # The tracker is a submodule of the SAM3 video model; we reuse
            # its trained weights and borrow the detector's ViT backbone
            # (the tracker alone has no vision trunk).
            video_model = build_sam3_video_model(
                checkpoint_path=checkpoint_path,
                apply_temporal_disambiguation=False,
            )
            tracker = video_model.tracker
            tracker.backbone = video_model.detector.backbone
            tracker = tracker.cuda().eval()
            self._tracker = tracker
            # The detector part is not used for inference; drop the
            # reference so only the tracker (+backbone) stays on GPU.
            del video_model
            logger.info("SAM2-task tracker loaded successfully")
        self._semaphore = asyncio.Semaphore(max_concurrent_inference)
        # session_id -> tracker inference_state
        self._states: Dict[str, dict] = {}
        # session_id -> submitted prompt records (for undo / reset replay)
        self._prompt_log: Dict[str, List[dict]] = {}

    # ── Session lifecycle ──────────────────────────────────────────────

    async def start_session(self, session_id: str, frames_dir: str) -> Dict:
        """Initialize a tracker inference state on an extracted JPEG folder."""
        async with self._semaphore:
            state = await asyncio.to_thread(
                self._tracker.init_state,
                video_path=frames_dir,
                offload_video_to_cpu=False,
                async_loading_frames=False,
            )
        self._states[session_id] = state
        self._prompt_log[session_id] = []
        logger.info(
            f"Started SAM2-task session {session_id} "
            f"({state['num_frames']} frames, "
            f"{state['video_width']}x{state['video_height']})"
        )
        return {
            "session_id": session_id,
            "num_frames": state["num_frames"],
            "orig_height": state["video_height"],
            "orig_width": state["video_width"],
        }

    async def close_session(self, session_id: str) -> None:
        """Drop the inference state and prompt log, freeing GPU memory."""
        self._states.pop(session_id, None)
        self._prompt_log.pop(session_id, None)
        await asyncio.to_thread(torch.cuda.empty_cache)
        logger.info(f"Closed SAM2-task session {session_id}")

    def _get_state(self, session_id: str) -> dict:
        state = self._states.get(session_id)
        if state is None:
            raise KeyError(f"Session {session_id} not found in tracker service")
        return state

    def is_tracking_started(self, session_id: str) -> bool:
        """Whether propagation has started (new objects need reset first)."""
        state = self._states.get(session_id)
        return bool(state and state.get("tracking_has_started"))

    # ── Prompt submission ──────────────────────────────────────────────

    async def submit_prompt(
        self,
        session_id: str,
        frame_idx: int,
        obj_id: int,
        points: Optional[List[List[float]]] = None,
        point_labels: Optional[List[int]] = None,
        box: Optional[List[float]] = None,
    ) -> Dict:
        """Add a point / box / point+box prompt for ``obj_id`` on one frame.

        A new obj_id registers a new object; an existing obj_id refines
        that object's mask on this frame. Coordinates are normalized
        [0,1]; ``box`` is xywh (converted to the xyxy two-corner format
        the tracker expects).

        Returns the v1-style outputs dict (out_obj_ids / out_binary_masks
        / out_boxes_xywh) for the prompt frame, covering ALL objects.
        """
        state = self._get_state(session_id)

        # After propagation started, only refinement of existing objects
        # is allowed; brand-new ids require reset_tracking first.
        is_new_obj = obj_id not in state["obj_id_to_idx"]
        if state["tracking_has_started"] and is_new_obj:
            raise PermissionError(
                "跟踪已开始：不能新增目标，请先点击\"重置跟踪\"（会保留已提交的提示）"
                "后再添加新目标并重新传播"
            )

        def _submit():
            kwargs = {}
            if points is not None and len(points) > 0:
                kwargs["points"] = torch.tensor(points, dtype=torch.float32)
                kwargs["labels"] = torch.tensor(point_labels, dtype=torch.int32)
            if box is not None:
                # frontend xywh (normalized) -> tracker xyxy (normalized)
                x, y, w, h = box
                kwargs["box"] = torch.tensor(
                    [x, y, x + w, y + h], dtype=torch.float32
                )
            with _bf16():
                return self._tracker.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=frame_idx,
                    obj_id=obj_id,
                    clear_old_points=True,
                    rel_coordinates=True,
                    **kwargs,
                )

        async with self._semaphore:
            frame_idx, obj_ids, _, video_res_masks = await asyncio.to_thread(_submit)

        self._prompt_log[session_id].append(
            {
                "frame_idx": frame_idx,
                "obj_id": obj_id,
                "points": list(points) if points else None,
                "point_labels": list(point_labels) if point_labels else None,
                "box": list(box) if box else None,
            }
        )
        logger.info(
            f"Added prompt to session {session_id[:8]}: obj {obj_id} "
            f"on frame {frame_idx} ({len(points or [])} point(s)"
            f"{', box' if box is not None else ''})"
        )
        return self._format_outputs(state, frame_idx, obj_ids, video_res_masks)

    async def undo_prompt(self, session_id: str, obj_id: int, frame_idx: int) -> Dict:
        """Undo the latest submission of ``obj_id`` on ``frame_idx``.

        Implemented as a full rebuild (clear + replay of the remaining
        log): the tracker's clear_all_points_in_frame would wipe ALL
        objects' inputs when no conditioning frame exists yet (i.e.
        before the first propagation), losing the whole session state.
        """
        state = self._get_state(session_id)
        log = self._prompt_log[session_id]

        # Drop the latest record for this (obj_id, frame_idx)
        last_idx = None
        for i in range(len(log) - 1, -1, -1):
            if log[i]["obj_id"] == obj_id and log[i]["frame_idx"] == frame_idx:
                last_idx = i
                break
        if last_idx is None:
            masks = await self.get_frame_masks(session_id, frame_idx)
            return {"frame_index": frame_idx, **masks}
        log.pop(last_idx)

        def _undo():
            self._tracker.clear_all_points_in_video(inference_state=state)
            for rec in log:
                self._replay_record(state, rec)

        async with self._semaphore:
            await asyncio.to_thread(_undo)
        logger.info(
            f"Undid prompt for obj {obj_id} frame {frame_idx} "
            f"in session {session_id[:8]} ({len(log)} prompts replayed)"
        )
        masks = await self.get_frame_masks(session_id, frame_idx)
        return {"frame_index": frame_idx, **masks}

    async def remove_object(self, session_id: str, obj_id: int) -> Dict:
        """Remove a tracked object (tracker state + prompt records)."""
        state = self._get_state(session_id)
        log = self._prompt_log[session_id]
        # remember the object's last prompted frame for the response
        last_frame = None
        for rec in log:
            if rec["obj_id"] == obj_id:
                last_frame = rec["frame_idx"]
        self._prompt_log[session_id] = [r for r in log if r["obj_id"] != obj_id]
        log = self._prompt_log[session_id]

        def _remove():
            if obj_id not in state["obj_id_to_idx"]:
                return
            if state["tracking_has_started"]:
                # Propagated: the tracker's native remove slices this object
                # out of the packed tensors, preserving every other object's
                # tracking results (and its internal clear path is safe here
                # because cond_frame_outputs is non-empty after propagation).
                with _bf16():
                    self._tracker.remove_object(
                        inference_state=state, obj_id=obj_id
                    )
            else:
                # Not yet propagated: native remove could hit the
                # clear_all_points_in_frame edge case that wipes every
                # object's inputs; rebuild from the remaining log instead.
                self._tracker.clear_all_points_in_video(inference_state=state)
                for rec in log:
                    self._replay_record(state, rec)

        async with self._semaphore:
            await asyncio.to_thread(_remove)
        logger.info(f"Removed object {obj_id} from session {session_id[:8]}")
        if last_frame is None:
            last_frame = 0
        masks = await self.get_frame_masks(session_id, last_frame)
        return {"frame_index": last_frame, **masks}

    async def reset_tracking(self, session_id: str) -> Dict:
        """Clear all tracking results and replay the confirmed prompts.

        After this, new objects may be added again (before the next
        propagation). Existing prompts are preserved and replayed, so
        only the propagation itself has to be redone.
        """
        state = self._get_state(session_id)
        log = list(self._prompt_log[session_id])

        def _reset():
            self._tracker.clear_all_points_in_video(inference_state=state)
            for rec in log:
                self._replay_record(state, rec)

        async with self._semaphore:
            await asyncio.to_thread(_reset)
        logger.info(
            f"Reset tracking for session {session_id[:8]} "
            f"({len(log)} prompts replayed)"
        )
        return {"status": "reset", "num_prompts_replayed": len(log)}

    def _replay_record(self, state: dict, rec: dict) -> None:
        """Re-apply a single prompt record (sync, caller holds semaphore)."""
        kwargs = {}
        if rec["points"]:
            kwargs["points"] = torch.tensor(rec["points"], dtype=torch.float32)
            kwargs["labels"] = torch.tensor(rec["point_labels"], dtype=torch.int32)
        if rec["box"]:
            x, y, w, h = rec["box"]
            kwargs["box"] = torch.tensor(
                [x, y, x + w, y + h], dtype=torch.float32
            )
        with _bf16():
            self._tracker.add_new_points_or_box(
                inference_state=state,
                frame_idx=rec["frame_idx"],
                obj_id=rec["obj_id"],
                clear_old_points=True,
                rel_coordinates=True,
                **kwargs,
            )

    # ── Propagation ────────────────────────────────────────────────────

    async def propagate_in_video(
        self,
        session_id: str,
        direction: str = "both",
    ) -> AsyncIterator[Dict]:
        """Propagate tracking through the video, yielding results per frame.

        Each yielded dict carries a "segment" key ("forward" / "backward"
        / "backfill"). After the tracking segments, a backfill pass
        re-infers objects missing from conditioning-frame snapshots and
        yields refreshed frames.
        """
        if direction not in ("forward", "backward", "both"):
            raise ValueError(f"invalid propagation direction: {direction}")
        state = self._get_state(session_id)
        plan = self.get_propagation_plan(session_id, direction)
        logger.info(
            f"Propagation plan for session {session_id[:8]} ({direction}): "
            f"forward={plan.get('forward', 0)}, "
            f"backward={plan.get('backward', 0)}, "
            f"backfill={plan['backfill']}, "
            f"expected_total={plan['expected_total']}"
        )

        backfilled: Dict[int, List[int]] = {}

        async with self._semaphore:
            # One segment: (reverse, start_frame_idx or None)
            segments = [("forward", False)]
            if direction == "backward":
                segments = [("backward", True)]
            elif direction == "both":
                segments.append(("backward", True))

            for seg_name, reverse in segments:
                # backward starts from the latest prompted frame (after the
                # forward pass, cond_frame_outputs holds every prompted
                # frame); forward defaults to the earliest one (None).
                start = None
                if reverse:
                    cond_frames = state["output_dict"]["cond_frame_outputs"]
                    if not cond_frames:
                        continue
                    start = max(cond_frames)
                    if start <= 0:
                        continue

                logger.info(
                    f"Propagation segment '{seg_name}' starting for session "
                    f"{session_id[:8]} "
                    f"(start_frame={'auto' if start is None else start})"
                )
                gen = self._tracker.propagate_in_video(
                    inference_state=state,
                    start_frame_idx=start,
                    max_frame_num_to_track=None,
                    reverse=reverse,
                    tqdm_disable=True,
                    propagate_preflight=True,
                )
                with _bf16():
                    for (
                        frame_idx,
                        obj_ids,
                        _,
                        video_res_masks,
                        _obj_scores,
                    ) in gen:
                        # Yield control to the event loop between frames; each
                        # iteration does GPU work inside this generator.
                        await asyncio.sleep(0)
                        out = self._format_outputs(
                            state, frame_idx, obj_ids, video_res_masks,
                            include_boxes=False,
                        )
                        out["segment"] = seg_name
                        yield out

            # Backfill pass: the preflight snapshot of a conditioning frame
            # leaves objects prompted on *other* frames as empty masks, and
            # the propagation loop skips cond frames (it replays snapshots).
            # Re-infer those (frame, object) pairs now, with the full memory
            # chain from both segments available.
            backfilled = await asyncio.to_thread(
                self._backfill_cond_frames, session_id
            )
            for frame_idx in sorted(backfilled):
                await asyncio.sleep(0)
                masks = await self.get_frame_masks(session_id, frame_idx)
                yield {
                    "frame_index": frame_idx,
                    "outputs": masks,
                    "segment": "backfill",
                }

        logger.info(
            f"Propagation ({direction}) completed for session {session_id[:8]} "
            f"(planned {plan['expected_total']} items, "
            f"{len(backfilled)} cond frame(s) backfilled: "
            f"{backfilled or '{}'})"
        )

    @torch.inference_mode()
    def _backfill_cond_frames(self, session_id: str) -> Dict[int, List[int]]:
        """Re-infer objects missing from conditioning-frame snapshots.

        Called after propagation (sync; caller holds the semaphore).
        Objects prompted on other frames hold a NO_OBJ_SCORE placeholder
        in a cond frame's snapshot; the propagation loop never re-infers
        cond frames. For each missing (frame, object) pair we run a
        single-frame inference with the object's memory chain and splice
        the result into the batched cond output in place -- the per-object
        slices are views of the same tensors and update too.

        The memory anchors (maskmem_features) are deliberately left
        untouched: a later refinement still replays from the original
        prompts via reset_tracking, so the snapshot semantics stay stable.

        Returns {frame_idx: [obj_id, ...]} of successfully backfilled
        objects per frame.
        """
        state = self._get_state(session_id)
        tracker = self._tracker
        output_dict = state["output_dict"]
        obj_count = len(state["obj_ids"])
        result: Dict[int, List[int]] = {}

        for frame_idx in sorted(output_dict["cond_frame_outputs"].keys()):
            current_out = output_dict["cond_frame_outputs"][frame_idx]
            missing = []
            for obj_idx in range(obj_count):
                has_input = (
                    frame_idx in state["point_inputs_per_obj"].get(obj_idx, {})
                    or frame_idx in state["mask_inputs_per_obj"].get(obj_idx, {})
                )
                if not has_input:
                    missing.append(obj_idx)
            if not missing:
                continue

            filled_ids = []
            for obj_idx in missing:
                obj_id = state["obj_ids"][obj_idx]
                obj_output_dict = state["output_dict_per_obj"][obj_idx]
                prompt_frames = sorted(
                    state["point_inputs_per_obj"].get(obj_idx, {}).keys()
                )
                # Track in the direction that reaches this frame from the
                # object's own first prompt frame.
                reverse = bool(prompt_frames) and frame_idx < prompt_frames[0]
                try:
                    with _bf16():
                        new_out, _ = tracker._run_single_frame_inference(
                            inference_state=state,
                            output_dict=obj_output_dict,
                            frame_idx=frame_idx,
                            batch_size=1,
                            is_init_cond_frame=False,
                            point_inputs=None,
                            mask_inputs=None,
                            reverse=reverse,
                            run_mem_encoder=False,
                        )
                    # Splice into the batched snapshot in place; per-object
                    # slices share memory and update automatically.
                    current_out["pred_masks"][obj_idx : obj_idx + 1] = (
                        new_out["pred_masks"].to(
                            current_out["pred_masks"].device, non_blocking=True
                        )
                    )
                    current_out["obj_ptr"][obj_idx : obj_idx + 1] = new_out[
                        "obj_ptr"
                    ]
                    current_out["object_score_logits"][obj_idx : obj_idx + 1] = (
                        new_out["object_score_logits"]
                    )
                    if "iou_score" in current_out and "iou_score" in new_out:
                        current_out["iou_score"][obj_idx : obj_idx + 1] = (
                            new_out["iou_score"]
                        )
                    filled_ids.append(int(obj_id))
                except Exception as e:
                    logger.warning(
                        f"Backfill failed for obj {obj_id} on frame {frame_idx} "
                        f"in session {session_id[:8]}: "
                        f"{type(e).__name__}: {e}"
                    )
            if filled_ids:
                result[frame_idx] = filled_ids
                logger.info(
                    f"Backfilled cond frame {frame_idx} for objects {filled_ids} "
                    f"in session {session_id[:8]}"
                )
        return result

    def get_propagation_plan(self, session_id: str, direction: str = "both") -> Dict:
        """Frame budget per segment, for honest progress reporting.

        "both" propagation processes more frames than the video has
        (the forward and backward segments overlap), and a backfill
        pass re-infers cond-frame snapshots afterwards. The plan tells
        the frontend what "100%" actually means:

        - forward: earliest prompted frame -> last frame
        - backward: latest prompted frame -> frame 0
        - backfill: cond frames that are missing at least one object
          (each yields one refreshed message, frame-granular like the
          other segments)
        """
        state = self._get_state(session_id)
        log = self._prompt_log.get(session_id, [])
        prompt_frames = sorted({rec["frame_idx"] for rec in log})
        obj_ids = {rec["obj_id"] for rec in log}
        first_cond = prompt_frames[0] if prompt_frames else 0
        last_cond = prompt_frames[-1] if prompt_frames else 0

        plan: Dict[str, int] = {}
        if direction in ("forward", "both"):
            plan["forward"] = state["num_frames"] - first_cond
        if direction in ("backward", "both") and last_cond > 0:
            plan["backward"] = last_cond + 1
        pairs = {(rec["frame_idx"], rec["obj_id"]) for rec in log}
        plan["backfill"] = sum(
            1 for f in prompt_frames
            if any((f, o) not in pairs for o in obj_ids)
        )
        plan["expected_total"] = (
            plan.get("forward", 0) + plan.get("backward", 0) + plan["backfill"]
        )
        return plan

    # ── Frame mask retrieval ───────────────────────────────────────────

    async def get_frame_masks(self, session_id: str, frame_idx: int) -> Dict:
        """Read the tracked masks for one frame from the tracker state."""
        state = self._get_state(session_id)

        def _read():
            out, obj_ids = self._read_frame_output(state, frame_idx)
            if out is None:
                return {"out_obj_ids": [], "out_binary_masks": [], "out_boxes_xywh": []}
            with _bf16():
                _, video_res = self._tracker._get_orig_video_res_output(
                    state, out["pred_masks"]
                )
            masks = (video_res > 0.0).cpu().numpy()[:, 0]
            boxes = _masks_to_boxes_xywh(
                masks, state["video_height"], state["video_width"]
            )
            return {
                "out_obj_ids": obj_ids,
                "out_binary_masks": [m for m in masks],
                "out_boxes_xywh": boxes,
            }

        result = await asyncio.to_thread(_read)
        return result

    @staticmethod
    def _read_frame_output(state: dict, frame_idx: int):
        """Batched frame output plus the obj_ids its pred_masks align with.

        Consolidated outputs are checked first; prompts that were added
        but not yet propagated (e.g. right after reset_tracking) live in
        per-object temp outputs, which are merged into a batched view.
        """
        output_dict = state["output_dict"]
        for key in ("cond_frame_outputs", "non_cond_frame_outputs"):
            out = output_dict[key].get(frame_idx)
            if out is not None:
                return out, list(state["obj_ids"])

        per_obj = state.get("temp_output_dict_per_obj", {})
        obj_ids, masks = [], []
        for obj_idx in sorted(per_obj.keys()):
            obj_out = (
                per_obj[obj_idx].get("cond_frame_outputs", {}).get(frame_idx)
                or per_obj[obj_idx].get("non_cond_frame_outputs", {}).get(frame_idx)
            )
            if obj_out is not None and obj_out.get("pred_masks") is not None:
                obj_ids.append(state["obj_ids"][obj_idx])
                masks.append(obj_out["pred_masks"])
        if not masks:
            return None, []
        return {"pred_masks": torch.cat(masks, dim=0)}, obj_ids

    def _format_outputs(
        self, state: dict, frame_idx: int, obj_ids, video_res_masks,
        include_boxes: bool = True,
    ) -> Dict:
        """Convert tracker outputs into the v1-style outputs dict.

        include_boxes=False skips the tight-box computation (a full-frame
        np.nonzero scan per object) for the propagation hot path, where
        the frontend only consumes the rendered mask PNG.
        """
        obj_ids = [int(o) for o in obj_ids]
        masks = (video_res_masks > 0.0).cpu().numpy()[:, 0]
        boxes = (
            _masks_to_boxes_xywh(
                masks, state["video_height"], state["video_width"]
            )
            if include_boxes
            else []
        )
        return {
            "frame_index": int(frame_idx),
            "outputs": {
                "out_obj_ids": obj_ids,
                "out_binary_masks": [m for m in masks],
                "out_boxes_xywh": boxes,
            },
        }

    # ── Export ─────────────────────────────────────────────────────────

    async def export_video(
        self,
        session_id: str,
        frames_dir: str,
        num_frames: int,
        fps: float = 30.0,
    ) -> str:
        """Export the annotated video with mask overlays composited."""
        from pathlib import Path

        import cv2

        from .mask_utils import get_color_for_obj_id

        state = self._get_state(session_id)

        def _export() -> str:
            export_dir = Path("/tmp/sam3_exports")
            export_dir.mkdir(parents=True, exist_ok=True)
            output_path = export_dir / f"{session_id}_annotated.mp4"

            first_frame = cv2.imread(str(Path(frames_dir) / "00000.jpg"))
            if first_frame is None:
                raise RuntimeError("Cannot read first frame for export")
            h, w = first_frame.shape[:2]

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
            if not writer.isOpened():
                raise RuntimeError("Failed to open VideoWriter for export")

            written = 0
            for frame_idx in range(num_frames):
                frame = cv2.imread(str(Path(frames_dir) / f"{frame_idx:05d}.jpg"))
                if frame is None:
                    continue

                out, obj_ids = self._read_frame_output(state, frame_idx)
                if out is not None:
                    with _bf16():
                        _, video_res = self._tracker._get_orig_video_res_output(
                            state, out["pred_masks"]
                        )
                    masks = (video_res > 0.0).cpu().numpy()[:, 0]
                    mask_layer = np.zeros((h, w, 3), dtype=np.uint8)
                    for oid, m in zip(obj_ids, masks):
                        if m.shape != (h, w):
                            m = cv2.resize(
                                m.astype(np.uint8), (w, h),
                                interpolation=cv2.INTER_NEAREST,
                            ).astype(bool)
                        color = get_color_for_obj_id(int(oid))
                        mask_layer[m] = color
                    frame = cv2.addWeighted(frame, 1.0, mask_layer, 0.5, 0)
                    for oid, m in zip(obj_ids, masks):
                        if m.shape != (h, w):
                            m = cv2.resize(
                                m.astype(np.uint8), (w, h),
                                interpolation=cv2.INTER_NEAREST,
                            ).astype(bool)
                        color = get_color_for_obj_id(int(oid))
                        contours, _ = cv2.findContours(
                            m.astype(np.uint8),
                            cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE,
                        )
                        cv2.drawContours(frame, contours, -1, color, 2)

                writer.write(frame)
                written += 1
            writer.release()
            logger.info(f"Exported {written}/{num_frames} frames to {output_path}")
            return str(output_path)

        async with self._semaphore:
            output_path = await asyncio.to_thread(_export)
        return output_path
