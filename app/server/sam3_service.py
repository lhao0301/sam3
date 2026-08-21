# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""SAM3 service layer.

Wraps the SAM3 predictor as a singleton, providing async-friendly methods
for session lifecycle (start, add_prompt, propagate, close) with
GPU concurrency control via asyncio.Semaphore.

All sessions share the same model weights. Concurrency isolation is achieved
through session_id and the semaphore controlling simultaneous GPU inference.
"""

import asyncio
from typing import AsyncIterator, Dict, List, Optional

from sam3.logger import get_logger
from sam3.model_builder import build_sam3_predictor

logger = get_logger(__name__)


class Sam3Service:
    """Singleton wrapper around the SAM3 predictor.

    The model is loaded once at startup and shared across all sessions.
    An asyncio.Semaphore limits concurrent GPU inference to prevent OOM.
    """

    _instance: Optional["Sam3Service"] = None
    _predictor = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        model_version: str = "sam3",
        max_concurrent_inference: int = 1,
        checkpoint_path: Optional[str] = None,
    ):
        if self._predictor is None:
            logger.info(f"Loading SAM3 model (version={model_version})...")
            self._predictor = build_sam3_predictor(
                version=model_version,
                checkpoint_path=checkpoint_path,
                apply_temporal_disambiguation=False,
                # Load frames synchronously: start_session must not return
                # before the GPU has every frame, otherwise prompt requests
                # racing the background loader fail with opaque errors.
                async_loading_frames=False,
            )
            logger.info("SAM3 model loaded successfully")
        self._semaphore = asyncio.Semaphore(max_concurrent_inference)
        self._model_version = model_version

    @property
    def predictor(self):
        return self._predictor

    async def start_session(
        self,
        session_id: str,
        resource_path: str,
        resource_type: str = "image_folder",
    ) -> Dict:
        """Start a SAM3 inference session.

        Args:
            session_id: Unique session identifier.
            resource_path: Path to video file or JPEG image folder.
            resource_type: "video_file" or "image_folder".

        Returns:
            Dict with session metadata (num_frames, image_size, etc.)
        """
        request = {
            "type": "start_session",
            "resource_path": resource_path,
            "resource_type": resource_type,
            "session_id": session_id,
        }
        async with self._semaphore:
            result = await asyncio.to_thread(
                self._predictor.handle_request, request
            )
        logger.info(f"Started SAM3 session for {session_id}")
        return result

    async def add_prompt(
        self,
        session_id: str,
        frame_idx: int,
        text: Optional[str] = None,
        points: Optional[List[List[float]]] = None,
        point_labels: Optional[List[int]] = None,
        bounding_boxes: Optional[List[List[float]]] = None,
        bounding_box_labels: Optional[List[int]] = None,
        obj_id: Optional[int] = None,
        rel_coordinates: bool = True,
    ) -> Dict:
        """Add a text / point / box prompt to a session at a given frame.

        Args:
            session_id: The SAM3 session ID.
            frame_idx: The frame index to add the prompt to.
            text: Text prompt (open-vocabulary detection, may produce
                  multiple tracklets with auto-assigned ids).
            points: List of [x, y] points, normalized to [0,1] if
                    rel_coordinates=True.
            point_labels: Point labels, 1 = positive, 0 = negative.
            bounding_boxes: List of [x, y, w, h] boxes, normalized to [0,1]
                            if rel_coordinates=True.
            bounding_box_labels: Tracklet id per box (SAM3 treats box labels
                                 as object ids).
            obj_id: User-specified tracklet id (used for point prompts;
                    ignored for text which auto-assigns ids).
            rel_coordinates: Whether points/boxes are normalized.

        Returns:
            Dict with prompt response including masks on the prompt frame.
        """
        request = {
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": frame_idx,
            "text": text,
            "points": points,
            "point_labels": point_labels,
            "bounding_boxes": bounding_boxes,
            "bounding_box_labels": bounding_box_labels,
            "obj_id": obj_id,
            "rel_coordinates": rel_coordinates,
        }
        async with self._semaphore:
            result = await asyncio.to_thread(
                self._predictor.handle_request, request
            )
        prompt_desc = (
            f"text={text!r}" if text
            else f"{len(bounding_boxes or [])} box(es)"
            if bounding_boxes
            else f"{len(points or [])} point(s)"
        )
        logger.info(
            f"Added prompt to session {session_id} at frame {frame_idx}: "
            f"{prompt_desc}"
        )
        return result

    async def remove_object(
        self,
        session_id: str,
        obj_id: int,
        frame_idx: int = 0,
    ) -> Dict:
        """Remove a tracklet from tracking by its object id.

        Args:
            session_id: The SAM3 session ID.
            obj_id: The tracklet id to remove.
            frame_idx: The frame the prompt was added on.

        Returns:
            Dict with masks of the remaining objects on that frame.
        """
        request = {
            "type": "remove_object",
            "session_id": session_id,
            "frame_index": frame_idx,
            "obj_id": obj_id,
        }
        async with self._semaphore:
            result = await asyncio.to_thread(
                self._predictor.handle_request, request
            )
        logger.info(f"Removed object {obj_id} from session {session_id}")
        return result

    # ── Prompt lifecycle with replay ─────────────────────────────────────
    #
    # SAM3's detection-path prompts (text / box) reset all tracker state on
    # every add_prompt call, while point prompts are incremental with a
    # user-specified obj_id. The methods below replay the full confirmed
    # prompt set around every submission so confirmed targets survive.

    async def submit_prompt(self, session_id: str, info, prompt: Dict) -> Dict:
        """Preview a prompt (== commit to the model), replaying confirmed
        prompts so previously confirmed targets survive the reset.

        Args:
            prompt: A dict from PromptHistory's documented formats.

        Returns:
            The outputs dict on the prompt frame (all tracked objects).
        """
        history = info.prompt_history
        history.set_pending(prompt)
        frame_idx = prompt["frame_idx"]

        last_outputs = {}
        det_payload = history.detection_replay_payload()
        if det_payload is not None:
            # Detection path: resets state; the payload already includes the
            # pending text/box plus all confirmed detection prompts.
            # The detection anchors on ITS OWN prompt frame: a point
            # submission must NOT move the anchored frame (the box would
            # otherwise re-detect on the point's frame and find a
            # different set of objects).
            if prompt["type"] in ("text", "box"):
                history.det_frame = frame_idx
            result = await self.add_prompt(
                session_id=session_id,
                frame_idx=history.det_frame,
                **det_payload,
            )
            last_outputs = result.get("outputs", {})
            raw_ids = last_outputs.get("out_obj_ids")
            out_ids = (
                [int(o) for o in raw_ids] if raw_ids is not None else []
            )
            history.record_detection_ids(out_ids)

            # The reset wiped confirmed point objects: restore them through
            # the incremental tracker path (the pending point, if any, is
            # added below as a fresh object).
            for pp in history.points:
                result = await self.add_prompt(
                    session_id=session_id,
                    frame_idx=pp["frame_idx"],
                    points=pp["points"],
                    point_labels=pp["point_labels"],
                    obj_id=pp["obj_id"],
                )
                last_outputs = result.get("outputs", {})

        if prompt["type"] == "point":
            # Pure point submission (no reset above when there are no
            # confirmed detection prompts), or the restore-after-reset case.
            result = await self.add_prompt(
                session_id=session_id,
                frame_idx=frame_idx,
                points=prompt["points"],
                point_labels=prompt["point_labels"],
                obj_id=prompt["obj_id"],
            )
            last_outputs = result.get("outputs", {})

        return last_outputs

    async def confirm_pending(self, session_id: str, info) -> Dict:
        """Move the previewed prompt into the confirmed set (no model call)."""
        history = info.prompt_history
        history.confirm_pending()
        return {
            "confirmed_obj_ids": sorted(history.confirmed_obj_ids),
            "pending": None,
        }

    async def cancel_pending(self, session_id: str, info) -> Dict:
        """Discard the previewed prompt and restore the confirmed state."""
        history = info.prompt_history
        history.clear_pending()
        return await self._replay_confirmed(session_id, history)

    async def remove_tracklet(self, session_id: str, info, obj_id: int) -> Dict:
        """Remove a confirmed tracklet and restore the remaining state."""
        history = info.prompt_history
        removed = history.remove_by_obj_id(obj_id)
        if not removed:
            logger.warning(
                f"remove_tracklet: obj {obj_id} not found in history "
                f"for session {session_id}; replaying anyway"
            )
        outputs = await self._replay_confirmed(session_id, history)
        logger.info(
            f"Removed tracklet {obj_id} from session {session_id} "
            f"(removed_from_history={removed})"
        )
        return outputs

    async def _replay_confirmed(self, session_id: str, history) -> Dict:
        """Rebuild the model state from the confirmed prompt set only."""
        last_outputs = {}
        det_payload = history.detection_replay_payload()
        if det_payload is None:
            # No confirmed detection prompts left: explicitly clear stale
            # detection state (reset_session preserves precomputed ViT feats)
            request = {"type": "reset_session", "session_id": session_id}
            async with self._semaphore:
                await asyncio.to_thread(
                    self._predictor.handle_request, request
                )
        else:
            result = await self.add_prompt(
                session_id=session_id,
                frame_idx=history.det_frame,
                **det_payload,
            )
            last_outputs = result.get("outputs", {})
        for pp in history.points:
            result = await self.add_prompt(
                session_id=session_id,
                frame_idx=pp["frame_idx"],
                points=pp["points"],
                point_labels=pp["point_labels"],
                obj_id=pp["obj_id"],
            )
            last_outputs = result.get("outputs", {})
        return last_outputs

    async def propagate_in_video(
        self,
        session_id: str,
        direction: str = "both",
    ) -> AsyncIterator[Dict]:
        """Propagate tracking through the video, yielding results per frame.

        This is an async generator that wraps the synchronous propagate_in_video
        generator from SAM3. It acquires the semaphore for the entire propagation
        to ensure GPU exclusivity during the tracking pass.

        Args:
            session_id: The SAM3 session ID.
            direction: "forward" (prompt frame -> end), "backward"
                       (prompt frame -> start) or "both".

        Yields:
            Dict per frame: {"frame_index": int, "outputs": {...}}
        """
        if direction not in ("forward", "backward", "both"):
            raise ValueError(f"invalid propagation direction: {direction}")
        request = {
            "type": "propagate_in_video",
            "session_id": session_id,
            "propagation_direction": direction,
        }
        async with self._semaphore:
            # Get the generator from SAM3 (runs in thread to avoid blocking)
            gen = await asyncio.to_thread(
                self._predictor.handle_stream_request, request
            )
            if gen is None:
                logger.error(f"propagate_in_video returned None for {session_id}")
                return

            for frame_result in gen:
                # Each call to next(gen) does GPU work; yield control to event loop
                await asyncio.sleep(0)
                yield frame_result

        logger.info(f"Propagation completed for session {session_id}")

    async def encode_features(
        self,
        session_id: str,
        batch_size: int = 4,
    ) -> Dict:
        """Pre-compute ViT backbone features for all frames in a session.

        Runs the ViT trunk on all video frames and caches the raw outputs so
        that subsequent ``add_prompt`` / ``propagate_in_video`` calls skip the
        expensive ViT forward and only run the lightweight FPN neck.

        Args:
            session_id: The SAM3 session ID.
            batch_size: Number of frames per ViT forward pass.

        Returns:
            Dict with ``num_frames_encoded``.
        """
        request = {
            "type": "encode_features",
            "session_id": session_id,
            "batch_size": batch_size,
        }
        async with self._semaphore:
            result = await asyncio.to_thread(
                self._predictor.handle_request, request
            )
        logger.info(f"Encoded features for session {session_id}: {result}")
        return result

    async def get_frame_masks(self, session_id: str, frame_idx: int) -> Dict:
        """Retrieve cached masks for a specific frame after propagation.

        Reads from inference_state["cached_frame_outputs"][frame_idx] which
        contains {obj_id: boolean_mask_tensor} for each tracked object.

        Returns:
            Dict with obj_ids (list[int]) and binary_masks (list[np.ndarray])
            or empty if no masks cached for this frame.
        """
        request = {
            "type": "get_frame_masks",
            "session_id": session_id,
            "frame_index": frame_idx,
        }
        result = await asyncio.to_thread(
            self._predictor.handle_request, request
        )
        return result

    async def close_session(self, session_id: str) -> None:
        """Close a SAM3 session and free its resources."""
        request = {
            "type": "close_session",
            "session_id": session_id,
        }
        try:
            await asyncio.to_thread(
                self._predictor.handle_request, request
            )
            logger.info(f"Closed SAM3 session {session_id}")
        except Exception as e:
            logger.warning(f"Error closing session {session_id}: {e}")

    async def export_video(
        self,
        session_id: str,
        frames_dir: str,
        num_frames: int,
        fps: float = 30.0,
    ) -> str:
        """Export an annotated video with mask overlays composited on each frame.

        Iterates through all frames, loads the original JPEG, overlays the
        cached masks (with contours), and writes the result to an MP4 file.

        Args:
            session_id: The SAM3 session ID.
            frames_dir: Path to the directory of JPEG frames.
            num_frames: Total number of frames.
            fps: Output video frame rate.

        Returns:
            Path to the output MP4 file.
        """
        from pathlib import Path
        import cv2
        import numpy as np
        from .mask_utils import get_color_for_obj_id

        def _export() -> str:
            export_dir = Path("/tmp/sam3_exports")
            export_dir.mkdir(parents=True, exist_ok=True)
            output_path = export_dir / f"{session_id}_annotated.mp4"

            # Read first frame to get resolution
            first_frame_path = Path(frames_dir) / "00000.jpg"
            first_frame = cv2.imread(str(first_frame_path))
            if first_frame is None:
                raise RuntimeError(f"Cannot read first frame: {first_frame_path}")
            h, w = first_frame.shape[:2]

            # Access cached frame outputs directly from inference state
            state = self._predictor._all_inference_states[session_id]["state"]
            cached = state.get("cached_frame_outputs", {})

            # Create video writer (mp4v codec for broad compatibility)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
            if not writer.isOpened():
                raise RuntimeError("Failed to open VideoWriter for export")

            frames_written = 0
            for frame_idx in range(num_frames):
                frame_path = Path(frames_dir) / f"{frame_idx:05d}.jpg"
                frame = cv2.imread(str(frame_path))
                if frame is None:
                    logger.warning(f"Cannot read frame {frame_idx}, skipping")
                    continue

                # Get cached masks for this frame
                frame_masks = cached.get(frame_idx, {})
                if frame_masks:
                    # Build mask overlay
                    mask_layer = np.zeros((h, w, 3), dtype=np.uint8)
                    for obj_id, mask in frame_masks.items():
                        mask_np = mask
                        if hasattr(mask_np, "cpu"):
                            mask_np = mask_np.cpu().numpy()
                        if mask_np.shape != (h, w):
                            mask_np = cv2.resize(
                                mask_np.astype(np.uint8), (w, h),
                                interpolation=cv2.INTER_NEAREST,
                            ).astype(bool)
                        color = get_color_for_obj_id(int(obj_id))
                        mask_layer[mask_np] = color

                    # Blend overlay with alpha=0.5
                    frame = cv2.addWeighted(frame, 1.0, mask_layer, 0.5, 0)

                    # Draw contours for visibility
                    for obj_id, mask in frame_masks.items():
                        mask_np = mask
                        if hasattr(mask_np, "cpu"):
                            mask_np = mask_np.cpu().numpy()
                        if mask_np.shape != (h, w):
                            mask_np = cv2.resize(
                                mask_np.astype(np.uint8), (w, h),
                                interpolation=cv2.INTER_NEAREST,
                            ).astype(bool)
                        color = get_color_for_obj_id(int(obj_id))
                        contours, _ = cv2.findContours(
                            mask_np.astype(np.uint8),
                            cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE,
                        )
                        cv2.drawContours(frame, contours, -1, color, 2)

                writer.write(frame)
                frames_written += 1

            writer.release()
            logger.info(
                f"Exported {frames_written}/{num_frames} frames to {output_path} "
                f"({fps:.2f} fps, {w}x{h})"
            )
            return str(output_path)

        async with self._semaphore:
            output_path = await asyncio.to_thread(_export)
        logger.info(f"Video export completed for session {session_id}: {output_path}")
        return output_path
