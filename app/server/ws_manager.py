# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""WebSocket connection manager.

Manages the lifecycle of WebSocket connections during propagation.
Each propagation request gets its own async task that can be cancelled
if the client disconnects or sends a cancel message.
"""

import asyncio
import base64
import json
from typing import Dict, Optional

import cv2
import numpy as np
from fastapi import WebSocket

from sam3.logger import get_logger

from .mask_utils import render_masks_only, get_color_for_obj_id

logger = get_logger(__name__)


class WSManager:
    """Manages WebSocket connections and propagation tasks."""

    def __init__(self):
        # Map session_id -> active propagation task
        self._propagation_tasks: Dict[str, asyncio.Task] = {}
        # Map session_id -> WebSocket connection
        self._connections: Dict[str, WebSocket] = {}
        # Map session_id -> frames streamed so far (for cancel reporting)
        self._propagation_counts: Dict[str, int] = {}

    async def handle_connection(
        self,
        websocket: WebSocket,
        session_id: str,
        sam3_service,
        session_manager,
    ):
        """Handle a WebSocket connection for video propagation.

        Listens for messages from the client and manages the propagation
        lifecycle. When the client sends "start", it begins propagation
        and streams results frame by frame. "cancel" stops propagation.
        """
        await websocket.accept()
        self._connections[session_id] = websocket

        info = session_manager.get_session(session_id)
        if info is None:
            await websocket.send_json({
                "type": "error",
                "message": f"Session {session_id} not found",
            })
            await websocket.close()
            return

        info.is_propagating = False
        logger.info(f"WebSocket connected for session {session_id}")

        try:
            while True:
                # Wait for client messages
                try:
                    msg = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
                except asyncio.TimeoutError:
                    # This 1s loop doubles as a heartbeat: get_session
                    # refreshes last_active_at so an open connection (long
                    # propagation) is never expired by the cleanup task.
                    info = session_manager.get_session(session_id)
                    if info is None:
                        # Session expired / removed (client gone): stop
                        # streaming and drop the connection
                        task = self._propagation_tasks.get(session_id)
                        if task is not None and not task.done():
                            task.cancel()
                            try:
                                await asyncio.wait_for(
                                    asyncio.shield(task), timeout=5.0
                                )
                            except (asyncio.CancelledError, asyncio.TimeoutError):
                                pass
                        break
                    task = self._propagation_tasks.get(session_id)
                    if task is not None and task.done():
                        break
                    continue

                data = json.loads(msg)
                msg_type = data.get("type")

                if msg_type == "start":
                    if session_id in self._propagation_tasks:
                        await websocket.send_json({
                            "type": "error",
                            "message": "Propagation already in progress",
                        })
                        continue

                    direction = data.get("direction", "both")
                    if direction not in ("forward", "backward", "both"):
                        await websocket.send_json({
                            "type": "error",
                            "message": f"invalid propagation direction: {direction}",
                        })
                        continue

                    task = asyncio.create_task(
                        self._run_propagation(
                            websocket, session_id, sam3_service, session_manager,
                            direction=direction,
                        )
                    )
                    self._propagation_tasks[session_id] = task

                elif msg_type == "cancel":
                    task = self._propagation_tasks.get(session_id)
                    if task is not None and not task.done():
                        task.cancel()
                        try:
                            await asyncio.wait_for(
                                asyncio.shield(task), timeout=5.0
                            )
                        except (asyncio.CancelledError, asyncio.TimeoutError):
                            pass
                        # Send from this (non-cancelled) context: awaiting
                        # inside the cancelled task would re-raise immediately
                        await websocket.send_json({
                            "type": "cancelled",
                            "frames_processed": self._propagation_counts.get(
                                session_id, 0
                            ),
                        })
                    self._propagation_tasks.pop(session_id, None)
                    self._propagation_counts.pop(session_id, None)

                elif msg_type == "ping":
                    await websocket.send_json({"type": "pong"})

        except Exception as e:
            logger.error(f"WebSocket error for {session_id}: {e}")
        finally:
            # Cleanup
            task = self._propagation_tasks.pop(session_id, None)
            self._propagation_counts.pop(session_id, None)
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            self._connections.pop(session_id, None)
            info = session_manager.get_session(session_id)
            if info is not None:
                info.is_propagating = False
            logger.info(f"WebSocket disconnected for session {session_id}")

    async def _run_propagation(
        self,
        websocket: WebSocket,
        session_id: str,
        sam3_service,
        session_manager,
        direction: str = "both",
    ):
        """Run the SAM3 propagation and stream results via WebSocket.

        direction: "forward" (prompt frame -> end), "backward"
        (prompt frame -> start) or "both".

        Progress is reported against the propagation plan (forward +
        backward + backfill budget) rather than the video's frame count,
        so "both" propagation no longer shows a fake 100% while the
        backward segment is still running.

        For each frame, sends a JSON message with:
        - frame_index: int
        - segment: "forward" | "backward" | "backfill"
        - out_obj_ids: list of int
        - mask_png: base64-encoded transparent PNG overlay
        - progress: float (0..1, relative to the plan)
        """
        info = session_manager.get_session(session_id)
        if info is None:
            await websocket.send_json({
                "type": "error",
                "message": "Session not found",
            })
            return

        info.is_propagating = True
        try:
            plan = sam3_service.get_propagation_plan(session_id, direction)
        except Exception:
            plan = {}
        expected_total = plan.get("expected_total", info.num_frames) or info.num_frames
        sent_count = 0
        segment_counts = {"forward": 0, "backward": 0, "backfill": 0}

        logger.info(
            f"Propagation '{direction}' starting for session {session_id[:8]} "
            f"(plan: {plan})"
        )
        await websocket.send_json({
            "type": "propagation_started",
            "total_frames": info.num_frames,
            "direction": direction,
            "plan": plan,
            "expected_total": expected_total,
        })

        try:
            async for frame_result in sam3_service.propagate_in_video(
                session_id, direction=direction
            ):
                frame_idx = frame_result.get("frame_index", sent_count)
                segment = frame_result.get("segment") or "forward"
                outputs = frame_result.get("outputs", {})

                obj_ids = outputs.get("out_obj_ids", [])
                binary_masks = outputs.get("out_binary_masks", [])

                # Render mask overlay as transparent PNG
                mask_png = None
                if binary_masks is not None and len(obj_ids) > 0:
                    try:
                        mask_png = render_masks_only(
                            np.array(obj_ids),
                            np.array(binary_masks),
                            info.orig_height,
                            info.orig_width,
                            alpha=0.5,
                        )
                    except Exception as e:
                        logger.warning(f"Mask rendering failed for frame {frame_idx}: {e}")

                # Convert obj_ids to list
                obj_ids_list = []
                for oid in obj_ids:
                    if hasattr(oid, "item"):
                        obj_ids_list.append(oid.item())
                    else:
                        obj_ids_list.append(int(oid))

                sent_count += 1
                segment_counts[segment] = segment_counts.get(segment, 0) + 1
                self._propagation_counts[session_id] = sent_count
                progress = (
                    min(1.0, sent_count / expected_total)
                    if expected_total > 0 else 0.0
                )

                msg = {
                    "type": "frame_result",
                    "frame_index": int(frame_idx),
                    "segment": segment,
                    "out_obj_ids": obj_ids_list,
                    "mask_png": mask_png,
                    "progress": round(progress, 4),
                    "sent_count": sent_count,
                    "expected_total": expected_total,
                }
                await websocket.send_json(msg)

            await websocket.send_json({
                "type": "propagation_complete",
                "total_frames": info.num_frames,
                "sent_count": sent_count,
                "expected_total": expected_total,
                "segment_counts": segment_counts,
            })
            logger.info(
                f"Propagation '{direction}' streamed {sent_count}/{expected_total} "
                f"items for session {session_id[:8]} (segments: {segment_counts})"
            )

        except asyncio.CancelledError:
            logger.info(f"Propagation cancelled for session {session_id}")
            # The "cancelled" message is sent by the connection handler
            # (this task's context cannot await after being cancelled)
            raise
        except Exception as e:
            logger.error(
                f"Propagation error for {session_id}: "
                f"{type(e).__name__}: {e}"
            )
            try:
                await websocket.send_json({
                    "type": "error",
                    "message": str(e),
                })
            except Exception:
                pass
        finally:
            info.is_propagating = False
