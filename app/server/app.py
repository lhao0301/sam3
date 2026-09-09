# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""FastAPI application entry point (SAM2-task mode).

Provides:
- REST endpoints for session management and prompt submission
- WebSocket endpoint for streaming propagation results
- Static file serving for the frontend
- CORS for cross-origin access (frontend on a different machine)

Inference runs on the SAM2-task tracker (``Sam3TrackerPredictor`` inside
the SAM3 checkpoint): prompts are incremental (no reset / replay), every
object carries a user-specified id, and points + boxes may be combined.

Usage:
    conda run -n sam3 uvicorn app.server.app:app --host 0.0.0.0 --port 8000

GPU selection happens automatically at import time (app/server/__init__.py
calls gpu_utils.configure_gpu BEFORE torch/CUDA is initialized): the card
with the most free memory wins. Set SAM3_GPU=<index> (or CUDA_VISIBLE_DEVICES
in the launcher) to pin a specific card instead.
"""

import os
import shutil
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from sam3.logger import get_logger

from .frame_utils import extract_frames, get_frames_dir, cleanup_session_frames
from .mask_utils import render_masks_only
from .session_manager import SessionManager
from .sam2_service import Sam2TrackerService
from .ws_manager import WSManager
from . import SELECTED_GPU

logger = get_logger(__name__)


def _render_prompt_frame_outputs(outputs: dict, info) -> dict:
    """Extract obj ids / boxes / rendered mask PNG from a prompt or mask
    response for a prompt frame."""
    import numpy as np

    obj_ids = outputs.get("out_obj_ids", [])
    binary_masks = outputs.get("out_binary_masks", [])
    boxes_xywh = outputs.get("out_boxes_xywh", [])

    obj_ids_list = []
    mask_png = None
    if obj_ids is not None and len(obj_ids) > 0:
        obj_ids_list = [int(o) for o in obj_ids]
        if binary_masks is not None and len(binary_masks) > 0:
            try:
                mask_png = render_masks_only(
                    np.array(obj_ids_list),
                    np.array(binary_masks),
                    info.orig_height,
                    info.orig_width,
                    alpha=0.5,
                )
            except Exception as e:
                logger.warning(f"Mask rendering failed: {e}")

    boxes_list = []
    if boxes_xywh is not None:
        for box in boxes_xywh:
            if hasattr(box, "tolist"):
                boxes_list.append(box.tolist())
            else:
                boxes_list.append(list(box))

    return {"obj_ids": obj_ids_list, "boxes_xywh": boxes_list, "mask_png": mask_png}


# Configuration
MAX_CONCURRENT_SESSIONS = int(os.environ.get("SAM3_MAX_SESSIONS", "4"))
MAX_CONCURRENT_INFERENCE = int(os.environ.get("SAM3_MAX_INFERENCE", "1"))
CHECKPOINT_PATH = os.environ.get("SAM3_CHECKPOINT_PATH", "")
UPLOAD_DIR = Path("/tmp/sam3_uploads")
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
LOG_DIR = Path(os.environ.get("SAM3_LOG_DIR", "logs"))

# Global instances
session_manager: SessionManager = None
sam2_service: Sam2TrackerService = None
ws_manager: WSManager = None


def _purge_stale_tmp_dirs():
    """Delete leftover session frames, uploads and exports from previous
    runs so restarting the service yields a clean slate."""
    for d in ("/tmp/sam3_sessions", "/tmp/sam3_uploads", "/tmp/sam3_exports"):
        if os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)
            logger.info(f"Purged stale directory {d}")


async def _close_session_full(session_id: str):
    """Close the tracker session, drop metadata and delete the frames dir."""
    await sam2_service.close_session(session_id)
    session_manager.remove_session(session_id)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle."""
    global session_manager, sam2_service, ws_manager

    # Startup: reset all state left over from a previous run
    _purge_stale_tmp_dirs()
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Initializing SAM3 annotation service (SAM2-task mode)...")

    # The GPU was already picked and pinned via CUDA_VISIBLE_DEVICES at
    # package import time (app/server/__init__.py): importing the sam3
    # package initializes CUDA, which pins the process to a device —
    # at that point it must already point at the idlest card.
    logger.info(f"Service runs on GPU {SELECTED_GPU}")

    sam2_service = Sam2TrackerService(
        max_concurrent_inference=MAX_CONCURRENT_INFERENCE,
        checkpoint_path=CHECKPOINT_PATH or None,
    )

    async def _expire_session(sid: str):
        """Release the tracker inference state (GPU memory, prompt log)
        together with the session metadata and frames on expiry."""
        await _close_session_full(sid)

    session_manager = SessionManager(
        max_concurrent_sessions=MAX_CONCURRENT_SESSIONS,
        on_session_expired=_expire_session,
        gpu_index=SELECTED_GPU,
    )
    ws_manager = WSManager()

    await session_manager.start_cleanup_task()
    logger.info("Service ready")

    yield

    # Shutdown
    await session_manager.stop_cleanup_task()
    logger.info("Service shut down")


app = FastAPI(
    title="SAM3 Video Annotation Service (SAM2-task mode)",
    description="Frontend-backend separated video annotation system using SAM3",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS: allow the frontend (on any machine) to access this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── REST Endpoints ──────────────────────────────────────────────────────


@app.get("/api/status")
async def get_status(session_id: str = None):
    """Return service status including GPU memory and active sessions.

    When `session_id` is passed its activity timestamp is refreshed,
    serving as the client heartbeat for the idle-expiry cleanup.
    """
    if session_id:
        session_manager.get_session(session_id)  # touches last_active_at
    return JSONResponse(session_manager.get_gpu_status())


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    """Upload a video file. Returns the saved path for later session creation."""
    if not file.content_type or not file.content_type.startswith("video/"):
        # Also allow common video extensions
        ext = Path(file.filename).suffix.lower()
        if ext not in (".mp4", ".avi", ".mov", ".mkv", ".webm"):
            raise HTTPException(status_code=400, detail="File must be a video")

    # Store under a unique generated name, NOT the original filename:
    # two sessions uploading files with the same name would otherwise
    # clobber each other (the second upload overwrites the first session's
    # video while its frame extraction may still be reading it). This also
    # neutralizes path-like or non-ASCII filenames. The original name is
    # still returned as `filename` for display.
    import uuid

    store_ext = Path(file.filename).suffix.lower() or ".mp4"
    upload_path = UPLOAD_DIR / f"{uuid.uuid4().hex[:12]}{store_ext}"
    with open(upload_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    logger.info(
        f"Uploaded video saved to {upload_path} "
        f"(original filename: {file.filename!r})"
    )
    return {"filename": file.filename, "path": str(upload_path)}


@app.post("/api/session/start")
async def start_session(request: dict):
    """Start a new annotation session.

    Request body:
        {
            "video_path": "/tmp/sam3_uploads/xxx.mp4",
            "video_filename": "xxx.mp4"  // optional, for display
        }

    Extracts video frames into a per-session JPEG folder, then starts a
    SAM2-task tracker session with that folder as resource_path.

    Returns:
        {
            "session_id": "uuid",
            "num_frames": 137,
            "orig_height": 1080,
            "orig_width": 1920
        }
    """
    video_path = request.get("video_path")
    if not video_path or not os.path.exists(video_path):
        raise HTTPException(status_code=400, detail="video_path is required and must exist")

    # Step 1: Create session metadata (to get a session_id for frames dir)
    import uuid
    temp_session_id = str(uuid.uuid4())

    # Step 2: Extract frames using ffmpeg
    try:
        num_frames, orig_height, orig_width, video_fps = extract_frames(video_path, temp_session_id)
    except Exception as e:
        cleanup_session_frames(temp_session_id)
        raise HTTPException(status_code=500, detail=f"Frame extraction failed: {e}")

    if num_frames == 0:
        cleanup_session_frames(temp_session_id)
        raise HTTPException(status_code=500, detail="No frames extracted from video")

    # Step 3: Register session in SessionManager
    frames_dir = get_frames_dir(temp_session_id)
    try:
        session_id = session_manager.create_session(
            frames_dir=frames_dir,
            num_frames=num_frames,
            orig_height=orig_height,
            orig_width=orig_width,
            video_fps=video_fps,
        )
    except RuntimeError as e:
        cleanup_session_frames(temp_session_id)
        raise HTTPException(status_code=503, detail=str(e))

    # Step 4: Start tracker session with the frames directory
    try:
        await sam2_service.start_session(session_id, frames_dir)
    except Exception as e:
        session_manager.remove_session(session_id)
        raise HTTPException(status_code=500, detail=f"Tracker start_session failed: {e}")

    return {
        "session_id": session_id,
        "num_frames": num_frames,
        "orig_height": orig_height,
        "orig_width": orig_width,
    }


@app.post("/api/prompt/add")
async def add_prompt(request: dict):
    """Add a point / box / point+box prompt for one object on one frame.

    The obj_id is always user-specified: a new id registers a new object,
    an existing id refines that object's mask on this frame. Submissions
    are incremental (no reset) and points + boxes may be combined.

    Request body:
        {
            "session_id": "uuid",
            "frame_idx": 0,
            "obj_id": 1,
            // any of the following (at least one), coordinates in [0,1]:
            "points": [[x, y], ...],
            "point_labels": [1, 0, ...],       // 1=positive, 0=negative
            "box": [x, y, w, h]                // normalized xywh
        }

    Returns:
        {
            "session_id": "uuid",
            "frame_idx": 0,
            "new_obj_ids": [1],
            "obj_ids": [1],                    // all objects on this frame
            "boxes_xywh": [[x, y, w, h]],      // normalized tight boxes
            "mask_png": "base64 ..."           // masks on the prompt frame
        }
    """
    session_id = request.get("session_id")
    frame_idx = int(request.get("frame_idx", 0))
    obj_id = request.get("obj_id")

    info = session_manager.get_session(session_id)
    if info is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if obj_id is None:
        raise HTTPException(status_code=400, detail="obj_id is required")
    obj_id = int(obj_id)

    points = request.get("points")
    point_labels = request.get("point_labels")
    box = request.get("box")

    if points is not None:
        if point_labels is None or len(points) != len(point_labels):
            raise HTTPException(
                status_code=400, detail="points and point_labels must have same length"
            )
        points = [[float(x), float(y)] for x, y in points]
        point_labels = [int(l) for l in point_labels]
    if box is not None:
        if len(box) != 4:
            raise HTTPException(status_code=400, detail="box must be [x, y, w, h]")
        box = [float(v) for v in box]
    if not points and box is None:
        raise HTTPException(
            status_code=400, detail="at least one of points or box is required"
        )

    try:
        result = await sam2_service.submit_prompt(
            session_id=session_id,
            frame_idx=frame_idx,
            obj_id=obj_id,
            points=points,
            point_labels=point_labels if points else None,
            box=box,
        )
    except PermissionError as e:
        # propagation started; brand-new objects need reset_tracking first
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"add_prompt failed: {type(e).__name__}: {e}",
        )

    rendered = _render_prompt_frame_outputs(result["outputs"], info)
    info.touch()
    return {
        "session_id": session_id,
        "frame_idx": frame_idx,
        "new_obj_ids": [obj_id],
        **rendered,
    }


@app.post("/api/prompt/confirm")
async def confirm_prompt(request: dict):
    """Confirm the previewed prompt.

    In SAM2-task mode submissions are already applied to the tracker
    state, so this is pure bookkeeping (kept for frontend compatibility).
    """
    session_id = request.get("session_id")
    info = session_manager.get_session(session_id)
    if info is None:
        raise HTTPException(status_code=404, detail="Session not found")

    info.touch()
    return {"session_id": session_id, "confirmed": True}


@app.delete("/api/prompt/{session_id}/pending")
async def cancel_pending(session_id: str, obj_id: int, frame_idx: int):
    """Undo the previewed prompt for obj_id on frame_idx.

    Restores the previous prompt version of that object/frame (if any);
    removes the object entirely when nothing is left.
    """
    info = session_manager.get_session(session_id)
    if info is None:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        result = await sam2_service.undo_prompt(
            session_id, obj_id=int(obj_id), frame_idx=int(frame_idx)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"undo failed: {type(e).__name__}: {e}"
        )

    rendered = _render_prompt_frame_outputs(result, info)
    info.touch()
    return {
        "session_id": session_id,
        "frame_idx": result["frame_index"],
        **rendered,
    }


@app.delete("/api/prompt/{session_id}/{obj_id}")
async def remove_object(session_id: str, obj_id: int):
    """Remove a tracked object (tracker state + prompt records)."""
    info = session_manager.get_session(session_id)
    if info is None:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        result = await sam2_service.remove_object(session_id, int(obj_id))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"remove_object failed: {type(e).__name__}: {e}"
        )

    rendered = _render_prompt_frame_outputs(result, info)
    info.touch()
    return {
        "session_id": session_id,
        "frame_idx": result["frame_index"],
        **rendered,
    }


@app.post("/api/session/{session_id}/reset-tracking")
async def reset_tracking(session_id: str):
    """Clear all tracking results and replay the submitted prompts.

    Afterwards new objects can be added again (before the next
    propagation). The propagation itself has to be redone.
    """
    info = session_manager.get_session(session_id)
    if info is None:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        result = await sam2_service.reset_tracking(session_id)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"reset_tracking failed: {type(e).__name__}: {e}"
        )

    info.touch()
    return {"session_id": session_id, **result}


@app.post("/api/logs")
async def receive_logs(request: dict):
    """Persist frontend terminal log lines to a local file (append).

    The frontend batches lines (flush on 10 lines / 3s / pagehide beacon).

    Body: {"session_id": str|null, "lines": [{"ts": "HH:MM:SS",
           "level": "INFO", "msg": str}, ...]}
    """
    lines = request.get("lines") or []
    if not lines:
        return {"saved": 0}

    session_id = request.get("session_id")
    sess_tag = f"sess {str(session_id)[:8]}" if session_id else "-"
    date_str = time.strftime("%Y-%m-%d")
    path = LOG_DIR / f"annotation_{time.strftime('%Y%m%d')}.log"

    entries = []
    for ln in lines[:200]:
        ts = str(ln.get("ts", ""))[:8]
        level = str(ln.get("level", "INFO"))[:4].upper()
        msg = str(ln.get("msg", "")).replace("\n", " ")[:2000]
        entries.append(f"{date_str} {ts} [{level}] [{sess_tag}] {msg}\n")

    try:
        with open(path, "a", encoding="utf-8") as f:
            f.writelines(entries)
    except Exception as e:
        logger.warning(f"Failed to persist terminal logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return {"saved": len(entries)}


@app.delete("/api/session/{session_id}")
async def close_session(session_id: str):
    """Close a session and clean up all resources."""
    info = session_manager.get_session(session_id)
    if info is None:
        raise HTTPException(status_code=404, detail="Session not found")

    await _close_session_full(session_id)

    return {"status": "closed", "session_id": session_id}


@app.post("/api/session/{session_id}/close")
async def close_session_beacon(session_id: str):
    """Beacon-compatible session close (navigator.sendBeacon can only POST).

    Sent by the frontend on pagehide so closing the tab releases the
    tracker state, prompt log and frame files immediately. Idempotent:
    a no-op when the session is already gone (e.g. the expiry cleanup
    raced the beacon).
    """
    info = session_manager.get_session(session_id)
    if info is not None:
        await _close_session_full(session_id)

    return {"status": "closed", "session_id": session_id}


@app.get("/api/session/{session_id}/info")
async def get_session_info(session_id: str):
    """Get session metadata."""
    info = session_manager.get_session(session_id)
    if info is None:
        raise HTTPException(status_code=404, detail="Session not found")

    return {
        "session_id": info.session_id,
        "num_frames": info.num_frames,
        "orig_height": info.orig_height,
        "orig_width": info.orig_width,
        "is_propagating": info.is_propagating,
        "tracking_started": sam2_service.is_tracking_started(session_id),
    }


@app.get("/api/frame/{session_id}/{frame_idx}")
async def get_frame(session_id: str, frame_idx: int):
    """Serve a single video frame as a JPEG file."""
    info = session_manager.get_session(session_id)
    if info is None:
        raise HTTPException(status_code=404, detail="Session not found")

    from pathlib import Path
    frame_path = Path(info.frames_dir) / f"{frame_idx:05d}.jpg"
    if not frame_path.exists():
        raise HTTPException(status_code=404, detail=f"Frame {frame_idx} not found")

    info.touch()
    return FileResponse(str(frame_path), media_type="image/jpeg")


@app.get("/api/thumb/{session_id}/{frame_idx}")
async def get_thumbnail(session_id: str, frame_idx: int, w: int = 160):
    """Serve a downscaled JPEG thumbnail of a video frame.

    Used by the frontend filmstrip; much cheaper than fetching full frames.
    """
    info = session_manager.get_session(session_id)
    if info is None:
        raise HTTPException(status_code=404, detail="Session not found")

    from pathlib import Path
    frame_path = Path(info.frames_dir) / f"{frame_idx:05d}.jpg"
    if not frame_path.exists():
        raise HTTPException(status_code=404, detail=f"Frame {frame_idx} not found")

    width = max(40, min(int(w), 320))
    try:
        import cv2

        img = cv2.imread(str(frame_path))
        if img is None:
            raise HTTPException(status_code=500, detail=f"Cannot read frame {frame_idx}")
        scale = width / img.shape[1]
        thumb = cv2.resize(
            img,
            (width, max(1, int(img.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
        ok, buf = cv2.imencode(
            ".jpg", thumb, [int(cv2.IMWRITE_JPEG_QUALITY), 80]
        )
        if not ok:
            raise HTTPException(status_code=500, detail="Thumbnail encoding failed")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Thumbnail failed: {e}")

    info.touch()
    return Response(content=buf.tobytes(), media_type="image/jpeg")


@app.get("/api/mask/{session_id}/{frame_idx}")
async def get_mask(session_id: str, frame_idx: int):
    """Get the mask overlay for a specific frame as a base64 PNG.

    Reads the tracker's per-frame outputs (conditioning and tracked
    results) and renders them as a transparent PNG overlay.
    """
    info = session_manager.get_session(session_id)
    if info is None:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        result = await sam2_service.get_frame_masks(session_id, frame_idx)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"get_frame_masks failed: {type(e).__name__}: {e}"
        )

    obj_ids = result.get("out_obj_ids", [])
    binary_masks = result.get("out_binary_masks", [])

    mask_png = None
    if binary_masks and len(obj_ids) > 0:
        try:
            import numpy as np
            mask_png = render_masks_only(
                np.array(obj_ids),
                np.array(binary_masks),
                info.orig_height,
                info.orig_width,
                alpha=0.5,
            )
        except Exception as e:
            logger.warning(f"Mask rendering failed for frame {frame_idx}: {e}")

    info.touch()
    return {"frame_index": frame_idx, "obj_ids": obj_ids, "mask_png": mask_png}


@app.get("/api/export/{session_id}")
async def export_video(session_id: str):
    """Export the annotated video with mask overlays as an MP4 file.

    Composites each frame with its tracked masks and writes the result
    to a video file. Returns the file for download.

    Returns:
        FileResponse with the MP4 video.
    """
    info = session_manager.get_session(session_id)
    if info is None:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        output_path = await sam2_service.export_video(
            session_id=session_id,
            frames_dir=info.frames_dir,
            num_frames=info.num_frames,
            fps=info.video_fps,
        )
    except Exception as e:
        logger.error(f"Video export failed for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Export failed: {e}")

    filename = f"annotated_{session_id[:8]}.mp4"
    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename=filename,
    )


# ─── WebSocket Endpoint ─────────────────────────────────────────────────


@app.websocket("/ws/propagate/{session_id}")
async def ws_propagate(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for streaming propagation results.

    Client sends JSON messages:
        {"type": "start", "direction": "forward"|"backward"|"both"}
        {"type": "cancel"}   - cancel ongoing propagation
        {"type": "ping"}     - health check

    Server sends JSON messages:
        {"type": "propagation_started", "total_frames": N, "direction": "..."}
        {"type": "frame_result", "frame_index": i, "mask_png": "...", "progress": 0.5}
        {"type": "propagation_complete", "total_frames": N}
        {"type": "cancelled"}
        {"type": "error", "message": "..."}
    """
    await ws_manager.handle_connection(
        websocket, session_id, sam2_service, session_manager
    )


# ─── Static Frontend Serving ────────────────────────────────────────────


if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    async def serve_index():
        return FileResponse(str(FRONTEND_DIR / "index.html"))

    @app.get("/manifest")
    async def serve_manifest():
        """Return frontend file list for dynamic loading."""
        return {
            "css": ["static/css/style.css"],
            "js": [
                "static/js/ws-client.js",
                "static/js/video-player.js",
                "static/js/prompt-canvas.js",
                "static/js/mask-overlay.js",
                "static/js/app.js",
            ],
        }
