# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""FastAPI application entry point.

Provides:
- REST endpoints for session management and prompt addition
- WebSocket endpoint for streaming propagation results
- Static file serving for the frontend
- CORS for cross-origin access (frontend on a different machine)

Usage:
    conda run -n sam3 uvicorn app.server.app:app --host 0.0.0.0 --port 8000
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

from .frame_utils import extract_frames, get_frame_path, get_frames_dir, cleanup_session_frames
from .mask_utils import render_masks_only, get_color_for_obj_id
from .session_manager import SessionManager
from .sam3_service import Sam3Service
from .ws_manager import WSManager

logger = get_logger(__name__)


def _render_prompt_frame_outputs(outputs: dict, info) -> dict:
    """Extract obj ids / boxes / rendered mask PNG from an add_prompt or
    remove_object response for a prompt frame."""
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


def _to_jsonable(obj):
    """Recursively convert numpy/tensor objects to JSON-serializable types."""
    import numpy as np
    import torch
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    elif hasattr(obj, 'item') and callable(obj.item):
        try:
            return obj.item()
        except Exception:
            pass
    return obj

# Configuration
MODEL_VERSION = os.environ.get("SAM3_MODEL_VERSION", "sam3")
MAX_CONCURRENT_SESSIONS = int(os.environ.get("SAM3_MAX_SESSIONS", "4"))
MAX_CONCURRENT_INFERENCE = int(os.environ.get("SAM3_MAX_INFERENCE", "1"))
CHECKPOINT_PATH = os.environ.get("SAM3_CHECKPOINT_PATH", "")
UPLOAD_DIR = Path("/tmp/sam3_uploads")
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
# Frontend terminal logs are persisted here (one file per day)
LOG_DIR = Path(os.environ.get("SAM3_LOG_DIR", "logs"))

# Global instances
session_manager: SessionManager = None
sam3_service: Sam3Service = None
ws_manager: WSManager = None


def _purge_stale_tmp_dirs():
    """Delete leftover session frames, uploads and exports from previous
    runs so restarting the service yields a clean slate."""
    for d in ("/tmp/sam3_sessions", "/tmp/sam3_uploads", "/tmp/sam3_exports"):
        if os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)
            logger.info(f"Purged stale directory {d}")


async def _close_session_full(session_id: str):
    """Close the SAM3 session, drop metadata and delete the frames dir."""
    await sam3_service.close_session(session_id)
    session_manager.remove_session(session_id)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle."""
    global session_manager, sam3_service, ws_manager

    # Startup: reset all state left over from a previous run
    _purge_stale_tmp_dirs()
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Initializing SAM3 annotation service...")

    sam3_service = Sam3Service(
        model_version=MODEL_VERSION,
        max_concurrent_inference=MAX_CONCURRENT_INFERENCE,
        checkpoint_path=CHECKPOINT_PATH or None,
    )

    async def _expire_session(sid: str):
        """Release the SAM3 inference state (GPU memory, prompt history)
        together with the session metadata and frames on expiry."""
        await _close_session_full(sid)

    session_manager = SessionManager(
        max_concurrent_sessions=MAX_CONCURRENT_SESSIONS,
        on_session_expired=_expire_session,
    )
    ws_manager = WSManager()

    await session_manager.start_cleanup_task()
    logger.info("Service ready")

    yield

    # Shutdown
    await session_manager.stop_cleanup_task()
    logger.info("Service shut down")


app = FastAPI(
    title="SAM3 Video Annotation Service",
    description="Frontend-backend separated video annotation system using SAM3",
    version="1.0.0",
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


@app.get("/api/debug/mem")
async def debug_memory():
    """Detailed GPU memory breakdown for debugging."""
    import torch

    # PyTorch allocator stats
    alloc = torch.cuda.memory_allocated(0)
    reserved = torch.cuda.memory_reserved(0)
    max_alloc = torch.cuda.max_memory_allocated(0)
    max_reserved = torch.cuda.max_memory_reserved(0)

    # Per-session breakdown
    sessions = []
    for sid, session in sam3_service._predictor._all_inference_states.items():
        state = session["state"]
        num_frames = state.get("num_frames", 0)

        # precomputed_vit size
        pcv = state.get("feature_cache", {}).get("precomputed_vit")
        pcv_gb = 0.0
        pcv_shape = None
        if pcv is not None:
            pcv_bytes = sum(
                f.element_size() * f.nelement() for f in pcv
            )
            pcv_gb = pcv_bytes / 1024**3
            pcv_shape = [list(f.shape) for f in pcv]

        # cached_frame_outputs size
        cached = state.get("cached_frame_outputs", {})
        cached_frames = len(cached)
        cached_gb = 0.0
        for fidx, m in cached.items():
            for oid, mask in m.items():
                if hasattr(mask, "element_size"):
                    cached_gb += mask.element_size() * mask.nelement()
        cached_gb /= 1024**3

        # output_dict size (raw predictions)
        out_dict = state.get("output_dict", {})
        od_frames = sum(
            len(v) for v in out_dict.values() if isinstance(v, dict)
        )

        # obj_ids
        obj_ids = state.get("obj_ids", [])

        # img_batch type
        img_batch = state.get("input_batch", None)
        if img_batch is not None:
            ib = img_batch.img_batch
            ib_type = type(ib).__name__
        else:
            ib_type = "N/A"

        # tracker states
        tracker_states = state.get("tracker_inference_states", [])
        num_tracker_states = len(tracker_states)

        sessions.append({
            "session_id": sid[:12] + "...",
            "num_frames": num_frames,
            "img_batch_type": ib_type,
            "precomputed_vit_gb": round(pcv_gb, 3),
            "precomputed_vit_shape": pcv_shape,
            "cached_frame_outputs": {
                "frames_with_masks": cached_frames,
                "total_mask_gb": round(cached_gb, 3),
            },
            "output_dict_frames": od_frames,
            "num_obj_ids": len(obj_ids),
            "num_tracker_states": num_tracker_states,
        })

    # Model param count
    model = sam3_service._predictor.model
    total_params = sum(p.numel() for p in model.parameters())
    param_gb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**3

    return JSONResponse({
        "pytorch": {
            "allocated_gb": round(alloc / 1024**3, 3),
            "reserved_gb": round(reserved / 1024**3, 3),
            "max_allocated_gb": round(max_alloc / 1024**3, 3),
            "max_reserved_gb": round(max_reserved / 1024**3, 3),
            "fragmentation_gb": round((reserved - alloc) / 1024**3, 3),
        },
        "model": {
            "total_params_M": round(total_params / 1e6, 1),
            "params_gb": round(param_gb, 3),
        },
        "sessions": sessions,
    })


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    """Upload a video file. Returns the saved path for later session creation."""
    if not file.content_type or not file.content_type.startswith("video/"):
        # Also allow common video extensions
        ext = Path(file.filename).suffix.lower()
        if ext not in (".mp4", ".avi", ".mov", ".mkv", ".webm"):
            raise HTTPException(status_code=400, detail="File must be a video")

    upload_path = UPLOAD_DIR / file.filename
    with open(upload_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    logger.info(f"Uploaded video saved to {upload_path}")
    return {"filename": file.filename, "path": str(upload_path)}


@app.post("/api/session/start")
async def start_session(request: dict):
    """Start a new annotation session.

    Request body:
        {
            "video_path": "/tmp/sam3_uploads/xxx.mp4",
            "video_filename": "xxx.mp4"  // optional, for display
        }

    Extracts video frames into a per-session JPEG folder, then starts
    a SAM3 session with that folder as the resource_path.

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

    # Step 4: Start SAM3 session with the frames directory
    try:
        await sam3_service.start_session(
            session_id=session_id,
            resource_path=frames_dir,
            resource_type="image_folder",
        )
    except Exception as e:
        session_manager.remove_session(session_id)
        raise HTTPException(status_code=500, detail=f"SAM3 start_session failed: {e}")

    return {
        "session_id": session_id,
        "num_frames": num_frames,
        "orig_height": orig_height,
        "orig_width": orig_width,
    }


@app.post("/api/prompt/add")
async def add_prompt(request: dict):
    """Add a text / point / box prompt to a session (preview == commit).

    One target per request. The tracklet id is user-specified for point
    prompts (obj_id) and box prompts (box_labels), and auto-assigned by
    SAM3 for text prompts (open-vocabulary detection, possibly multiple
    instances).

    Request body:
        {
            "session_id": "uuid",
            "frame_idx": 0,
            "prompt_type": "box" | "point" | "text",
            // box prompt (normalized [0,1], one box per request):
            "boxes": [[x, y, w, h]],
            "box_labels": [tracklet_id],
            // point prompt (normalized [0,1], one target per request):
            "points": [[x, y], ...],
            "point_labels": [1, 0, ...],       // 1=positive, 0=negative
            "obj_id": tracklet_id,
            // text prompt:
            "text": "the red car"
        }

    Returns:
        {
            "session_id": "uuid",
            "frame_idx": 0,
            "obj_ids": [1],
            "boxes_xywh": [[x, y, w, h]],
            "mask_png": "base64 ..."          // masks on the prompt frame
        }
    """
    session_id = request.get("session_id")
    frame_idx = int(request.get("frame_idx", 0))
    prompt_type = request.get("prompt_type", "box")

    info = session_manager.get_session(session_id)
    if info is None:
        raise HTTPException(status_code=404, detail="Session not found")

    # Build the prompt dict in PromptHistory's documented format. Tracklet
    # ids are user-specified for point prompts and auto-assigned by SAM3 for
    # text / box detection prompts.
    prompt = {"type": prompt_type, "frame_idx": frame_idx}
    if prompt_type == "box":
        boxes = request.get("boxes", [])
        box_labels = request.get("box_labels", [])
        if not boxes:
            raise HTTPException(status_code=400, detail="boxes is required for box prompt")
        if len(boxes) != len(box_labels):
            raise HTTPException(
                status_code=400, detail="boxes and box_labels must have same length"
            )
        prompt["boxes"] = boxes
        prompt["box_labels"] = [int(l) for l in box_labels]
    elif prompt_type == "point":
        points = request.get("points", [])
        point_labels = request.get("point_labels", [])
        obj_id = request.get("obj_id")
        if not points:
            raise HTTPException(status_code=400, detail="points is required for point prompt")
        if len(points) != len(point_labels):
            raise HTTPException(
                status_code=400, detail="points and point_labels must have same length"
            )
        if obj_id is None:
            raise HTTPException(status_code=400, detail="obj_id is required for point prompt")
        prompt["points"] = points
        prompt["point_labels"] = [int(l) for l in point_labels]
        prompt["obj_id"] = int(obj_id)
    elif prompt_type == "text":
        text = (request.get("text") or "").strip()
        if not text:
            raise HTTPException(status_code=400, detail="text is required for text prompt")
        prompt["text"] = text
    else:
        raise HTTPException(status_code=400, detail=f"invalid prompt_type: {prompt_type}")

    try:
        outputs = await sam3_service.submit_prompt(session_id, info, prompt)
    except HTTPException:
        raise
    except Exception as e:
        info.prompt_history.clear_pending()
        raise HTTPException(
            status_code=500,
            detail=f"add_prompt failed: {type(e).__name__}: {e}",
        )

    rendered = _render_prompt_frame_outputs(outputs, info)

    # Ids newly introduced by this submission (for the frontend pending set)
    if prompt_type == "point":
        new_obj_ids = [prompt["obj_id"]]
    else:
        new_obj_ids = list(info.prompt_history.pending.get("assigned_ids", []))

    info.touch()
    return {
        "session_id": session_id,
        "frame_idx": frame_idx,
        "new_obj_ids": new_obj_ids,
        **rendered,
    }


@app.post("/api/prompt/confirm")
async def confirm_prompt(request: dict):
    """Confirm the previewed prompt: it joins the confirmed set that gets
    replayed on every subsequent submission."""
    session_id = request.get("session_id")
    info = session_manager.get_session(session_id)
    if info is None:
        raise HTTPException(status_code=404, detail="Session not found")

    result = await sam3_service.confirm_pending(session_id, info)
    info.touch()
    return {"session_id": session_id, **result}


@app.delete("/api/prompt/{session_id}/pending")
async def cancel_pending(session_id: str):
    """Undo the previewed prompt and restore the confirmed state."""
    info = session_manager.get_session(session_id)
    if info is None:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        outputs = await sam3_service.cancel_pending(session_id, info)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"cancel failed: {e}")

    rendered = _render_prompt_frame_outputs(outputs, info)
    info.touch()
    return {
        "session_id": session_id,
        "frame_idx": info.prompt_history.det_frame,
        **rendered,
    }


@app.delete("/api/prompt/{session_id}/{obj_id}")
async def remove_object(session_id: str, obj_id: int):
    """Remove a confirmed tracklet and restore the remaining state."""
    info = session_manager.get_session(session_id)
    if info is None:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        outputs = await sam3_service.remove_tracklet(session_id, info, obj_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"remove_object failed: {e}")

    rendered = _render_prompt_frame_outputs(outputs, info)
    info.touch()
    return {
        "session_id": session_id,
        "frame_idx": info.prompt_history.det_frame,
        **rendered,
    }


@app.post("/api/encode/{session_id}")
async def encode_features(session_id: str):
    """Pre-compute ViT backbone features for all frames in a session.

    Runs the ViT trunk on every video frame and caches the raw outputs so
    that subsequent prompt additions and propagations skip the expensive ViT
    forward pass and only run the lightweight FPN neck per frame.

    Returns:
        {"session_id": "...", "num_frames_encoded": N}
    """
    info = session_manager.get_session(session_id)
    if info is None:
        raise HTTPException(status_code=404, detail="Session not found")

    if info.is_encoded:
        return {
            "session_id": session_id,
            "num_frames_encoded": info.num_frames,
            "already_encoded": True,
        }

    try:
        result = await sam3_service.encode_features(session_id)
        info.is_encoded = True
        info.touch()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"encode_features failed: {e}")

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
    SAM3 state, prompt history and frame files immediately. Idempotent:
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
        "is_encoded": info.is_encoded,
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

    Retrieves cached masks from SAM3's inference state after propagation
    and renders them as a transparent PNG overlay.
    """
    info = session_manager.get_session(session_id)
    if info is None:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        result = await sam3_service.get_frame_masks(session_id, frame_idx)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"get_frame_masks failed: {e}")

    obj_ids = result.get("obj_ids", [])
    binary_masks = result.get("binary_masks", [])

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

    Composites each frame with its cached masks (from propagation) and
    writes the result to a video file. Returns the file for download.

    Returns:
        FileResponse with the MP4 video.
    """
    info = session_manager.get_session(session_id)
    if info is None:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        output_path = await sam3_service.export_video(
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
        {"type": "start"}    - begin propagation
        {"type": "cancel"}   - cancel ongoing propagation
        {"type": "ping"}     - health check

    Server sends JSON messages:
        {"type": "propagation_started", "total_frames": N}
        {"type": "frame_result", "frame_index": i, "mask_png": "...", "progress": 0.5}
        {"type": "propagation_complete", "total_frames": N}
        {"type": "cancelled"}
        {"type": "error", "message": "..."}
    """
    await ws_manager.handle_connection(
        websocket, session_id, sam3_service, session_manager
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
