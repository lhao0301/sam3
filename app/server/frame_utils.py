# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Video frame extraction utilities.

Extracts MP4 video frames into a per-session JPEG folder so that:
1. The frontend can preview individual frames via GET /api/frame/{sid}/{idx}
2. The JPEG folder path can be passed directly to SAM3's start_session as resource_path
"""

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Tuple

from sam3.logger import get_logger

logger = get_logger(__name__)

# Base directory for all session frame storage
SESSIONS_BASE_DIR = Path("/tmp/sam3_sessions")


def get_session_frames_dir(session_id: str) -> Path:
    """Return the frames directory for a given session_id."""
    return SESSIONS_BASE_DIR / session_id / "frames"


def extract_frames(video_path: str, session_id: str) -> Tuple[int, int, int, float]:
    """Extract video frames as JPEG files into the session's frames directory.

    Uses ffmpeg to extract frames at original resolution.
    The JPEG folder path is suitable as resource_path for SAM3's start_session.

    Args:
        video_path: Path to the input MP4 (or other video file)
        session_id: Unique session identifier

    Returns:
        Tuple of (num_frames, orig_height, orig_width, fps)
    """
    frames_dir = get_session_frames_dir(session_id)
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Clean any existing frames in the directory
    for f in frames_dir.glob("*.jpg"):
        f.unlink()

    # Use ffprobe to get video metadata (frame count, resolution, fps).
    # `nb_frames` comes from container metadata, so probing does NOT decode
    # the video (unlike `-count_frames`, which decodes every frame just to
    # count them and roughly doubles the session-start latency). Containers
    # without this metadata (some mkv/webm) report "N/A"; the frame count
    # then falls back to counting the extracted files below. Output is JSON
    # and parsed by key, so ffprobe's field ordering doesn't matter.
    probe_cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=nb_frames,width,height,r_frame_rate",
        "-of", "json",
        video_path,
    ]
    fps = 30.0
    try:
        probe_result = subprocess.run(
            probe_cmd, capture_output=True, text=True, check=True, timeout=60
        )
        stream = json.loads(probe_result.stdout)["streams"][0]
        orig_width = int(stream["width"])
        orig_height = int(stream["height"])
        # nb_frames may be "N/A" or missing on containers that don't store it
        nb_frames = stream.get("nb_frames") or ""
        num_frames = int(nb_frames) if nb_frames.isdigit() else 0
        # r_frame_rate is a fraction like "30/1" or "30000/1001"
        rate = stream.get("r_frame_rate") or ""
        if "/" in rate:
            num, den = rate.split("/")
            den = int(den) if int(den) != 0 else 1
            fps = int(num) / den
        elif rate:
            fps = float(rate)
    except (
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        json.JSONDecodeError,
        ValueError,
        IndexError,
        KeyError,
    ):
        logger.warning("ffprobe failed, falling back to ffmpeg frame extraction without metadata")
        num_frames = 0
        orig_width = 0
        orig_height = 0

    # Extract frames using ffmpeg: 00000.jpg, 00001.jpg, ...
    output_pattern = str(frames_dir / "%05d.jpg")
    extract_cmd = [
        "ffmpeg",
        "-i", video_path,
        "-q:v", "2",          # high quality JPEG
        "-start_number", "0", # start from 0 to match frame indices
        "-y",                 # overwrite output files
        output_pattern,
    ]
    try:
        subprocess.run(
            extract_cmd, capture_output=True, check=True, timeout=300
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"ffmpeg frame extraction failed: {e.stderr}")
        raise RuntimeError(f"Failed to extract video frames: {e}")

    # If we didn't get metadata from ffprobe, count extracted frames and get resolution
    if num_frames == 0:
        extracted = sorted(frames_dir.glob("*.jpg"))
        num_frames = len(extracted)
        if num_frames > 0:
            # Get resolution from first frame using cv2
            import cv2
            cap = cv2.VideoCapture(str(extracted[0]))
            orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

    logger.info(
        f"Extracted {num_frames} frames ({orig_width}x{orig_height}, {fps:.2f} fps) "
        f"for session {session_id}"
    )
    return num_frames, orig_height, orig_width, fps


def get_frame_path(session_id: str, frame_idx: int) -> Path:
    """Return the JPEG file path for a specific frame in a session."""
    return get_session_frames_dir(session_id) / f"{frame_idx:05d}.jpg"


def cleanup_session_frames(session_id: str) -> None:
    """Delete the entire session frames directory."""
    session_dir = SESSIONS_BASE_DIR / session_id
    if session_dir.exists():
        shutil.rmtree(session_dir, ignore_errors=True)
        logger.info(f"Cleaned up frames for session {session_id}")


def get_frames_dir(session_id: str) -> str:
    """Return the frames directory path as string (for SAM3 resource_path)."""
    return str(get_session_frames_dir(session_id))
