# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Session manager for multi-user concurrency isolation.

Manages per-session metadata (frame directory, creation time, GPU memory estimate),
background cleanup of expired sessions, and GPU memory monitoring.
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Dict, Optional

import torch

from sam3.logger import get_logger

from .frame_utils import cleanup_session_frames

logger = get_logger(__name__)

# Session expiration timeout (seconds). Pages send a close beacon on
# unload; this timeout is only the fallback for crashed / disconnected
# clients, so it stays short: a session with no activity (no REST call,
# no WebSocket propagation loop) for 60s is treated as gone.
SESSION_EXPIRATION_SEC = 60
# How often the background cleanup task runs (seconds)
CLEANUP_INTERVAL_SEC = 10
# Estimated GPU memory per frame (image_size=1008, 3 channels, float16 = ~6MB)
# This is a rough estimate; actual usage includes feature caches.
EST_MEM_PER_FRAME_MB = 6


@dataclass
class SessionInfo:
    """Metadata for a single user session."""
    session_id: str
    frames_dir: str
    num_frames: int
    orig_height: int
    orig_width: int
    video_fps: float = 30.0
    created_at: float = field(default_factory=time.time)
    last_active_at: float = field(default_factory=time.time)
    est_gpu_mem_mb: float = 0.0
    is_propagating: bool = False
    tracking_started: bool = False

    def touch(self):
        """Update last active time."""
        self.last_active_at = time.time()

    def is_expired(self, expiration_sec: float = SESSION_EXPIRATION_SEC) -> bool:
        """Check if the session has expired due to inactivity."""
        return (time.time() - self.last_active_at) > expiration_sec


class SessionManager:
    """Manages session metadata and lifecycle for multi-user concurrency."""

    def __init__(
        self,
        max_concurrent_sessions: int = 4,
        max_gpu_mem_pct: float = 90.0,
        on_session_expired: Optional[Callable[[str], Awaitable[None]]] = None,
        gpu_index: Optional[int] = None,
    ):
        self._sessions: Dict[str, SessionInfo] = {}
        self._max_concurrent_sessions = max_concurrent_sessions
        self._max_gpu_mem_pct = max_gpu_mem_pct
        # Physical index of the GPU the service runs on (chosen at startup
        # by gpu_utils.configure_gpu; None when unknown / CPU-only)
        self._gpu_index = gpu_index
        # Async callback invoked on expiry so the SAM3 inference state
        # (GPU memory, prompt history) is released together with the
        # session metadata and frames directory.
        self._on_session_expired = on_session_expired
        self._cleanup_task: Optional[asyncio.Task] = None

    def create_session(
        self,
        frames_dir: str,
        num_frames: int,
        orig_height: int,
        orig_width: int,
        video_fps: float = 30.0,
    ) -> str:
        """Create a new session and return its ID.

        Raises:
            RuntimeError: If max concurrent sessions exceeded or GPU memory insufficient.
        """
        if len(self._sessions) >= self._max_concurrent_sessions:
            active = len(self._sessions)
            raise RuntimeError(
                f"Max concurrent sessions ({self._max_concurrent_sessions}) reached. "
                f"Currently active: {active}. Please try again later."
            )

        # Check GPU memory availability
        if torch.cuda.is_available():
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            free_mb = free_bytes / (1024 * 1024)
            total_mb = total_bytes / (1024 * 1024)
            used_pct = (1.0 - free_bytes / total_bytes) * 100.0
            est_needed_mb = num_frames * EST_MEM_PER_FRAME_MB
            if used_pct > self._max_gpu_mem_pct or free_mb < est_needed_mb:
                raise RuntimeError(
                    f"Insufficient GPU memory: {free_mb:.0f} MB free, "
                    f"~{est_needed_mb:.0f} MB needed. Used: {used_pct:.1f}%"
                )

        session_id = str(uuid.uuid4())
        info = SessionInfo(
            session_id=session_id,
            frames_dir=frames_dir,
            num_frames=num_frames,
            orig_height=orig_height,
            orig_width=orig_width,
            video_fps=video_fps,
            est_gpu_mem_mb=num_frames * EST_MEM_PER_FRAME_MB,
        )
        self._sessions[session_id] = info
        logger.info(f"Created session {session_id} ({num_frames} frames)")
        return session_id

    def get_session(self, session_id: str) -> Optional[SessionInfo]:
        """Get session info, or None if not found."""
        info = self._sessions.get(session_id)
        if info is not None:
            info.touch()
        return info

    def remove_session(self, session_id: str) -> bool:
        """Remove a session and clean up its frames directory."""
        import shutil
        from pathlib import Path
        info = self._sessions.pop(session_id, None)
        if info is None:
            return False
        # Use the stored frames_dir to find and delete the session directory
        session_dir = Path(info.frames_dir).parent
        if session_dir.exists():
            shutil.rmtree(session_dir, ignore_errors=True)
        logger.info(f"Removed session {session_id}")
        return True

    def get_active_count(self) -> int:
        """Return the number of active sessions."""
        return len(self._sessions)

    def get_gpu_status(self) -> dict:
        """Return current GPU memory status and active session count."""
        if torch.cuda.is_available():
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            free_mb = free_bytes / (1024 * 1024)
            total_mb = total_bytes / (1024 * 1024)
            used_mb = total_mb - free_mb
            used_pct = (used_mb / total_mb) * 100.0
        else:
            free_mb = total_mb = used_mb = used_pct = 0.0
        return {
            "active_sessions": len(self._sessions),
            "max_sessions": self._max_concurrent_sessions,
            "gpu_index": self._gpu_index,
            "gpu_free_mb": round(free_mb, 1),
            "gpu_total_mb": round(total_mb, 1),
            "gpu_used_mb": round(used_mb, 1),
            "gpu_used_pct": round(used_pct, 1),
        }

    async def start_cleanup_task(self):
        """Start the background cleanup task."""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def stop_cleanup_task(self):
        """Stop the background cleanup task."""
        if self._cleanup_task is not None and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None

    async def _cleanup_loop(self):
        """Background loop that cleans up expired sessions."""
        while True:
            await asyncio.sleep(CLEANUP_INTERVAL_SEC)
            now = time.time()
            expired_ids = [
                sid for sid, info in self._sessions.items()
                if (now - info.last_active_at) > SESSION_EXPIRATION_SEC
            ]
            for sid in expired_ids:
                logger.warning(f"Session {sid} expired, cleaning up")
                if self._on_session_expired is not None:
                    try:
                        await self._on_session_expired(sid)
                    except Exception as e:
                        logger.warning(
                            f"Expiry callback failed for {sid}: {e}; "
                            f"removing metadata anyway"
                        )
                        self.remove_session(sid)
                else:
                    self.remove_session(sid)
