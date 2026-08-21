// video-player.js - Frame loading, display, and navigation

class VideoPlayer {
    constructor(frameCanvas, overlayCanvas, drawCanvas) {
        this.frameCanvas = frameCanvas;
        this.overlayCanvas = overlayCanvas;
        this.drawCanvas = drawCanvas;
        this.frameCtx = frameCanvas.getContext("2d");
        this.overlayCtx = overlayCanvas.getContext("2d");
        this.drawCtx = drawCanvas.getContext("2d");

        this.sessionId = null;
        this.numFrames = 0;
        this.currentFrameIdx = 0;
        this.origWidth = 0;
        this.origHeight = 0;

        // Display dimensions (fit within viewport)
        this.displayWidth = 0;
        this.displayHeight = 0;

        // Frame image cache
        this._frameCache = new Map();
        this._maxCacheSize = 50;

        // Callbacks
        this.onFrameChange = null;

        // Loading state
        this._isLoading = false;
    }

    /**
     * Initialize the player with session metadata.
     */
    initSession(sessionId, numFrames, origHeight, origWidth) {
        this.sessionId = sessionId;
        this.numFrames = numFrames;
        this.origHeight = origHeight;
        this.origWidth = origWidth;
        this._frameCache.clear();

        // Calculate display dimensions to fit viewport
        this._fitToViewport();

        // Set canvas dimensions
        const w = this.displayWidth;
        const h = this.displayHeight;
        this.frameCanvas.width = w;
        this.frameCanvas.height = h;
        this.overlayCanvas.width = w;
        this.overlayCanvas.height = h;
        this.drawCanvas.width = w;
        this.drawCanvas.height = h;

        this.currentFrameIdx = 0;
    }

    /**
     * Calculate display dimensions to fit within the viewport while
     * maintaining aspect ratio.
     */
    _fitToViewport() {
        const viewport = document.getElementById("viewport");
        const maxW = viewport.clientWidth - 40;
        const maxH = viewport.clientHeight - 40;
        const aspect = this.origWidth / this.origHeight;

        if (maxW / aspect <= maxH) {
            this.displayWidth = Math.floor(maxW);
            this.displayHeight = Math.floor(maxW / aspect);
        } else {
            this.displayHeight = Math.floor(maxH);
            this.displayWidth = Math.floor(maxH * aspect);
        }
    }

    /**
     * Load and display a frame at the given index.
     * @param {number} frameIdx
     */
    async loadFrame(frameIdx) {
        if (this._isLoading) return;
        if (frameIdx < 0 || frameIdx >= this.numFrames) return;

        this._isLoading = true;
        this.currentFrameIdx = frameIdx;

        // Check cache first
        let img = this._frameCache.get(frameIdx);

        if (!img) {
            img = new Image();
            img.crossOrigin = "anonymous";
            img.src = `/api/frame/${this.sessionId}/${frameIdx}`;
            await new Promise((resolve, reject) => {
                img.onload = resolve;
                img.onerror = reject;
            });

            // Cache the image
            this._frameCache.set(frameIdx, img);
            // Prune cache if too large
            if (this._frameCache.size > this._maxCacheSize) {
                const oldestKey = this._frameCache.keys().next().value;
                this._frameCache.delete(oldestKey);
            }
        }

        // Draw the frame
        this.frameCtx.clearRect(0, 0, this.displayWidth, this.displayHeight);
        this.frameCtx.drawImage(img, 0, 0, this.displayWidth, this.displayHeight);

        // Clear overlay canvases
        this.overlayCtx.clearRect(0, 0, this.displayWidth, this.displayHeight);
        this.drawCtx.clearRect(0, 0, this.displayWidth, this.displayHeight);

        this._isLoading = false;

        if (this.onFrameChange) {
            this.onFrameChange(frameIdx);
        }
    }

    /**
     * Navigate to the next frame.
     */
    nextFrame() {
        if (this.currentFrameIdx < this.numFrames - 1) {
            this.loadFrame(this.currentFrameIdx + 1);
        }
    }

    /**
     * Navigate to the previous frame.
     */
    prevFrame() {
        if (this.currentFrameIdx > 0) {
            this.loadFrame(this.currentFrameIdx - 1);
        }
    }

    /**
     * Navigate to a specific frame.
     */
    goToFrame(frameIdx) {
        this.loadFrame(frameIdx);
    }

    /**
     * Get the current frame index.
     */
    getCurrentFrame() {
        return this.currentFrameIdx;
    }

    /**
     * Convert canvas display coordinates to normalized [0,1] coordinates.
     */
    toNormalized(x, y) {
        return {
            x: Math.max(0, Math.min(1, x / this.displayWidth)),
            y: Math.max(0, Math.min(1, y / this.displayHeight)),
        };
    }

    /**
     * Convert normalized [0,1] coordinates to canvas display coordinates.
     */
    fromNormalized(nx, ny) {
        return {
            x: nx * this.displayWidth,
            y: ny * this.displayHeight,
        };
    }

    /**
     * Show the canvas container (hide empty state).
     */
    show() {
        document.getElementById("canvas-container").classList.add("active");
        document.getElementById("viewport-empty").classList.add("hidden");
    }

    /**
     * Hide the canvas container (show empty state).
     */
    hide() {
        document.getElementById("canvas-container").classList.remove("active");
        document.getElementById("viewport-empty").classList.remove("hidden");
    }
}

// Export for global access
window.VideoPlayer = VideoPlayer;
