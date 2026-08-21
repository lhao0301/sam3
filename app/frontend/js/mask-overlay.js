// mask-overlay.js - Render mask overlays from propagation results

class MaskOverlay {
    constructor(overlayCanvas, videoPlayer) {
        this.overlayCanvas = overlayCanvas;
        this.overlayCtx = overlayCanvas.getContext("2d");
        this.player = videoPlayer;

        // Cache: frame_index -> Image object (from base64 PNG)
        this._maskCache = new Map();
        this._maxCacheSize = 500;

        // Current displayed mask image
        this._currentMaskImage = null;

        // Track which frames have results
        this._framesWithMasks = new Set();

        // Track object IDs seen
        this._objectIds = new Set();

        // Session ID for REST API calls
        this._sessionId = null;

        // Frames currently being fetched (to avoid duplicate requests)
        this._fetchingFrames = new Set();
    }

    /**
     * Clear the mask cache and state.
     */
    reset() {
        this._maskCache.clear();
        this._framesWithMasks.clear();
        this._objectIds.clear();
        this._currentMaskImage = null;
        this.overlayCtx.clearRect(
            0, 0, this.player.displayWidth, this.player.displayHeight
        );
    }

    /**
     * Set the session ID for REST API mask fetching.
     */
    setSession(sessionId) {
        this._sessionId = sessionId;
    }

    /**
     * Fetch mask for a frame from the server if not cached.
     * Returns true if a mask was found (cached or fetched).
     */
    async fetchAndDisplayMask(frameIdx) {
        // If already cached, just display
        if (this._maskCache.has(frameIdx)) {
            this.displayMask(frameIdx);
            return true;
        }

        // If we know this frame has no mask, skip
        // (but allow re-fetch if we haven't tried yet)

        // Avoid duplicate fetches
        if (this._fetchingFrames.has(frameIdx)) {
            return false;
        }

        if (!this._sessionId) {
            return false;
        }

        this._fetchingFrames.add(frameIdx);
        try {
            const resp = await fetch(
                `/api/mask/${this._sessionId}/${frameIdx}`
            );
            if (!resp.ok) {
                return false;
            }
            const data = await resp.json();

            if (data.obj_ids && data.obj_ids.length > 0 && data.mask_png) {
                this.storeMask(frameIdx, data.mask_png, data.obj_ids);
                this.displayMask(frameIdx);
                return true;
            } else {
                // No mask for this frame — clear overlay
                this.displayMask(frameIdx);
                return false;
            }
        } catch (e) {
            console.warn(`[MaskOverlay] Failed to fetch mask for frame ${frameIdx}:`, e);
            return false;
        } finally {
            this._fetchingFrames.delete(frameIdx);
        }
    }

    /**
     * Drop the cached mask for one frame (e.g. after an object removal),
     * so the next display call clears the overlay.
     */
    evictFrame(frameIdx) {
        this._maskCache.delete(frameIdx);
        this._framesWithMasks.delete(frameIdx);
    }

    /**
     * Drop all cached masks (objects changed, every frame may be stale).
     * Keeps the session binding intact.
     */
    evictAll() {
        this._maskCache.clear();
        this._framesWithMasks.clear();
        this._objectIds.clear();
        this._currentMaskImage = null;
        this.overlayCtx.clearRect(
            0, 0, this.player.displayWidth, this.player.displayHeight
        );
    }

    /**
     * Store a mask result from propagation for a given frame.
     * @param {number} frameIdx - Frame index
     * @param {string|null} maskPngBase64 - Base64-encoded PNG, or null
     * @param {Array} objIds - Object IDs in this frame
     */
    storeMask(frameIdx, maskPngBase64, objIds) {
        this._framesWithMasks.add(frameIdx);
        if (objIds) {
            for (const id of objIds) {
                this._objectIds.add(id);
            }
        }

        if (maskPngBase64) {
            // Preload the image and cache it
            const img = new Image();
            img.onload = () => {
                this._maskCache.set(frameIdx, img);
                // Prune cache
                if (this._maskCache.size > this._maxCacheSize) {
                    const oldestKey = this._maskCache.keys().next().value;
                    this._maskCache.delete(oldestKey);
                }
                // Decoding is asynchronous: callers that call displayMask()
                // right after storeMask() drew nothing (the cache entry
                // didn't exist yet). Re-render once the image is ready so the
                // mask actually shows on the current frame.
                if (this.player.getCurrentFrame() === frameIdx) {
                    this.displayMask(frameIdx);
                }
            };
            img.src = "data:image/png;base64," + maskPngBase64;
        }
    }

    /**
     * Display the mask overlay for a given frame.
     * If no mask is cached for this frame, clears the overlay.
     * @param {number} frameIdx
     */
    displayMask(frameIdx) {
        const img = this._maskCache.get(frameIdx);
        this._currentMaskImage = img;

        this.overlayCtx.clearRect(
            0, 0, this.player.displayWidth, this.player.displayHeight
        );

        if (img) {
            this.overlayCtx.drawImage(
                img, 0, 0,
                this.player.displayWidth, this.player.displayHeight
            );
        }
    }

    /**
     * Check if a frame has a mask result.
     */
    hasMask(frameIdx) {
        return this._framesWithMasks.has(frameIdx);
    }

    /**
     * Get all object IDs seen across all frames.
     */
    getObjectIds() {
        return Array.from(this._objectIds).sort((a, b) => a - b);
    }

    /**
     * Get the total number of frames with masks.
     */
    getMaskedFrameCount() {
        return this._framesWithMasks.size;
    }

    /**
     * Get color info for an object ID (must match backend palette).
     */
    getObjectColor(objId) {
        const palette = [
            { r: 255, g: 0, b: 0 },      // red
            { r: 0, g: 255, b: 0 },      // green
            { r: 0, g: 0, b: 255 },      // blue
            { r: 0, g: 255, b: 255 },    // cyan
            { r: 255, g: 0, b: 255 },    // magenta
            { r: 255, g: 255, b: 0 },    // yellow
            { r: 128, g: 0, b: 0 },      // maroon
            { r: 0, g: 128, b: 0 },      // dark green
            { r: 0, g: 0, b: 128 },      // navy
            { r: 128, g: 128, b: 0 },    // olive
            { r: 128, g: 0, b: 128 },    // purple
            { r: 0, g: 128, b: 128 },    // teal
            { r: 255, g: 128, b: 0 },    // orange
            { r: 128, g: 255, b: 0 },    // lime
            { r: 0, g: 128, b: 255 },    // sky blue
            { r: 255, g: 0, b: 128 },    // pink
            { r: 128, g: 255, b: 255 },  // light yellow
            { r: 255, g: 128, b: 128 },  // light red
            { r: 128, g: 128, b: 255 },  // light blue
            { r: 200, g: 200, b: 200 },  // gray
        ];
        const c = palette[objId % palette.length];
        return { ...c, hex: `#${c.r.toString(16).padStart(2, "0")}${c.g.toString(16).padStart(2, "0")}${c.b.toString(16).padStart(2, "0")}` };
    }
}

// Export for global access
window.MaskOverlay = MaskOverlay;
