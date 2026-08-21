// prompt-canvas.js - Canvas mouse interaction for point / box prompts
//
// Manages the *pending* (not yet submitted) prompt for one target at a
// time. Modes are mutually exclusive; switching modes clears the pending
// content. Text prompts are entered in the toolbar, so the canvas ignores
// mouse events in text mode.

class PromptCanvas {
    constructor(drawCanvas, videoPlayer) {
        this.drawCanvas = drawCanvas;
        this.drawCtx = drawCanvas.getContext("2d");
        this.player = videoPlayer;

        this.mode = "box";       // "box" | "point" | "text"
        this.pointType = 1;      // 1 = positive, 0 = negative

        // Pending prompt content (normalized [0,1] coordinates)
        this.pendingBox = null;   // { x, y, w, h } — single box, new box replaces old
        this.pendingPoints = [];  // [{ x, y, label }] — one group describing one target

        // Box dragging state (display coordinates)
        this._isDrawing = false;
        this._startX = 0;
        this._startY = 0;
        this._currentX = 0;
        this._currentY = 0;

        // Callbacks
        this.onChange = null;

        this._bindEvents();
    }

    /**
     * Switch prompt mode; clears the pending content (point and box are
     * mutually exclusive).
     */
    setMode(mode) {
        if (!["box", "point", "text"].includes(mode)) return;
        this.mode = mode;
        this.clearPending();
    }

    /**
     * Set the point type to add on click (1 = positive, 0 = negative).
     */
    setPointType(label) {
        this.pointType = label ? 1 : 0;
    }

    /**
     * Clear the pending prompt content.
     */
    clearPending() {
        this.pendingBox = null;
        this.pendingPoints = [];
        this._isDrawing = false;
        if (this.onChange) this.onChange();
        this._render();
    }

    /**
     * Remove the most recently added point.
     */
    undoLastPoint() {
        this.pendingPoints.pop();
        if (this.onChange) this.onChange();
        this._render();
    }

    /**
     * Whether there is pending prompt content to submit (text handled by app).
     */
    hasPending() {
        if (this.mode === "box") return this.pendingBox !== null;
        if (this.mode === "point") return this.pendingPoints.length > 0;
        return false;
    }

    /**
     * Get the pending prompt data for API submission (normalized).
     * Returns null if nothing pending; returns { type: "text" } for text mode.
     */
    getPending() {
        if (this.mode === "box") {
            if (!this.pendingBox) return null;
            const b = this.pendingBox;
            return { type: "box", boxes: [[b.x, b.y, b.w, b.h]] };
        }
        if (this.mode === "point") {
            if (this.pendingPoints.length === 0) return null;
            return {
                type: "point",
                points: this.pendingPoints.map((p) => [p.x, p.y]),
                pointLabels: this.pendingPoints.map((p) => p.label),
            };
        }
        return { type: "text" };
    }

    // ─── Event handling ──────────────────────────────────────────────

    _bindEvents() {
        this.drawCanvas.addEventListener("mousedown", (e) => this._onMouseDown(e));
        this.drawCanvas.addEventListener("mousemove", (e) => this._onMouseMove(e));
        this.drawCanvas.addEventListener("mouseup", (e) => this._onMouseUp(e));
        this.drawCanvas.addEventListener("mouseleave", (e) => this._onMouseUp(e));
    }

    _getMousePos(e) {
        const rect = this.drawCanvas.getBoundingClientRect();
        return {
            x: e.clientX - rect.left,
            y: e.clientY - rect.top,
        };
    }

    _onMouseDown(e) {
        if (this.mode === "text") return;
        const pos = this._getMousePos(e);

        if (this.mode === "point") {
            // Add a point of the currently selected type
            const norm = this.player.toNormalized(pos.x, pos.y);
            this.pendingPoints.push({ x: norm.x, y: norm.y, label: this.pointType });
            if (this.onChange) this.onChange();
            this._render();
            return;
        }

        // box mode: start dragging a new box (replaces the old one on release)
        this._isDrawing = true;
        this._startX = pos.x;
        this._startY = pos.y;
        this._currentX = pos.x;
        this._currentY = pos.y;
        this._render();
    }

    _onMouseMove(e) {
        if (!this._isDrawing || this.mode !== "box") return;
        const pos = this._getMousePos(e);
        this._currentX = pos.x;
        this._currentY = pos.y;
        this._render();
    }

    _onMouseUp(e) {
        if (!this._isDrawing || this.mode !== "box") return;
        this._isDrawing = false;

        const pos = this._getMousePos(e);
        this._currentX = pos.x;
        this._currentY = pos.y;

        const x = Math.min(this._startX, this._currentX);
        const y = Math.min(this._startY, this._currentY);
        const w = Math.abs(this._currentX - this._startX);
        const h = Math.abs(this._currentY - this._startY);

        // Only accept the box if it is large enough (at least 5x5 pixels)
        if (w >= 5 && h >= 5) {
            const norm = this.player.toNormalized(x, y);
            this.pendingBox = {
                x: norm.x,
                y: norm.y,
                w: w / this.player.displayWidth,
                h: h / this.player.displayHeight,
            };
            if (this.onChange) this.onChange();
        }

        this._render();
    }

    // ─── Rendering ───────────────────────────────────────────────────

    _render() {
        const ctx = this.drawCtx;
        const W = this.player.displayWidth;
        const H = this.player.displayHeight;
        ctx.clearRect(0, 0, W, H);

        // Pending box: solid yellow rect
        if (this.mode === "box" && this.pendingBox) {
            const b = this.pendingBox;
            const p = this.player.fromNormalized(b.x, b.y);
            ctx.strokeStyle = "#ffd400";
            ctx.lineWidth = 2;
            ctx.strokeRect(p.x, p.y, b.w * W, b.h * H);
        }

        // Pending points: green filled circle (+) / red cross (−)
        if (this.mode === "point") {
            for (const pt of this.pendingPoints) {
                const p = this.player.fromNormalized(pt.x, pt.y);
                if (pt.label === 1) {
                    ctx.fillStyle = "#00e676";
                    ctx.strokeStyle = "#ffffff";
                    ctx.lineWidth = 1.5;
                    ctx.beginPath();
                    ctx.arc(p.x, p.y, 6, 0, Math.PI * 2);
                    ctx.fill();
                    ctx.stroke();
                } else {
                    ctx.strokeStyle = "#ff1744";
                    ctx.lineWidth = 2.5;
                    const r = 6;
                    ctx.beginPath();
                    ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
                    ctx.stroke();
                    ctx.beginPath();
                    ctx.moveTo(p.x - r, p.y - r);
                    ctx.lineTo(p.x + r, p.y + r);
                    ctx.moveTo(p.x + r, p.y - r);
                    ctx.lineTo(p.x - r, p.y + r);
                    ctx.stroke();
                }
            }
        }

        // Box dragging preview: dashed yellow rect
        if (this._isDrawing && this.mode === "box") {
            const x = Math.min(this._startX, this._currentX);
            const y = Math.min(this._startY, this._currentY);
            const w = Math.abs(this._currentX - this._startX);
            const h = Math.abs(this._currentY - this._startY);
            ctx.strokeStyle = "#ffff00";
            ctx.lineWidth = 2;
            ctx.setLineDash([5, 5]);
            ctx.strokeRect(x, y, w, h);
            ctx.setLineDash([]);
        }
    }

    // ─── Enable / disable ────────────────────────────────────────────

    disable() {
        this.drawCanvas.style.pointerEvents = "none";
        this.drawCanvas.style.cursor = "default";
    }

    enable() {
        this.drawCanvas.style.pointerEvents = "auto";
        this.drawCanvas.style.cursor = "crosshair";
    }
}

// Export for global access
window.PromptCanvas = PromptCanvas;
