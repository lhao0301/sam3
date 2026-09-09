// prompt-canvas.js - Canvas mouse interaction for point / box prompts
//
// Manages the *pending* (not yet submitted) prompt for one target.
// Points and a box may coexist in the pending set (SAM2-task mode
// accepts combined point+box prompts); switching tools keeps the pending
// content so a box can be drawn first and then refined with clicks.
//
// Pending content is anchored to the frame it was drawn on
// (pendingFrameIdx): after switching frames it stays hidden and cannot
// be hit-tested, so a box drawn on another frame neither "reappears"
// under the cursor nor gets silently submitted against the wrong frame.
//
// Existing prompt elements can be selected (click a point or the box edge)
// and deleted with the Delete / Backspace key — no need for the undo
// button. Escape drops the selection.

class PromptCanvas {
    constructor(drawCanvas, videoPlayer) {
        this.drawCanvas = drawCanvas;
        this.drawCtx = drawCanvas.getContext("2d");
        this.player = videoPlayer;

        this.mode = "box";       // "box" | "point" | "text"
        this.pointType = 1;      // 1 = positive, 0 = negative

        // Pending prompt content (normalized [0,1] coordinates), all
        // belonging to pendingFrameIdx (the frame it was drawn on).
        this.pendingBox = null;   // { x, y, w, h } — single box, new box replaces old
        this.pendingPoints = [];  // [{ x, y, label }] — one group describing one target
        this.pendingFrameIdx = null; // null when nothing is pending

        // Box dragging state (display coordinates)
        this._isDrawing = false;
        this._startX = 0;
        this._startY = 0;
        this._currentX = 0;
        this._currentY = 0;

        // Selection state for delete-with-keyboard: a clicked element is
        // locked in _selected (persistent highlight); _hovered is the
        // transient highlight under the cursor. Shape: {type:"point",index}
        // or {type:"box"}.
        this._enabled = false;
        this._selected = null;
        this._hovered = null;

        // Callbacks
        this.onChange = null;

        this._bindEvents();
    }

    /**
     * Switch the active tool; the pending content is kept so points and
     * a box can be combined before submission.
     */
    setMode(mode) {
        if (!["box", "point"].includes(mode)) return;
        this.mode = mode;
        this._render();
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
        this.pendingFrameIdx = null;
        this._isDrawing = false;
        this._selected = null;
        this._hovered = null;
        if (this.onChange) this.onChange();
        this._render();
    }

    /**
     * The frame index the pending prompt belongs to (null when empty).
     */
    getPendingFrameIdx() {
        return this.pendingFrameIdx;
    }

    /**
     * Anchor new pending content to the current frame. Starting content
     * on a different frame discards the old pending set: one submission
     * carries a single frame_idx, so prompts from two frames can never
     * combine.
     */
    _syncPendingFrame() {
        const cur = this.player.getCurrentFrame();
        if (this.pendingFrameIdx !== null && this.pendingFrameIdx !== cur) {
            this.pendingBox = null;
            this.pendingPoints = [];
            this._selected = null;
            this._hovered = null;
        }
        this.pendingFrameIdx = cur;
    }

    /**
     * Frame switch notification: the drawing canvas was cleared by the
     * player, so re-render. Pending content from another frame stays
     * hidden (see _pendingVisible).
     */
    notifyFrameChanged() {
        this._hovered = null;
        this._render();
    }

    /** Whether the pending content should be drawn / hit-tested now. */
    _pendingVisible() {
        return this.pendingFrameIdx === null ||
            this.player.getCurrentFrame() === this.pendingFrameIdx;
    }

    /**
     * Remove the most recently added point.
     */
    undoLastPoint() {
        this.pendingPoints.pop();
        this._selected = null;
        this._hovered = null;
        if (!this.hasPending()) this.pendingFrameIdx = null;
        if (this.onChange) this.onChange();
        this._render();
    }

    /**
     * Whether there is pending prompt content to submit (pending box,
     * pending points, or both).
     */
    hasPending() {
        return this.pendingBox !== null || this.pendingPoints.length > 0;
    }

    /**
     * Get the pending prompt data for API submission (normalized).
     * Returns null if nothing pending; box and points are combined
     * when both exist.
     */
    getPending() {
        const hasBox = this.pendingBox !== null;
        const hasPoints = this.pendingPoints.length > 0;
        if (!hasBox && !hasPoints) return null;
        const out = { points: null, pointLabels: null, box: null };
        if (hasPoints) {
            out.points = this.pendingPoints.map((p) => [p.x, p.y]);
            out.pointLabels = this.pendingPoints.map((p) => p.label);
        }
        if (hasBox) {
            const b = this.pendingBox;
            out.box = [b.x, b.y, b.w, b.h];
        }
        return out;
    }

    // ─── Event handling ──────────────────────────────────────────────

    _bindEvents() {
        this.drawCanvas.addEventListener("mousedown", (e) => this._onMouseDown(e));
        this.drawCanvas.addEventListener("mousemove", (e) => this._onMouseMove(e));
        this.drawCanvas.addEventListener("mouseup", (e) => this._onMouseUp(e));
        this.drawCanvas.addEventListener("mouseleave", (e) => this._onMouseUp(e));
        document.addEventListener("keydown", (e) => this._onKeyDown(e));
    }

    _getMousePos(e) {
        const rect = this.drawCanvas.getBoundingClientRect();
        return {
            x: e.clientX - rect.left,
            y: e.clientY - rect.top,
        };
    }

    /**
     * Which pending element (if any) is under the given display-space
     * position. Points are tested first (last drawn wins), then the box
     * *edge band* — the box interior is left free for drawing a new box
     * or placing clicks.
     */
    _hitTest(pos) {
        // Pending content is anchored to its frame: on any other frame
        // there is nothing to select (this is what keeps a box drawn on
        // another frame from "reappearing" under the cursor there).
        if (!this._pendingVisible()) return null;
        const W = this.player.displayWidth;
        const H = this.player.displayHeight;
        const POINT_HIT_R = 9;
        for (let i = this.pendingPoints.length - 1; i >= 0; i--) {
            const pt = this.pendingPoints[i];
            const p = this.player.fromNormalized(pt.x, pt.y);
            const dx = pos.x - p.x;
            const dy = pos.y - p.y;
            if (dx * dx + dy * dy <= POINT_HIT_R * POINT_HIT_R) {
                return { type: "point", index: i };
            }
        }
        if (this.pendingBox) {
            const b = this.pendingBox;
            const p = this.player.fromNormalized(b.x, b.y);
            const bw = b.w * W;
            const bh = b.h * H;
            const tol = 6;
            const inOuter = pos.x >= p.x - tol && pos.x <= p.x + bw + tol &&
                pos.y >= p.y - tol && pos.y <= p.y + bh + tol;
            const inInner = pos.x >= p.x + tol && pos.x <= p.x + bw - tol &&
                pos.y >= p.y + tol && pos.y <= p.y + bh - tol;
            if (inOuter && !inInner) return { type: "box" };
        }
        return null;
    }

    /** Whether a selection reference ({type,index}) matches another one. */
    _matchSel(sel, ref) {
        if (!sel || !ref) return false;
        if (sel.type !== ref.type) return false;
        if (sel.type === "point") return sel.index === ref.index;
        return true;
    }

    _onKeyDown(e) {
        if (!this._enabled) return;
        const tag = (e.target && e.target.tagName || "").toLowerCase();
        if (tag === "input" || tag === "select" || tag === "textarea") return;
        if (e.key === "Escape") {
            if (this._selected) {
                this._selected = null;
                this._render();
            }
            return;
        }
        if ((e.key === "Delete" || e.key === "Backspace") && this._selected) {
            e.preventDefault();
            const sel = this._selected;
            this._selected = null;
            this._hovered = null;
            if (sel.type === "box") {
                this.pendingBox = null;
            } else {
                this.pendingPoints.splice(sel.index, 1);
            }
            if (!this.hasPending()) this.pendingFrameIdx = null;
            if (this.onChange) this.onChange();
            this._render();
        }
    }

    _onMouseDown(e) {
        const pos = this._getMousePos(e);

        // Clicking an existing point or the box edge selects that element
        // (Delete removes it) instead of adding new prompt content.
        const hit = this._hitTest(pos);
        if (hit) {
            this._selected = hit;
            this._render();
            return;
        }
        this._selected = null;

        if (this.mode === "point") {
            // Add a point of the currently selected type (anchored to the
            // current frame; drops pending content from another frame)
            this._syncPendingFrame();
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
        if (this._isDrawing && this.mode === "box") {
            const pos = this._getMousePos(e);
            this._currentX = pos.x;
            this._currentY = pos.y;
            this._render();
            return;
        }
        if (!this._enabled) return;
        // Hover highlight + pointer cursor over selectable elements
        const pos = this._getMousePos(e);
        const hit = this._hitTest(pos);
        const changed = JSON.stringify(hit) !== JSON.stringify(this._hovered);
        this._hovered = hit;
        this.drawCanvas.style.cursor = hit ? "pointer" : "crosshair";
        if (changed) this._render();
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
            // Anchor to the current frame (drops pending content from
            // another frame — one submission cannot mix two frames)
            this._syncPendingFrame();
            const norm = this.player.toNormalized(x, y);
            this.pendingBox = {
                x: norm.x,
                y: norm.y,
                w: w / this.player.displayWidth,
                h: h / this.player.displayHeight,
            };
            this._selected = null;
            this._hovered = null;
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

        // Pending content renders only on the frame it belongs to
        const showPending = this._pendingVisible();

        // Pending box: solid yellow rect (highlighted when selected/hovered)
        if (this.pendingBox && showPending) {
            const b = this.pendingBox;
            const p = this.player.fromNormalized(b.x, b.y);
            const hl = this._matchSel(this._selected, { type: "box" }) ||
                this._matchSel(this._hovered, { type: "box" });
            if (hl) {
                ctx.strokeStyle = "#ffffff";
                ctx.lineWidth = 4;
                ctx.strokeRect(p.x - 2, p.y - 2, b.w * W + 4, b.h * H + 4);
            }
            ctx.strokeStyle = hl ? "#ffea00" : "#ffd400";
            ctx.lineWidth = hl ? 3 : 2;
            ctx.strokeRect(p.x, p.y, b.w * W, b.h * H);
        }

        // Pending points: green filled circle (+) / red cross (−);
        // selected/hovered points get an outer white ring.
        if (showPending) {
            for (let i = 0; i < this.pendingPoints.length; i++) {
                const pt = this.pendingPoints[i];
                const p = this.player.fromNormalized(pt.x, pt.y);
                const hl = this._matchSel(this._selected, { type: "point", index: i }) ||
                    this._matchSel(this._hovered, { type: "point", index: i });
                if (hl) {
                    ctx.strokeStyle = "#ffffff";
                    ctx.lineWidth = 2;
                    ctx.beginPath();
                    ctx.arc(p.x, p.y, 10, 0, Math.PI * 2);
                    ctx.stroke();
                }
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
        this._enabled = false;
        this._selected = null;
        this._hovered = null;
        this.drawCanvas.style.pointerEvents = "none";
        this.drawCanvas.style.cursor = "default";
        this._render();
    }

    enable() {
        this._enabled = true;
        this.drawCanvas.style.pointerEvents = "auto";
        this.drawCanvas.style.cursor = "crosshair";
    }
}

// Export for global access
window.PromptCanvas = PromptCanvas;
