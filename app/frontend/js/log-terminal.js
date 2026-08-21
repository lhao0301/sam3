// log-terminal.js - Docked log terminal at the bottom-left of the page
//
// Shows key operation logs (upload progress, session lifecycle, prompt
// actions, propagation progress). The panel is docked flush against the
// page's left and bottom edges: its height reaches up to the video frame
// bottom and its width matches the tracklet sidebar's width (both synced
// by the app on layout changes).
//
// Levels: INFO (blue-gray) / OK (green) / WARN (orange) / ERR (red).
// Named progress entries update a single line in place instead of
// flooding the log stream.
//
// Every finalized line (plain logs + progress conclusions, not the
// in-place progress ticks) is also batched to POST /api/logs so the
// backend persists the terminal history to a local file. Batches flush
// on 10 lines, every 3s, or via sendBeacon on pagehide.

class LogTerminal {
    static MAX_LINES = 500;
    static BATCH_SIZE = 10;
    static BATCH_INTERVAL_MS = 3000;

    /**
     * @param {HTMLElement} root  #log-terminal container element
     */
    constructor(root) {
        this.root = root;
        this.body = root.querySelector(".lt-body");
        this.collapseBtn = root.querySelector(".lt-collapse");
        this.clearBtn = root.querySelector(".lt-clear");
        this.latestBtn = root.querySelector(".lt-latest");
        this.previewEl = root.querySelector(".lt-preview");

        // Named in-place progress entries: id -> line element
        this._progressEls = new Map();
        this._follow = true;
        this._collapsed = false;

        // Server persistence: reported context + pending batch
        this._ctx = { session_id: null };
        this._pending = [];
        this._flushTimer = null;

        this._bindEvents();
    }

    _bindEvents() {
        // Auto-follow only while the user sits at the bottom; scrolling
        // up pauses following so history can be inspected
        this.body.addEventListener("scroll", () => {
            const atBottom =
                this.body.scrollHeight - this.body.scrollTop - this.body.clientHeight < 8;
            this._setFollow(atBottom);
        });

        this.collapseBtn.addEventListener("click", () => this.toggleCollapse());
        this.clearBtn.addEventListener("click", () => this.clear());
        this.latestBtn.addEventListener("click", () => {
            this._setFollow(true);
            this.body.scrollTop = this.body.scrollHeight;
        });
    }

    _setFollow(on) {
        this._follow = on;
        this.latestBtn.classList.toggle("visible", on ? false : !this._collapsed);
    }

    /**
     * Dock the panel: top at `topPx` (video bottom), right edge at
     * `rightPx` (bottom-nav left). Both are page coordinates.
     */
    dock(topPx, rightPx) {
        const h = Math.max(72, window.innerHeight - topPx);
        const w = Math.max(150, rightPx);
        this.root.style.height = `${h}px`;
        this.root.style.width = `${w}px`;
    }

    toggleCollapse(force) {
        this._collapsed = force !== undefined ? force : !this._collapsed;
        this.root.classList.toggle("collapsed", this._collapsed);
        this.collapseBtn.textContent = this._collapsed ? "▸" : "▾";
        this.latestBtn.classList.remove("visible");
        if (!this._collapsed) {
            // Dropping the stale preview; the full body is visible again
            this.previewEl.textContent = "";
            if (this._follow) {
                this.body.scrollTop = this.body.scrollHeight;
            }
        }
    }

    clear() {
        this.body.innerHTML = "";
        this._progressEls.clear();
        this.previewEl.textContent = "";
    }

    /**
     * Set the reporting context (e.g. current session id) attached to
     * every persisted batch.
     */
    setContext(ctx) {
        this._ctx = ctx || { session_id: null };
    }

    _timestamp() {
        const d = new Date();
        const p = (n) => String(n).padStart(2, "0");
        return `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
    }

    // ── Server persistence ─────────────────────────────────────

    _queue(level, msg) {
        this._pending.push({ ts: this._timestamp(), level, msg });
        if (this._pending.length >= LogTerminal.BATCH_SIZE) {
            this._flush();
        } else if (!this._flushTimer) {
            this._flushTimer = setTimeout(() => {
                this._flushTimer = null;
                this._flush();
            }, LogTerminal.BATCH_INTERVAL_MS);
        }
    }

    _flush() {
        if (this._flushTimer) {
            clearTimeout(this._flushTimer);
            this._flushTimer = null;
        }
        if (this._pending.length === 0) return;
        const payload = JSON.stringify({
            session_id: this._ctx.session_id || null,
            lines: this._pending,
        });
        this._pending = [];
        fetch("/api/logs", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: payload,
        }).catch(() => {
            // Best effort persistence; failed batches are dropped
        });
    }

    /**
     * Flush the pending batch via sendBeacon (survives page unload).
     */
    flushBeacon() {
        if (this._flushTimer) {
            clearTimeout(this._flushTimer);
            this._flushTimer = null;
        }
        if (this._pending.length === 0) return;
        const payload = JSON.stringify({
            session_id: this._ctx.session_id || null,
            lines: this._pending,
        });
        this._pending = [];
        if (navigator.sendBeacon) {
            navigator.sendBeacon(
                "/api/logs",
                new Blob([payload], { type: "application/json" })
            );
        }
    }

    _makeLine(level, msg) {
        const line = document.createElement("div");
        line.className = "lt-line";
        line.innerHTML =
            `<span class="lt-ts">${this._timestamp()}</span>` +
            `<span class="lt-lv lt-${level.toLowerCase()}">[${level}]</span>` +
            `<span class="lt-msg"></span>`;
        line.querySelector(".lt-msg").textContent = msg;
        return line;
    }

    _append(line) {
        this.body.appendChild(line);
        // Cap the number of retained lines (progress entries age out too)
        while (this.body.childElementCount > LogTerminal.MAX_LINES) {
            const first = this.body.firstElementChild;
            for (const [id, el] of this._progressEls) {
                if (el === first) this._progressEls.delete(id);
            }
            this.body.removeChild(first);
        }
        if (this._follow && !this._collapsed) {
            this.body.scrollTop = this.body.scrollHeight;
        }
        // Collapsed: surface the newest message in the title bar preview
        if (this._collapsed) {
            const lv = line.querySelector(".lt-lv");
            const msg = line.querySelector(".lt-msg");
            this.previewEl.textContent = `${lv.textContent} ${msg.textContent}`;
        }
    }

    /**
     * Append one log line. level: "INFO" | "OK" | "WARN" | "ERR".
     */
    log(level, msg) {
        this._append(this._makeLine(level, msg));
        this._queue(level, msg);
    }

    /**
     * Start a named progress entry (own line, updated in place).
     */
    progressStart(id, msg) {
        this.progressEnd(id, "INFO", msg); // finalize any stale entry
        const line = this._makeLine("INFO", msg);
        line.classList.add("lt-progress");
        this._progressEls.set(id, line);
        this._append(line);
    }

    /**
     * Update a named progress entry in place (pct: 0..100).
     */
    progressUpdate(id, pct, msg) {
        let line = this._progressEls.get(id);
        if (!line) {
            line = this._makeLine("INFO", msg);
            line.classList.add("lt-progress");
            this._progressEls.set(id, line);
            this._append(line);
            return;
        }
        line.querySelector(".lt-ts").textContent = this._timestamp();
        const lv = line.querySelector(".lt-lv");
        lv.textContent = "[INFO]";
        lv.className = "lt-lv lt-info";
        const m = line.querySelector(".lt-msg");
        m.textContent = `${msg}  ${pct}%`;
        if (this._follow && !this._collapsed) {
            this.body.scrollTop = this.body.scrollHeight;
        }
    }

    /**
     * Finalize a named progress entry: the line becomes a normal log
     * line with the final level and message.
     */
    progressEnd(id, level, msg) {
        const line = this._progressEls.get(id);
        if (line) {
            line.querySelector(".lt-ts").textContent = this._timestamp();
            const lv = line.querySelector(".lt-lv");
            lv.textContent = `[${level}]`;
            lv.className = `lt-lv lt-${level.toLowerCase()}`;
            line.querySelector(".lt-msg").textContent = msg;
            line.classList.remove("lt-progress");
            this._progressEls.delete(id);
            if (this._follow && !this._collapsed) {
                this.body.scrollTop = this.body.scrollHeight;
            }
            // Persist the finalized conclusion (in-place ticks are not)
            this._queue(level, msg);
        }
    }
}

window.LogTerminal = LogTerminal;
