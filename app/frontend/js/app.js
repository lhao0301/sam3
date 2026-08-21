// app.js - Main application logic for SAM3 video annotation system
//
// Interaction flow:
//   upload → (encode) → navigate frames → pick prompt mode (text/point/box)
//   → draw prompt → segment preview (== commit with tracklet id)
//   → confirm (keep) / undo (remove_object) → accumulate tracklets
//   → propagate (forward / backward / both) → playback preview

class AnnotationApp {
    constructor() {
        // Application state
        this.sessionId = null;
        this.numFrames = 0;
        this.videoPath = null;       // Path returned by upload endpoint
        this.videoFilename = null;
        this.isPropagating = false;

        // Prompt state
        this.promptMode = "box";     // "text" | "point" | "box"
        this.pointType = 1;          // 1 = positive, 0 = negative
        this.tracklets = new Map();  // id -> {id, className, color, promptType, frameIdx}
        this.pendingObjIds = [];     // obj ids from last preview, awaiting confirm/undo
        this.pendingFrameIdx = -1;

        // Filmstrip state
        this._filmThumbs = [];       // [{ frameIdx, el }]
        this._filmStep = 1;
        this._lastFilmTs = 0;

        // Playback state
        this._isPlaying = false;
        this._playTimer = null;
        this._playFps = 8;

        // Components (initialized in init())
        this.player = null;
        this.promptCanvas = null;
        this.maskOverlay = null;
        this.wsClient = null;
        this.logTerminal = null;   // docked bottom-left log panel

        // Throttle timestamp for propagation progress log lines
        this._lastPropLogTs = 0;

        // UI elements
        this.els = {};

        // Status poll interval
        this._statusInterval = null;
    }

    /**
     * Initialize the application: grab DOM elements, set up components, bind events.
     */
    init() {
        this._cacheElements();
        this._initComponents();
        this._bindEvents();
        this._startStatusPolling();
        this._initLogTerminal();
        console.log("[App] Initialized");
    }

    /**
     * Create the docked log terminal and place it in the blank strip
     * left of the bottom nav, below the video frame.
     */
    _initLogTerminal() {
        this.logTerminal = new LogTerminal(
            document.getElementById("log-terminal")
        );
        this._dockLogTerminal();
        this.logTerminal.log("INFO", "系统就绪，等待上传视频");
    }

    /**
     * Re-dock the log terminal: top edge at the video frame bottom (or
     * the bottom nav top before any video is loaded); the right edge
     * aligns with the tracklet sidebar's right edge so the terminal
     * width always equals the sidebar width.
     */
    _dockLogTerminal() {
        if (!this.logTerminal) return;
        const right = this.els.resultSidebar.getBoundingClientRect().right;
        let top;
        if (this.sessionId && this.player.displayWidth > 0) {
            top = this.els.canvasContainer.getBoundingClientRect().bottom;
        } else {
            top = this.els.bottomNav.getBoundingClientRect().top;
        }
        this.logTerminal.dock(top, right);
    }

    /**
     * Cache all DOM element references.
     */
    _cacheElements() {
        this.els = {
            // Upload
            fileInput: document.getElementById("video-file-input"),
            btnUpload: document.getElementById("btn-upload"),
            uploadStatus: document.getElementById("upload-status"),
            videoInfo: document.getElementById("video-info"),
            // Encoding
            btnEncode: document.getElementById("btn-encode"),
            encodeStatus: document.getElementById("encode-status"),
            // Navigation
            btnPlay: document.getElementById("btn-play"),
            btnPrev: document.getElementById("btn-prev-frame"),
            btnNext: document.getElementById("btn-next-frame"),
            frameSlider: document.getElementById("frame-slider"),
            frameIdxLabel: document.getElementById("frame-index-label"),
            frameTotalLabel: document.getElementById("frame-total-label"),
            filmstrip: document.getElementById("filmstrip"),
            navInner: document.getElementById("nav-inner"),
            // Prompt tools (text input + point+/-/box buttons)
            textInput: document.getElementById("text-prompt-input"),
            btnPointPositive: document.getElementById("btn-point-positive"),
            btnPointNegative: document.getElementById("btn-point-negative"),
            btnBoxTool: document.getElementById("btn-box-tool"),
            btnUndoPoint: document.getElementById("btn-undo-point"),
            btnClearPoints: document.getElementById("btn-clear-points"),
            // Confirm row
            classNameInput: document.getElementById("class-name-input"),
            trackletIdInput: document.getElementById("tracklet-id-input"),
            btnSegmentPreview: document.getElementById("btn-segment-preview"),
            btnConfirmPrompt: document.getElementById("btn-confirm-prompt"),
            btnUndoPrompt: document.getElementById("btn-undo-prompt"),
            promptStatus: document.getElementById("prompt-status"),
            // Propagation
            directionSelect: document.getElementById("propagation-direction"),
            btnPropagate: document.getElementById("btn-propagate"),
            btnCancel: document.getElementById("btn-cancel"),
            progressFill: document.getElementById("progress-fill"),
            progressText: document.getElementById("progress-text"),
            // Status bar
            gpuStatus: document.getElementById("gpu-status"),
            sessionStatus: document.getElementById("session-status"),
            // Canvases
            frameCanvas: document.getElementById("frame-canvas"),
            overlayCanvas: document.getElementById("overlay-canvas"),
            drawCanvas: document.getElementById("draw-canvas"),
            canvasContainer: document.getElementById("canvas-container"),
            viewportEmpty: document.getElementById("viewport-empty"),
            // Results
            resultInfo: document.getElementById("result-info"),
            trackletList: document.getElementById("tracklet-list"),
            resultSidebar: document.getElementById("result-sidebar"),
            // Log terminal docking anchors
            bottomNav: document.getElementById("bottom-nav"),
            // Error modal
            errorModal: document.getElementById("error-modal"),
            modalTitle: document.getElementById("modal-title"),
            modalText: document.getElementById("modal-text"),
            btnModalClose: document.getElementById("btn-modal-close"),
        };
    }

    /**
     * Initialize player, prompt canvas, and mask overlay components.
     */
    _initComponents() {
        this.player = new VideoPlayer(
            this.els.frameCanvas,
            this.els.overlayCanvas,
            this.els.drawCanvas
        );

        this.promptCanvas = new PromptCanvas(
            this.els.drawCanvas,
            this.player
        );

        this.maskOverlay = new MaskOverlay(
            this.els.overlayCanvas,
            this.player
        );

        // When frame changes: update labels, filmstrip highlight, and mask
        this.player.onFrameChange = (frameIdx) => {
            this.els.frameIdxLabel.textContent = frameIdx;
            this.els.frameSlider.value = frameIdx;
            this._updateFilmstripHighlight(frameIdx);

            if (this.sessionId) {
                this.maskOverlay.fetchAndDisplayMask(frameIdx);
            }
        };

        // When pending prompt content changes, refresh button states
        this.promptCanvas.onChange = () => this._updatePromptButtons();
    }

    /**
     * Bind all event listeners.
     */
    _bindEvents() {
        // One-click upload: picking a file uploads it and starts the session
        this.els.fileInput.addEventListener("change", () => {
            if (this.els.fileInput.files[0]) {
                this._handleUpload();
            }
        });

        // Feature encoding
        if (this.els.btnEncode) {
            this.els.btnEncode.addEventListener("click", () => this._handleEncode());
        }

        // Playback / navigation
        this.els.btnPlay.addEventListener("click", () => this._togglePlay());
        this.els.btnPrev.addEventListener("click", () => this.player.prevFrame());
        this.els.btnNext.addEventListener("click", () => this.player.nextFrame());
        this.els.frameSlider.addEventListener("input", (e) => {
            this.player.loadFrame(parseInt(e.target.value));
        });

        // Keyboard shortcuts: space = play/pause, arrows = step frames
        window.addEventListener("keydown", (e) => {
            if (!this.sessionId || this.isPropagating) return;
            const tag = (e.target.tagName || "").toLowerCase();
            if (tag === "input" || tag === "select" || tag === "textarea") return;
            if (e.code === "Space") {
                e.preventDefault();
                this._togglePlay();
            } else if (e.code === "ArrowLeft") {
                e.preventDefault();
                this.player.prevFrame();
            } else if (e.code === "ArrowRight") {
                e.preventDefault();
                this.player.nextFrame();
            }
        });

        // Prompt tools: mutually exclusive point+ / point- / box; focusing
        // the text input switches to text mode
        this.els.btnPointPositive.addEventListener("click", () => {
            this._setPromptMode("point");
            this._setPointType(1);
        });
        this.els.btnPointNegative.addEventListener("click", () => {
            this._setPromptMode("point");
            this._setPointType(0);
        });
        this.els.btnBoxTool.addEventListener("click", () => this._setPromptMode("box"));
        this.els.btnUndoPoint.addEventListener("click", () => this.promptCanvas.undoLastPoint());
        this.els.btnClearPoints.addEventListener("click", () => this.promptCanvas.clearPending());

        // Text input: focusing it switches to text mode; typing updates buttons
        this.els.textInput.addEventListener("focus", () => this._setPromptMode("text"));
        this.els.textInput.addEventListener("input", () => this._updatePromptButtons());

        // Segment preview / confirm / undo
        this.els.btnSegmentPreview.addEventListener("click", () => this._handleSegmentPreview());
        this.els.btnConfirmPrompt.addEventListener("click", () => this._handleConfirmPrompt());
        this.els.btnUndoPrompt.addEventListener("click", () => this._handleUndoPrompt());

        // Propagation
        this.els.btnPropagate.addEventListener("click", () => this._handlePropagate());
        this.els.btnCancel.addEventListener("click", () => this._handleCancel());

        // Window resize: re-fit viewport
        let resizeTimer = null;
        window.addEventListener("resize", () => {
            clearTimeout(resizeTimer);
            resizeTimer = setTimeout(() => {
                if (this.sessionId) {
                    this.player._fitToViewport();
                    const w = this.player.displayWidth;
                    const h = this.player.displayHeight;
                    this.els.frameCanvas.width = w;
                    this.els.frameCanvas.height = h;
                    this.els.overlayCanvas.width = w;
                    this.els.overlayCanvas.height = h;
                    this.els.drawCanvas.width = w;
                    this.els.drawCanvas.height = h;
                    this.player.loadFrame(this.player.getCurrentFrame());
                    this._syncBottomNavWidth();
                    this._buildFilmstrip();
                }
                this._dockLogTerminal();
            }, 200);
        });

        // Error modal: close via button, backdrop click, or Escape
        this.els.btnModalClose.addEventListener("click", () => this._closeErrorModal());
        this.els.errorModal.addEventListener("click", (e) => {
            if (e.target === this.els.errorModal) this._closeErrorModal();
        });
        window.addEventListener("keydown", (e) => {
            if (e.key === "Escape" && !this.els.errorModal.hidden) {
                this._closeErrorModal();
            }
        });

        // Page closed / navigated away: release the backend session
        // immediately (SAM3 state, prompt history, frame files) so
        // reopening the browser starts from a clean slate. sendBeacon
        // survives page unload, unlike fetch. pagehide also covers tab
        // close, browser quit and refresh (refresh starts a new session).
        // Page closed / navigated away: flush pending terminal logs to
        // the server (beacon survives unload) BEFORE closing the session
        // so the final batch is tagged with the right session id.
        window.addEventListener("pagehide", () => {
            if (this.logTerminal) this.logTerminal.flushBeacon();
            if (this.sessionId && navigator.sendBeacon) {
                navigator.sendBeacon(`/api/session/${this.sessionId}/close`);
            }
        });
    }

    // ─── Upload & Session Start ──────────────────────────────────────

    // The upload button is a <label>, which has no disabled attribute
    _setUploadEnabled(enabled) {
        this.els.btnUpload.classList.toggle("disabled", !enabled);
    }

    // Upload button visual state: primary (default) / success / danger
    _setUploadState(cls) {
        this.els.btnUpload.classList.remove("btn-primary", "btn-success", "btn-danger");
        this.els.btnUpload.classList.add(cls);
    }

    _showErrorModal(title, message) {
        this.els.modalTitle.textContent = title;
        this.els.modalText.textContent = message;
        this.els.errorModal.hidden = false;
    }

    _closeErrorModal() {
        this.els.errorModal.hidden = true;
    }

    // Align the bottom navigation width with the displayed video width,
    // and align the filmstrip row with the slider track (the buttons stay
    // on the slider's row, so the strip must be indented to the slider's
    // left edge and share its exact width).
    _syncBottomNavWidth() {
        if (!this.sessionId) return;
        const w = Math.round(this.els.canvasContainer.getBoundingClientRect().width);
        this.els.navInner.style.width = `${w}px`;

        const navRect = this.els.navInner.getBoundingClientRect();
        const sRect = this.els.frameSlider.getBoundingClientRect();
        const left = Math.round(sRect.left - navRect.left);
        this.els.filmstrip.style.marginLeft = `${left}px`;
        this.els.filmstrip.style.width = `${Math.round(sRect.width)}px`;

        // The terminal shares the same layout anchors as the nav column
        this._dockLogTerminal();
    }

    async _handleUpload() {
        const file = this.els.fileInput.files[0];
        if (!file) return;

        this._setUploadEnabled(false);
        this._setUploadState("btn-primary");
        this.els.btnUpload.textContent = "上传中…";

        // A fresh upload replaces any previous session: close it
        // server-side first so its GPU state and frames are released
        // immediately instead of lingering until the idle timeout.
        if (this.sessionId) {
            const oldSid = this.sessionId;
            this.sessionId = null;
            this.logTerminal.log("INFO", "新上传替换旧会话，正在关闭...");
            try {
                await fetch(`/api/session/${oldSid}`, { method: "DELETE" });
                this.logTerminal.log("OK", `旧会话 ${oldSid.substring(0, 8)} 已关闭`);
            } catch (e) {
                // Best effort: the idle cleanup will catch it anyway
                this.logTerminal.log("WARN", `旧会话关闭失败（将由超时清理）: ${e.message}`);
            }
        }

        try {
            const formData = new FormData();
            formData.append("file", file);

            const sizeMb = (file.size / 1024 / 1024).toFixed(1);
            this.logTerminal.progressStart("upload", `上传 ${file.name} (${sizeMb} MB)`);

            // XHR (not fetch) so upload.onprogress gives real percentages
            const resp = await new Promise((resolve, reject) => {
                const xhr = new XMLHttpRequest();
                xhr.open("POST", "/api/upload");
                xhr.upload.onprogress = (e) => {
                    if (e.lengthComputable) {
                        const pct = Math.round((e.loaded / e.total) * 100);
                        this.logTerminal.progressUpdate(
                            "upload", pct, `上传 ${file.name} (${sizeMb} MB)`
                        );
                    }
                };
                xhr.onload = () => {
                    let body = null;
                    try { body = JSON.parse(xhr.responseText); } catch (err) { /* ignore */ }
                    resolve({
                        ok: xhr.status >= 200 && xhr.status < 300,
                        json: async () => body,
                    });
                };
                xhr.onerror = () => reject(new Error("网络错误，上传失败"));
                xhr.send(formData);
            });

            if (!resp.ok) {
                const err = await resp.json();
                throw new Error(err.detail || "Upload failed");
            }

            const data = await resp.json();
            this.videoPath = data.path;
            this.videoFilename = data.filename;
            this.logTerminal.progressEnd("upload", "OK", `上传完成: ${file.name}`);

            // Automatically start the session
            await this._startSession();
        } catch (e) {
            this.logTerminal.progressEnd("upload", "ERR", `上传失败: ${e.message}`);
            this._setUploadState("btn-danger");
            this._showErrorModal("上传失败", e.message);
        } finally {
            // Allow re-uploading (e.g. picking another video) afterwards;
            // resetting the input lets the same file be picked again
            this.els.fileInput.value = "";
            this._setUploadEnabled(true);
            this.els.btnUpload.textContent = "⬆ 上传视频";
        }
    }

    async _startSession() {
        this.els.btnUpload.textContent = "加载帧…";
        this.logTerminal.progressStart("session", "抽取帧并启动 SAM3 会话...");

        try {
            const resp = await fetch("/api/session/start", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ video_path: this.videoPath }),
            });

            if (!resp.ok) {
                const err = await resp.json();
                throw new Error(err.detail || "Session start failed");
            }

            const data = await resp.json();
            this.sessionId = data.session_id;
            this.numFrames = data.num_frames;

            // Reset per-session state
            this.tracklets.clear();
            this.pendingObjIds = [];
            this._renderTrackletList();

            // Initialize player
            this.player.initSession(
                this.sessionId,
                data.num_frames,
                data.orig_height,
                data.orig_width
            );
            this.player.show();

            // Enable navigation
            this.els.frameSlider.min = 0;
            this.els.frameSlider.max = data.num_frames - 1;
            this.els.frameSlider.value = 0;
            this.els.frameSlider.disabled = false;
            this.els.btnPrev.disabled = false;
            this.els.btnNext.disabled = false;
            this.els.btnPlay.disabled = false;
            this.els.frameTotalLabel.textContent = data.num_frames - 1;

            // Load first frame, sync the nav width, then build the filmstrip
            // (adaptive count needs the final strip width)
            await this.player.loadFrame(0);
            this._syncBottomNavWidth();
            this._buildFilmstrip();

            // Enable prompt interaction
            this.promptCanvas.enable();
            if (this.els.btnEncode) {
                this.els.btnEncode.disabled = false;
            }
            this.els.trackletIdInput.value = this._suggestTrackletId();
            this._updatePromptButtons();

            this.els.uploadStatus.textContent = "";
            this._setUploadState("btn-success");
            this.els.videoInfo.textContent =
                `${data.num_frames} 帧 · ${data.orig_width}×${data.orig_height}`;
            this.els.videoInfo.hidden = false;
            this.els.sessionStatus.textContent = `会话: ${this.sessionId.substring(0, 8)}...`;

            // Set session on mask overlay for REST API fetching
            this.maskOverlay.setSession(this.sessionId);

            // Persisted terminal logs from now on carry the session id
            this.logTerminal.setContext({ session_id: this.sessionId });

            this.logTerminal.progressEnd(
                "session", "OK",
                `会话 ${this.sessionId.substring(0, 8)} · ${data.num_frames} 帧 · ` +
                `${data.orig_width}×${data.orig_height}`
            );

            console.log("[App] Session started:", this.sessionId);
        } catch (e) {
            this.logTerminal.progressEnd("session", "ERR", `会话启动失败: ${e.message}`);
            this._setUploadState("btn-danger");
            this._showErrorModal("会话启动失败", e.message);
        }
    }

    // ─── Feature Encoding ─────────────────────────────────────────

    async _handleEncode() {
        if (!this.sessionId) return;

        this.els.btnEncode.disabled = true;
        this.els.encodeStatus.textContent = "正在编码特征（ViT backbone），请稍候...";
        this.els.encodeStatus.style.color = "var(--accent)";
        this.logTerminal.progressStart("encode", "ViT 特征编码中...");

        try {
            const resp = await fetch(`/api/encode/${this.sessionId}`, {
                method: "POST",
            });

            if (!resp.ok) {
                const err = await resp.json();
                throw new Error(err.detail || "Encode failed");
            }

            const data = await resp.json();
            if (data.already_encoded) {
                this.els.encodeStatus.textContent = `特征已编码 (${data.num_frames_encoded} 帧)`;
                this.logTerminal.progressEnd(
                    "encode", "INFO",
                    `特征已编码（缓存命中，${data.num_frames_encoded} 帧）`
                );
            } else {
                this.els.encodeStatus.textContent =
                    `✓ 编码完成: ${data.num_frames_encoded} 帧`;
                this.logTerminal.progressEnd(
                    "encode", "OK", `特征编码完成: ${data.num_frames_encoded} 帧`
                );
            }
            this.els.encodeStatus.style.color = "#4caf50";
            this.els.btnEncode.textContent = "✓ 已编码";
            this.els.btnEncode.classList.add("encoded");
        } catch (e) {
            this.els.encodeStatus.textContent = `编码失败: ${e.message}`;
            this.els.encodeStatus.style.color = "#e53935";
            this.els.btnEncode.disabled = false;
            this.logTerminal.progressEnd("encode", "ERR", `特征编码失败: ${e.message}`);
        }
    }

    // ─── Prompt Mode ──────────────────────────────────────────────

    _setPromptMode(mode) {
        // Same-mode calls (e.g. switching P+ -> N-) must not clear pending points
        if (this.promptMode === mode) return;

        this.promptMode = mode;
        this.promptCanvas.setMode(mode);

        // Text / box prompts get auto-assigned ids from SAM3's detection
        // path; only point prompts accept a user-specified tracklet id.
        this.els.trackletIdInput.hidden = mode !== "point";

        this._updateToolButtons();
        this._setPromptStatus("");
        this._updatePromptButtons();
    }

    // Highlight the currently active prompt tool (text input / P+ / N- / box)
    _updateToolButtons() {
        const isPoint = this.promptMode === "point";
        this.els.textInput.classList.toggle("active-tool", this.promptMode === "text");
        this.els.btnPointPositive.classList.toggle("active", isPoint && this.pointType === 1);
        this.els.btnPointNegative.classList.toggle("active", isPoint && this.pointType === 0);
        this.els.btnBoxTool.classList.toggle("active", this.promptMode === "box");
    }

    _setPointType(label) {
        this.pointType = label;
        this.promptCanvas.setPointType(label);
        this._updateToolButtons();
    }

    _suggestTrackletId() {
        let maxId = 0;
        for (const id of this.tracklets.keys()) {
            if (id > maxId) maxId = id;
        }
        return maxId + 1;
    }

    _setPromptStatus(text, color) {
        this.els.promptStatus.textContent = text;
        this.els.promptStatus.style.color = color || "";
    }

    /**
     * Refresh enable/disable state of the prompt action buttons.
     */
    _updatePromptButtons() {
        if (!this.sessionId) return;

        const hasPending =
            (this.promptMode === "box" && this.promptCanvas.hasPending()) ||
            (this.promptMode === "point" && this.promptCanvas.hasPending()) ||
            (this.promptMode === "text" && this.els.textInput.value.trim() !== "");

        const locked = this.pendingObjIds.length > 0;
        this.els.btnSegmentPreview.disabled = !hasPending || locked || this.isPropagating;
        this.els.btnConfirmPrompt.disabled = !locked;
        this.els.btnUndoPrompt.disabled = !locked;

        // Box and point prompts are mutually exclusive: once one type is
        // pending, the other tool stays disabled until cleared
        const hasBox = this.promptCanvas.pendingBox !== null;
        const hasPoints = this.promptCanvas.pendingPoints.length > 0;
        this.els.btnPointPositive.disabled = locked || hasBox;
        this.els.btnPointNegative.disabled = locked || hasBox;
        this.els.btnBoxTool.disabled = locked || hasPoints;
        this.els.btnPropagate.disabled =
            this.tracklets.size === 0 || locked || this.isPropagating;

        // Point utilities: undo removes the last point; clear removes any
        // pending prompt (points or box) and unlocks the other tool
        this.els.btnUndoPoint.disabled =
            !(this.promptMode === "point" && hasPoints) || this.isPropagating;
        this.els.btnClearPoints.disabled = !(hasBox || hasPoints) || this.isPropagating;
    }

    // ─── Segment Preview / Confirm / Undo ────────────────────────────

    async _handleSegmentPreview() {
        if (!this.sessionId || this.pendingObjIds.length > 0) return;

        const frameIdx = this.player.getCurrentFrame();
        const trackletId = parseInt(this.els.trackletIdInput.value);

        // Build the request payload by mode
        const payload = {
            session_id: this.sessionId,
            frame_idx: frameIdx,
            prompt_type: this.promptMode,
        };

        if (this.promptMode === "text") {
            payload.text = this.els.textInput.value.trim();
            if (!payload.text) {
                this._setPromptStatus("请输入文本描述", "#e53935");
                return;
            }
        } else if (this.promptMode === "point") {
            const pending = this.promptCanvas.getPending();
            if (!pending || pending.points.length === 0) {
                this._setPromptStatus("请先在画面上点击添加点", "#e53935");
                return;
            }
            if (!pending.pointLabels.includes(1)) {
                this._setPromptStatus("至少需要一个正点", "#e53935");
                return;
            }
            if (isNaN(trackletId) || trackletId < 0) {
                this._setPromptStatus("请填写有效的 tracklet id", "#e53935");
                return;
            }
            payload.points = pending.points;
            payload.point_labels = pending.pointLabels;
            payload.obj_id = trackletId;
        } else {
            const pending = this.promptCanvas.getPending();
            if (!pending || pending.boxes.length === 0) {
                this._setPromptStatus("请先在画面上拖拽画框", "#e53935");
                return;
            }
            payload.boxes = pending.boxes;
            payload.box_labels = [1]; // positive box; SAM3 auto-assigns ids
        }

        this.els.btnSegmentPreview.disabled = true;
        this._setPromptStatus("分割中（模型处理中）...", "var(--accent)");
        const typeNames = { text: "文本", point: "点", box: "框" };
        this.logTerminal.log(
            "INFO",
            `${typeNames[this.promptMode]}提示分割 @帧 ${frameIdx}...`
        );

        try {
            const resp = await fetch("/api/prompt/add", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });

            if (!resp.ok) {
                const err = await resp.json();
                throw new Error(err.detail || "add_prompt failed");
            }

            const data = await resp.json();
            // Ids newly introduced by this submission (user id for points,
            // model-assigned ids for text / box detection prompts)
            const objIds = data.new_obj_ids || [];

            if (objIds.length === 0) {
                this._setPromptStatus(
                    "未检出目标，请调整 prompt 后重试", "#ff9800"
                );
                this.logTerminal.log(
                    "WARN", `${typeNames[this.promptMode]}提示 @帧 ${frameIdx} 未检出目标`
                );
                return;
            }

            // Store the preview masks on the prompt frame
            this.pendingObjIds = objIds;
            this.pendingFrameIdx = frameIdx;
            this.maskOverlay.storeMask(frameIdx, data.mask_png, data.obj_ids);
            this.maskOverlay.displayMask(frameIdx);

            this._setPromptStatus(
                `已分割出目标 (id: ${objIds.join(", ")})，请确认或撤销`,
                "#4caf50"
            );
            this.logTerminal.log(
                "OK",
                `分割成功 @帧 ${frameIdx} → 检出 id: ${objIds.join(", ")}`
            );
        } catch (e) {
            this._setPromptStatus(`分割失败: ${e.message}`, "#e53935");
            this.logTerminal.log("ERR", `分割失败: ${e.message}`);
        } finally {
            this._updatePromptButtons();
        }
    }

    async _handleConfirmPrompt() {
        if (this.pendingObjIds.length === 0) return;

        const className = this.els.classNameInput.value.trim();
        if (!className) {
            this._setPromptStatus("请填写类别名后再确认", "#e53935");
            return;
        }

        // Tell the backend to move the pending prompt into the confirmed
        // set (it will be replayed on every subsequent submission)
        try {
            const resp = await fetch("/api/prompt/confirm", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ session_id: this.sessionId }),
            });
            if (!resp.ok) {
                const err = await resp.json();
                throw new Error(err.detail || "confirm failed");
            }
        } catch (e) {
            this._setPromptStatus(`确认失败: ${e.message}`, "#e53935");
            return;
        }

        const frameIdx = this.pendingFrameIdx;

        // Register every id produced by this submission (point: the single
        // user-specified id; text: possibly multiple auto-assigned instances)
        const confirmedIds = [];
        for (const id of this.pendingObjIds) {
            confirmedIds.push(id);
            this.tracklets.set(id, {
                id,
                className,
                promptType: this.promptMode,
                frameIdx,
            });
        }

        // Reset the pending state for the next target
        this.pendingObjIds = [];
        this.pendingFrameIdx = -1;
        this.promptCanvas.clearPending();
        this.els.classNameInput.value = "";
        this.els.textInput.value = "";
        this.els.trackletIdInput.value = this._suggestTrackletId();
        this._setPromptStatus("已确认，可继续标注下一个目标", "#4caf50");
        this.logTerminal.log(
            "OK",
            `已确认 tracklet #${confirmedIds.join(" #")} (${className}) @帧 ${frameIdx}`
        );

        this._renderTrackletList();
        this._updatePromptButtons();
    }

    async _handleUndoPrompt() {
        if (this.pendingObjIds.length === 0 || !this.sessionId) return;

        this.els.btnUndoPrompt.disabled = true;
        this._setPromptStatus("撤销中...", "var(--accent)");

        const frameIdx = this.pendingFrameIdx;
        let respData = null;

        try {
            const resp = await fetch(
                `/api/prompt/${this.sessionId}/pending`,
                { method: "DELETE" }
            );
            if (!resp.ok) {
                const err = await resp.json();
                throw new Error(err.detail || "cancel failed");
            }
            respData = await resp.json();
        } catch (e) {
            this._setPromptStatus(`撤销失败: ${e.message}`, "#e53935");
            this._updatePromptButtons();
            return;
        }

        // Refresh the prompt frame with the remaining objects (or clear it)
        this.maskOverlay.evictFrame(frameIdx);
        if (respData && respData.mask_png && (respData.obj_ids || []).length > 0) {
            this.maskOverlay.storeMask(frameIdx, respData.mask_png, respData.obj_ids);
        }
        this.maskOverlay.displayMask(frameIdx);

        this.pendingObjIds = [];
        this.pendingFrameIdx = -1;
        this._setPromptStatus("已撤销，可重新绘制 prompt", "#ff9800");
        this.logTerminal.log("INFO", `已撤销预览提示 @帧 ${frameIdx}`);
        this._updatePromptButtons();
    }

    // ─── Tracklet List ──────────────────────────────────────────────

    _renderTrackletList() {
        const listEl = this.els.trackletList;
        listEl.innerHTML = "";

        const sorted = Array.from(this.tracklets.values()).sort((a, b) => a.id - b.id);
        if (sorted.length === 0) {
            this.els.resultInfo.innerHTML =
                '<p class="hint">分割确认后的 tracklet 将在此显示。</p>';
            return;
        }

        this.els.resultInfo.innerHTML =
            `<p class="hint">共 ${sorted.length} 个目标</p>`;

        const typeNames = { text: "文本", point: "点", box: "框" };
        for (const t of sorted) {
            const color = this.maskOverlay.getObjectColor(t.id);
            const item = document.createElement("div");
            item.className = "tracklet-item";
            item.innerHTML = `
                <span class="box-color" style="background:${color.hex}"></span>
                <span class="tracklet-info">
                    <strong>#${t.id}</strong> ${t.className}
                    <span class="tracklet-meta">${typeNames[t.promptType] || t.promptType} · 帧${t.frameIdx}</span>
                </span>
                <span class="tracklet-delete" data-id="${t.id}" title="删除">&times;</span>
            `;
            item.querySelector(".tracklet-delete").addEventListener("click", () => {
                this._removeTracklet(t.id);
            });
            listEl.appendChild(item);
        }
    }

    async _removeTracklet(id) {
        if (!this.sessionId) return;
        const t = this.tracklets.get(id);
        if (!t) return;

        try {
            const resp = await fetch(
                `/api/prompt/${this.sessionId}/${id}`,
                { method: "DELETE" }
            );
            if (!resp.ok) {
                const err = await resp.json();
                throw new Error(err.detail || "remove failed");
            }
        } catch (e) {
            this._setPromptStatus(`删除目标 #${id} 失败: ${e.message}`, "#e53935");
            return;
        }

        this.tracklets.delete(id);
        this._renderTrackletList();

        // Cached masks on every frame contain the removed object: drop them
        this.maskOverlay.evictAll();
        const current = this.player.getCurrentFrame();
        if (this.sessionId) {
            this.maskOverlay.fetchAndDisplayMask(current);
        }
        this._setPromptStatus(`已删除目标 #${id}`, "#ff9800");
        this.logTerminal.log("INFO", `已删除 tracklet #${id} (${t.className})`);
        this._updatePromptButtons();
    }

    // ─── Filmstrip ──────────────────────────────────────────────────

    _buildFilmstrip() {
        const strip = this.els.filmstrip;
        strip.innerHTML = "";
        this._filmThumbs = [];

        const n = this.numFrames;
        if (n <= 0) return;

        // Adaptive count: fit as many thumbnails as possible into the strip
        // (which shares the slider column width). Thumb width follows the
        // video aspect ratio at the fixed CSS height of 52px.
        const aspect = (this.player.origWidth && this.player.origHeight)
            ? this.player.origWidth / this.player.origHeight
            : 4 / 3;
        const thumbW = Math.max(24, Math.round(52 * aspect));
        const minGap = 6;
        const avail = strip.clientWidth;
        let count = avail > 0
            ? Math.floor((avail + minGap) / (thumbW + minGap))
            : 12;
        count = Math.max(2, Math.min(count, 40, n));

        const denom = count > 1 ? count - 1 : 1;
        this._filmStep = Math.max(1, Math.floor((n - 1) / denom));

        for (let i = 0; i < count; i++) {
            // Last thumbnail always represents the final frame
            const frameIdx = (i === count - 1) ? n - 1 : i * this._filmStep;

            const thumb = document.createElement("div");
            thumb.className = "film-thumb";

            const img = document.createElement("img");
            img.loading = "lazy";
            img.alt = `帧 ${frameIdx}`;
            img.src = `/api/thumb/${this.sessionId}/${frameIdx}?w=160`;

            const idx = document.createElement("span");
            idx.className = "thumb-idx";
            idx.textContent = frameIdx;

            thumb.appendChild(img);
            thumb.appendChild(idx);
            thumb.addEventListener("click", () => this.player.goToFrame(frameIdx));

            strip.appendChild(thumb);
            this._filmThumbs.push({ frameIdx, el: thumb });
        }

        this._updateFilmstripHighlight(this.player.getCurrentFrame());
    }

    _updateFilmstripHighlight(frameIdx) {
        if (this._filmThumbs.length === 0) return;

        // Throttle DOM updates during playback
        const now = performance.now();
        if (this._isPlaying && now - this._lastFilmTs < 200) return;
        this._lastFilmTs = now;

        const count = this._filmThumbs.length;
        const step = this._filmStep;
        let thumbIdx = Math.floor(frameIdx / step);
        if (thumbIdx >= count) thumbIdx = count - 1;

        for (let i = 0; i < count; i++) {
            this._filmThumbs[i].el.classList.toggle("active", i === thumbIdx);
        }

        // Keep the highlighted thumbnail visible when scrolling is needed
        const el = this._filmThumbs[thumbIdx].el;
        const strip = this.els.filmstrip;
        const left = el.offsetLeft;
        const right = left + el.offsetWidth;
        if (left < strip.scrollLeft || right > strip.scrollLeft + strip.clientWidth) {
            strip.scrollTo({
                left: left - (strip.clientWidth - el.offsetWidth) / 2,
                behavior: "smooth",
            });
        }
    }

    // ─── Playback ───────────────────────────────────────────────────

    _togglePlay() {
        if (this.isPropagating) return;
        if (this._isPlaying) {
            this._stopPlay();
        } else {
            this._startPlay();
        }
    }

    _startPlay() {
        if (!this.sessionId || this.numFrames === 0) return;
        if (this._isPlaying || this.isPropagating) return;

        this._isPlaying = true;
        this.els.btnPlay.textContent = "⏸";

        // Disable prompt drawing during playback
        this.promptCanvas.disable();

        const interval = 1000 / this._playFps;
        const playStep = () => {
            if (!this._isPlaying) return;
            let nextIdx = this.player.getCurrentFrame() + 1;
            if (nextIdx >= this.numFrames) {
                nextIdx = 0; // loop
            }
            this.player.loadFrame(nextIdx);
        };

        this._playTimer = setInterval(playStep, interval);
    }

    _stopPlay() {
        this._isPlaying = false;
        if (this._playTimer) {
            clearInterval(this._playTimer);
            this._playTimer = null;
        }
        this.els.btnPlay.textContent = "▶";
        if (this.sessionId && !this.isPropagating) {
            this.promptCanvas.enable();
        }
    }

    // ─── Propagation ─────────────────────────────────────────────────

    _getPropagationDirection() {
        return this.els.directionSelect.value;
    }

    async _handlePropagate() {
        if (!this.sessionId || this.tracklets.size === 0) return;

        const direction = this._getPropagationDirection();
        const dirNames = { forward: "前向", backward: "后向", both: "双向" };

        // Stop playback before propagating
        this._stopPlay();

        // Disable UI
        this.isPropagating = true;
        this.els.btnPropagate.disabled = true;
        this.els.btnCancel.disabled = false;
        this.promptCanvas.disable();
        this.els.frameSlider.disabled = true;
        this.els.btnPrev.disabled = true;
        this.els.btnNext.disabled = true;
        this.els.btnPlay.disabled = true;

        // Reset mask overlay state
        this.maskOverlay.reset();

        this.logTerminal.progressStart(
            "prop", `${dirNames[direction]}跟踪 ${this.numFrames} 帧`
        );

        // Create WebSocket client
        this.wsClient = new WSClient(this.sessionId, {
            onOpen: () => {
                console.log("[App] WS connected, starting propagation:", direction);
                this.wsClient.startPropagation(direction);
            },
            onPropagationStart: (totalFrames, data) => {
                const dir = dirNames[(data && data.direction) || direction] || direction;
                this.els.progressText.textContent = `0 / ${totalFrames}`;
                this.els.progressFill.style.width = "0%";
                this.els.resultInfo.innerHTML =
                    `<p class="hint">${dir}跟踪 ${totalFrames} 帧中...</p>`;
            },
            onFrameResult: (data) => {
                // Store the mask for this frame
                this.maskOverlay.storeMask(
                    data.frame_index,
                    data.mask_png,
                    data.out_obj_ids
                );

                // Update progress
                const total = this.numFrames;
                const processed = Math.round(data.progress * total);
                this.els.progressFill.style.width = `${data.progress * 100}%`;
                this.els.progressText.textContent = `${processed} / ${total}`;

                // Terminal progress line: throttled in-place update
                const now = performance.now();
                if (now - this._lastPropLogTs >= 500 || data.progress >= 1.0) {
                    this._lastPropLogTs = now;
                    this.logTerminal.progressUpdate(
                        "prop", Math.round(data.progress * 100),
                        `${dirNames[direction]}跟踪 ${processed} / ${total} 帧`
                    );
                }
            },
            onPropagationComplete: (totalFrames) => {
                this.els.progressFill.style.width = "100%";
                this.els.progressText.textContent = `${totalFrames} / ${totalFrames}`;
                this.els.resultInfo.innerHTML =
                    `<p class="hint" style="color:#4caf50">跟踪完成: ${totalFrames} 帧，可播放预览</p>`;
                this.logTerminal.progressEnd(
                    "prop", "OK", `${dirNames[direction]}跟踪完成: ${totalFrames} 帧`
                );

                this._endPropagation();
            },
            onCancelled: (framesProcessed) => {
                this.els.resultInfo.innerHTML =
                    `<p class="hint">已取消 (已处理 ${framesProcessed} 帧)</p>`;
                this.logTerminal.progressEnd(
                    "prop", "WARN", `跟踪已取消 (已处理 ${framesProcessed} 帧)`
                );
                this._endPropagation();
            },
            onError: (error) => {
                this.els.resultInfo.innerHTML =
                    `<p style="color:#e53935">错误: ${error.message}</p>`;
                this.logTerminal.progressEnd(
                    "prop", "ERR", `跟踪失败: ${error.message}`
                );
                this._endPropagation();
            },
            onClose: () => {
                if (this.isPropagating) {
                    this.logTerminal.progressEnd(
                        "prop", "WARN", "连接中断，跟踪中止"
                    );
                    this._endPropagation();
                }
            },
        });

        try {
            await this.wsClient.connect();
        } catch (e) {
            this.els.resultInfo.innerHTML =
                `<p style="color:#e53935">WebSocket 连接失败: ${e.message}</p>`;
            this.logTerminal.progressEnd(
                "prop", "ERR", `WebSocket 连接失败: ${e.message}`
            );
            this._endPropagation();
        }
    }

    async _handleCancel() {
        if (this.wsClient) {
            this.wsClient.cancelPropagation();
        }
    }

    _endPropagation() {
        this.isPropagating = false;
        this.els.btnCancel.disabled = true;
        this.promptCanvas.enable();
        this.els.frameSlider.disabled = false;
        this.els.btnPrev.disabled = false;
        this.els.btnNext.disabled = false;
        this.els.btnPlay.disabled = false;

        // Disconnect WebSocket
        if (this.wsClient) {
            this.wsClient.disconnect();
            this.wsClient = null;
        }

        // Display mask for the current frame if available
        const currentFrame = this.player.getCurrentFrame();
        if (this.sessionId) {
            this.maskOverlay.fetchAndDisplayMask(currentFrame);
        }
    }

    // ─── Status Polling ──────────────────────────────────────────────

    _startStatusPolling() {
        // The poll doubles as the session heartbeat: passing session_id
        // refreshes the backend activity timestamp so an open page keeps
        // its session alive, and a closed page (no polls) lets the
        // backend expire it quickly.
        const poll = async () => {
            try {
                const url = this.sessionId
                    ? `/api/status?session_id=${encodeURIComponent(this.sessionId)}`
                    : "/api/status";
                const resp = await fetch(url);
                const data = await resp.json();
                this.els.gpuStatus.textContent =
                    `GPU: ${data.gpu_used_pct.toFixed(0)}% ` +
                    `(${data.gpu_free_mb.toFixed(0)}MB 空闲)`;
                if (this.sessionId) {
                    this.els.sessionStatus.textContent =
                        `会话: ${this.sessionId.substring(0, 8)}... ` +
                        `(活跃: ${data.active_sessions}/${data.max_sessions})`;
                } else {
                    this.els.sessionStatus.textContent =
                        `会话: 无 (空闲: ${data.active_sessions}/${data.max_sessions})`;
                }
            } catch (e) {
                // Ignore polling errors
            }
        };
        poll();
        this._statusInterval = setInterval(poll, 5000);
    }
}

// Initialize on DOM ready
document.addEventListener("DOMContentLoaded", () => {
    const app = new AnnotationApp();
    app.init();
    window.annotationApp = app; // for debugging
});
