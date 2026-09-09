// ws-client.js - WebSocket connection manager for SAM3 propagation streaming

class WSClient {
    constructor(sessionId, callbacks) {
        this.sessionId = sessionId;
        this.callbacks = callbacks || {};
        this.ws = null;
        this.isConnected = false;
        this._reconnectAttempts = 0;
        this._maxReconnect = 3;
    }

    /**
     * Build WebSocket URL based on current location.
     * Uses the same host as the page but with ws:// or wss:// protocol.
     */
    _buildURL(sessionId) {
        const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
        return `${proto}//${window.location.host}/ws/propagate/${sessionId}`;
    }

    /**
     * Connect to the WebSocket server.
     * @returns {Promise<void>} Resolves on open, rejects on error.
     */
    connect() {
        return new Promise((resolve, reject) => {
            const url = this._buildURL(this.sessionId);
            this.ws = new WebSocket(url);

            this.ws.onopen = () => {
                this.isConnected = true;
                this._reconnectAttempts = 0;
                console.log("[WS] Connected:", url);
                if (this.callbacks.onOpen) this.callbacks.onOpen();
                resolve();
            };

            this.ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this._handleMessage(data);
                } catch (e) {
                    console.error("[WS] Failed to parse message:", e);
                }
            };

            this.ws.onerror = (error) => {
                console.error("[WS] Error:", error);
                if (this.callbacks.onError) this.callbacks.onError(error);
                if (!this.isConnected) reject(error);
            };

            this.ws.onclose = () => {
                this.isConnected = false;
                console.log("[WS] Disconnected");
                if (this.callbacks.onClose) this.callbacks.onClose();
            };
        });
    }

    /**
     * Send a JSON message to the server.
     */
    send(message) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(message));
            return true;
        }
        console.warn("[WS] Cannot send: not connected");
        return false;
    }

    /**
     * Start propagation in the given direction.
     * @param {string} direction - "forward" | "backward" | "both"
     */
    startPropagation(direction = "both") {
        return this.send({ type: "start", direction: direction });
    }

    /**
     * Cancel propagation.
     */
    cancelPropagation() {
        return this.send({ type: "cancel" });
    }

    /**
     * Send a ping to keep the connection alive.
     */
    ping() {
        return this.send({ type: "ping" });
    }

    /**
     * Close the WebSocket connection.
     */
    disconnect() {
        if (this.ws) {
            this.ws.onclose = null; // prevent callback
            this.ws.close();
            this.ws = null;
        }
        this.isConnected = false;
    }

    /**
     * Handle incoming messages from the server.
     */
    _handleMessage(data) {
        const { type } = data;

        switch (type) {
            case "propagation_started":
                if (this.callbacks.onPropagationStart) {
                    this.callbacks.onPropagationStart(data.total_frames, data);
                }
                break;

            case "frame_result":
                if (this.callbacks.onFrameResult) {
                    this.callbacks.onFrameResult(data);
                }
                break;

            case "propagation_complete":
                if (this.callbacks.onPropagationComplete) {
                    this.callbacks.onPropagationComplete(
                        data.total_frames, data
                    );
                }
                break;

            case "cancelled":
                if (this.callbacks.onCancelled) {
                    this.callbacks.onCancelled(data.frames_processed);
                }
                break;

            case "error":
                console.error("[WS] Server error:", data.message);
                if (this.callbacks.onError) {
                    this.callbacks.onError(new Error(data.message));
                }
                break;

            case "pong":
                // Heartbeat response, ignore
                break;

            default:
                console.warn("[WS] Unknown message type:", type, data);
        }
    }
}

// Export for global access (no module system in plain HTML)
window.WSClient = WSClient;
