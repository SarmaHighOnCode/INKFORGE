/**
 * INKFORGE — Network Shield
 *
 * Backend-protective fetch wrapper. Every outbound request is cancellable
 * via AbortSignal. WebSocket connections return a teardown handle.
 *
 * Rule #1: Never fire a request the user can't cancel.
 * Rule #2: Never hold a connection the user has abandoned.
 */

const API_BASE = "/api";

/**
 * Submit a handwriting generation job.
 *
 * @param {object} payload — { text, style_id, params, paper_texture, ink_color, font_size }
 * @param {AbortSignal} signal — caller-owned AbortSignal for cancellation
 * @returns {Promise<{ job_id: string, ws_url: string }>}
 */
export async function submitJob(payload, signal) {
    const res = await fetch(`${API_BASE}/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        signal, // ← the entire point: caller can abort this at any time
    });

    if (!res.ok) {
        const body = await res.text().catch(() => "");
        throw new Error(`Generate failed (${res.status}): ${body || res.statusText}`);
    }

    return res.json();
}

/**
 * Open a WebSocket to stream stroke data from a running job.
 *
 * Returns a close() handle so the caller can tear down the socket
 * when the user cancels or navigates away.
 *
 * @param {string} wsPath — relative WebSocket path, e.g. "/ws/abc123"
 * @param {object} callbacks
 * @param {(data: {dx:number, dy:number, p1:number, p2:number, p3:number}) => void} callbacks.onStroke
 * @param {(totalStrokes: number) => void} callbacks.onComplete
 * @param {(error: string) => void} callbacks.onError
 * @returns {{ close: () => void }}
 */
export function connectStream(wsPath, { onStroke, onComplete, onError } = {}) {
    // Build absolute WS URL from current host (Vite proxy handles routing)
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}${wsPath}`;

    const ws = new WebSocket(wsUrl);

    ws.onmessage = (event) => {
        try {
            const msg = JSON.parse(event.data);

            if (msg.type === "stroke" && msg.data) {
                onStroke?.(msg.data);
            } else if (msg.type === "complete") {
                onComplete?.(msg.total_strokes ?? 0);
            } else if (msg.type === "error") {
                // Handle server-side errors (e.g., model failure, job not found)
                onError?.(msg.message || "Server error during generation");
            }
        } catch {
            // Malformed JSON — log but don't crash
            console.warn("[inkforge] unparseable WS message:", event.data);
        }
    };

    ws.onerror = () => {
        onError?.("WebSocket connection error");
    };

    ws.onclose = (event) => {
        // Code 1000 = normal closure, anything else is unexpected
        if (event.code !== 1000 && event.code !== 1005) {
            onError?.(`WebSocket closed unexpectedly (code ${event.code})`);
        }
    };

    return {
        close: () => {
            if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) {
                ws.close(1000, "Client cancelled");
            }
        },
    };
}
