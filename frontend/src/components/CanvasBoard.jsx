/**
 * INKFORGE — CanvasBoard
 *
 * Imperative canvas renderer. Zero React re-renders during streaming.
 * All drawing happens via exposed imperative methods that the network
 * layer calls directly, completely bypassing React's render cycle.
 *
 * Stroke data (dx, dy) is accumulated in a plain mutable ref —
 * never in useState — so React's reconciler never fires.
 */

import { forwardRef, useRef, useEffect, useImperativeHandle } from "react";

/** Ink color CSS values keyed by backend enum */
const INK_COLORS = {
    black: "#1a1a2e",
    blue: "#0066cc",
    dark_blue: "#003366",
    sepia: "#704214",
};

const DEFAULT_WIDTH = 800;
const DEFAULT_HEIGHT = 600;

const CanvasBoard = forwardRef(function CanvasBoard(
    { width = DEFAULT_WIDTH, height = DEFAULT_HEIGHT, inkColor = "black", strokeWidthVariation = 0.5 },
    ref
) {
    const canvasRef = useRef(null);
    const ctxRef = useRef(null);

    // Mutable cursor position — never triggers a render
    const cursorRef = useRef({ x: 40, y: 40 });

    // Helper to calculate line width
    const getLineWidth = (variation) => {
        // Map 0-1 variation to 1.0 - 2.8 context line width
        return 1.0 + (variation * 1.8);
    };

    // Initialize canvas context once on mount or when dimensions change
    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;

        const ctx = canvas.getContext("2d");
        ctx.lineCap = "round";
        ctx.lineJoin = "round";
        ctx.lineWidth = getLineWidth(strokeWidthVariation);
        ctx.strokeStyle = INK_COLORS[inkColor] || INK_COLORS.black;
        ctxRef.current = ctx;

        // Start with a clean slate
        ctx.clearRect(0, 0, width, height);
        ctx.beginPath();
        ctx.moveTo(cursorRef.current.x, cursorRef.current.y);
    }, [width, height]); // Exclude style paints so we don't clear midway

    // Update styling properties dynamically without clearing canvas
    useEffect(() => {
        const ctx = ctxRef.current;
        if (ctx) {
            ctx.strokeStyle = INK_COLORS[inkColor] || INK_COLORS.black;
            ctx.lineWidth = getLineWidth(strokeWidthVariation);
        }
    }, [inkColor, strokeWidthVariation]);

    // Expose imperative methods to parent via ref
    useImperativeHandle(
        ref,
        () => ({
            /**
             * Draw a single stroke segment.
             *
             * @param {number} dx — horizontal delta
             * @param {number} dy — vertical delta
             * @param {boolean} penDown — true = draw, false = lift pen (move without ink)
             */
            drawStroke(dx, dy, penDown) {
                const ctx = ctxRef.current;
                if (!ctx) return;

                const cursor = cursorRef.current;
                const nx = cursor.x + dx;
                const ny = cursor.y + dy;

                if (penDown) {
                    ctx.lineTo(nx, ny);
                    ctx.stroke();
                    // Continue the path from the new point
                    ctx.beginPath();
                    ctx.moveTo(nx, ny);
                } else {
                    // Pen-up: reposition without drawing
                    ctx.beginPath();
                    ctx.moveTo(nx, ny);
                }

                cursor.x = nx;
                cursor.y = ny;
            },

            /**
             * Clear the canvas and reset cursor to origin.
             */
            clear() {
                const ctx = ctxRef.current;
                if (!ctx) return;

                ctx.clearRect(0, 0, width, height);
                ctx.beginPath();
                cursorRef.current = { x: 40, y: 40 };
                ctx.moveTo(40, 40);
            },
        }),
        [width, height]
    );

    return (
        <canvas
            ref={canvasRef}
            width={width}
            height={height}
            style={{
                imageRendering: "auto",
                background: "#fefefe",
                borderRadius: "6px",
            }}
        />
    );
});

export default CanvasBoard;
