/**
 * INKFORGE — Application Shell
 *
 * Clean, research-lab aesthetic. DeepMind-grade restraint.
 * The UI should feel like it was designed by people who
 * spend their time on inference, not on gradients.
 */

import { useState, useRef, useCallback, useEffect } from "react";
import CanvasBoard from "./components/CanvasBoard";
import { submitJob, connectStream } from "./network/api";

const STATUS = {
    IDLE: "idle",
    SUBMITTING: "submitting",
    STREAMING: "streaming",
    DONE: "done",
    ERROR: "error",
    CANCELLED: "cancelled",
};

const STYLE_OPTIONS = [
    { id: "neat_cursive", name: "Neat Cursive" },
    { id: "casual_print", name: "Casual Print" },
    { id: "rushed_notes", name: "Rushed Notes" },
    { id: "doctors_scrawl", name: "Doctor's Scrawl" },
    { id: "elegant_formal", name: "Elegant Formal" },
];

const INK_OPTIONS = [
    { id: "black", label: "Black" },
    { id: "blue", label: "Blue" },
    { id: "dark_blue", label: "Dark Blue" },
    { id: "sepia", label: "Sepia" },
];

const TEXTURE_OPTIONS = [
    { id: "blank", label: "Blank" },
    { id: "lined", label: "Lined" },
    { id: "graph", label: "Graph" },
    { id: "aged_parchment", label: "Aged Parchment" },
];

const SIZE_OPTIONS = [
    { id: "small", label: "Small" },
    { id: "medium", label: "Medium" },
    { id: "large", label: "Large" },
];

const MAX_TEXT = 2000;

function App() {
    const [status, setStatus] = useState(STATUS.IDLE);
    const [error, setError] = useState(null);
    const [charCount, setCharCount] = useState(0);
    const [inkColor, setInkColor] = useState("black");

    const canvasRef = useRef(null);
    const abortRef = useRef(null);
    const wsRef = useRef(null);
    const formRef = useRef({
        text: "",
        style_id: "neat_cursive",
        ink_color: "black",
        paper_texture: "blank",
        font_size: "medium",
        stroke_width_variation: 0.5,
        character_inconsistency: 0.4,
        slant_angle: 5,
        baseline_drift: 0.3,
        ink_bleed: 0.2,
        ligature_enabled: true,
    });

    const isActive = status === STATUS.SUBMITTING || status === STATUS.STREAMING;

    const teardown = useCallback(() => {
        abortRef.current?.abort();
        abortRef.current = null;
        wsRef.current?.close();
        wsRef.current = null;
    }, []);

    // Ensure cleanup on unmount
    useEffect(() => {
        return () => teardown();
    }, [teardown]);

    const handleCancel = useCallback(() => {
        teardown();
        setStatus(STATUS.CANCELLED);
        setError(null);
    }, [teardown]);

    const handleGenerate = useCallback(async () => {
        const f = formRef.current;
        if (!f.text.trim()) {
            setError("Please enter some text.");
            setStatus(STATUS.ERROR);
            return;
        }

        teardown();
        const controller = new AbortController();
        abortRef.current = controller;
        canvasRef.current?.clear();
        setError(null);
        setStatus(STATUS.SUBMITTING);

        try {
            const payload = {
                text: f.text,
                style_id: f.style_id,
                params: {
                    stroke_width_variation: f.stroke_width_variation,
                    character_inconsistency: f.character_inconsistency,
                    slant_angle: f.slant_angle,
                    baseline_drift: f.baseline_drift,
                    ink_bleed: f.ink_bleed,
                    ligature_enabled: f.ligature_enabled,
                },
                paper_texture: f.paper_texture,
                ink_color: f.ink_color,
                font_size: f.font_size,
            };

            const { ws_url } = await submitJob(payload, controller.signal);
            if (controller.signal.aborted) return;

            setStatus(STATUS.STREAMING);

            let wsPath;
            try {
                wsPath = new URL(ws_url).pathname;
            } catch {
                wsPath = ws_url;
            }

            wsRef.current = connectStream(wsPath, {
                onStroke: (d) => {
                    canvasRef.current?.drawStroke(d.dx, d.dy, d.p1 === 1);
                },
                onComplete: () => {
                    wsRef.current = null;
                    setStatus(STATUS.DONE);
                },
                onError: (msg) => {
                    wsRef.current = null;
                    if (!controller.signal.aborted) {
                        setError(msg);
                        setStatus(STATUS.ERROR);
                    }
                },
            });
        } catch (err) {
            if (err.name === "AbortError") {
                setStatus(STATUS.CANCELLED);
                return;
            }
            setError(err.message || "Request failed");
            setStatus(STATUS.ERROR);
        }
    }, [teardown]);

    const setField = (key) => (e) => {
        const val = e.target.type === "checkbox" ? e.target.checked : e.target.value;
        formRef.current[key] = e.target.type === "range" ? parseFloat(val) : val;
        // Sync ink color to state so CanvasBoard gets the updated value
        if (key === "ink_color") {
            setInkColor(val);
        }
    };

    const handleTextChange = (e) => {
        formRef.current.text = e.target.value;
        setCharCount(e.target.value.length);
    };

    return (
        <div className="min-h-screen" style={{ background: "var(--bg-primary)" }}>
            {/* ─── Header ─── */}
            <header
                className="flex items-center justify-between px-6 py-3.5"
                style={{ borderBottom: "1px solid var(--border-subtle)" }}
            >
                <div className="flex items-center gap-3">
                    <span className="text-base font-semibold tracking-tight" style={{ color: "var(--text-primary)" }}>
                        inkforge
                    </span>
                    <span
                        className="text-[10px] font-mono px-1.5 py-0.5 rounded"
                        style={{ background: "var(--bg-tertiary)", color: "var(--text-tertiary)" }}
                    >
                        v1.0
                    </span>
                </div>
                <span className="text-xs" style={{ color: "var(--text-muted)" }}>
                    Handwriting Synthesis
                </span>
            </header>

            {/* ─── Main Layout ─── */}
            <div className="flex min-h-[calc(100vh-52px)]">
                {/* ─── Sidebar ─── */}
                <aside
                    className="w-80 shrink-0 overflow-y-auto p-5"
                    style={{ borderRight: "1px solid var(--border-subtle)" }}
                >
                    {/* Text Input */}
                    <Section title="Input">
                        <textarea
                            onChange={handleTextChange}
                            maxLength={MAX_TEXT}
                            rows={7}
                            placeholder="Start typing…"
                            className="w-full px-3.5 py-3 text-[13px] leading-relaxed resize-y rounded-lg"
                            style={{
                                background: "var(--bg-secondary)",
                                border: "1px solid var(--border-default)",
                                minHeight: "120px",
                            }}
                        />
                        <div className="flex justify-end mt-1.5">
                            <span className="text-[11px] font-mono" style={{ color: "var(--text-muted)" }}>
                                {charCount}<span style={{ color: "var(--text-tertiary)" }}>/{MAX_TEXT}</span>
                            </span>
                        </div>
                    </Section>

                    {/* Configuration */}
                    <Section title="Configuration">
                        <FieldGroup label="Style">
                            <Select
                                options={STYLE_OPTIONS}
                                defaultValue="neat_cursive"
                                onChange={setField("style_id")}
                            />
                        </FieldGroup>
                        <FieldGroup label="Ink">
                            <Select
                                options={INK_OPTIONS}
                                defaultValue="black"
                                onChange={setField("ink_color")}
                            />
                        </FieldGroup>
                        <FieldGroup label="Paper">
                            <Select
                                options={TEXTURE_OPTIONS}
                                defaultValue="blank"
                                onChange={setField("paper_texture")}
                            />
                        </FieldGroup>
                        <FieldGroup label="Size">
                            <Select
                                options={SIZE_OPTIONS}
                                defaultValue="medium"
                                onChange={setField("font_size")}
                            />
                        </FieldGroup>
                    </Section>

                    {/* Humanization */}
                    <Section title="Humanization">
                        <Slider label="Stroke Width" field="stroke_width_variation" min={0} max={1} step={0.05} defaultValue={0.5} onChange={setField} />
                        <Slider label="Inconsistency" field="character_inconsistency" min={0} max={1} step={0.05} defaultValue={0.4} onChange={setField} />
                        <Slider label="Slant" field="slant_angle" min={-30} max={30} step={1} defaultValue={5} onChange={setField} />
                        <Slider label="Baseline Drift" field="baseline_drift" min={0} max={1} step={0.05} defaultValue={0.3} onChange={setField} />
                        <Slider label="Ink Bleed" field="ink_bleed" min={0} max={1} step={0.05} defaultValue={0.2} onChange={setField} />

                        <label className="flex items-center gap-2.5 mt-3 cursor-pointer group">
                            <input
                                type="checkbox"
                                defaultChecked
                                onChange={setField("ligature_enabled")}
                                className="w-3.5 h-3.5"
                            />
                            <span className="text-xs" style={{ color: "var(--text-secondary)" }}>
                                Ligature connections
                            </span>
                        </label>
                    </Section>
                </aside>

                {/* ─── Main Content ─── */}
                <main className="flex-1 flex flex-col p-6">
                    {/* Action Bar */}
                    <div className="flex items-center gap-3 mb-5">
                        <button
                            onClick={handleGenerate}
                            disabled={isActive}
                            className="h-9 px-5 text-[13px] font-medium rounded-lg transition-all duration-200"
                            style={{
                                background: isActive ? "var(--bg-tertiary)" : "var(--text-primary)",
                                color: isActive ? "var(--text-muted)" : "var(--bg-primary)",
                                cursor: isActive ? "not-allowed" : "pointer",
                                opacity: isActive ? 0.5 : 1,
                            }}
                        >
                            {status === STATUS.SUBMITTING ? "Submitting…" : "Generate"}
                        </button>

                        {isActive && (
                            <button
                                onClick={handleCancel}
                                className="h-9 px-4 text-[13px] rounded-lg transition-all duration-200 animate-fade-in"
                                style={{
                                    background: "transparent",
                                    border: "1px solid var(--border-default)",
                                    color: "var(--text-secondary)",
                                }}
                                onMouseEnter={(e) => {
                                    e.currentTarget.style.borderColor = "var(--error)";
                                    e.currentTarget.style.color = "var(--error)";
                                }}
                                onMouseLeave={(e) => {
                                    e.currentTarget.style.borderColor = "var(--border-default)";
                                    e.currentTarget.style.color = "var(--text-secondary)";
                                }}
                            >
                                Cancel
                            </button>
                        )}

                        <StatusIndicator status={status} />
                    </div>

                    {/* Error */}
                    {error && (
                        <div
                            className="mb-4 px-4 py-2.5 rounded-lg text-[13px] animate-fade-in"
                            style={{
                                background: "rgba(248, 113, 113, 0.08)",
                                border: "1px solid rgba(248, 113, 113, 0.2)",
                                color: "var(--error)",
                            }}
                        >
                            {error}
                        </div>
                    )}

                    {/* Canvas Area */}
                    <div
                        className="flex-1 flex items-start justify-center rounded-xl p-6"
                        style={{
                            background: "var(--bg-secondary)",
                            border: "1px solid var(--border-subtle)",
                        }}
                    >
                        <CanvasBoard
                            ref={canvasRef}
                            width={800}
                            height={600}
                            inkColor={inkColor}
                        />
                    </div>
                </main>
            </div>
        </div>
    );
}

/* ─── Sub-components ─── */

function Section({ title, children }) {
    return (
        <div className="mb-6">
            <h3
                className="text-[11px] font-medium uppercase tracking-widest mb-3"
                style={{ color: "var(--text-tertiary)" }}
            >
                {title}
            </h3>
            {children}
        </div>
    );
}

function FieldGroup({ label, children }) {
    return (
        <div className="mb-3">
            <label className="block text-[11px] mb-1.5" style={{ color: "var(--text-secondary)" }}>
                {label}
            </label>
            {children}
        </div>
    );
}

function Select({ options, defaultValue, onChange }) {
    return (
        <select
            defaultValue={defaultValue}
            onChange={onChange}
            className="w-full h-8 px-2.5 text-[13px] rounded-md cursor-pointer"
            style={{
                background: "var(--bg-tertiary)",
                border: "1px solid var(--border-default)",
                color: "var(--text-primary)",
            }}
        >
            {options.map((o) => (
                <option key={o.id} value={o.id}>
                    {o.name || o.label}
                </option>
            ))}
        </select>
    );
}

function Slider({ label, field, min, max, step, defaultValue, onChange }) {
    const [displayValue, setDisplayValue] = useState(defaultValue);

    const handleChange = (e) => {
        const val = parseFloat(e.target.value);
        setDisplayValue(val);
        onChange(field)(e);
    };

    return (
        <div className="mb-3">
            <div className="flex items-center justify-between mb-1">
                <span className="text-[11px]" style={{ color: "var(--text-secondary)" }}>
                    {label}
                </span>
                <span className="text-[11px] font-mono" style={{ color: "var(--text-tertiary)" }}>
                    {displayValue}
                </span>
            </div>
            <input
                type="range"
                min={min}
                max={max}
                step={step}
                defaultValue={defaultValue}
                onChange={handleChange}
            />
        </div>
    );
}

function StatusIndicator({ status }) {
    if (status === STATUS.IDLE) return null;

    const config = {
        [STATUS.SUBMITTING]: { color: "var(--warning)", label: "Submitting", pulse: true },
        [STATUS.STREAMING]: { color: "var(--success)", label: "Streaming", pulse: true },
        [STATUS.DONE]: { color: "var(--success)", label: "Complete", pulse: false },
        [STATUS.ERROR]: { color: "var(--error)", label: "Error", pulse: false },
        [STATUS.CANCELLED]: { color: "var(--text-muted)", label: "Cancelled", pulse: false },
    };

    const c = config[status];
    if (!c) return null;

    return (
        <div className="flex items-center gap-2 animate-fade-in">
            <span
                className={`w-1.5 h-1.5 rounded-full ${c.pulse ? "animate-pulse-dot" : ""}`}
                style={{ background: c.color }}
            />
            <span className="text-xs font-mono" style={{ color: c.color }}>
                {c.label}
            </span>
        </div>
    );
}

export default App;
