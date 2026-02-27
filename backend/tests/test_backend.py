"""
INKFORGE — Comprehensive Backend Test Script

Tests all endpoints against a running server.
Run with: python tests/test_backend.py

Requires the server to be running on localhost:8000
"""

import json
import sys
import urllib.request
import urllib.error

BASE = "http://localhost:8000"
PASSED = 0
FAILED = 0
ERRORS = []


def get(path):
    req = urllib.request.Request(f"{BASE}{path}")
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode()), resp.status


def post(path, body):
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode()), resp.status


def get_error_code(path):
    req = urllib.request.Request(f"{BASE}{path}")
    try:
        urllib.request.urlopen(req, timeout=10)
        return 200
    except urllib.error.HTTPError as e:
        return e.code


def post_error_code(path, body):
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        urllib.request.urlopen(req, timeout=10)
        return 200
    except urllib.error.HTTPError as e:
        return e.code


def stream_sse(path, max_events=200, timeout=60):
    req = urllib.request.Request(f"{BASE}{path}")
    events = []
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for _ in range(5000):  # safety limit on raw lines
            raw_line = resp.readline()
            if not raw_line:  # EOF
                break
            line = raw_line.decode().strip()
            if not line:
                continue
            if line.startswith("data: "):
                event = json.loads(line[6:])
                events.append(event)
                if event.get("type") == "complete" or len(events) >= max_events:
                    break
    return events


def test(name, fn):
    global PASSED, FAILED
    try:
        fn()
        PASSED += 1
        print(f"  PASS  {name}")
    except Exception as e:
        FAILED += 1
        ERRORS.append((name, str(e)))
        print(f"  FAIL  {name} -- {e}")


# ============================================================
print()
print("=" * 60)
print("INKFORGE BACKEND -- Full Test Suite")
print("=" * 60)

# --- 1. Root ---
print("\n[1] Root Endpoint")


def t_root():
    data, status = get("/")
    assert status == 200
    assert data["name"] == "Inkforge API"
    assert data["version"] == "1.0.0"
    assert data["status"] == "operational"
    assert "engine" in data  # May be "mock", "lstm", "vllm", etc.

test("GET / returns app info", t_root)


# --- 2. Health ---
print("\n[2] Health Endpoint")


def t_health():
    data, status = get("/health")
    assert status == 200
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["engine"]["backend"] in ("mock", "lstm")
    assert data["engine"]["device"] in ("cpu", "cuda")
    assert data["engine"]["max_seq_len"] == 2048
    assert isinstance(data["inference"]["active_requests"], int)
    assert isinstance(data["inference"]["uptime_seconds"], float)

test("GET /health returns full engine status", t_health)


# --- 3. Generate ---
print("\n[3] Generate Endpoint")


def t_gen_minimal():
    data, status = post("/api/generate", {"text": "Hello world"})
    assert status == 202, f"Expected 202, got {status}"
    assert "job_id" in data
    assert "ws_url" in data or "stream_url" in data  # At least one transport URL
    assert data["status"] == "queued"
    assert len(data["job_id"]) == 36

test("POST /api/generate with minimal body", t_gen_minimal)


def t_gen_full():
    data, status = post("/api/generate", {
        "text": "Dear Ms. Johnson,\n\nThank you for the meeting.",
        "style_id": "elegant_formal",
        "params": {
            "stroke_width_variation": 0.7,
            "character_inconsistency": 0.3,
            "slant_angle": 10.0,
            "baseline_drift": 0.5,
            "ligature_enabled": True,
            "fatigue_simulation": 0.6,
            "ink_bleed": 0.3,
        },
        "paper_texture": "blank",
        "ink_color": "blue",
        "font_size": "large",
    })
    assert status == 202
    assert data["status"] == "queued"

test("POST /api/generate with full params", t_gen_full)


def t_gen_empty():
    code = post_error_code("/api/generate", {"text": ""})
    assert code == 422, f"Expected 422, got {code}"

test("POST /api/generate rejects empty text (422)", t_gen_empty)


def t_gen_missing():
    code = post_error_code("/api/generate", {})
    assert code == 422, f"Expected 422, got {code}"

test("POST /api/generate rejects missing text (422)", t_gen_missing)


def t_gen_bad_params():
    code = post_error_code("/api/generate", {
        "text": "Hello",
        "params": {"slant_angle": 999},
    })
    assert code == 422, f"Expected 422, got {code}"

test("POST /api/generate rejects out-of-range params (422)", t_gen_bad_params)


# --- 4. SSE Streaming ---
print("\n[4] SSE Streaming")


def t_stream_basic():
    data, _ = post("/api/generate", {"text": "Hi there"})
    job_id = data["job_id"]
    events = stream_sse(f"/api/stream/{job_id}")

    strokes = [e for e in events if e["type"] == "stroke"]
    completes = [e for e in events if e["type"] == "complete"]

    assert len(strokes) > 0, "No stroke events"
    assert len(completes) == 1, f"Expected 1 complete event, got {len(completes)}"

    first = strokes[0]
    assert first["index"] == 0
    d = first["data"]
    for key in ("dx", "dy", "p1", "p2", "p3", "char", "x", "y"):
        assert key in d, f"Missing key: {key}"

    comp = completes[0]
    assert comp["total_strokes"] > 0

test("SSE stream returns strokes + completion event", t_stream_basic)


def t_stream_long():
    text = "Hello world, this is a test of multi-word layout."
    data, _ = post("/api/generate", {"text": text})
    events = stream_sse(f"/api/stream/{data['job_id']}", max_events=2000, timeout=120)

    strokes = [e for e in events if e["type"] == "stroke"]
    completes = [e for e in events if e["type"] == "complete"]

    assert len(strokes) > 20, f"Expected 20+ strokes, got {len(strokes)}"
    assert len(completes) == 1

test("SSE stream with long text -- multi-word layout", t_stream_long)


def t_stream_paragraphs():
    data, _ = post("/api/generate", {
        "text": "Hello.\n\nWorld.",
    })
    events = stream_sse(f"/api/stream/{data['job_id']}", max_events=500, timeout=60)

    strokes = [e for e in events if e["type"] == "stroke"]
    completes = [e for e in events if e["type"] == "complete"]
    assert len(strokes) > 0, f"No strokes produced"
    assert len(completes) == 1, f"Expected 1 complete event, got {len(completes)}"
    assert completes[0]["lines"] >= 2, f"Expected >= 2 lines, got {completes[0]['lines']}"

test("SSE stream handles paragraph breaks", t_stream_paragraphs)


def t_stream_404():
    code = get_error_code("/api/stream/nonexistent-job-id")
    assert code == 404, f"Expected 404, got {code}"

test("SSE stream returns 404 for bad job_id", t_stream_404)


# --- 5. Job Status ---
print("\n[5] Job Status")


def t_job_complete():
    data, _ = post("/api/generate", {"text": "Test"})
    job_id = data["job_id"]

    # Stream to completion first
    stream_sse(f"/api/stream/{job_id}")

    status_data, code = get(f"/api/job/{job_id}")
    assert code == 200
    assert status_data["job_id"] == job_id
    assert status_data["status"] == "complete"

test("GET /api/job/{id} shows complete after streaming", t_job_complete)


def t_job_404():
    code = get_error_code("/api/job/fake-id-12345")
    assert code == 404

test("GET /api/job with bad id returns 404", t_job_404)


# --- 6. OpenAPI ---
print("\n[6] OpenAPI Schema")


def t_openapi():
    data, status = get("/openapi.json")
    assert status == 200
    assert data["info"]["title"] == "Inkforge API"
    paths = data["paths"]
    assert "/api/generate" in paths, "Missing /api/generate"
    assert "/api/stream/{job_id}" in paths, "Missing /api/stream"
    assert "/api/job/{job_id}" in paths, "Missing /api/job"
    assert "/health" in paths, "Missing /health"
    assert "/" in paths, "Missing /"

test("GET /openapi.json has all endpoints", t_openapi)


# ============================================================
print()
print("=" * 60)
print(f"RESULTS: {PASSED} passed, {FAILED} failed")
if ERRORS:
    print()
    print("Failed tests:")
    for name, err in ERRORS:
        print(f"  FAIL  {name}: {err}")
print("=" * 60)
print()

sys.exit(1 if FAILED > 0 else 0)
