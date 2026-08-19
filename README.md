# Trinetra Vision — Complete Python AI Safety System

Trinetra is a modular Python real-time vision and safety system for the current **phone-camera → MJPEG → Google Colab GPU** environment. The phone is only the camera source. The existing YOLO and VLM objects must remain loaded in the same Colab runtime; this repository wraps them and never retrains, replaces, reinstalls, or downloads duplicate models.

> **Safety precedence:** immediate physical collision evidence outranks spatial evidence, tracking, YOLO semantics, and VLM interpretation. A slow or incorrect VLM result cannot downgrade a close collision-path warning.

## Integrated pipeline

```text
PHONE CAMERA
    ↓
MJPEG CAMERA READER + RECONNECT
    ↓
LATEST-FRAME BUFFER
    ↓
FRAME QUALITY / ORIENTATION PROCESSING
    ↓
YOLO FAST PERCEPTION (existing instance)
    ↓
TEMPORAL TRACKING
    ↓
MOTION / APPROACH ANALYSIS
    ↓
SPATIAL / DISTANCE / WALKING CORRIDOR
    ↓
GROUND HAZARD + COLLISION RISK
    ↓
VLM TRIGGER MANAGER (event-driven)
    ↓
ASYNC VLM VERIFICATION (existing instance)
    ↓
MULTIMODAL FUSION
    ↓
CENTRAL SAFETY ENGINE
    ↓
ALERT MANAGER
    ↓
LATEST STRUCTURED RESULT
    ↓
FASTAPI /health /analyze /metrics
```

## Repository structure

| Path | Purpose |
| --- | --- |
| `colab/trinetra/config.py` | Environment-backed thresholds and runtime settings. |
| `colab/trinetra/models.py` | Unified frame-analysis and API data structures. |
| `colab/trinetra/camera.py` | Reconnecting MJPEG reader with camera FPS and invalid-frame handling. |
| `colab/trinetra/frame_buffer.py` | Thread-safe one-slot latest-frame buffer. |
| `colab/trinetra/preprocess.py` | Orientation handling and brightness/contrast/blur/visibility estimates. |
| `colab/trinetra/yolo_engine.py` | Adapter for the already-loaded YOLO object and normalized detections. |
| `colab/trinetra/tracker.py` | Stable-ID temporal tracker with an explicit ByteTrack replacement seam. |
| `colab/trinetra/motion.py` | Smoothed approaching/receding/lateral motion states. |
| `colab/trinetra/spatial.py` | Approximate distance sources, direction, corridor overlap, and future depth injection. |
| `colab/trinetra/hazards.py` | Safety-relevant hazard classification and conservative ground candidates. |
| `colab/trinetra/risk.py` | Collision risk, TTC, approaching vehicle, and hazard priority calculations. |
| `colab/trinetra/vlm.py` | One-active/one-pending asynchronous VLM trigger manager and JSON validator. |
| `colab/trinetra/fusion.py` | Weighted multimodal evidence fusion. |
| `colab/trinetra/safety.py` | Sole producer of final SAFE/CAUTION/WARNING/DANGER/EMERGENCY state. |
| `colab/trinetra/alerts.py` | Deduplication, cooldown, escalation, and machine-readable events. |
| `colab/trinetra/metrics.py` | Rolling 10/50/100-frame performance metrics. |
| `colab/trinetra/runtime.py` | Persistent background inference loop and latest-result cache. |
| `colab/trinetra/server.py` | FastAPI `/health`, `/analyze`, `/metrics`. |
| `colab/run_server.py` | Colab integration entry point. |
| `demo.py` | Optional local/debug frame injection mode. |
| `backend/` | Production-style backend facade for the persistent Colab runtime. |
| `frontend/` | Static responsive browser dashboard served by FastAPI. |
| `scripts/benchmark.py` | Runtime benchmark with injected fake adapters. |
| `tests/` | Automated safety, parsing, API, WebSocket, frontend-contract, and failure tests. |

## Colab startup

Install only the service dependencies into the current Colab runtime. Do not reinstall the working model stack or download new weights:

```python
%pip install -r colab/requirements.txt
```

The existing notebook must already have working model objects, for example `yolo_model` and `vlm_model`. From the repository root:

```python
from colab.run_server import build_app

app = build_app(
    existing_yolo_model=yolo_model,
    existing_vlm_model=vlm_model,
    camera_url="http://PHONE_IP:8080/video",
)
```

Start the persistent API in the same runtime:

```python
import nest_asyncio
import uvicorn

nest_asyncio.apply()
uvicorn.run(app, host="0.0.0.0", port=8000)
```

Expose port `8000` using the existing Cloudflare/trycloudflare tunnel method. The tunnel URL must be configured or printed as `BASE_URL`; it is not hardcoded in this repository.

### Exact phone camera URL format

```text
http://PHONE_IP:8080/video
```

Use the phone's actual reachable IP or the existing tunnel/network address. The current system does not build an Android application.

## API requests

```bash
curl "$BASE_URL/health"
curl "$BASE_URL/analyze"
curl "$BASE_URL/metrics"
```

Example health response:

```json
{
  "status": "healthy",
  "camera": true,
  "camera_status": "CONNECTED",
  "yolo": true,
  "vlm": true,
  "gpu": true,
  "fps": 14.3,
  "system_state": "SAFE"
}
```

Example analysis response:

```json
{
  "timestamp": "2026-08-19T00:00:00+00:00",
  "risk_level": "DANGER",
  "danger": true,
  "hazard": "pothole",
  "hazard_confidence": 0.91,
  "distance_m": 1.1,
  "distance_confidence": 0.78,
  "direction": "center",
  "severity": "high",
  "recommended_action": "stop",
  "message": "Danger. pothole ahead approximately 1.1 meters.",
  "objects": [],
  "hazards": [],
  "yolo": {"latency_ms": 55.0, "fps": 14.2},
  "tracking": {},
  "motion": {},
  "spatial": {},
  "vlm": {"used": true, "valid": true, "hazard_type": "pothole", "confidence": 0.91, "latency_ms": 260.0},
  "alert": {"should_alert": true, "priority": "DANGER", "message": "Danger. pothole ahead approximately 1.1 meters."},
  "performance": {"total_latency_ms": 330.0},
  "system_state": "DANGER",
  "frame_id": 42
}
```

`/analyze` reads the latest cached result. It does not start another camera or model inference for each HTTP request.

## Browser dashboard

The FastAPI process serves the static dashboard at `/`. Open the same public tunnel base URL in a browser after starting the server:

```text
https://<tunnel-host>/
```

The dashboard maintains safety state, detections, alerts, metrics, pipeline status, and camera state without reloading the page. It uses `/ws` for low-latency safety, alert, metrics, and system-state updates, with REST polling as a fallback. The backend owns the single camera connection and exposes the cached frame relay at `/stream`; the browser never processes the phone MJPEG stream directly.

Additional endpoints are available for the dashboard:

```text
POST /camera/start
POST /camera/stop
POST /camera/config     {"url": "http://PHONE_IP:8080/video"}
POST /settings
GET  /events
GET  /stream
GET  /ws
```

The frontend is deliberately lightweight HTML/CSS/JavaScript, responsive on desktop and mobile, high-contrast, keyboard-friendly, and uses text alongside color for every safety state. Debug overlays are optional and disabled by default.

## Configuration

The main settings are configurable with environment variables, including `TRINETRA_CAMERA_URL`, `TRINETRA_YOLO_CONFIDENCE`, `TRINETRA_PROCESSING_FPS`, `TRINETRA_VLM_MIN_INTERVAL_MS`, `TRINETRA_EMERGENCY_DISTANCE_M`, `TRINETRA_DANGER_DISTANCE_M`, `TRINETRA_WARNING_DISTANCE_M`, `TRINETRA_ALERT_COOLDOWN_S`, `TRINETRA_API_HOST`, `TRINETRA_API_PORT`, and `TRINETRA_DEBUG`.

## Validation and demo mode

Run automated tests and syntax checks:

```bash
python -m pytest -q
python -m compileall -q colab backend demo.py run.py scripts
python scripts/benchmark.py
```

Use the optional demo mode for fake frames and adapter checks without loading any model weights:

```bash
python demo.py
```

This mode validates control flow only. Real-world accuracy, false positives, false negatives, alert delay, VLM trigger frequency, and latency must be measured using actual phone-camera scenarios.

## Startup options

In the existing Colab notebook, use the injected-model path shown above. For a reusable process, set a provider function that returns the already-loaded `(yolo_model, vlm_model)` tuple:

```bash
export TRINETRA_MODEL_PROVIDER=my_models:loaded_models
python run.py
```

The provider must reuse the working Colab model objects. `run.py` never downloads or initializes model weights. With the camera URL configured, the server prints the API base and serves the dashboard at the same URL.

## Explicit non-goals

This fresh repository intentionally contains no Android code, no ARCore runtime, no on-device YOLO/VLM, no model weights, no model downloads, and no fabricated depth data. ARCore/depth is represented only by an injectable interface. OCR, navigation, product/currency recognition, voice assistant, and future local deployment remain later modules that must not compromise the core safety loop.

## Safety limitation

Trinetra is a research-stage prototype, not a certified medical, mobility, or collision-avoidance device. It must not be used as the sole basis for real-world navigation or emergency decisions.
