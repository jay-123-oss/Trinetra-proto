# Trinetra Vision Engineering Audit

## Audit scope

This audit follows the selected **Scope A** decision: preserve the current Python/Colab prototype. The phone remains the camera source, the existing YOLO and VLM objects remain in the same Google Colab GPU runtime, and the browser client visualizes backend results. No Android project, CameraX implementation, ARCore session, on-device model, TTS engine, command parser, or navigation subsystem exists in the current repository, so none is claimed as implemented.

## Repository reality

The current project is a Python backend plus static web dashboard. There are no Kotlin or Java files, no Gradle wrapper, no AndroidManifest, no Compose source, and no Android assets. Therefore Android-first requirements from the larger master prompt are classified as **not applicable to the selected current prototype scope**, rather than fabricated.

## Actual runtime map

| Subsystem | Source exists | Instantiated | Initialized | Connected | Runs at runtime | Output used |
|---|---:|---:|---:|---:|---:|---:|
| Phone MJPEG camera | Yes | Yes, when `camera_url` is supplied | Yes | Network-dependent | Yes | Latest-frame buffer and stream relay |
| Latest-frame buffer | Yes | Yes, in `TrinetraRuntime` | Yes | Camera publishes into it | Yes | Inference worker consumes newest frame |
| Frame quality/orientation | Yes | Yes | Yes | Receives buffered frame | Yes | Degraded-state decision and hazard confidence |
| Existing YOLO | Yes, injected object | Yes, in `YoloEngine` | Optional warmup | Receives latest processed frame | Yes | Normalized detections |
| Temporal tracking | Yes | Yes | Yes | Receives YOLO detections | Yes | Stable IDs and motion history |
| Motion engine | Yes | Yes | Yes | Receives tracked detections | Yes | Approach/lateral state |
| Spatial/corridor engine | Yes | Yes | Yes | Receives frame and detections | Yes | Direction, approximate distance, path relevance |
| Ground-hazard engine | Yes | Yes | Yes | Receives lower-frame quality | Yes | Conservative ground candidates |
| Collision risk | Yes | Yes | Yes | Receives spatial detections | Yes | TTC and collision score |
| Existing VLM | Yes, injected object | Yes, in `VLMTriggerManager` | Worker initialized | Event-triggered only | Yes when required | Validated semantic evidence |
| Multimodal fusion | Yes | Yes | Yes | Receives hazards and current-frame VLM result | Yes | Fused hazards |
| Safety engine | Yes | Yes | Yes | Receives physical and semantic evidence | Yes | Sole final risk/action decision |
| Alert manager | Yes | Yes | Yes | Receives final safety state | Yes | Deduplicated alert events |
| FastAPI | Yes | Yes through `create_app` | Lifespan-managed | Exposes runtime | Yes | Health, analysis, camera, metrics, WebSocket |
| Browser dashboard | Yes | Served by FastAPI | Browser initializes it | WebSocket/REST/stream | Yes | Visualization only |

## Actual fast and slow paths

The fast path is:

```text
MJPEG → one-slot buffer → preprocess → existing YOLO → tracking → motion → spatial/corridor → risk → SafetyEngine → AlertManager → latest API result
```

The slow path is:

```text
Current frame + safety context → trigger policy → one active VLM request + one replaceable pending request → strict JSON validation → frame-age check → fusion → SafetyEngine
```

A delayed VLM response is ignored when it is more than two processed frames older than the current frame. Immediate physical collision evidence is evaluated before VLM output and is not downgraded by VLM semantics.

## Repairs made during this audit

The runtime now converts invalid frames and YOLO failures into an explicit **CAUTION/degraded** safety state instead of returning a default SAFE state with an error field. This is fail-safe behavior: perception failure is not interpreted as free space. The runtime also validates VLM frame age before fusion so a late semantic response cannot contaminate a newer safety decision.

## Verified

The following are verified through local Python tests and compilation: latest-frame replacement, normalized YOLO results, stable tracking IDs, spatial corridor behavior, TTC, immediate-collision precedence, VLM JSON validation, alert cooldown, FastAPI health/analyze, dashboard static serving, settings update, WebSocket updates, and model-boundary failure behavior.

## Partially verified

Actual camera decoding and reconnect behavior are implemented but require a reachable phone MJPEG stream to validate end-to-end. Actual GPU usage and latency require the real Colab-loaded YOLO/VLM objects. Tunnel behavior requires a live Colab runtime and public tunnel. Real-world detection accuracy, distance accuracy, thermal adaptation, and human-factors alert quality require physical scenarios.

## Not verified / intentionally out of scope

Android CameraX, ARCore depth/point-cloud, on-device YOLO/VLM, Kotlin lifecycle, Compose HUD, TTS/audio priority, command parsing, object search, navigation, OCR, GPS, and local mobile inference are not present in the current Python/Colab prototype and are not claimed as operational.
