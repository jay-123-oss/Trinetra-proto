# Trinetra Vision Architecture

## Current boundary

The fresh repository is Python-only. The phone publishes an MJPEG camera stream, while the same Google Colab GPU process owns the already-loaded YOLO and VLM instances. No Android module or on-device inference is included.

## Data flow

`MjpegCamera` publishes a timestamped `Frame` into `LatestFrameBuffer`. `TrinetraRuntime` consumes the newest frame, validates quality, runs persistent YOLO, assigns temporal track IDs, estimates motion, enriches detections with spatial/corridor information, scores collision risk, and creates hazards. It asks `VLMTriggerManager` for semantic verification only when ambiguity or meaningful scene risk requires it. `FusionEngine` combines evidence, then `SafetyEngine` produces exactly one final current safety state. `AlertManager` converts state changes into prioritized events.

## Safety precedence

The immediate-collision fast path is computed from physical proximity, corridor relevance, motion, and collision risk before VLM output is considered. VLM can add a hazard name or clarify a ground region, but it cannot turn a physically close collision-path object into SAFE. This rule is intentional and tested.

## Latest-frame and async guarantees

`LatestFrameBuffer` has exactly one pending slot. Publishing a new camera image replaces an older pending image and increments the dropped-frame count. The VLM manager has at most one active request and one replaceable pending request. It never creates an unbounded VLM queue and never blocks the YOLO loop while a semantic request is running.

## Model reuse

`YoloEngine` and `VLMTriggerManager` require objects passed by the existing notebook. They do not import a model hub, resolve weights, call `from_pretrained`, or create a second CUDA model. Warmup is optional and delegates to an already-loaded model's own warmup method when available.

## Failure isolation

Camera errors trigger reconnect attempts and appear in health status. Invalid frames become degraded frame-quality results. YOLO errors produce a degraded result while the background loop remains alive. Invalid or failed VLM output is represented as `valid: false` and does not stop YOLO, spatial analysis, risk, or SafetyEngine. GPU memory is observed when PyTorch is available; the runtime otherwise remains CPU-compatible for basic control-flow testing.

## Future seams

A `DepthProvider` can be injected into `SpatialEngine` for future depth or ARCore data; no fake ARCore values are generated in this Colab version. A production ByteTrack adapter can replace the dependency-light tracker without changing normalized detections or the safety layers. The API layer remains separate from perception and safety so future local inference can reuse the same contracts.
