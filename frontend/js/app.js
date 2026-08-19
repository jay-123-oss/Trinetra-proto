const $ = (id) => document.getElementById(id);
const api = (path, options = {}) => fetch(path, { headers: { "Content-Type": "application/json" }, ...options });
const state = { debug: false, paused: false, ws: null, alerts: [], connected: false };

function setText(id, value) { const node = $(id); if (node) node.textContent = value ?? "—"; }
function formatDistance(value) { return value == null ? "—" : `${Number(value).toFixed(1)} m`; }
function formatMs(value) { return value == null ? "—" : `${Number(value).toFixed(0)} ms`; }
function formatTime(timestamp) { try { return new Date(timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" }); } catch { return "now"; } }
function riskClass(level) { return String(level || "SAFE").toLowerCase(); }

function setSystemState(status, online = false) {
  setText("systemStatus", status || "OFFLINE");
  $("systemDot").className = `status-dot ${online ? "online" : "warn"}`;
  setText("pipelineState", status || "OFFLINE");
}

function updateSafety(data) {
  const level = data.risk_level || "SAFE";
  const cls = riskClass(level);
  const display = $("riskDisplay");
  display.className = `risk-display risk-${cls}`;
  $("stateChip").className = `state-chip risk-${cls}`;
  setText("riskLevel", level);
  setText("stateChip", level);
  setText("riskMessage", data.message || "Path clear.");
  setText("hazardValue", data.hazard ? String(data.hazard).replaceAll("_", " ") : "—");
  setText("distanceValue", formatDistance(data.distance_m));
  setText("directionValue", data.direction ? String(data.direction).replaceAll("_", " ") : "—");
  setText("actionValue", data.recommended_action || "CONTINUE");
  setText("debugFrame", `FRAME ${data.frame_id ?? "—"}`);
  if (data.system_state) setSystemState(data.system_state, true);
}

function updateDetections(data) {
  const objects = Array.isArray(data.objects) ? data.objects : [];
  setText("trackCount", `${(data.tracking?.active_tracks ?? objects.length)} ACTIVE`);
  const list = $("detectionList");
  if (!objects.length) { list.innerHTML = '<div class="empty-state">No tracked objects.</div>'; return; }
  const relevant = objects.filter(item => (item.path_relevance ?? 0) > 0.15 || ["vehicle", "car", "motorcycle", "bicycle", "pothole", "stairs"].includes(String(item.class_name).toLowerCase())).slice(0, 12);
  list.innerHTML = relevant.map(item => `<div class="detection-row"><div><div class="detection-name">${escapeHtml(item.class_name || "object")}</div><div class="motion-tag">${escapeHtml(item.motion_state || "UNKNOWN")}</div></div><div class="detection-meta">${Math.round((item.confidence || 0) * 100)}%<br>${formatDistance(item.distance_m)}</div><div class="detection-meta">${escapeHtml(item.direction || "—")}<br><span class="muted">#${item.track_id ?? "—"}</span></div></div>`).join("");
}

function updateMetrics(data) {
  setText("cameraFps", data.camera_fps == null ? "—" : Number(data.camera_fps).toFixed(1));
  setText("processingFps", data.processing_fps == null ? "—" : Number(data.processing_fps).toFixed(1));
  setText("yoloLatency", formatMs(data.yolo_latency_ms));
  setText("vlmLatency", formatMs(data.vlm_latency_ms));
  setText("totalLatency", formatMs(data.total_latency_ms));
  setText("droppedFrames", data.dropped_frames ?? "—");
  setText("gpuMemory", data.gpu_memory == null ? "—" : `${Number(data.gpu_memory).toFixed(0)} MB`);
  setText("activeTracks", data.active_tracks ?? "—");
}

function updateHealth(data) {
  const online = data.status === "healthy" || data.status === "ok";
  setSystemState(online ? "SYSTEM ONLINE" : String(data.status || "DEGRADED").toUpperCase(), online);
  setText("cameraState", `CAMERA: ${String(data.camera_status || (data.camera ? "CONNECTED" : "DISCONNECTED"))}`);
  setText("cameraReady", data.camera ? "CONNECTED" : "OFFLINE");
  setText("yoloReady", data.yolo ? "READY" : "ERROR");
  setText("vlmReady", data.vlm ? "READY" : "UNAVAILABLE");
  setText("gpuReady", data.gpu ? "ACTIVE" : "CPU MODE");
}

function updateAlerts(events) {
  if (!Array.isArray(events)) return;
  state.alerts = events.slice(0, 20);
  setText("alertCount", state.alerts.length);
  const list = $("alertList");
  if (!state.alerts.length) { list.innerHTML = '<div class="empty-state">No priority events yet.</div>'; return; }
  list.innerHTML = state.alerts.map(event => `<div class="alert-row ${riskClass(event.priority)}"><div class="alert-priority">${escapeHtml(event.priority || "INFO")}</div><div class="alert-message">${escapeHtml(event.message || "Safety event")}</div><div class="alert-time">${formatTime(event.timestamp)}</div></div>`).join("");
}

function applyAnalysis(data) { if (!state.paused) updateSafety(data); updateDetections(data); if (data.performance) updateMetrics(data.performance); }
function escapeHtml(value) { return String(value).replace(/[&<>'"]/g, char => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" }[char])); }

async function refresh() {
  try {
    const [healthResponse, analysisResponse, metricsResponse, eventsResponse] = await Promise.all([api("/health"), api("/analyze"), api("/metrics"), api("/events")]);
    if (healthResponse.ok) updateHealth(await healthResponse.json());
    if (analysisResponse.ok) applyAnalysis(await analysisResponse.json());
    if (metricsResponse.ok) updateMetrics(await metricsResponse.json());
    if (eventsResponse.ok) updateAlerts((await eventsResponse.json()).events);
    state.connected = true;
  } catch (error) { state.connected = false; setSystemState("BACKEND OFFLINE", false); }
}

function connectWebSocket() {
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  try { state.ws = new WebSocket(`${protocol}://${location.host}/ws`); } catch { return; }
  state.ws.onopen = () => setSystemState("SYSTEM ONLINE", true);
  state.ws.onmessage = event => {
    try {
      const data = JSON.parse(event.data);
      if (data.type === "safety") updateSafety(data);
      if (data.type === "metrics") updateMetrics(data);
      if (data.type === "alert") updateAlerts([data, ...state.alerts].slice(0, 20));
    } catch { /* ignore malformed update in non-debug mode */ }
  };
  state.ws.onclose = () => { state.ws = null; setTimeout(connectWebSocket, 3000); };
  state.ws.onerror = () => state.ws?.close();
}

async function cameraAction(path) { try { const response = await api(path, { method: "POST" }); if (!response.ok) throw new Error(await response.text()); await refresh(); } catch { setText("cameraState", "CAMERA: ERROR"); } }
function setStream(active) { const image = $("cameraStream"); if (active) { image.src = `/stream?ts=${Date.now()}`; image.classList.add("visible"); $("cameraPlaceholder").classList.add("hidden"); } else { image.removeAttribute("src"); image.classList.remove("visible"); $("cameraPlaceholder").classList.remove("hidden"); } }

$("startCamera").addEventListener("click", () => { state.paused = false; setStream(true); cameraAction("/camera/start"); });
$("stopCamera").addEventListener("click", () => { setStream(false); cameraAction("/camera/stop"); });
$("pauseCamera").addEventListener("click", event => { state.paused = !state.paused; event.currentTarget.textContent = state.paused ? "RESUME" : "PAUSE"; });
$("debugToggle").addEventListener("click", event => { state.debug = !state.debug; $("debugOverlay").classList.toggle("hidden", !state.debug); event.currentTarget.classList.toggle("btn-primary", state.debug); });
$("settingsToggle").addEventListener("click", () => { $("settingsDrawer").classList.add("open"); $("drawerBackdrop").classList.add("open"); $("settingsDrawer").setAttribute("aria-hidden", "false"); });
function closeSettings() { $("settingsDrawer").classList.remove("open"); $("drawerBackdrop").classList.remove("open"); $("settingsDrawer").setAttribute("aria-hidden", "true"); }
$("settingsClose").addEventListener("click", closeSettings); $("drawerBackdrop").addEventListener("click", closeSettings);
$("settingsForm").addEventListener("submit", async event => { event.preventDefault(); const payload = { url: $("cameraUrl").value, yolo_confidence: Number($("yoloConfidence").value), processing_fps: Number($("processingFps").value), vlm_interval_ms: Number($("vlmInterval").value), warning_distance_m: Number($("warningDistance").value), danger_distance_m: Number($("dangerDistance").value), alert_cooldown_s: Number($("alertCooldown").value), debug: $("debugMode").checked }; try { const response = await api("/settings", { method: "POST", body: JSON.stringify(payload) }); if (!response.ok) throw new Error(await response.text()); $("settingsNote").textContent = "Settings saved to backend."; } catch { $("settingsNote").textContent = "Could not save settings. Check backend connection."; } });

setInterval(() => setText("clock", new Date().toLocaleTimeString([], { hour12: false })), 1000);
setInterval(refresh, 1500);
refresh(); connectWebSocket();
