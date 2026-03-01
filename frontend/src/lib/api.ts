// FastAPI client

import type {
  ChatResponse,
  MeshInfo,
  MonthlyGHI,
  PresetsResponse,
  ReconstructionStatus,
  ReportResponse,
  ShadowTimelineResponse,
  SimulationRequest,
  SimulationResult,
  SunPositionsResponse,
} from "./types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

async function fetchJSON<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...init?.headers,
    },
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`API error ${res.status}: ${body}`);
  }
  return res.json();
}

// Simulation
export async function startSimulation(req: SimulationRequest): Promise<SimulationResult> {
  return fetchJSON("/api/simulation/run", {
    method: "POST",
    body: JSON.stringify(req),
  });
}

export async function getSimulationStatus(taskId: string): Promise<SimulationResult> {
  return fetchJSON(`/api/simulation/${taskId}`);
}

export async function getMonthlyGHI(taskId: string): Promise<MonthlyGHI> {
  return fetchJSON(`/api/simulation/${taskId}/monthly`);
}

// Mesh
export async function uploadMesh(file: File): Promise<MeshInfo> {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch(`${API_BASE}/api/mesh/upload`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) throw new Error(`Upload failed: ${res.status}`);
  return res.json();
}

export function getDemoMeshURL(type: "simple" | "complex"): string {
  return `${API_BASE}/api/mesh/demo/${type}`;
}

export function getHeatmapMeshURL(type: "simple" | "complex", taskId: string): string {
  return `${API_BASE}/api/mesh/demo/${type}/heatmap?task_id=${taskId}`;
}

// Chat
export async function sendChatMessage(message: string, sessionId?: string): Promise<ChatResponse> {
  return fetchJSON("/api/chat", {
    method: "POST",
    body: JSON.stringify({ message, session_id: sessionId }),
  });
}

// Report
export async function generateReport(taskId?: string): Promise<ReportResponse> {
  return fetchJSON("/api/report/generate", {
    method: "POST",
    body: JSON.stringify({ task_id: taskId }),
  });
}

export function getReportDownloadURL(taskId: string, format: "md" | "json"): string {
  return `${API_BASE}/api/report/${taskId}/download/${format}`;
}

// Config
export async function getConfig(): Promise<Record<string, unknown>> {
  return fetchJSON("/api/config");
}

// Solar Animation
export async function getSolarPositions(
  date: string,
  lat = 34.69,
  lng = 135.50,
  freq = 15,
): Promise<SunPositionsResponse> {
  return fetchJSON(
    `/api/solar/positions?date=${date}&lat=${lat}&lng=${lng}&freq=${freq}`,
  );
}

export async function getShadowTimeline(
  date: string,
  meshSource = "complex",
  lat = 34.69,
  lng = 135.50,
  freq = 15,
): Promise<ShadowTimelineResponse> {
  return fetchJSON(
    `/api/solar/shadow-timeline?date=${date}&mesh_source=${meshSource}&lat=${lat}&lng=${lng}&freq=${freq}`,
  );
}

export function getMonthlyHeatmapURL(
  type: "simple" | "complex",
  month: number,
): string {
  return `${API_BASE}/api/mesh/demo/${type}/heatmap/monthly?month=${month}`;
}

// Presets (pre-built H100 reconstruction data)
export async function getPresets(): Promise<PresetsResponse> {
  return fetchJSON("/api/reconstruction/presets");
}

export async function loadPreset(name: string): Promise<ReconstructionStatus> {
  const formData = new FormData();
  formData.append("name", name);
  const res = await fetch(`${API_BASE}/api/reconstruction/load-preset`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`API error ${res.status}: ${body}`);
  }
  return res.json();
}

// Reconstruction
export async function startReconstruction(
  files: File[],
  method: "vggt" | "colmap" = "vggt",
): Promise<ReconstructionStatus> {
  const formData = new FormData();
  for (const file of files) {
    formData.append("files", file);
  }
  formData.append("method", method);
  const res = await fetch(`${API_BASE}/api/reconstruction/start`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`API error ${res.status}: ${body}`);
  }
  return res.json();
}

export async function getReconstructionStatus(
  taskId: string,
): Promise<ReconstructionStatus> {
  return fetchJSON(`/api/reconstruction/${taskId}`);
}

export function getMeshGlbURL(meshId: string): string {
  return `${API_BASE}/api/mesh/${meshId}/glb`;
}

// WebSocket
export function createSimulationWS(taskId: string): WebSocket {
  const wsBase = API_BASE.replace(/^http/, "ws");
  return new WebSocket(`${wsBase}/api/ws/simulation/${taskId}`);
}
