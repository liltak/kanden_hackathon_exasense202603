// TypeScript types mirroring Pydantic schemas

export interface SimulationRequest {
  latitude: number;
  longitude: number;
  year: number;
  time_resolution_minutes: number;
  panel_efficiency: number;
  electricity_price_jpy: number;
  mesh_source: "uploaded" | "simple" | "complex";
}

export interface FaceIrradiance {
  face_id: number;
  annual_irradiance_kwh_m2: number;
  annual_direct_kwh_m2: number;
  annual_diffuse_kwh_m2: number;
  area_m2: number;
  normal: [number, number, number];
  sun_hours: number;
}

export interface PanelProposal {
  face_id: number;
  area_m2: number;
  annual_generation_kwh: number;
  installed_capacity_kw: number;
  installation_cost_jpy: number;
  annual_savings_jpy: number;
  payback_years: number;
  npv_25y_jpy: number;
  irr_percent: number;
  priority_rank: number;
}

export interface ROIReport {
  proposals: PanelProposal[];
  total_area_m2: number;
  total_capacity_kw: number;
  total_annual_generation_kwh: number;
  total_installation_cost_jpy: number;
  total_annual_savings_jpy: number;
  overall_payback_years: number;
  overall_npv_25y_jpy: number;
}

export interface SimulationResult {
  task_id: string;
  status: "pending" | "running" | "complete" | "failed";
  progress: number;
  step: string | null;
  message: string | null;
  irradiance: FaceIrradiance[] | null;
  roi_report: ROIReport | null;
  elapsed_seconds: number | null;
  monthly_ghi: number[] | null;
}

export interface MeshInfo {
  mesh_id: string;
  num_vertices: number;
  num_faces: number;
  surface_area_m2: number;
  bounds_min: number[];
  bounds_max: number[];
  download_url: string;
}

export interface WSProgress {
  task_id: string;
  step: string;
  progress: number;
  message: string;
}

export interface ChatResponse {
  response: string;
  session_id: string | null;
}

export interface MonthlyGHI {
  months: string[];
  ghi_kwh_m2: number[];
}

export interface ReportResponse {
  markdown: string;
  download_urls: Record<string, string>;
}

// Solar animation types

export interface SunPositionEntry {
  time: string;
  azimuth: number;
  elevation: number;
  direction_y_up: [number, number, number];
}

export interface SunPositionsResponse {
  date: string;
  latitude: number;
  longitude: number;
  freq_minutes: number;
  positions: SunPositionEntry[];
}

export interface ShadowTimelineResponse {
  date: string;
  mesh_source: string;
  n_faces: number;
  n_steps: number;
  times: string[];
  shadow_matrix: boolean[][];
}

// Preset / pre-built reconstruction data

export interface PresetInfo {
  name: string;
  n_faces: number;
  n_vertices: number;
  surface_area_m2: number;
}

export interface PresetsResponse {
  presets: PresetInfo[];
}

// Reconstruction types

export interface ReconstructionStatus {
  task_id: string;
  status: "pending" | "running" | "complete" | "failed";
  progress: number;
  step: string | null;
  message: string | null;
  mesh_id: string | null;
}

export type WaypointSeedSource = "demo" | "upload" | "viewer";
export type WaypointViewName = "bird" | "south" | "west";

export interface WaypointGenerateRequest {
  prompt: string;
  negative_prompt?: string | null;
  seed_source: WaypointSeedSource;
  seed_image_data_url?: string | null;
  template_key?: string | null;
  view_name: WaypointViewName;
  steps: number;
  guidance_scale: number;
  strength: number;
}

export interface WaypointMetric {
  annual_generation_kwh: number;
  co2_reduction_tons: number;
  installed_capacity_kw: number;
  estimated_payback_years: number;
}

export interface WaypointVariant {
  id: string;
  view_name: WaypointViewName;
  label: string;
  image_data_url: string;
}

export interface WaypointGenerateResponse {
  request_id: string;
  status: "queued" | "running" | "complete" | "failed";
  mock_mode: boolean;
  prompt: string;
  negative_prompt: string | null;
  view_name: WaypointViewName;
  seed_image_data_url: string;
  result_image_data_url: string;
  metrics: WaypointMetric;
  variants: WaypointVariant[];
  latency_ms: number;
}

export interface WaypointStatus {
  service_status: "ready" | "loading" | "unavailable";
  mock_mode: boolean;
  model_name: string;
  device: string;
  gpu_available: boolean;
  model_loaded: boolean;
  queue_depth: number;
  vram_used_gb: number | null;
}

// Rust Inspection (OpenVLA)

export interface RustInspectionRunRequest {
  seed: number;
  grid_rows: number;
  grid_cols: number;
  max_steps: number;
  coverage_threshold: number;
}

export interface RustInspectionMetrics {
  coverage_rate: number;
  total_steps: number;
  backtrack_count: number;
  component_jumps: number;
  rust_patch_count: number;
  visited_rust_count: number;
  grid_rows: number;
  grid_cols: number;
}

export interface RustInspectionResult {
  trajectory_image_data_url: string;
  source_image_data_url: string | null;
  metrics: RustInspectionMetrics;
}

export interface RustInspectionRunResponse {
  request_id: string;
  status: "complete" | "failed";
  mock_mode: boolean;
  result: RustInspectionResult;
}

export interface RustInspectionStatus {
  service_status: "ready" | "loading" | "unavailable";
  mock_mode: boolean;
  model_name: string;
  device: string;
  gpu_available: boolean;
  model_loaded: boolean;
  vram_used_gb: number | null;
}
