// Preset locations and other constants

export const PRESET_LOCATIONS: Record<string, { lat: number; lon: number }> = {
  "大阪（関西電力エリア）": { lat: 34.69, lon: 135.5 },
  東京: { lat: 35.68, lon: 139.77 },
  名古屋: { lat: 35.18, lon: 136.91 },
  福岡: { lat: 33.59, lon: 130.4 },
  札幌: { lat: 43.06, lon: 141.35 },
};

export const PIPELINE_PHASES = [
  { id: 1, name: "3D再構築", status: "pending" as const },
  { id: 2, name: "メッシュ処理", status: "pending" as const },
  { id: 3, name: "日照シミュレーション", status: "ready" as const },
  { id: 4, name: "AI分析 (VLM)", status: "pending" as const },
  { id: 5, name: "WebUI", status: "active" as const },
];

export const CHAT_EXAMPLES = [
  "この施設で太陽光パネルの設置に最も適した場所はどこですか？",
  "エネルギーコスト削減のための改善提案を3つ挙げてください。",
  "投資回収期間とROIの詳細を教えてください。",
  "この施設の分析結果を要約してください。",
];

export const MONTHLY_COLORS = [
  "#4fc3f7", "#4fc3f7", "#81c784", "#81c784", "#81c784",
  "#ffb74d", "#ffb74d", "#ffb74d", "#81c784", "#81c784",
  "#4fc3f7", "#4fc3f7",
];

export const MONTHS_JA = [
  "1月", "2月", "3月", "4月", "5月", "6月",
  "7月", "8月", "9月", "10月", "11月", "12月",
];
