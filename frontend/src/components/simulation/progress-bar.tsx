"use client";

import type { WSProgress } from "@/lib/types";

const STEP_LABELS: Record<string, string> = {
  solar_positions: "太陽位置を計算中...",
  clear_sky: "クリアスカイ日射量を計算中...",
  ray_casting: "影のレイキャスティング中...",
  irradiance: "年間日射量を計算中...",
  done: "完了",
  error: "エラー",
};

interface ProgressBarProps {
  progress: WSProgress | null;
  isRunning: boolean;
}

export function ProgressBar({ progress, isRunning }: ProgressBarProps) {
  if (!isRunning && !progress) return null;

  const pct = (progress?.progress ?? 0) * 100;
  const label = progress?.step ? STEP_LABELS[progress.step] || progress.message : "待機中...";

  return (
    <div className="space-y-2 rounded-lg border bg-blue-50 p-4">
      <div className="flex justify-between text-xs text-gray-600">
        <span>{label}</span>
        <span>{pct.toFixed(0)}%</span>
      </div>
      <div className="h-2 overflow-hidden rounded-full bg-blue-100">
        <div
          className="h-full rounded-full bg-blue-500 transition-all duration-500"
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}
