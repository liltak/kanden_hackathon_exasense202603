"use client";

import { useState } from "react";
import { useRustInspectionStatus, useRustInspectionRun } from "@/hooks/use-rust-inspection";
import type { RustInspectionRunResponse } from "@/lib/types";

export default function RustInspectionPage() {
  const statusQuery = useRustInspectionStatus();
  const runMutation = useRustInspectionRun();

  const [gridRows, setGridRows] = useState(5);
  const [gridCols, setGridCols] = useState(5);
  const [seed, setSeed] = useState(42);
  const [maxSteps, setMaxSteps] = useState(200);
  const [coverageThreshold, setCoverageThreshold] = useState(0.95);
  const [result, setResult] = useState<RustInspectionRunResponse | null>(null);

  const handleRun = () => {
    runMutation.mutate(
      {
        seed,
        grid_rows: gridRows,
        grid_cols: gridCols,
        max_steps: maxSteps,
        coverage_threshold: coverageThreshold,
      },
      {
        onSuccess: (data) => setResult(data),
      },
    );
  };

  const status = statusQuery.data;
  const metrics = result?.result?.metrics;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-xl font-bold text-gray-900">設備診断 (Rust Inspection)</h2>
        <p className="mt-1 text-sm text-gray-500">
          OpenVLA エージェントによるインフラ画像のサビ経路自動トレーシング
        </p>
      </div>

      {/* Status Badge */}
      {status && (
        <div className="flex items-center gap-3 rounded-lg border bg-white p-4">
          <div
            className={`h-2.5 w-2.5 rounded-full ${
              status.service_status === "ready"
                ? "bg-green-500"
                : status.service_status === "loading"
                  ? "bg-yellow-500 animate-pulse"
                  : "bg-red-500"
            }`}
          />
          <div className="flex-1">
            <p className="text-sm font-medium text-gray-700">{status.model_name}</p>
            <p className="text-xs text-gray-400">
              {status.device}
              {status.mock_mode && " (mock)"}
              {status.vram_used_gb != null && ` \u00b7 VRAM ${status.vram_used_gb} GB`}
            </p>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        {/* Left: Controls */}
        <div className="space-y-4 rounded-lg border bg-white p-5">
          <h3 className="text-sm font-semibold text-gray-800">パラメータ設定</h3>

          <div>
            <label className="mb-1 block text-xs font-medium text-gray-600">
              グリッドサイズ (行 x 列)
            </label>
            <div className="flex gap-2">
              <input
                type="number"
                min={3}
                max={10}
                value={gridRows}
                onChange={(e) => setGridRows(Number(e.target.value))}
                className="w-full rounded-md border px-3 py-2 text-sm"
              />
              <span className="flex items-center text-gray-400">x</span>
              <input
                type="number"
                min={3}
                max={10}
                value={gridCols}
                onChange={(e) => setGridCols(Number(e.target.value))}
                className="w-full rounded-md border px-3 py-2 text-sm"
              />
            </div>
          </div>

          <div>
            <label className="mb-1 block text-xs font-medium text-gray-600">シード値</label>
            <input
              type="number"
              value={seed}
              onChange={(e) => setSeed(Number(e.target.value))}
              className="w-full rounded-md border px-3 py-2 text-sm"
            />
          </div>

          <div>
            <label className="mb-1 block text-xs font-medium text-gray-600">
              最大ステップ数: {maxSteps}
            </label>
            <input
              type="range"
              min={50}
              max={500}
              step={10}
              value={maxSteps}
              onChange={(e) => setMaxSteps(Number(e.target.value))}
              className="w-full"
            />
          </div>

          <div>
            <label className="mb-1 block text-xs font-medium text-gray-600">
              目標カバレッジ: {(coverageThreshold * 100).toFixed(0)}%
            </label>
            <input
              type="range"
              min={50}
              max={100}
              step={5}
              value={coverageThreshold * 100}
              onChange={(e) => setCoverageThreshold(Number(e.target.value) / 100)}
              className="w-full"
            />
          </div>

          <button
            onClick={handleRun}
            disabled={runMutation.isPending}
            className="w-full rounded-lg bg-blue-600 px-4 py-2.5 text-sm font-medium text-white transition hover:bg-blue-700 disabled:opacity-50"
          >
            {runMutation.isPending ? "実行中..." : "診断実行"}
          </button>

          {runMutation.isError && (
            <p className="text-xs text-red-500">
              エラー: {(runMutation.error as Error).message}
            </p>
          )}
        </div>

        {/* Center: Trajectory Visualization */}
        <div className="rounded-lg border bg-white p-5 lg:col-span-2">
          <h3 className="mb-3 text-sm font-semibold text-gray-800">トラジェクトリ可視化</h3>
          {result ? (
            <div className="flex flex-col items-center gap-4">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={result.result.trajectory_image_data_url}
                alt="Rust tracing trajectory"
                className="w-full max-w-lg rounded-lg"
              />
              {result.mock_mode && (
                <span className="rounded-full bg-yellow-100 px-3 py-1 text-xs font-medium text-yellow-700">
                  Mock Mode
                </span>
              )}
            </div>
          ) : (
            <div className="flex h-80 items-center justify-center rounded-lg bg-gray-50">
              <p className="text-sm text-gray-400">
                左のパネルで設定を調整して「診断実行」を押してください
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Metrics Cards */}
      {metrics && (
        <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
          <MetricCard
            label="カバレッジ率"
            value={`${(metrics.coverage_rate * 100).toFixed(1)}%`}
            sub={`${metrics.visited_rust_count} / ${metrics.rust_patch_count} パッチ`}
            color={metrics.coverage_rate >= 0.9 ? "green" : metrics.coverage_rate >= 0.7 ? "yellow" : "red"}
          />
          <MetricCard
            label="総ステップ数"
            value={metrics.total_steps.toLocaleString()}
            sub={`${metrics.grid_rows}x${metrics.grid_cols} グリッド`}
            color="blue"
          />
          <MetricCard
            label="バックトラック"
            value={metrics.backtrack_count.toLocaleString()}
            sub="行き止まりからの引き返し"
            color="amber"
          />
          <MetricCard
            label="成分間ジャンプ"
            value={metrics.component_jumps.toLocaleString()}
            sub="途切れたサビ間の移動"
            color="purple"
          />
        </div>
      )}

      {/* Architecture Info */}
      <div className="rounded-lg border bg-white p-5">
        <h3 className="mb-3 text-sm font-semibold text-gray-800">アーキテクチャ</h3>
        <div className="space-y-2 text-xs text-gray-600">
          <p>
            <span className="font-medium">パイプライン:</span>{" "}
            特大画像 (1120x1120) → 224x224パッチ分割 → OpenVLA 7B + LoRA → 9方向アクション → DFS探索
          </p>
          <p>
            <span className="font-medium">アクション空間:</span>{" "}
            8方向移動 + バックトラック (z軸フラグ方式)
          </p>
          <p>
            <span className="font-medium">探索戦略:</span>{" "}
            連結成分ベース二段構え (成分内DFS + 成分間最近傍ジャンプ)
          </p>
        </div>
      </div>
    </div>
  );
}

function MetricCard({
  label,
  value,
  sub,
  color,
}: {
  label: string;
  value: string;
  sub: string;
  color: "green" | "blue" | "yellow" | "red" | "amber" | "purple";
}) {
  const colorMap = {
    green: "border-green-200 bg-green-50",
    blue: "border-blue-200 bg-blue-50",
    yellow: "border-yellow-200 bg-yellow-50",
    red: "border-red-200 bg-red-50",
    amber: "border-amber-200 bg-amber-50",
    purple: "border-purple-200 bg-purple-50",
  };
  const textMap = {
    green: "text-green-700",
    blue: "text-blue-700",
    yellow: "text-yellow-700",
    red: "text-red-700",
    amber: "text-amber-700",
    purple: "text-purple-700",
  };

  return (
    <div className={`rounded-lg border p-4 ${colorMap[color]}`}>
      <p className="text-xs font-medium text-gray-500">{label}</p>
      <p className={`mt-1 text-2xl font-bold ${textMap[color]}`}>{value}</p>
      <p className="mt-0.5 text-xs text-gray-400">{sub}</p>
    </div>
  );
}
