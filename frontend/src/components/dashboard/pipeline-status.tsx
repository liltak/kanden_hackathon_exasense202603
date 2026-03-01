"use client";

import { PIPELINE_PHASES } from "@/lib/constants";

export function PipelineStatus() {
  return (
    <div className="rounded-xl border bg-gray-50 p-5">
      <h3 className="mb-4 text-sm font-semibold text-gray-700">パイプライン概要</h3>
      <div className="flex items-center gap-2">
        {PIPELINE_PHASES.map((phase, i) => (
          <div key={phase.id} className="flex items-center gap-2">
            <div
              className={`flex-1 rounded-lg border-2 px-4 py-3 text-center transition-all ${
                phase.status === "ready"
                  ? "border-green-500 bg-green-50"
                  : phase.status === "active"
                  ? "border-green-500 bg-green-50"
                  : "border-blue-300 bg-blue-50"
              }`}
            >
              <div className="text-xs font-bold text-gray-700">Phase {phase.id}</div>
              <div className="text-[11px] text-gray-600">{phase.name}</div>
              {phase.status === "ready" && (
                <div className="mt-1 text-[10px] font-semibold text-green-600">READY</div>
              )}
              {phase.status === "active" && (
                <div className="mt-1 text-[10px] font-semibold text-green-600">ACTIVE</div>
              )}
            </div>
            {i < PIPELINE_PHASES.length - 1 && (
              <span className="text-xl text-gray-300">→</span>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
