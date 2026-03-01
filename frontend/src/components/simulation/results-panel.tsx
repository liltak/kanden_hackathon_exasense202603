"use client";

import { KPICard } from "@/components/dashboard/kpi-card";
import type { SimulationResult } from "@/lib/types";

interface ResultsPanelProps {
  result: SimulationResult;
}

export function ResultsPanel({ result }: ResultsPanelProps) {
  const roi = result.roi_report;
  if (!roi) return null;

  const co2 = roi.total_annual_generation_kwh * 0.000453;

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-4 gap-3">
        <KPICard title="設置可能容量" value={`${roi.total_capacity_kw.toFixed(0)}`} unit="kW" gradient="blue" />
        <KPICard title="年間発電量" value={`${(roi.total_annual_generation_kwh / 1000).toFixed(0)}`} unit="MWh/年" gradient="green" />
        <KPICard title="年間削減額" value={`¥${(roi.total_annual_savings_jpy / 10000).toFixed(0)}万`} unit="円/年" gradient="orange" />
        <KPICard title="投資回収" value={`${roi.overall_payback_years.toFixed(1)}`} unit="年" gradient="red" />
      </div>
      <div className="grid grid-cols-4 gap-3">
        <KPICard title="25年NPV" value={`¥${(roi.overall_npv_25y_jpy / 10000).toFixed(0)}万`} unit="円" gradient="green" />
        <KPICard title="CO2削減" value={co2.toFixed(1)} unit="t-CO2/年" gradient="blue" />
        <KPICard title="適合面数" value={`${roi.proposals.length}`} unit="面" gradient="purple" />
        <KPICard title="計算時間" value={result.elapsed_seconds?.toFixed(1) ?? "-"} unit="秒" gradient="default" />
      </div>

      {/* Proposal table */}
      {roi.proposals.length > 0 && (
        <div className="overflow-x-auto rounded-lg border">
          <table className="w-full text-xs">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-3 py-2 text-left font-medium">順位</th>
                <th className="px-3 py-2 text-left font-medium">面ID</th>
                <th className="px-3 py-2 text-right font-medium">面積(m²)</th>
                <th className="px-3 py-2 text-right font-medium">年間発電量</th>
                <th className="px-3 py-2 text-right font-medium">NPV</th>
                <th className="px-3 py-2 text-right font-medium">回収期間</th>
              </tr>
            </thead>
            <tbody className="divide-y">
              {roi.proposals.slice(0, 10).map((p) => (
                <tr key={p.face_id} className="hover:bg-gray-50">
                  <td className="px-3 py-2">{p.priority_rank}</td>
                  <td className="px-3 py-2">{p.face_id}</td>
                  <td className="px-3 py-2 text-right">{p.area_m2.toFixed(0)}</td>
                  <td className="px-3 py-2 text-right">{p.annual_generation_kwh.toLocaleString()} kWh</td>
                  <td className="px-3 py-2 text-right">¥{(p.npv_25y_jpy / 10000).toFixed(0)}万</td>
                  <td className="px-3 py-2 text-right">{p.payback_years.toFixed(1)}年</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
