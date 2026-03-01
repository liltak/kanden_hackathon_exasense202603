"use client";

import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { MONTHLY_COLORS, MONTHS_JA } from "@/lib/constants";

interface MonthlyChartProps {
  data: number[];
}

export function MonthlyChart({ data }: MonthlyChartProps) {
  const chartData = MONTHS_JA.map((month, i) => ({
    month,
    ghi: data[i] ?? 0,
    fill: MONTHLY_COLORS[i],
  }));

  return (
    <div className="rounded-lg border p-4">
      <h4 className="mb-3 text-sm font-semibold">月別水平面日射量 (GHI)</h4>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
          <XAxis dataKey="month" tick={{ fontSize: 11 }} />
          <YAxis tick={{ fontSize: 11 }} label={{ value: "kWh/m²", angle: -90, position: "insideLeft", style: { fontSize: 11 } }} />
          <Tooltip
            formatter={(value) => [`${Number(value).toFixed(0)} kWh/m²`, "GHI"]}
            contentStyle={{ fontSize: 12 }}
          />
          <Bar dataKey="ghi" radius={[4, 4, 0, 0]}>
            {chartData.map((entry, i) => (
              <rect key={i} fill={entry.fill} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
