"use client";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

const MONTHS = [
  "1月", "2月", "3月", "4月", "5月", "6月",
  "7月", "8月", "9月", "10月", "11月", "12月",
];

interface HeatmapControlsProps {
  selectedMonth: number | null;
  onMonthChange: (month: number | null) => void;
}

export function HeatmapControls({
  selectedMonth,
  onMonthChange,
}: HeatmapControlsProps) {
  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-sm">月別ヒートマップ</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="grid grid-cols-4 gap-1">
          {MONTHS.map((label, i) => {
            const month = i + 1;
            const isActive = selectedMonth === month;
            return (
              <Button
                key={month}
                variant={isActive ? "default" : "outline"}
                size="sm"
                className="h-7 text-[10px]"
                onClick={() => onMonthChange(isActive ? null : month)}
              >
                {label}
              </Button>
            );
          })}
        </div>

        {selectedMonth !== null && (
          <div className="space-y-1">
            <p className="text-[11px] text-gray-500">
              {MONTHS[selectedMonth - 1]}の日射量ヒートマップを表示中
            </p>
            {/* Color legend */}
            <div className="flex items-center gap-1 text-[10px] text-gray-500">
              <div
                className="h-3 w-full rounded"
                style={{
                  background: "linear-gradient(to right, #ffffb2, #fd8d3c, #bd0026)",
                }}
              />
            </div>
            <div className="flex justify-between text-[10px] text-gray-400">
              <span>低</span>
              <span>日射量 (kWh/m²)</span>
              <span>高</span>
            </div>
          </div>
        )}

        {selectedMonth !== null && (
          <Button
            variant="ghost"
            size="sm"
            className="w-full text-xs"
            onClick={() => onMonthChange(null)}
          >
            ヒートマップを非表示
          </Button>
        )}
      </CardContent>
    </Card>
  );
}
