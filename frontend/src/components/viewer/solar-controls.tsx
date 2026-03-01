"use client";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Slider } from "@/components/ui/slider";
import type { SolarAnimationActions, SolarAnimationState } from "@/hooks/use-solar-animation";

type SolarControlsProps = Pick<
  SolarAnimationState,
  "positions" | "currentIndex" | "playing" | "speed" | "date" | "loading" | "currentPosition"
> &
  Pick<SolarAnimationActions, "setDate" | "setIndex" | "togglePlay" | "setSpeed" | "fetchData">;

export function SolarAnimationControls({
  positions,
  currentIndex,
  playing,
  speed,
  date,
  loading,
  currentPosition,
  setDate,
  setIndex,
  togglePlay,
  setSpeed,
  fetchData,
}: SolarControlsProps) {
  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-sm">太陽アニメーション</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        {/* Date picker + fetch */}
        <div>
          <label className="mb-1 block text-xs text-gray-500">日付</label>
          <div className="flex gap-1">
            <input
              type="date"
              value={date}
              onChange={(e) => setDate(e.target.value)}
              className="flex-1 rounded border px-2 py-1 text-xs"
            />
            <Button
              variant="default"
              size="sm"
              className="shrink-0 text-xs"
              disabled={loading}
              onClick={() => fetchData(date)}
            >
              {loading ? "..." : "取得"}
            </Button>
          </div>
        </div>

        {/* Time slider */}
        {positions.length > 0 && (
          <div>
            <div className="mb-1 flex items-center justify-between">
              <label className="text-xs text-gray-500">時刻</label>
              <span className="text-xs font-medium">
                {currentPosition?.time ?? "--:--"}
              </span>
            </div>
            <Slider
              value={[currentIndex]}
              min={0}
              max={Math.max(positions.length - 1, 0)}
              step={1}
              onValueChange={([v]) => setIndex(v)}
              className="w-full"
            />
          </div>
        )}

        {/* Play / speed controls */}
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={togglePlay}
            disabled={positions.length === 0 || loading}
            className="flex-1 text-xs"
          >
            {loading ? "読込中..." : playing ? "一時停止" : "再生"}
          </Button>
          <select
            value={speed}
            onChange={(e) => setSpeed(Number(e.target.value))}
            className="rounded border px-1 py-1 text-xs"
          >
            <option value={0.5}>0.5x</option>
            <option value={1}>1x</option>
            <option value={2}>2x</option>
            <option value={4}>4x</option>
          </select>
        </div>

        {/* Current info */}
        {currentPosition && (
          <div className="rounded bg-gray-50 p-2 text-[11px] text-gray-600">
            <p>方位角: {currentPosition.azimuth}°</p>
            <p>仰角: {currentPosition.elevation}°</p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
