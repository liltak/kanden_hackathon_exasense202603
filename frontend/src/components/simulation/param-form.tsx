"use client";

import { useState } from "react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { PRESET_LOCATIONS } from "@/lib/constants";
import type { SimulationRequest } from "@/lib/types";

interface ParamFormProps {
  onSubmit: (params: SimulationRequest) => void;
  isRunning: boolean;
}

export function ParamForm({ onSubmit, isRunning }: ParamFormProps) {
  const [location, setLocation] = useState("大阪（関西電力エリア）");
  const [lat, setLat] = useState(34.69);
  const [lon, setLon] = useState(135.5);
  const [year, setYear] = useState(2025);
  const [freq, setFreq] = useState(60);
  const [efficiency, setEfficiency] = useState(20);
  const [price, setPrice] = useState(30);
  const [meshSource, setMeshSource] = useState<"simple" | "complex">("complex");

  const handleLocationChange = (name: string) => {
    setLocation(name);
    const loc = PRESET_LOCATIONS[name];
    if (loc) {
      setLat(loc.lat);
      setLon(loc.lon);
    }
  };

  const handleSubmit = () => {
    onSubmit({
      latitude: lat,
      longitude: lon,
      year,
      time_resolution_minutes: freq,
      panel_efficiency: efficiency,
      electricity_price_jpy: price,
      mesh_source: meshSource,
    });
  };

  return (
    <div className="space-y-4">
      <h3 className="text-sm font-semibold">パラメータ設定</h3>

      <div className="space-y-2">
        <Label className="text-xs">プリセット地点</Label>
        <Select value={location} onValueChange={handleLocationChange}>
          <SelectTrigger className="text-xs">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {Object.keys(PRESET_LOCATIONS).map((name) => (
              <SelectItem key={name} value={name} className="text-xs">
                {name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      <div className="grid grid-cols-2 gap-2">
        <div className="space-y-1">
          <Label className="text-xs">緯度</Label>
          <Input type="number" step="0.01" value={lat} onChange={(e) => setLat(Number(e.target.value))} className="text-xs" />
        </div>
        <div className="space-y-1">
          <Label className="text-xs">経度</Label>
          <Input type="number" step="0.01" value={lon} onChange={(e) => setLon(Number(e.target.value))} className="text-xs" />
        </div>
      </div>

      <div className="space-y-1">
        <Label className="text-xs">シミュレーション年</Label>
        <Input type="number" value={year} onChange={(e) => setYear(Number(e.target.value))} className="text-xs" />
      </div>

      <div className="space-y-1">
        <Label className="text-xs">時間分解能</Label>
        <Select value={String(freq)} onValueChange={(v) => setFreq(Number(v))}>
          <SelectTrigger className="text-xs">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="60" className="text-xs">1時間</SelectItem>
            <SelectItem value="30" className="text-xs">30分</SelectItem>
          </SelectContent>
        </Select>
      </div>

      <div className="space-y-2">
        <Label className="text-xs">パネル効率: {efficiency}%</Label>
        <Slider min={15} max={25} step={0.5} value={[efficiency]} onValueChange={([v]) => setEfficiency(v)} />
      </div>

      <div className="space-y-1">
        <Label className="text-xs">電気料金 (¥/kWh)</Label>
        <Input type="number" value={price} onChange={(e) => setPrice(Number(e.target.value))} className="text-xs" />
      </div>

      <div className="space-y-1">
        <Label className="text-xs">対象建物</Label>
        <Select value={meshSource} onValueChange={(v) => setMeshSource(v as "simple" | "complex")}>
          <SelectTrigger className="text-xs">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="simple" className="text-xs">単棟工場</SelectItem>
            <SelectItem value="complex" className="text-xs">工場コンプレックス（4棟）</SelectItem>
          </SelectContent>
        </Select>
      </div>

      <Button onClick={handleSubmit} disabled={isRunning} className="w-full" size="lg">
        {isRunning ? "実行中..." : "シミュレーション実行"}
      </Button>
    </div>
  );
}
