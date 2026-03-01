"use client";

import dynamic from "next/dynamic";
import { useCallback, useEffect, useRef, useState } from "react";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { HeatmapControls } from "@/components/viewer/heatmap-controls";
import { MeshAdjustControls, type MeshRotation } from "@/components/viewer/mesh-adjust-controls";
import { SolarAnimationControls } from "@/components/viewer/solar-controls";
import { useSolarAnimation } from "@/hooks/use-solar-animation";
import {
  getDemoMeshURL,
  getMeshGlbURL,
  getMonthlyHeatmapURL,
  getPresets,
  getReconstructionStatus,
  loadPreset,
  startReconstruction,
  uploadMesh,
} from "@/lib/api";
import type { MeshInfo, PresetInfo, ReconstructionStatus } from "@/lib/types";

// Dynamic import to avoid SSR issues with Three.js
const MeshCanvas = dynamic(
  () => import("@/components/viewer/mesh-canvas").then((m) => ({ default: m.MeshCanvas })),
  { ssr: false, loading: () => <div className="flex h-[600px] items-center justify-center rounded-xl border bg-gray-50 text-gray-400">Loading 3D viewer...</div> }
);

export default function ViewerPage() {
  const [meshUrl, setMeshUrl] = useState<string | null>(null);
  const [meshInfo, setMeshInfo] = useState<MeshInfo | null>(null);
  const [loading, setLoading] = useState(false);
  const [activeMeshType, setActiveMeshType] = useState<"simple" | "complex">("complex");
  const [heatmapMonth, setHeatmapMonth] = useState<number | null>(null);
  const [meshRotation, setMeshRotation] = useState<MeshRotation>({ heading: 0, tiltX: 0, tiltZ: 0 });
  const fileRef = useRef<HTMLInputElement>(null);

  // Reconstruction state
  const [reconMethod, setReconMethod] = useState<"vggt" | "colmap">("vggt");
  const [reconStatus, setReconStatus] = useState<ReconstructionStatus | null>(null);
  const [reconLoading, setReconLoading] = useState(false);
  const reconFileRef = useRef<HTMLInputElement>(null);

  // Preset state
  const [presets, setPresets] = useState<PresetInfo[]>([]);
  const [presetLoading, setPresetLoading] = useState(false);

  const solar = useSolarAnimation(activeMeshType);

  // Fetch presets on mount
  useEffect(() => {
    getPresets()
      .then((res) => setPresets(res.presets))
      .catch(() => {}); // H100 may be offline
  }, []);

  // Load preset mesh from H100
  const handleLoadPreset = useCallback(async (name: string) => {
    setPresetLoading(true);
    try {
      const status = await loadPreset(name);
      setReconStatus(status);
    } catch (err) {
      alert(`プリセット読み込みエラー: ${err}`);
    } finally {
      setPresetLoading(false);
    }
  }, []);

  // Reconstruction: start
  const handleReconStart = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFiles = e.target.files;
    if (!selectedFiles || selectedFiles.length === 0) return;
    setReconLoading(true);
    try {
      const files = Array.from(selectedFiles);
      const status = await startReconstruction(files, reconMethod);
      setReconStatus(status);
    } catch (err) {
      alert(`復元開始エラー: ${err}`);
    } finally {
      setReconLoading(false);
      if (reconFileRef.current) reconFileRef.current.value = "";
    }
  }, [reconMethod]);

  // Reconstruction: poll status
  useEffect(() => {
    if (!reconStatus) return;
    if (reconStatus.status === "complete" || reconStatus.status === "failed") return;

    const interval = setInterval(async () => {
      try {
        const updated = await getReconstructionStatus(reconStatus.task_id);
        setReconStatus(updated);
        if (updated.status === "complete" && updated.mesh_id) {
          setMeshUrl(getMeshGlbURL(updated.mesh_id));
          setMeshInfo(null);
          setHeatmapMonth(null);
        }
      } catch {
        // ignore transient errors
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [reconStatus]);

  const loadDemo = useCallback((type: "simple" | "complex") => {
    setActiveMeshType(type);
    setMeshUrl(getDemoMeshURL(type));
    setMeshInfo(null);
    setHeatmapMonth(null);
  }, []);

  const handleUpload = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setLoading(true);
    try {
      const info = await uploadMesh(file);
      setMeshInfo(info);
      setMeshUrl(`${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"}${info.download_url}`);
      setHeatmapMonth(null);
    } catch (err) {
      alert(`アップロードエラー: ${err}`);
    } finally {
      setLoading(false);
    }
  }, []);

  const handleMonthChange = useCallback(
    (month: number | null) => {
      setHeatmapMonth(month);
      if (month !== null) {
        setMeshUrl(getMonthlyHeatmapURL(activeMeshType, month));
      } else {
        setMeshUrl(getDemoMeshURL(activeMeshType));
      }
    },
    [activeMeshType],
  );

  // Determine the effective URL (heatmap overrides base mesh)
  const displayUrl = meshUrl;
  const solarActive = solar.positions.length > 0;

  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold">3D建物モデル</h3>
      <div className="flex gap-6">
        {/* Left panel */}
        <div className="w-64 shrink-0 space-y-4">
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm">メッシュファイル読み込み</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <input
                ref={fileRef}
                type="file"
                accept=".ply,.obj,.stl,.glb"
                onChange={handleUpload}
                className="hidden"
              />
              <Button
                variant="default"
                className="w-full"
                onClick={() => fileRef.current?.click()}
                disabled={loading}
              >
                {loading ? "読み込み中..." : "ファイルを選択"}
              </Button>
              <p className="text-[11px] text-gray-500">PLY / OBJ / STL / GLB</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm">デモメッシュ</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              <Button variant="outline" className="w-full text-xs" onClick={() => loadDemo("simple")}>
                単棟工場
              </Button>
              <Button variant="outline" className="w-full text-xs" onClick={() => loadDemo("complex")}>
                工場コンプレックス（4棟）
              </Button>
            </CardContent>
          </Card>

          {presets.length > 0 && (
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-sm">H100 復元済みデータ</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                {presets.map((p) => (
                  <Button
                    key={p.name}
                    variant="outline"
                    className="w-full text-xs"
                    disabled={presetLoading || (reconStatus !== null && reconStatus.status === "running")}
                    onClick={() => handleLoadPreset(p.name)}
                  >
                    {p.name}
                    {p.n_faces > 0 && (
                      <span className="ml-1 text-gray-400">({p.n_faces.toLocaleString()}面)</span>
                    )}
                  </Button>
                ))}
                <p className="text-[11px] text-gray-500">H100で復元済みのメッシュを読み込み</p>
              </CardContent>
            </Card>
          )}

          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm">3D復元 (H100)</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="space-y-1">
                <label className="text-[11px] text-gray-500">復元方法</label>
                <Select value={reconMethod} onValueChange={(v) => setReconMethod(v as "vggt" | "colmap")}>
                  <SelectTrigger className="w-full">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="vggt">VGGT</SelectItem>
                    <SelectItem value="colmap">COLMAP</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <input
                ref={reconFileRef}
                type="file"
                accept="image/*"
                multiple
                onChange={handleReconStart}
                className="hidden"
              />
              <Button
                variant="default"
                className="w-full"
                onClick={() => reconFileRef.current?.click()}
                disabled={reconLoading || (reconStatus !== null && reconStatus.status === "running")}
              >
                {reconLoading ? "アップロード中..." : "画像を選択して復元開始"}
              </Button>
              <p className="text-[11px] text-gray-500">複数画像を選択 (JPEG/PNG)</p>

              {reconStatus && (
                <div className="space-y-2">
                  <div className="flex items-center justify-between text-xs">
                    <span className={
                      reconStatus.status === "complete" ? "text-green-600" :
                      reconStatus.status === "failed" ? "text-red-600" :
                      "text-blue-600"
                    }>
                      {reconStatus.message}
                    </span>
                    <span className="text-gray-400">
                      {Math.round(reconStatus.progress * 100)}%
                    </span>
                  </div>
                  <div className="h-2 w-full overflow-hidden rounded-full bg-gray-200">
                    <div
                      className={`h-full rounded-full transition-all duration-500 ${
                        reconStatus.status === "failed" ? "bg-red-500" :
                        reconStatus.status === "complete" ? "bg-green-500" :
                        "bg-blue-500"
                      }`}
                      style={{ width: `${reconStatus.progress * 100}%` }}
                    />
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          {meshInfo && (
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-sm">メッシュ情報</CardTitle>
              </CardHeader>
              <CardContent className="space-y-1 text-xs text-gray-600">
                <p>頂点数: <strong>{meshInfo.num_vertices.toLocaleString()}</strong></p>
                <p>面数: <strong>{meshInfo.num_faces.toLocaleString()}</strong></p>
                <p>表面積: <strong>{meshInfo.surface_area_m2.toFixed(1)} m²</strong></p>
              </CardContent>
            </Card>
          )}

          {/* Mesh Rotation Adjustment */}
          <MeshAdjustControls
            rotation={meshRotation}
            onChange={setMeshRotation}
          />

          {/* Solar Animation Controls */}
          <SolarAnimationControls
            positions={solar.positions}
            currentIndex={solar.currentIndex}
            playing={solar.playing}
            speed={solar.speed}
            date={solar.date}
            loading={solar.loading}
            currentPosition={solar.currentPosition}
            setDate={solar.setDate}
            setIndex={solar.setIndex}
            togglePlay={solar.togglePlay}
            setSpeed={solar.setSpeed}
            fetchData={solar.fetchData}
          />

          {/* Heatmap Controls */}
          <HeatmapControls
            selectedMonth={heatmapMonth}
            onMonthChange={handleMonthChange}
          />

          <div className="rounded-lg border p-3 text-xs text-gray-500">
            <p className="mb-1 font-semibold">色の凡例:</p>
            <div className="space-y-1">
              <div className="flex items-center gap-2">
                <span className="h-3 w-3 rounded-sm bg-blue-400" />
                <span>屋根面（パネル設置候補）</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="h-3 w-3 rounded-sm bg-gray-400" />
                <span>壁面</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="h-3 w-3 rounded-sm bg-amber-700" />
                <span>床面・その他</span>
              </div>
            </div>
          </div>
        </div>

        {/* 3D Canvas */}
        <div className="flex-1">
          <MeshCanvas
            url={displayUrl}
            sunPositions={solar.positions}
            sunIndex={solar.currentIndex}
            currentSunPosition={solar.currentPosition}
            currentShadow={solar.currentShadow}
            solarActive={solarActive}
            meshRotation={meshRotation}
          />
        </div>
      </div>
    </div>
  );
}
