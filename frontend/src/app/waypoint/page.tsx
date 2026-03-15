"use client";

import Image from "next/image";
import Link from "next/link";
import { startTransition, useMemo, useRef, useState, type ChangeEvent } from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
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
import { Textarea } from "@/components/ui/textarea";
import { useWaypointGenerate, useWaypointStatus } from "@/hooks/use-waypoint";
import type { WaypointGenerateRequest, WaypointViewName } from "@/lib/types";

type TemplateOption = {
  key: string;
  label: string;
  prompt: string;
};

const TEMPLATE_OPTIONS: TemplateOption[] = [
  {
    key: "roof",
    label: "屋根面パネル最大化",
    prompt:
      "Add high-efficiency rooftop solar arrays aligned to the existing roof geometry, preserve industrial realism, and keep maintenance lanes visible.",
  },
  {
    key: "carport",
    label: "カーポート併設",
    prompt:
      "Extend the site with solar carports near the parking area, maintain truck circulation, and keep the factory facade realistic.",
  },
  {
    key: "presentation",
    label: "提案資料向け",
    prompt:
      "Render a clean proposal-ready factory scene with installed solar panels, balanced daylight, and clear visibility of major roof zones.",
  },
];

const VIEW_OPTIONS: Array<{ value: WaypointViewName; label: string }> = [
  { value: "bird", label: "鳥瞰" },
  { value: "south", label: "南面" },
  { value: "west", label: "西面" },
];

const DEMO_SEEDS: Record<WaypointViewName, { label: string; imageDataUrl: string }> = {
  bird: {
    label: "鳥瞰シード",
    imageDataUrl: buildDemoSeed("BIRD SEED", "#0f766e", "#5eead4"),
  },
  south: {
    label: "南面シード",
    imageDataUrl: buildDemoSeed("SOUTH SEED", "#1d4ed8", "#93c5fd"),
  },
  west: {
    label: "西面シード",
    imageDataUrl: buildDemoSeed("WEST SEED", "#b45309", "#fdba74"),
  },
};

function buildDemoSeed(title: string, baseColor: string, accentColor: string): string {
  const svg = `
<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="720" viewBox="0 0 1280 720" fill="none">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1280" y2="720" gradientUnits="userSpaceOnUse">
      <stop stop-color="${baseColor}"/>
      <stop offset="1" stop-color="#020617"/>
    </linearGradient>
  </defs>
  <rect width="1280" height="720" rx="36" fill="url(#bg)"/>
  <rect x="212" y="194" width="860" height="324" rx="28" fill="#0F172A" fill-opacity="0.56" stroke="#E2E8F0" stroke-opacity="0.14"/>
  <path d="M280 470L500 270L712 350L980 232" stroke="#E2E8F0" stroke-opacity="0.22" stroke-width="12"/>
  <path d="M280 488L500 288L712 368L980 250" stroke="${accentColor}" stroke-width="20" stroke-linecap="round"/>
  <circle cx="1070" cy="132" r="76" fill="${accentColor}" fill-opacity="0.32"/>
  <text x="112" y="624" fill="#F8FAFC" font-size="54" font-weight="700" font-family="Arial, Helvetica, sans-serif">${title}</text>
  <text x="112" y="666" fill="#CBD5E1" font-size="28" font-family="Arial, Helvetica, sans-serif">Demo seed image for Waypoint UI integration</text>
</svg>
`.trim();
  return `data:image/svg+xml;charset=UTF-8,${encodeURIComponent(svg)}`;
}

function readFileAsDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(reader.error ?? new Error("Failed to read file"));
    reader.readAsDataURL(file);
  });
}

export default function WaypointPage() {
  const fileRef = useRef<HTMLInputElement>(null);
  const [seedSource, setSeedSource] = useState<"demo" | "upload">("demo");
  const [selectedView, setSelectedView] = useState<WaypointViewName>("bird");
  const [templateKey, setTemplateKey] = useState<string>(TEMPLATE_OPTIONS[0].key);
  const [prompt, setPrompt] = useState<string>(TEMPLATE_OPTIONS[0].prompt);
  const [negativePrompt, setNegativePrompt] = useState<string>(
    "distorted roof, extra buildings, heavy fog, motion blur, duplicated solar panels",
  );
  const [steps, setSteps] = useState<number[]>([12]);
  const [guidanceScale, setGuidanceScale] = useState<number[]>([6.5]);
  const [strength, setStrength] = useState<number[]>([0.65]);
  const [uploadedSeed, setUploadedSeed] = useState<string | null>(null);
  const [uploadedFileName, setUploadedFileName] = useState<string | null>(null);

  const statusQuery = useWaypointStatus();
  const generateMutation = useWaypointGenerate();

  const activeSeedImage = useMemo(() => {
    if (seedSource === "upload") {
      return uploadedSeed;
    }
    return DEMO_SEEDS[selectedView].imageDataUrl;
  }, [seedSource, selectedView, uploadedSeed]);

  const activeResultImage = useMemo(() => {
    const result = generateMutation.data;
    if (!result) return null;
    return (
      result.variants.find((variant) => variant.view_name === selectedView)?.image_data_url
      ?? result.result_image_data_url
    );
  }, [generateMutation.data, selectedView]);

  async function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;
    const imageDataUrl = await readFileAsDataUrl(file);
    setUploadedSeed(imageDataUrl);
    setUploadedFileName(file.name);
    setSeedSource("upload");
  }

  function handleTemplateChange(nextTemplateKey: string) {
    const template = TEMPLATE_OPTIONS.find((option) => option.key === nextTemplateKey);
    setTemplateKey(nextTemplateKey);
    if (!template) return;
    startTransition(() => {
      setPrompt(template.prompt);
    });
  }

  function handleGenerate() {
    if (!activeSeedImage) return;
    const request: WaypointGenerateRequest = {
      prompt,
      negative_prompt: negativePrompt,
      seed_source: seedSource,
      seed_image_data_url: activeSeedImage,
      template_key: templateKey,
      view_name: selectedView,
      steps: steps[0] ?? 12,
      guidance_scale: guidanceScale[0] ?? 6.5,
      strength: strength[0] ?? 0.65,
    };
    generateMutation.mutate(request);
  }

  const status = statusQuery.data;
  const metrics = generateMutation.data?.metrics;

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-3 lg:flex-row lg:items-end lg:justify-between">
        <div className="space-y-2">
          <div className="flex flex-wrap items-center gap-2">
            <Badge className="bg-emerald-600 text-white hover:bg-emerald-600">Issue #52</Badge>
            <Badge variant="outline">Waypoint UI Foundation</Badge>
            {status?.mock_mode ? <Badge variant="secondary">Mock Mode</Badge> : null}
          </div>
          <div>
            <h3 className="text-lg font-semibold">World Model</h3>
            <p className="text-sm text-gray-500">
              シード画像選択から Waypoint 生成結果表示までを、Next.js 上でモック接続します。
            </p>
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <Badge variant={status?.service_status === "ready" ? "default" : "secondary"}>
            {status?.service_status ?? "loading"}
          </Badge>
          <Badge variant="outline">{status?.device ?? "status unavailable"}</Badge>
          <Badge variant="outline">
            GPU {status?.gpu_available ? "available" : "offline"}
          </Badge>
        </div>
      </div>

      <div className="grid gap-6 xl:grid-cols-[420px_minmax(0,1fr)]">
        <div className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Seed Image</CardTitle>
              <CardDescription>
                まずはデモシードまたはアップロード画像で UI フローを確認します。
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-2">
                <Button
                  type="button"
                  variant={seedSource === "demo" ? "default" : "outline"}
                  onClick={() => setSeedSource("demo")}
                >
                  デモシード
                </Button>
                <Button
                  type="button"
                  variant={seedSource === "upload" ? "default" : "outline"}
                  onClick={() => setSeedSource("upload")}
                >
                  画像アップロード
                </Button>
              </div>

              <div className="space-y-2">
                <Label>カメラ視点</Label>
                <Select
                  value={selectedView}
                  onValueChange={(value) => setSelectedView(value as WaypointViewName)}
                >
                  <SelectTrigger className="w-full">
                    <SelectValue placeholder="視点を選択" />
                  </SelectTrigger>
                  <SelectContent>
                    {VIEW_OPTIONS.map((view) => (
                      <SelectItem key={view.value} value={view.value}>
                        {view.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              {seedSource === "upload" ? (
                <div className="space-y-3 rounded-xl border border-dashed p-4">
                  <input
                    ref={fileRef}
                    type="file"
                    accept="image/png,image/jpeg,image/webp"
                    className="hidden"
                    onChange={handleFileChange}
                  />
                  <Button type="button" className="w-full" onClick={() => fileRef.current?.click()}>
                    シード画像を選択
                  </Button>
                  <p className="text-xs text-gray-500">
                    {uploadedFileName ?? "PNG / JPEG / WebP"}
                  </p>
                </div>
              ) : (
                <div className="rounded-xl border border-dashed p-4 text-sm text-gray-600">
                  <p>{DEMO_SEEDS[selectedView].label} を使用します。</p>
                  <p className="mt-1 text-xs text-gray-500">
                    `/viewer` からのスナップショット連携は次段階で追加します。
                  </p>
                </div>
              )}

              <div className="overflow-hidden rounded-2xl border bg-slate-950">
                {activeSeedImage ? (
                  <Image
                    src={activeSeedImage}
                    alt="Waypoint seed preview"
                    width={1280}
                    height={720}
                    className="h-auto w-full"
                    unoptimized
                  />
                ) : (
                  <div className="flex h-[220px] items-center justify-center text-sm text-slate-400">
                    シード画像を選択してください
                  </div>
                )}
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Prompt + Params</CardTitle>
              <CardDescription>モック生成でもテンプレートと制御値を UI に通します。</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label>テンプレート</Label>
                <Select value={templateKey} onValueChange={handleTemplateChange}>
                  <SelectTrigger className="w-full">
                    <SelectValue placeholder="テンプレートを選択" />
                  </SelectTrigger>
                  <SelectContent>
                    {TEMPLATE_OPTIONS.map((option) => (
                      <SelectItem key={option.key} value={option.key}>
                        {option.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label>Prompt</Label>
                <Textarea value={prompt} onChange={(event) => setPrompt(event.target.value)} />
              </div>

              <div className="space-y-2">
                <Label>Negative Prompt</Label>
                <Input
                  value={negativePrompt}
                  onChange={(event) => setNegativePrompt(event.target.value)}
                />
              </div>

              <div className="space-y-3">
                <div className="flex items-center justify-between text-sm">
                  <span>Sampling Steps</span>
                  <span className="font-medium">{steps[0]}</span>
                </div>
                <Slider value={steps} min={4} max={24} step={1} onValueChange={setSteps} />
              </div>

              <div className="space-y-3">
                <div className="flex items-center justify-between text-sm">
                  <span>Guidance Scale</span>
                  <span className="font-medium">{guidanceScale[0]?.toFixed(1)}</span>
                </div>
                <Slider
                  value={guidanceScale}
                  min={1}
                  max={12}
                  step={0.5}
                  onValueChange={setGuidanceScale}
                />
              </div>

              <div className="space-y-3">
                <div className="flex items-center justify-between text-sm">
                  <span>Image Strength</span>
                  <span className="font-medium">{strength[0]?.toFixed(2)}</span>
                </div>
                <Slider value={strength} min={0.2} max={1} step={0.05} onValueChange={setStrength} />
              </div>

              <div className="flex gap-2">
                <Button
                  type="button"
                  className="flex-1"
                  disabled={!activeSeedImage || generateMutation.isPending || !prompt.trim()}
                  onClick={handleGenerate}
                >
                  {generateMutation.isPending ? "Generating..." : "Generate Mock Result"}
                </Button>
                <Button asChild variant="outline">
                  <Link href="/viewer">3Dビューア</Link>
                </Button>
              </div>

              {generateMutation.error ? (
                <p className="text-sm text-red-600">{String(generateMutation.error)}</p>
              ) : null}
            </CardContent>
          </Card>
        </div>

        <div className="space-y-6">
          <Card className="overflow-hidden">
            <CardHeader>
              <CardTitle>Generation Result</CardTitle>
              <CardDescription>
                API モックでも、seed → prompt → result の導線と KPI オーバーレイを確認できます。
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid gap-4 lg:grid-cols-2">
                <div className="space-y-2">
                  <p className="text-sm font-medium text-gray-700">Before / Seed</p>
                  <div className="overflow-hidden rounded-2xl border bg-slate-950">
                    {activeSeedImage ? (
                      <Image
                        src={activeSeedImage}
                        alt="Seed image"
                        width={1280}
                        height={720}
                        className="h-auto w-full"
                        unoptimized
                      />
                    ) : (
                      <div className="flex h-[260px] items-center justify-center text-sm text-slate-400">
                        シード画像未選択
                      </div>
                    )}
                  </div>
                </div>

                <div className="space-y-2">
                  <p className="text-sm font-medium text-gray-700">After / Waypoint</p>
                  <div className="relative overflow-hidden rounded-2xl border bg-slate-950">
                    {activeResultImage ? (
                      <>
                        <Image
                          src={activeResultImage}
                          alt="Waypoint generated result"
                          width={1280}
                          height={720}
                          className="h-auto w-full"
                          unoptimized
                        />
                        {metrics ? (
                          <div className="absolute left-4 top-4 flex flex-wrap gap-2">
                            <Badge className="bg-white/90 text-slate-900 hover:bg-white/90">
                              {metrics.annual_generation_kwh.toLocaleString()} kWh/年
                            </Badge>
                            <Badge className="bg-emerald-500/90 text-white hover:bg-emerald-500/90">
                              {metrics.co2_reduction_tons.toFixed(1)} t-CO2
                            </Badge>
                            <Badge className="bg-slate-900/80 text-white hover:bg-slate-900/80">
                              {metrics.installed_capacity_kw.toFixed(0)} kW
                            </Badge>
                          </div>
                        ) : null}
                      </>
                    ) : (
                      <div className="flex h-[260px] items-center justify-center text-sm text-slate-400">
                        生成結果はまだありません
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {generateMutation.data ? (
                <div className="grid gap-3 md:grid-cols-3">
                  {generateMutation.data.variants.map((variant) => (
                    <button
                      key={variant.id}
                      type="button"
                      onClick={() => setSelectedView(variant.view_name as WaypointViewName)}
                      className={`overflow-hidden rounded-2xl border text-left transition ${
                        selectedView === variant.view_name
                          ? "border-blue-500 ring-2 ring-blue-100"
                          : "border-slate-200"
                      }`}
                    >
                      <Image
                        src={variant.image_data_url}
                        alt={variant.label}
                        width={1280}
                        height={720}
                        className="h-auto w-full"
                        unoptimized
                      />
                      <div className="flex items-center justify-between px-3 py-2">
                        <span className="text-sm font-medium">{variant.label}</span>
                        <Badge variant="outline">{variant.view_name}</Badge>
                      </div>
                    </button>
                  ))}
                </div>
              ) : null}
            </CardContent>
          </Card>

          <div className="grid gap-6 lg:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>Runtime Status</CardTitle>
                <CardDescription>バックエンドの Waypoint 接続状態を表示します。</CardDescription>
              </CardHeader>
              <CardContent className="space-y-3 text-sm">
                <div className="flex items-center justify-between rounded-lg bg-slate-50 px-3 py-2">
                  <span>Model</span>
                  <span className="font-medium">{status?.model_name ?? "Loading..."}</span>
                </div>
                <div className="flex items-center justify-between rounded-lg bg-slate-50 px-3 py-2">
                  <span>Queue Depth</span>
                  <span className="font-medium">{status?.queue_depth ?? 0}</span>
                </div>
                <div className="flex items-center justify-between rounded-lg bg-slate-50 px-3 py-2">
                  <span>VRAM Used</span>
                  <span className="font-medium">
                    {status?.vram_used_gb != null ? `${status.vram_used_gb.toFixed(2)} GB` : "n/a"}
                  </span>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Current Output</CardTitle>
                <CardDescription>生成完了後に KPI と応答時間を確認できます。</CardDescription>
              </CardHeader>
              <CardContent className="space-y-3 text-sm">
                <div className="flex items-center justify-between rounded-lg bg-slate-50 px-3 py-2">
                  <span>Request ID</span>
                  <span className="font-medium">{generateMutation.data?.request_id ?? "-"}</span>
                </div>
                <div className="flex items-center justify-between rounded-lg bg-slate-50 px-3 py-2">
                  <span>Latency</span>
                  <span className="font-medium">
                    {generateMutation.data ? `${generateMutation.data.latency_ms} ms` : "-"}
                  </span>
                </div>
                <div className="flex items-center justify-between rounded-lg bg-slate-50 px-3 py-2">
                  <span>Payback</span>
                  <span className="font-medium">
                    {metrics ? `${metrics.estimated_payback_years.toFixed(1)} years` : "-"}
                  </span>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
}
