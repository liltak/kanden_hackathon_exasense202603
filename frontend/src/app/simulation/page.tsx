"use client";

import { MonthlyChart } from "@/components/simulation/monthly-chart";
import { ParamForm } from "@/components/simulation/param-form";
import { ProgressBar } from "@/components/simulation/progress-bar";
import { ResultsPanel } from "@/components/simulation/results-panel";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useSimulation } from "@/hooks/use-simulation";

export default function SimulationPage() {
  const { run, isRunning, result, progress } = useSimulation();

  const isComplete = result?.status === "complete";

  return (
    <div className="flex gap-6">
      {/* Left: Controls */}
      <div className="w-72 shrink-0">
        <ParamForm onSubmit={run} isRunning={!!isRunning} />
      </div>

      {/* Right: Results */}
      <div className="flex-1 space-y-4">
        {isRunning && <ProgressBar progress={progress} isRunning={!!isRunning} />}

        {result?.status === "failed" && (
          <div className="rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-700">
            エラー: {result.message}
          </div>
        )}

        {isComplete && result.roi_report && (
          <>
            <ResultsPanel result={result} />
            <Tabs defaultValue="monthly">
              <TabsList>
                <TabsTrigger value="monthly">月別日射量</TabsTrigger>
              </TabsList>
              <TabsContent value="monthly">
                {result.monthly_ghi && <MonthlyChart data={result.monthly_ghi} />}
              </TabsContent>
            </Tabs>
          </>
        )}

        {!isRunning && !isComplete && (
          <div className="flex h-64 items-center justify-center rounded-xl border text-sm text-gray-400">
            シミュレーションを実行すると結果が表示されます
          </div>
        )}
      </div>
    </div>
  );
}
