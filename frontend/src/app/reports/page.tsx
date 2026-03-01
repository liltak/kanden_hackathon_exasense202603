"use client";

import { useCallback, useState } from "react";

import { ReportPreview } from "@/components/reports/report-preview";
import { Button } from "@/components/ui/button";
import { generateReport } from "@/lib/api";
import type { ReportResponse } from "@/lib/types";

export default function ReportsPage() {
  const [report, setReport] = useState<ReportResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleGenerate = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await generateReport();
      setReport(res);
    } catch (err) {
      setError(`${err}`);
    } finally {
      setLoading(false);
    }
  }, []);

  const handleDownload = (url: string) => {
    const base = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
    window.open(`${base}${url}`, "_blank");
  };

  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold">太陽光パネル設置提案レポート</h3>

      <div className="flex gap-3">
        <Button onClick={handleGenerate} disabled={loading} size="lg">
          {loading ? "生成中..." : "レポート生成"}
        </Button>
        {report && (
          <>
            <Button
              variant="outline"
              onClick={() => handleDownload(report.download_urls.markdown)}
            >
              Markdown DL
            </Button>
            <Button
              variant="outline"
              onClick={() => handleDownload(report.download_urls.json)}
            >
              JSON DL
            </Button>
          </>
        )}
      </div>

      {error && (
        <div className="rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-700">
          {error}
        </div>
      )}

      {report ? (
        <ReportPreview markdown={report.markdown} />
      ) : (
        <div className="flex h-64 items-center justify-center rounded-xl border text-sm text-gray-400">
          シミュレーション実行後、レポートを生成できます。
        </div>
      )}
    </div>
  );
}
