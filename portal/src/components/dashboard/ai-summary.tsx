"use client";

import { useQuery } from "@tanstack/react-query";
import ReactMarkdown from "react-markdown";
import { Sparkles, RefreshCw, Clock } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { formatDate } from "@/lib/utils";
import type { AISummary } from "@/lib/types";

export function AISummaryCard() {
  const {
    data: summary,
    isLoading,
    refetch,
    isFetching,
  } = useQuery<AISummary>({
    queryKey: ["summary"],
    queryFn: async () => {
      const res = await fetch("/api/summaries");
      if (!res.ok) throw new Error("Failed to fetch summary");
      return res.json();
    },
    staleTime: 3600 * 1000,
  });

  return (
    <Card>
      <CardHeader className="pb-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Sparkles className="h-5 w-5 text-chart-4" />
            <CardTitle>AI週次レポート</CardTitle>
            <span className="text-xs text-muted-foreground">
              by Claude Sonnet 4.5
            </span>
          </div>
          <div className="flex items-center gap-2">
            {summary && (
              <div className="flex items-center gap-1 text-xs text-muted-foreground">
                <Clock className="h-3 w-3" />
                <span>{formatDate(summary.generatedAt)}</span>
              </div>
            )}
            <Button
              variant="ghost"
              size="sm"
              onClick={() => refetch()}
              disabled={isFetching}
            >
              <RefreshCw
                className={`h-4 w-4 ${isFetching ? "animate-spin" : ""}`}
              />
            </Button>
          </div>
        </div>
        {summary && (
          <p className="text-xs text-muted-foreground">
            対象期間: {formatDate(summary.periodStart)} 〜{" "}
            {formatDate(summary.periodEnd)}
          </p>
        )}
      </CardHeader>
      <CardContent>
        {isLoading || isFetching ? (
          <div className="flex flex-col items-center gap-4 py-8">
            <Sparkles className="h-6 w-6 text-chart-4 animate-pulse" />
            <p className="text-sm text-muted-foreground">
              {isLoading
                ? "AIレポートを生成しています..."
                : "最新データで再生成しています..."}
            </p>
            <div className="w-full max-w-md space-y-2">
              {[1, 2, 3, 4, 5].map((i) => (
                <div
                  key={i}
                  className="h-3 animate-pulse rounded bg-muted"
                  style={{
                    width: `${Math.max(30, 95 - i * 12)}%`,
                    animationDelay: `${i * 100}ms`,
                  }}
                />
              ))}
            </div>
          </div>
        ) : summary ? (
          <div className="prose prose-sm prose-neutral max-w-none dark:prose-invert">
            <ReactMarkdown>{summary.summary}</ReactMarkdown>
          </div>
        ) : (
          <p className="text-center text-sm text-muted-foreground py-4">
            サマリーの取得に失敗しました
          </p>
        )}
      </CardContent>
    </Card>
  );
}
