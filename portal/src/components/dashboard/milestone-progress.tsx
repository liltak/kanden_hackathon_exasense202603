"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { useMilestones } from "@/hooks/use-dashboard";

export function MilestoneProgress() {
  const { data: milestones, isLoading } = useMilestones();

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>マイルストーン進捗</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {[1, 2].map((i) => (
              <div key={i} className="h-12 animate-pulse rounded bg-muted" />
            ))}
          </div>
        </CardContent>
      </Card>
    );
  }

  if (!milestones?.length) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>マイルストーン進捗</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">
            マイルストーンが設定されていません
          </p>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>マイルストーン進捗</CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {milestones.map((ms) => {
          const total = ms.open_issues + ms.closed_issues;
          const progress = total > 0 ? Math.round((ms.closed_issues / total) * 100) : 0;
          return (
            <div key={ms.id} className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">{ms.title}</span>
                <span className="text-sm text-muted-foreground">
                  {ms.closed_issues}/{total} ({progress}%)
                </span>
              </div>
              <Progress value={progress} className="h-2" />
              {ms.due_on && (
                <p className="text-xs text-muted-foreground">
                  期限:{" "}
                  {new Date(ms.due_on).toLocaleDateString("ja-JP")}
                </p>
              )}
            </div>
          );
        })}
      </CardContent>
    </Card>
  );
}
