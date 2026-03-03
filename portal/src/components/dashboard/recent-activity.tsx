"use client";

import { GitCommit, CircleDot, GitPullRequest } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useActivityFeed } from "@/hooks/use-activity";
import { relativeTime } from "@/lib/utils";

const typeIcon = {
  commit: GitCommit,
  issue: CircleDot,
  pull_request: GitPullRequest,
};

export function RecentActivity() {
  const { data: items, isLoading } = useActivityFeed();

  const recent = items?.slice(0, 5);

  return (
    <Card>
      <CardHeader>
        <CardTitle>最新アクティビティ</CardTitle>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <div className="space-y-3">
            {[1, 2, 3].map((i) => (
              <div key={i} className="h-10 animate-pulse rounded bg-muted" />
            ))}
          </div>
        ) : !recent?.length ? (
          <p className="text-sm text-muted-foreground">アクティビティがありません</p>
        ) : (
          <div className="space-y-3">
            {recent.map((item) => {
              const Icon = typeIcon[item.type];
              return (
                <a
                  key={item.id}
                  href={item.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-start gap-3 rounded-md p-2 transition-colors hover:bg-muted"
                >
                  <Icon className="mt-0.5 h-4 w-4 shrink-0 text-muted-foreground" />
                  <div className="min-w-0 flex-1">
                    <p className="truncate text-sm font-medium">{item.title}</p>
                    <p className="text-xs text-muted-foreground">
                      {item.author} - {relativeTime(item.timestamp)}
                    </p>
                  </div>
                </a>
              );
            })}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
