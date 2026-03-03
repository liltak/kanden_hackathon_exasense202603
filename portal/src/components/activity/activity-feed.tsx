"use client";

import { useActivityFeed } from "@/hooks/use-activity";
import { ActivityItemComponent } from "./activity-item";

export function ActivityFeed() {
  const { data: items, isLoading } = useActivityFeed();

  if (isLoading) {
    return (
      <div className="space-y-3">
        {[1, 2, 3, 4, 5].map((i) => (
          <div key={i} className="h-20 animate-pulse rounded-lg bg-muted" />
        ))}
      </div>
    );
  }

  if (!items?.length) {
    return (
      <p className="text-center text-muted-foreground">
        アクティビティがありません
      </p>
    );
  }

  return (
    <div className="space-y-0">
      {items.map((item) => (
        <ActivityItemComponent key={item.id} item={item} />
      ))}
    </div>
  );
}
