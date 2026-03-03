"use client";

import { ActivityFeed } from "@/components/activity/activity-feed";

export default function ActivityPage() {
  return (
    <div className="space-y-6">
      <p className="text-sm text-muted-foreground">
        コミット、PR、Issueイベントの時系列フィード
      </p>
      <ActivityFeed />
    </div>
  );
}
