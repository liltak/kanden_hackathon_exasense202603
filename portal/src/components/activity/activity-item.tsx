import { GitCommit, CircleDot, GitPullRequest } from "lucide-react";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { relativeTime } from "@/lib/utils";
import type { ActivityItem as ActivityItemType } from "@/lib/types";

const typeConfig = {
  commit: { icon: GitCommit, color: "text-chart-2", bg: "bg-chart-2/10" },
  issue: { icon: CircleDot, color: "text-chart-1", bg: "bg-chart-1/10" },
  pull_request: { icon: GitPullRequest, color: "text-chart-4", bg: "bg-chart-4/10" },
};

export function ActivityItemComponent({ item }: { item: ActivityItemType }) {
  const config = typeConfig[item.type];
  const Icon = config.icon;

  return (
    <a
      href={item.url}
      target="_blank"
      rel="noopener noreferrer"
      className="flex gap-4 rounded-lg p-3 transition-colors hover:bg-muted"
    >
      <div className="flex flex-col items-center">
        <div className={`rounded-full p-2 ${config.bg}`}>
          <Icon className={`h-4 w-4 ${config.color}`} />
        </div>
        <div className="mt-2 h-full w-px bg-border" />
      </div>
      <div className="min-w-0 flex-1 pb-4">
        <div className="flex items-center gap-2">
          {item.avatarUrl && (
            <Avatar className="h-5 w-5">
              <AvatarImage src={item.avatarUrl} alt={item.author} />
              <AvatarFallback>{item.author[0].toUpperCase()}</AvatarFallback>
            </Avatar>
          )}
          <span className="text-sm font-medium">{item.author}</span>
          <span className="text-xs text-muted-foreground">
            {relativeTime(item.timestamp)}
          </span>
        </div>
        <p className="mt-1 text-sm">{item.title}</p>
        <p className="mt-0.5 text-xs text-muted-foreground">
          {item.description}
        </p>
      </div>
    </a>
  );
}
