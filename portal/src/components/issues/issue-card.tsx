import Link from "next/link";
import { MessageSquare } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { LabelBadge } from "./label-badge";
import { relativeTime } from "@/lib/utils";
import type { GitHubIssue } from "@/lib/types";

export function IssueCard({ issue }: { issue: GitHubIssue }) {
  return (
    <Link href={`/issues/${issue.number}`}>
      <Card className="transition-colors hover:bg-muted/50">
        <CardHeader className="pb-2">
          <div className="flex items-start justify-between gap-2">
            <CardTitle className="text-base leading-snug">
              <span
                className={`mr-2 inline-block h-2 w-2 rounded-full ${
                  issue.state === "open" ? "bg-green-500" : "bg-purple-500"
                }`}
              />
              {issue.title}
            </CardTitle>
            <span className="shrink-0 text-xs text-muted-foreground">
              #{issue.number}
            </span>
          </div>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-1.5">
            {issue.labels.map((label) => (
              <LabelBadge key={label.id} label={label} />
            ))}
          </div>
          <div className="mt-3 flex items-center justify-between">
            <div className="flex items-center gap-2">
              {issue.assignees.map((user) => (
                <Avatar key={user.login} className="h-6 w-6">
                  <AvatarImage src={user.avatar_url} alt={user.login} />
                  <AvatarFallback>{user.login[0].toUpperCase()}</AvatarFallback>
                </Avatar>
              ))}
            </div>
            <div className="flex items-center gap-3 text-xs text-muted-foreground">
              {issue.comments > 0 && (
                <span className="flex items-center gap-1">
                  <MessageSquare className="h-3 w-3" />
                  {issue.comments}
                </span>
              )}
              <span>{relativeTime(issue.updated_at)}</span>
            </div>
          </div>
        </CardContent>
      </Card>
    </Link>
  );
}
