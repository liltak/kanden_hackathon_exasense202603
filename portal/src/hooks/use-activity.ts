"use client";

import { useQuery } from "@tanstack/react-query";
import type {
  GitHubCommit,
  GitHubPullRequest,
  GitHubIssue,
  ActivityItem,
} from "@/lib/types";

export function useActivityFeed() {
  return useQuery<ActivityItem[]>({
    queryKey: ["activity-feed"],
    queryFn: async () => {
      const [commitsRes, pullsRes, issuesRes] = await Promise.all([
        fetch("/api/github/commits?per_page=20"),
        fetch("/api/github/pulls?state=all"),
        fetch("/api/github/issues?state=all"),
      ]);

      const commits: GitHubCommit[] = await commitsRes.json();
      const pulls: GitHubPullRequest[] = await pullsRes.json();
      const issues: GitHubIssue[] = await issuesRes.json();

      const items: ActivityItem[] = [
        ...commits.map((c) => ({
          id: `commit-${c.sha}`,
          type: "commit" as const,
          title: c.commit.message.split("\n")[0],
          description: `${c.sha.slice(0, 7)} にコミット`,
          author: c.commit.author.name,
          avatarUrl: c.author?.avatar_url,
          timestamp: c.commit.author.date,
          url: c.html_url,
        })),
        ...pulls.map((p) => ({
          id: `pr-${p.id}`,
          type: "pull_request" as const,
          title: p.title,
          description: p.merged_at
            ? "PRがマージされました"
            : `PR #${p.number} ${p.state}`,
          author: p.user.login,
          avatarUrl: p.user.avatar_url,
          timestamp: p.updated_at,
          url: p.html_url,
        })),
        ...issues
          .filter((i) => !i.pull_request && !/^\[PR #\d+ placeholder\]$/.test(i.title))
          .map((i) => ({
            id: `issue-${i.id}`,
            type: "issue" as const,
            title: i.title,
            description: `Issue #${i.number} ${i.state === "open" ? "オープン" : "クローズ"}`,
            author: i.user.login,
            avatarUrl: i.user.avatar_url,
            timestamp: i.updated_at,
            url: i.html_url,
          })),
      ];

      return items.sort(
        (a, b) =>
          new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
      );
    },
  });
}
