"use client";

import { useQuery } from "@tanstack/react-query";
import type {
  GitHubIssue,
  GitHubCommit,
  GitHubMilestone,
  DashboardStats,
} from "@/lib/types";

export function useDashboardStats() {
  return useQuery<DashboardStats>({
    queryKey: ["dashboard-stats"],
    queryFn: async () => {
      const [openRes, closedRes, commitsRes] = await Promise.all([
        fetch("/api/github/issues?state=open"),
        fetch("/api/github/issues?state=closed"),
        fetch(
          `/api/github/commits?since=${new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString()}`
        ),
      ]);

      const open: GitHubIssue[] = await openRes.json();
      const closed: GitHubIssue[] = await closedRes.json();
      const commits: GitHubCommit[] = await commitsRes.json();

      const total = open.length + closed.length;
      return {
        openIssues: open.length,
        closedIssues: closed.length,
        completionRate: total > 0 ? Math.round((closed.length / total) * 100) : 0,
        recentCommits: commits.length,
      };
    },
  });
}

export function useMilestones() {
  return useQuery<GitHubMilestone[]>({
    queryKey: ["milestones"],
    queryFn: async () => {
      const res = await fetch("/api/github/milestones");
      if (!res.ok) throw new Error("Failed to fetch milestones");
      return res.json();
    },
  });
}

export function useLabelDistribution() {
  return useQuery<{ name: string; count: number; color: string }[]>({
    queryKey: ["label-distribution"],
    queryFn: async () => {
      const res = await fetch("/api/github/issues?state=all");
      if (!res.ok) throw new Error("Failed to fetch issues");
      const issues: GitHubIssue[] = await res.json();

      const labelMap = new Map<string, { count: number; color: string }>();
      for (const issue of issues) {
        for (const label of issue.labels) {
          const existing = labelMap.get(label.name);
          if (existing) {
            existing.count++;
          } else {
            labelMap.set(label.name, { count: 1, color: `#${label.color}` });
          }
        }
      }

      return Array.from(labelMap.entries())
        .map(([name, { count, color }]) => ({ name, count, color }))
        .sort((a, b) => b.count - a.count);
    },
  });
}
