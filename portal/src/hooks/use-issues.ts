"use client";

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import type { GitHubIssue, GitHubComment } from "@/lib/types";

export function useIssues(params?: {
  state?: string;
  labels?: string;
  page?: number;
}) {
  const searchParams = new URLSearchParams();
  if (params?.state) searchParams.set("state", params.state);
  if (params?.labels) searchParams.set("labels", params.labels);
  if (params?.page) searchParams.set("page", String(params.page));

  return useQuery<GitHubIssue[]>({
    queryKey: ["issues", params],
    queryFn: async () => {
      const res = await fetch(`/api/github/issues?${searchParams}`);
      if (!res.ok) throw new Error("Failed to fetch issues");
      return res.json();
    },
  });
}

export function useIssue(number: number) {
  return useQuery<GitHubIssue>({
    queryKey: ["issue", number],
    queryFn: async () => {
      const res = await fetch(`/api/github/issues/${number}`);
      if (!res.ok) throw new Error("Failed to fetch issue");
      return res.json();
    },
  });
}

export function useIssueComments(number: number) {
  return useQuery<GitHubComment[]>({
    queryKey: ["comments", number],
    queryFn: async () => {
      const res = await fetch(`/api/github/issues/${number}/comments`);
      if (!res.ok) throw new Error("Failed to fetch comments");
      return res.json();
    },
  });
}

export function usePostComment(number: number) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async ({ name, body }: { name: string; body: string }) => {
      const res = await fetch(`/api/github/issues/${number}/comments`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, body }),
      });
      if (!res.ok) throw new Error("Failed to post comment");
      return res.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["comments", number] });
    },
  });
}

export function useCreateIssue() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (params: {
      title: string;
      body: string;
      name: string;
      labels?: string[];
    }) => {
      const res = await fetch("/api/github/issues", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(params),
      });
      if (!res.ok) throw new Error("Failed to create issue");
      return res.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["issues"] });
    },
  });
}
