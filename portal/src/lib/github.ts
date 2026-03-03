import {
  GITHUB_API_BASE,
  GITHUB_OWNER,
  GITHUB_REPO_NAME,
  GITHUB_TOKEN,
} from "./constants";
import type {
  GitHubIssue,
  GitHubComment,
  GitHubCommit,
  GitHubPullRequest,
  GitHubMilestone,
} from "./types";

function headers() {
  return {
    Accept: "application/vnd.github.v3+json",
    Authorization: `Bearer ${GITHUB_TOKEN}`,
  };
}

function repoUrl(path: string) {
  return `${GITHUB_API_BASE}/repos/${GITHUB_OWNER}/${GITHUB_REPO_NAME}${path}`;
}

export async function fetchIssues(params?: {
  state?: string;
  labels?: string;
  page?: number;
  per_page?: number;
}): Promise<GitHubIssue[]> {
  const searchParams = new URLSearchParams();
  searchParams.set("state", params?.state || "all");
  searchParams.set("per_page", String(params?.per_page || 30));
  searchParams.set("page", String(params?.page || 1));
  searchParams.set("sort", "updated");
  searchParams.set("direction", "desc");
  if (params?.labels) searchParams.set("labels", params.labels);

  const res = await fetch(`${repoUrl("/issues")}?${searchParams}`, {
    headers: headers(),
  });
  if (!res.ok) throw new Error(`GitHub API error: ${res.status}`);
  const issues: GitHubIssue[] = await res.json();
  return issues.filter(
    (i) => !i.pull_request && !i.title.match(/^\[PR #\d+ placeholder\]$/)
  );
}

export async function fetchIssue(number: number): Promise<GitHubIssue> {
  const res = await fetch(repoUrl(`/issues/${number}`), {
    headers: headers(),
  });
  if (!res.ok) throw new Error(`GitHub API error: ${res.status}`);
  return res.json();
}

export async function fetchIssueComments(
  number: number
): Promise<GitHubComment[]> {
  const res = await fetch(repoUrl(`/issues/${number}/comments`), {
    headers: headers(),
  });
  if (!res.ok) throw new Error(`GitHub API error: ${res.status}`);
  return res.json();
}

export async function postIssueComment(
  number: number,
  body: string
): Promise<GitHubComment> {
  const res = await fetch(repoUrl(`/issues/${number}/comments`), {
    method: "POST",
    headers: { ...headers(), "Content-Type": "application/json" },
    body: JSON.stringify({ body }),
  });
  if (!res.ok) throw new Error(`GitHub API error: ${res.status}`);
  return res.json();
}

export async function createIssue(params: {
  title: string;
  body: string;
  labels?: string[];
}): Promise<GitHubIssue> {
  const res = await fetch(repoUrl("/issues"), {
    method: "POST",
    headers: { ...headers(), "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  if (!res.ok) throw new Error(`GitHub API error: ${res.status}`);
  return res.json();
}

export async function fetchCommits(params?: {
  since?: string;
  per_page?: number;
}): Promise<GitHubCommit[]> {
  const searchParams = new URLSearchParams();
  searchParams.set("per_page", String(params?.per_page || 30));
  if (params?.since) searchParams.set("since", params.since);

  const res = await fetch(`${repoUrl("/commits")}?${searchParams}`, {
    headers: headers(),
  });
  if (!res.ok) throw new Error(`GitHub API error: ${res.status}`);
  return res.json();
}

export async function fetchPulls(params?: {
  state?: string;
}): Promise<GitHubPullRequest[]> {
  const searchParams = new URLSearchParams();
  searchParams.set("state", params?.state || "all");
  searchParams.set("sort", "updated");
  searchParams.set("direction", "desc");

  const res = await fetch(`${repoUrl("/pulls")}?${searchParams}`, {
    headers: headers(),
  });
  if (!res.ok) throw new Error(`GitHub API error: ${res.status}`);
  return res.json();
}

export async function fetchMilestones(): Promise<GitHubMilestone[]> {
  const searchParams = new URLSearchParams();
  searchParams.set("state", "all");
  searchParams.set("sort", "due_on");

  const res = await fetch(`${repoUrl("/milestones")}?${searchParams}`, {
    headers: headers(),
  });
  if (!res.ok) throw new Error(`GitHub API error: ${res.status}`);
  return res.json();
}
