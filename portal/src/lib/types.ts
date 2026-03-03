export interface GitHubUser {
  login: string;
  avatar_url: string;
  html_url: string;
}

export interface GitHubLabel {
  id: number;
  name: string;
  color: string;
  description?: string;
}

export interface GitHubMilestone {
  id: number;
  number: number;
  title: string;
  description: string | null;
  state: "open" | "closed";
  open_issues: number;
  closed_issues: number;
  due_on: string | null;
  created_at: string;
  updated_at: string;
}

export interface GitHubIssue {
  id: number;
  number: number;
  title: string;
  body: string | null;
  state: "open" | "closed";
  labels: GitHubLabel[];
  assignees: GitHubUser[];
  user: GitHubUser;
  comments: number;
  milestone: GitHubMilestone | null;
  created_at: string;
  updated_at: string;
  closed_at: string | null;
  html_url: string;
  pull_request?: { url: string };
}

export interface GitHubComment {
  id: number;
  body: string;
  user: GitHubUser;
  created_at: string;
  updated_at: string;
  html_url: string;
}

export interface GitHubCommit {
  sha: string;
  commit: {
    message: string;
    author: {
      name: string;
      date: string;
    };
  };
  author: GitHubUser | null;
  html_url: string;
}

export interface GitHubPullRequest {
  id: number;
  number: number;
  title: string;
  state: "open" | "closed";
  user: GitHubUser;
  merged_at: string | null;
  created_at: string;
  updated_at: string;
  html_url: string;
  labels: GitHubLabel[];
}

export interface DashboardStats {
  openIssues: number;
  closedIssues: number;
  completionRate: number;
  recentCommits: number;
}

export interface ActivityItem {
  id: string;
  type: "commit" | "issue" | "pull_request";
  title: string;
  description: string;
  author: string;
  avatarUrl?: string;
  timestamp: string;
  url: string;
}

export interface AISummary {
  summary: string;
  generatedAt: string;
  periodStart: string;
  periodEnd: string;
}
