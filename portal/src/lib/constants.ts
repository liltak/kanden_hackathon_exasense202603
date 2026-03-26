export const GITHUB_REPO =
  process.env.GITHUB_REPO ||
  "your-org/your-repo";

export const GITHUB_TOKEN = process.env.GITHUB_TOKEN || "";

export const GITHUB_API_BASE = "https://api.github.com";

export const [GITHUB_OWNER, GITHUB_REPO_NAME] = GITHUB_REPO.split("/");

export const NAV_ITEMS = [
  { href: "/", label: "ダッシュボード", icon: "LayoutDashboard" as const },
  { href: "/issues", label: "Issue", icon: "CircleDot" as const },
  { href: "/activity", label: "アクティビティ", icon: "Activity" as const },
];
