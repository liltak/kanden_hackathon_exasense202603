"use client";

import { usePathname } from "next/navigation";

const pageTitles: Record<string, string> = {
  "/": "ダッシュボード",
  "/issues": "Issue",
  "/activity": "アクティビティ",
};

export function Header() {
  const pathname = usePathname();
  const title =
    pageTitles[pathname] ||
    (pathname.startsWith("/issues/") ? "Issue詳細" : "ExaSense Portal");

  return (
    <header className="flex h-16 items-center border-b px-6">
      <h1 className="text-xl font-semibold">{title}</h1>
    </header>
  );
}
